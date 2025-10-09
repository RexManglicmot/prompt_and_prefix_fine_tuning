# app/train.py
"""
Train PEFT adapters (Prompt Tuning / Prefix Tuning) on PubMedQA.

Reads:
  - config.yaml (paths, hyperparams, peft settings)
  - data/pubmedqa_{train,val}.csv (id, question, contexts, final_decision)

Writes:
  - outputs/<method>-adapter/  (PEFT weights + trainer checkpoints)
  - outputs/metrics.json (loss curve appended by eval.py later)
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") # GPU Issues
import json
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,                     # Loads tokenizer by name/checkpoint
    AutoModelForSeq2SeqLM,             # Loads encoder-decoder (T5/FLAN) model
    DataCollatorForSeq2Seq,            # Pads/aligns seq2seq batches (handles labels too)
    Seq2SeqTrainer as Trainer,         # Thin alias for HF Trainer specialized for seq2seq
    Seq2SeqTrainingArguments as TrainingArguments,  # Training hyperparameters container
    set_seed,                          # Reproducibility
)
from peft import (
    get_peft_model,                    # Wrap backbone with PEFT adapter
    PromptTuningConfig,                # Config for virtual prompt tokens
    PrefixTuningConfig,                # Config for prefix-tuning vectors
    TaskType,                          # Tells PEFT what head/task type to target
)
from app.config import load_config


# ---------------------------
# Small helpers
# ---------------------------

@dataclass
class RowExample:
    # Convenience type annotation for a row (not directly used by Trainer)
    question: str
    contexts: str
    final_decision: str  # "yes" | "no"


class PubMedQADataset(Dataset):
    # Minimal torch Dataset backed by pandas DataFrame.
    def __init__(self, df: pd.DataFrame, 
                 tokenizer: AutoTokenizer, 
                 text_tmpl: str,
                 max_input_len: int, 
                 max_target_len: int):
                     
        # Keep a clean, contiguous index (safer for __getitem__)
        self.df = df.reset_index(drop=True)
        
        # Single tokenizer used for both inputs and targets
        self.tok = tokenizer
        
        # Prompt template, e.g., "Question: {question}\nContext: {contexts}\nAnswer:"
        self.tmpl = text_tmpl
        
        # Truncation limits for encoder/decoder sides
        self.max_in = max_input_len
        self.max_tgt = max_target_len

    # Required by PyTorch Dataset, so it returns the total number of samples.
    # Here it's just the number of rows in the backing DataFrame (self.df).
    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Grab the idx-th example from the pandas DataFrame backing this dataset
        row = self.df.iloc[idx]
        
        # Fill the text template with this example's fields to build the model input
        # e.g., "Question: ... Abstract: ... Instruction: Answer yes or no ..."
        prompt = self.tmpl.format(question=row["question"], contexts=row["contexts"])
        
        # Normalize target into {yes,no}
        target = str(row["final_decision"]).strip().lower()  # "yes"/"no"
        
        # Tokenize encoder input (no padding here; collator pads per-batch)
        enc = self.tok(
            prompt,
            max_length=self.max_in,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        # Tokenize decoder target using modern API (replaces as_target_tokenizer())
        dec = self.tok(
            text_target=target,       
            max_length=self.max_tgt,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        # Return fields expected by HF seq2seq models:
        # - input_ids/attention_mask -> encoder
        # - labels -> decoder (Trainer will compute cross-entropy)
        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": dec["input_ids"][0],
        }
        
        return item


def count_trainable_params(model) -> int:
    # Sum parameters that require gradients (PEFT exposes only adapter params)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_device_info():
    # Check if cuda is available, and order of priority
    # cuda > mps > cpu
    # SHOULD PRINT CUDA
    if torch.cuda.is_available():
        # Helpful runtime diagnostics for GPU environment
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA detected: {name} ({vram:.1f} GB)")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) detected")
    else:
        print("CPU mode")


# ---------------------------
# Metrics helpers (no sklearn dependency)
# ---------------------------

def _normalize_yn(text: str) -> str:
    # Map any decoded string to 'yes' or 'no'
    t = (text or "").strip().lower()
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    # fallback to 'no' to avoid crashes
    return "no"

def _macro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Macro-F1 for two classes: 'no' and 'yes'.
    def f1_for(label: str) -> float:
        
        # Manual confusion terms per label
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        # precision/recall with safe zero-division handling
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
    # Average F1 across the two labels
    return 0.5 * (f1_for("no") + f1_for("yes"))

def compute_metrics_fn(eval_pred, tokenizer):
    # Decode predictions/labels and compute accuracy + macro_f1.
    # Works with Seq2SeqTrainer(predict_with_generate=True).
    preds_ids = eval_pred.predictions
    
    # some HF versions return (sequences, …)
    if isinstance(preds_ids, tuple): 
        preds_ids = preds_ids[0]

    # decode predictions (ids -> text)
    pred_texts = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)

    # Convert label IDs back to text strings for metric computation.
    # Hugging Face sets ignored label positions to -100 so the loss function skips them.
    # We must replace those -100s with a real token ID (the pad token) so decoding works.
    
    # Raw label IDs from the evaluator (shape: [batch, seq_len])
    labels = eval_pred.label_ids
    
    # Get a valid padding token ID (fallback to 0 if tokenizer has no explicit pad token)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Replace all ignore indices (-100) with the pad token ID so they can be decoded
    labels = np.where(labels == -100, pad_id, labels)
    
    # Turn token IDs back into strings, dropping special tokens like <pad>, </s>, etc.
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Normalize to {yes,no}
    y_pred = np.array([_normalize_yn(t) for t in pred_texts])
    y_true = np.array([_normalize_yn(t) for t in label_texts])

    # Simple accuracy + macro-F1 (binary)
    acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    macro_f1 = _macro_f1_binary(y_true, y_pred) if len(y_true) else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


# ---------------------------
# PEFT factory
# ---------------------------

def attach_peft(model, cfg, method: str):
    # Tell PEFT what head to adapt (seq2seq language modeling)
    task = TaskType.SEQ_2_SEQ_LM

    if method == "prompt_tuning":
        # Virtual tokens prepended to the input (frozen backbone)
        peft_cfg = PromptTuningConfig(
            task_type=task,
            num_virtual_tokens=cfg.peft.prompt_num_virtual_tokens,
        )
    elif method == "prefix_tuning":
        # Learnable key/value prefixes injected into each transformer layer
        peft_cfg = PrefixTuningConfig(
            task_type=task,
            num_virtual_tokens=cfg.peft.prefix_num_virtual_tokens,
        )
    else:
        raise ValueError(f"Unknown PEFT method: {method}")
    
    # Wrap the base model with the chosen adapter
    peft_model = get_peft_model(model, peft_cfg)
    
    # Print PEFT parameter counts (debug/visibility)
    peft_model.print_trainable_parameters()
    print(f"Trainable params: {count_trainable_params(peft_model):,}")
    return peft_model


# ---------------------------
# Main training entry
# ---------------------------

def train_one(cfg, method: str):
    # Defensive: only allow supported PEFT methods
    assert method in {"prompt_tuning", "prefix_tuning"}

    # Reproducibility: affects dataloader shuffling, dropout seeds, etc
    set_seed(cfg.project.seed)
    
    # Ensure output root exists
    os.makedirs(cfg.project.output_dir, exist_ok=True)

    # Log device info and which method model is training on
    print_device_info()
    print(f"Training method: {method}")

    # Load tokenizer & backbone (weights downloaded/cached by Transformers)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.backbone)

    # ---- hygiene fixes ----
    
    # keep padding consistent
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # disable cache for training/gradient checkpointing
    model.config.use_cache = False

    # optional mixed precision flags from config (HF Trainer reads these)
    use_fp16 = str(cfg.compute.mixed_precision).lower() == "fp16"
    use_bf16 = str(cfg.compute.mixed_precision).lower() == "bf16"

    # datasets (pre-split CSVs with id, question, contexts, final_decision)
    train_df = pd.read_csv(cfg.data.train_csv)
    val_df   = pd.read_csv(cfg.data.val_csv)

    # basic column assertions to catch schema issues early
    for need in ["id", "question", "contexts", "final_decision"]:
        assert need in train_df.columns and need in val_df.columns, f"Missing '{need}' in CSVs"

    # Build torch Datasets (string -> token IDs happens inside __getitem__)
    ds_train = PubMedQADataset(
        train_df, tokenizer, cfg.data.text_template,
        cfg.data.max_input_length, cfg.data.max_target_length
    )
    ds_val = PubMedQADataset(
        val_df, tokenizer, cfg.data.text_template,
        cfg.data.max_input_length, cfg.data.max_target_length
    )

    # Collator pads to multiples of 8 for better tensor-core utilization on GPU
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8) # IMPORTANT: tensor shapes align with GPU

    # attach PEFT adapter on top of the frozen backbone
    model = attach_peft(model, cfg, method)

    # training args
    out_dir = os.path.join(cfg.project.output_dir, f"{method}-adapter")
    
    # Had to change the lr to a float type
    lr = float(cfg.train.lr_prompt if method == "prompt_tuning" else cfg.train.lr_prefix)

    # HF training arguments (checkpointing, eval cadence, precision, optimizer, etc.)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        learning_rate=lr,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        evaluation_strategy="epoch",
        save_strategy=cfg.train.save_strategy,
        lr_scheduler_type=cfg.train.scheduler,
        warmup_ratio=cfg.train.warmup_ratio,
        fp16=use_fp16,                                      # hard coded here
        bf16=use_bf16,                                      # hard coded here
        optim=cfg.train.optimizer,
        predict_with_generate=True,                         # enable generation for metrics
        generation_max_length=cfg.data.max_target_length,   # ensure labels/outputs are comparable
        generation_num_beams=1,                             # greedy
        report_to=[],
    )

    # Build Trainer with datasets, collator, tokenizer, and custom metrics
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics_fn(p, tokenizer),  # per-epoch eval metrics
    )

    # Launch training loop (handles epochs, eval, and checkpointing)
    trainer.train()
    
    # saves adapter + config
    trainer.save_model(out_dir)

    # lightweight training summary (loss per epoch + eval_* from log_history)
    hist = getattr(trainer, "state", None)
    
    summary = {
        "method": method,
        "backbone": cfg.model.backbone,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "learning_rate": args.learning_rate,
        "trainable_params": count_trainable_params(model),
        "log_history": hist.log_history if hist else [],    # Trainer keeps step/epoch logs here
    }
    
    # Persist compact summary alongside the adapter for later plotting
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f" Finished training {method}. Artifacts → {out_dir}")


def main():
    # Load hyperparameters and paths from config.yaml (via app.config)
    cfg = load_config()

    # Decide device preference (optional; Trainer will still pick automatically)
    if cfg.compute.device == "cuda" and torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)  # keep default
    elif cfg.compute.device == "mps":
        pass  # Trainer handles
    # (no-op for cpu/auto)

    # CORE methods from config
    methods: List[str] = [m for m in cfg.core.methods if m in {"prompt_tuning", "prefix_tuning"}]
    if not methods:
        raise ValueError("No PEFT methods specified in config.core.methods")

    # Train each requested adapter (separate runs & output dirs)
    for m in methods:
        train_one(cfg, m)


if __name__ == "__main__":
    main()


# 8/31/25
# Start 7:33pm
# End 7:50 error

# Start 8:02pm
# ITS WORKING, saw the epochs and all that
# Nevermind error
# Had to go back to v4

# Start 8:21pm
# End 8:26 pm WOOOOOOOOO!!!!
# SOMETHING wrong

#9/1/25 try again
# Start 9:07am
# model is downloading on Instance
# optimizer error and float error
# Start: 9:43 am...working!!
# End 9:47am quick

# Use the GPU instances  in the vastai_instances/...GPU was much faster and cleaner

# 10/8/25
# run python3 -m app.train
# Start: 4:58 pm
# End: 5:15
# Took around 17 minutes
# Next, is eval.py
