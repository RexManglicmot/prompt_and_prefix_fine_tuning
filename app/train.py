# app/train.py
"""
Train PEFT adapters (Prompt Tuning / Prefix Tuning) on PubMedQA.

Reads:
  - config.yaml (paths, hyperparams, peft settings)
  - data/pubmedqa_{train,val}.csv (id, question, context, label)

Writes:
  - outputs/<method>-adapter/  (PEFT weights + trainer checkpoints)
  - outputs/metrics.json (loss curve appended by eval.py later)
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import (
    get_peft_model,
    PromptTuningConfig,
    PrefixTuningConfig,
    TaskType,
)

from app.config import load_config


# ---------------------------
# Small helpers
# ---------------------------

@dataclass
class RowExample:
    question: str
    context: str
    label: str  # "yes" | "no"


class PubMedQADataset(Dataset):
    """Minimal torch Dataset backed by pandas DataFrame."""
    def __init__(self, df: pd.DataFrame, 
                 tokenizer: AutoTokenizer, 
                 text_tmpl: str,
                 max_input_len: int, 
                 max_target_len: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.tmpl = text_tmpl
        self.max_in = max_input_len
        self.max_tgt = max_target_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        prompt = self.tmpl.format(question=row["question"], context=row["contexts"])
        target = str(row["final_decision"]).strip().lower()  # "yes"/"no"

        enc = self.tok(
            prompt,
            max_length=self.max_in,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        with self.tok.as_target_tokenizer():
            dec = self.tok(
                target,
                max_length=self.max_tgt,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )

        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "final_decision": dec["input_ids"][0],
        }
        return item


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_device_info():
    """
    Check if cuda is available, and order of priorityL
    cuda > mps > cpu
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA detected: {name} ({vram:.1f} GB)")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) detected")
    else:
        print("CPU mode")


# ---------------------------
# PEFT factory
# ---------------------------

def attach_peft(model, cfg, method: str):
    task = TaskType.SEQ_2_SEQ_LM

    if method == "prompt_tuning":
        peft_cfg = PromptTuningConfig(
            task_type=task,
            num_virtual_tokens=cfg.peft.prompt_num_virtual_tokens,
        )
    elif method == "prefix_tuning":
        peft_cfg = PrefixTuningConfig(
            task_type=task,
            num_virtual_tokens=cfg.peft.prefix_num_virtual_tokens,
        )
    else:
        raise ValueError(f"Unknown PEFT method: {method}")

    peft_model = get_peft_model(model, peft_cfg)
    peft_model.print_trainable_parameters()
    print(f"Trainable params: {count_trainable_params(peft_model):,}")
    return peft_model


# ---------------------------
# Main training entry
# ---------------------------

def train_one(cfg, method: str):
    assert method in {"prompt_tuning", "prefix_tuning"}

    set_seed(cfg.project.seed)
    os.makedirs(cfg.project.output_dir, exist_ok=True)

    print_device_info()
    print(f"➡️  Training method: {method}")

    # tokenizer & backbone
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.backbone)

    # optional mixed precision flags from config
    use_fp16 = str(cfg.compute.mixed_precision).lower() == "fp16"
    use_bf16 = str(cfg.compute.mixed_precision).lower() == "bf16"

    # datasets
    train_df = pd.read_csv(cfg.data.train_csv)
    val_df   = pd.read_csv(cfg.data.val_csv)

    # basic column assertions
    for need in ["id", "question", "context", "label"]:
        assert need in train_df.columns and need in val_df.columns, f"Missing '{need}' in CSVs"

    ds_train = PubMedQADataset(
        train_df, tokenizer, cfg.data.text_template,
        cfg.data.max_input_length, cfg.data.max_target_length
    )
    ds_val = PubMedQADataset(
        val_df, tokenizer, cfg.data.text_template,
        cfg.data.max_input_length, cfg.data.max_target_length
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # attach PEFT adapter
    model = attach_peft(model, cfg, method)

    # training args
    out_dir = os.path.join(cfg.project.output_dir, f"{method}-adapter")
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        learning_rate=(cfg.train.lr_prompt if method == "prompt_tuning" else cfg.train.lr_prefix),
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        evaluation_strategy="epoch",
        save_strategy=cfg.train.save_strategy,
        lr_scheduler_type=cfg.train.scheduler,
        warmup_ratio=cfg.train.warmup_ratio,
        fp16=use_fp16,
        bf16=use_bf16,
        predict_with_generate=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)  # saves adapter + config

    # lightweight training summary (loss per epoch)
    hist = getattr(trainer, "state", None)
    summary = {
        "method": method,
        "backbone": cfg.model.backbone,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "learning_rate": args.learning_rate,
        "trainable_params": count_trainable_params(model),
        "log_history": hist.log_history if hist else [],
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Finished training {method}. Artifacts → {out_dir}")


def main():
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

    for m in methods:
        train_one(cfg, m)


if __name__ == "__main__":
    main()
