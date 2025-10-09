# app/eval.py
"""
Evaluate Base vs Prompt Tuning vs Prefix Tuning on PubMedQA (binary yes/no).

Inputs
  - config.yaml
  - data/pubmedqa_{val,test}.csv  (id, question, contexts, final_decision)
  - outputs/{prompt_tuning-adapter, prefix_tuning-adapter}/trained adapters)

Outputs (under outputs/<run>/eval/)
  - preds_val.csv, preds_test.csv  (adds latency_s)
  - metrics_val.json, metrics_test.json  (adds latency_mean_s, latency_p95_s)
  - cm_val.csv, cm_test.csv
"""

import os, json, re, time
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from app.config import load_config

# Hard coded labels for fixed order for binary reports
# Used in compute_metrics() and cm_table()
LABELS = ["no", "yes"]

# ---------------------------
# Small helpers
# ---------------------------

def get_device() -> torch.device:
    # Check if cuda is available, and order of priority
    # cuda > mps > cpu
    # SHOULD PRINT CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def norm_gold(s: str) -> str:
    # Normalize gold labels to exactly 'yes' or 'no'.
    # Any non-'yes' text is coerced to 'no' to keep the task strictly binary.
    
    # strip of whitespace and lowercase s
    s = (s or "").strip().lower()
    return "yes" if s == "yes" else "no"


def norm_pred(text: str) -> str:
    # Normalize model output strictly to {'yes','no'}.
    # Strategy:
      #- direct prefix match ('yes...' or 'no...')
      #- alphanumeric-only prefix match (strip punctuation/spaces)
      #- otherwise fallback to 'no' (to keep alignment with rows)
      
    s = (text or "").strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    zs = re.sub(r"[^a-z0-9]+", "", s)
    if zs.startswith("yes"):
        return "yes"
    if zs.startswith("no"):
        return "no"
    return "no"  # fallback keeps preds aligned with rows


@torch.no_grad()
def predict_yes_no(
    model: AutoModelForSeq2SeqLM,
    tok: AutoTokenizer,
    rows: pd.DataFrame,
    text_tmpl: str,
    max_input_len: int,
    gen_max_new_tokens: int = 2,    # hard-coded # cap generation to 2 new tokens because this task only needs "yes" or "no".
    batch_size: int = 8,            # hard-coded # latency depends on batch size; 8 is a safe, reasonably fast default that avoids OOM.
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[str], List[float]]:
     
    # Run batched generation to get 'yes'/'no' predictions and per-example latency.
    #   - Formats each row into a prompt using text_tmpl
    #   - Tokenizes inputs with truncation/padding
    #   - Greedy generate a few tokens (max_new_tokens small for speed)
    #   - Decode -> normalize to {'yes','no'} with norm_pred
    #   - Measure elapsed time per batch and attribute evenly to examples
    # Returns:
    #   preds: list of normalized labels (len == len(rows))
    #   latencies: list of per-example seconds (same length)
    
    
    # Created empty lists for preds and latencies that will eventually be filled up
    preds: List[str] = []
    latencies: List[float] = []
    
    # Switch to inference mode: disables Dropout and other train-time behaviors for stable outputs.
    model.eval()

    # Iterate in batches to avoid OOM and to measure throughput
    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start : start + batch_size]

        # Render prompts from template using question + contexts
        prompts = [
            text_tmpl.format(question=r["question"], contexts=r["contexts"])
            for _, r in chunk.iterrows()
        ]

        # Tokenize inputs (truncate to fit the model's max_input_len)
        enc = tok(
            prompts,
            max_length=max_input_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        
        # Move the tokenized batch (e.g., input_ids, attention_mask) onto the model’s device (GPU/CPU).
        # Prevents “tensors on different devices” errors and enables GPU acceleration; .to(device) is a no-op if already there.
        enc = {k: v.to(device) for k, v in enc.items()}

        # Timer end
        t0 = time.time()
        
        # Generate with deterministic settings (greedy, 1 beam)
        out = model.generate(
            **enc,
            max_new_tokens=gen_max_new_tokens,
            do_sample=False,
            num_beams=1,                    # Greedy decoding (no beam search): fastest, deterministic,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        
        # Timer end
        elapsed = time.time() - t0
        
        # Attribute the measured batch time evenly across items so each row gets a per-example latency.
        # Approximate per-item latency: batch wall time ÷ batch size (good for comparisons, not exact).
        per_example = float(elapsed) / len(chunk)

        # Decode generated token IDs into strings (e.g., "yes", "no"), skipping special tokens.
        texts = tok.batch_decode(out, skip_special_tokens=True)
        
        # Normalize any free-form outputs to {'yes','no'} labels for consistent metrics.
        preds.extend(norm_pred(t) for t in texts)
        
        # Store one latency value per example so preds[i] aligns with latencies[i].
        latencies.extend([per_example] * len(chunk))

    # Return parallel lists: predictions and per-example latencies (same length as input rows).
    return preds, latencies


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    # Compute headline and per-class metrics on binary labels:
    #   - Accuracy, Macro-F1, MCC
    #   - Per-class precision/recall/F1/support for 'no' and 'yes'
    # Assumes y_true and y_pred are already normalized to {'yes','no'}.
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "per_class": {
            "no":  {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f1[0]), "support": int(support[0])},
            "yes": {"precision": float(pr[1]), "recall": float(rc[1]), "f1": float(f1[1]), "support": int(support[1])},
        },
    }


def cm_table(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
    # Build a labeled confusion-matrix DataFrame with fixed label order.
    #  - rows  = true_no, true_yes
    #  - cols  = pred_no, pred_yes
    
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])


def load_backbone_and_tokenizer(cfg):
    # Load tokenizer and backbone model (no adapters).
    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model.backbone,
        low_cpu_mem_usage=True,
    )
    # Set pad_token_id and disable KV cache for training/eval compatibility.
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    return tok, model


def maybe_load_adapter(model, method: str, out_dir_root: str):
    # If method == 'base': return the raw backbone and a base eval folder.
    # Else: wrap the backbone with a PEFT adapter loaded from outputs/<method>-adapter/.
    # Returns the (possibly wrapped) model and the path to the eval output dir.
   
    if method == "base":
        return model, os.path.join(out_dir_root, "base-eval")
    adapter_dir = os.path.join(out_dir_root, f"{method}-adapter")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    return model, os.path.join(adapter_dir, "eval")


def eval_split(
    cfg,
    split_name: str,
    df: pd.DataFrame,
    tok: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    save_dir: str,
    device: torch.device,
):
    # Evaluate one split ('val' or 'test'):
    #   - generate predictions + per-example latency
    #   - compute metrics + confusion matrix + latency summary
    #   - save preds CSV, metrics JSON, and CM CSV under save_dir
    #   - print a concise summary line
    
    # Make dir if it does not already exist
    os.makedirs(save_dir, exist_ok=True)

    # Predict (one label per row) + latency
    preds, latencies = predict_yes_no(
        model, tok, df,
        text_tmpl=cfg.data.text_template,
        max_input_len=cfg.data.max_input_length,
        gen_max_new_tokens=2,
        batch_size=8,
        device=device,
    )
    
    # Sanity checks: lengths must match the dataframe
    assert len(preds) == len(df), f"preds={len(preds)} vs rows={len(df)}"
    assert len(latencies) == len(df), f"latencies={len(latencies)} vs rows={len(df)}"

    # Normalize gold labels and pair with predictions
    y_true = [norm_gold(x) for x in df["final_decision"].tolist()]
    
    # already normalized
    y_pred = [p for p in preds]

    # Save predictions table (+ latency_s) for downstream analysis
    out_preds = df[["id", "question", "contexts", "final_decision"]].copy()
    out_preds["prediction"] = y_pred
    out_preds["latency_s"] = latencies
    out_preds.to_csv(os.path.join(save_dir, f"preds_{split_name}.csv"), index=False)

    # Metrics + confusion matrix (+ latency summaries)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "latency_mean_s": float(np.mean(latencies)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
    })
    with open(os.path.join(save_dir, f"metrics_{split_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix as CSV
    cm_df = cm_table(y_true, y_pred)
    cm_df.to_csv(os.path.join(save_dir, f"cm_{split_name}.csv"))

    # Print one-line summary
    print(f"[{split_name}] n={len(df)} "
          f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"mcc={metrics['mcc']:.4f} "
          f"lat_mean={metrics['latency_mean_s']:.3f}s p95={metrics['latency_p95_s']:.3f}s")


def main():
    # Orchestrate evaluation:
    #   - Load config and choose device
    #   - Read val/test CSVs
    #   - Load backbone/tokenizer once
    #   - For each method (base/prompt_tuning/prefix_tuning):
    #       * maybe wrap backbone with adapter
    #       * evaluate on val and test
    #       * write outputs under outputs/<method>-adapter/eval (or outputs/base-eval)
    
    cfg = load_config()
    device = get_device()
    print(f"Eval device: {device}")

    # Load data
    val_df  = pd.read_csv(cfg.data.val_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    # Load backbone/tokenizer once (reused for all methods)
    tok, base_model = load_backbone_and_tokenizer(cfg)
    base_model.to(device)

    methods = []
    if getattr(cfg.core, "compare_base", False):
        methods.append("base")
    methods.extend([m for m in cfg.core.methods if m in {"prompt_tuning", "prefix_tuning"}])

    # Decide which methods to evaluate
    for method in methods:
        print(f"\n=== Evaluating: {method} ===")
        
        # Load adapter if needed and pick save root for outputs
        model, save_root = maybe_load_adapter(base_model, method, cfg.project.output_dir)
        model.to(device)
        model.eval()
        os.makedirs(save_root, exist_ok=True)

        # Echo minimal run settings for reproducibility/context
        print(f"[eval] lr_prompt: {cfg.train.lr_prompt}")
        print(f"[eval] lr_prefix: {cfg.train.lr_prefix}")
        print(f"[eval] epochs: {cfg.train.epochs}")
        nv = None
        if hasattr(model, "peft_config") and "default" in model.peft_config:
            nv = getattr(model.peft_config["default"], "num_virtual_tokens", None)
        print("[eval] num_virtual_tokens:", nv if nv is not None else "NONE")
        print()
        
        # Evaluate both splits and write artifacts
        eval_split(cfg, "val",  val_df,  tok, model, save_root, device)
        eval_split(cfg, "test", test_df, tok, model, save_root, device)
        print()

    print("\n Evaluation complete.")


if __name__ == "__main__":
    main()

# Run python3 -m app.eval
# Was quick
# Next, is stats.eval.py
