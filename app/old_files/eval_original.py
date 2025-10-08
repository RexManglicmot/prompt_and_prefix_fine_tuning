# app/eval.py
"""
Evaluate Base vs Prompt Tuning vs Prefix Tuning on PubMedQA.

Inputs
  - config.yaml
  - data/pubmedqa_{val,test}.csv  (id, question, contexts, final_decision)
  - outputs/{prompt_tuning-adapter, prefix_tuning-adapter}/   (trained adapters)

Outputs (under outputs/<run_name>/eval/)
  - preds_val.csv, preds_test.csv
  - metrics_val.json, metrics_test.json
  - cm_val.csv, cm_test.csv      (confusion matrices as tables)
"""

import os
import json
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from peft import PeftModel
from app.config import load_config


YES_TOKENS = ("yes", " Yes", "YES")
NO_TOKENS  = ("no", " No", "NO")


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_label(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith("yes"):
        return "yes"
    else:
        return "no"
    # if s.startswith("no"):            # Not dealing with unknown anymore
    #     return "no"
    # return "unknown"


@torch.no_grad()
def predict_yes_no(
    model: AutoModelForSeq2SeqLM,
    tok: AutoTokenizer,
    rows: pd.DataFrame,
    text_tmpl: str,
    max_input_len: int,
    gen_max_new_tokens: int = 3,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """
    Greedy-generate 1-3 tokens, map to 'yes'/'no'/unknown.    took out unknown
    Small, simple batches for CPU/MPS.
    """
    preds: List[str] = []
    model.eval()

    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start : start + batch_size]

        prompts = [
            text_tmpl.format(question=r["question"], contexts=r["contexts"])
            for _, r in chunk.iterrows()
        ]
        enc = tok(
            prompts,
            max_length=max_input_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model.generate(
            **enc,
            max_new_tokens=gen_max_new_tokens,
            do_sample=False,
        )
        texts = tok.batch_decode(out, skip_special_tokens=True)

        for t in texts:
            t = t.strip()
            low = t.lower()
            if low.startswith("yes"):
                preds.append("yes")
            elif low.startswith("no"):
                preds.append("no")
            elif any(t.startswith(y) for y in YES_TOKENS):  # sometimes models emit punctuation first; try simple fallbacks
                preds.append("yes")
            elif any(t.startswith(n) for n in NO_TOKENS):
                preds.append("no")
            else:
                # preds.append("unknown")                   # not dealring with unknown anymore
                pass

    return preds


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    y_true = [normalize_label(x) for x in y_true]
    y_pred = [normalize_label(x) for x in y_pred]

    # filter out "unknown" from predictions for binary metrics (optional)
    # Here, we keep them—they’ll just count as wrong unless the label is unknown.
    acc = accuracy_score(y_true, y_pred)

    # Use 'yes' as positive class
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["yes", "no"], average="binary", pos_label="yes", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision_yes": float(pr),
        "recall_yes": float(rc),
        "f1_yes": float(f1),
        "mcc": float(mcc),
        # "unknown_rate": float(np.mean([p == "unknown" for p in y_pred])),
    }


def cm_table(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
    labels = ["yes", "no"]                          # took out unknown
    y_true_n = [normalize_label(x) for x in y_true]
    y_pred_n = [normalize_label(x) for x in y_pred]
    cm = confusion_matrix(y_true_n, y_pred_n, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def load_backbone_and_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.backbone)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    return tok, model


def maybe_load_adapter(model, method: str, out_dir_root: str):
    """
    method: 'base' | 'prompt_tuning' | 'prefix_tuning'
    If not base, load adapter weights from outputs/<method>-adapter/
    """
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
    os.makedirs(save_dir, exist_ok=True)

    preds = predict_yes_no(
        model, tok, df,
        text_tmpl=cfg.data.text_template,
        max_input_len=cfg.data.max_input_length,
        gen_max_new_tokens=3,
        batch_size=8,
        device=device,
    )

    y_true = [normalize_label(x) for x in df["final_decision"].tolist()]
    y_pred = [normalize_label(x) for x in preds]

    # Save predictions
    out_preds = df[["id", "question", "contexts", "final_decision"]].copy()
    out_preds["prediction"] = y_pred
    out_preds.to_csv(os.path.join(save_dir, f"preds_{split_name}.csv"), index=False)

    # After built y_true and y_pred lists
    print("y_true uniques:", sorted(set(y_true)))
    print("y_pred uniques:", sorted(set(y_pred)))

    # Metrics
    metrics = compute_metrics(y_true, y_pred)
    with open(os.path.join(save_dir, f"metrics_{split_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix table
    cm_df = cm_table(y_true, y_pred)
    cm_df.to_csv(os.path.join(save_dir, f"cm_{split_name}.csv"))

    print(f"[{split_name}] accuracy={metrics['accuracy']:.4f} "
          f"f1_yes={metrics['f1_yes']:.4f} mcc={metrics['mcc']:.4f} ")
          # f"unknown_rate={metrics['unknown_rate']:.4f}")


def main():
    cfg = load_config()
    device = get_device()
    print(f"Eval device: {device}")

    # Load data
    val_df  = pd.read_csv(cfg.data.val_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    # Load backbone/tokenizer once
    tok, base_model = load_backbone_and_tokenizer(cfg)
    base_model.to(device)

    methods = []
    if getattr(cfg.core, "compare_base", False):
        methods.append("base")
    methods.extend([m for m in cfg.core.methods if m in {"prompt_tuning", "prefix_tuning"}])

    for method in methods:
        print(f"\n=== Evaluating: {method} ===")
        # Load adapter if needed
        model, save_root = maybe_load_adapter(base_model, method, cfg.project.output_dir)
        model.to(device)
        model.eval()

        # Each method gets its own eval/ folder
        os.makedirs(save_root, exist_ok=True)

        eval_split(cfg, "val",  val_df,  tok, model, save_root, device)
        eval_split(cfg, "test", test_df, tok, model, save_root, device)

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()

"""
Start 12:48

"""