# app/eval.py
"""
Evaluate Base vs Prompt Tuning vs Prefix Tuning on PubMedQA (binary yes/no).

Inputs
  - config.yaml
  - data/pubmedqa_{val,test}.csv  (id, question, contexts, final_decision)
  - outputs/{prompt_tuning-adapter, prefix_tuning-adapter}/   (trained adapters)

Outputs (under outputs/<run>/eval/)
  - preds_val.csv, preds_test.csv
  - metrics_val.json, metrics_test.json
  - cm_val.csv, cm_test.csv
"""

import os, json, re
from typing import List, Dict

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

LABELS = ["no", "yes"]  # fixed order for binary reports


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def norm_gold(s: str) -> str:
    s = (s or "").strip().lower()
    return "yes" if s == "yes" else "no"


def norm_pred(text: str) -> str:
    """
    Normalize model output strictly to {'yes','no'}.
    Anything else falls back to 'no' to keep binary length consistent.
    """
    s = (text or "").strip().lower()
    # quick wins for "yes", "yes.", "yes!" etc
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    # super-strict: strip non-alnum and re-check
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
    gen_max_new_tokens: int = 2,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """
    Greedy generate with small max_new_tokens and clamp to yes/no with norm_pred.
    Always appends exactly one label per row -> lengths match.
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
            num_beams=1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        texts = tok.batch_decode(out, skip_special_tokens=True)

        preds.extend(norm_pred(t) for t in texts)

    return preds


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    # All labels already normalized to {'yes','no'}
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
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])


def load_backbone_and_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model.backbone,
        low_cpu_mem_usage=True,
    )
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

    # Predict (one label per row)
    preds = predict_yes_no(
        model, tok, df,
        text_tmpl=cfg.data.text_template,
        max_input_len=cfg.data.max_input_length,
        gen_max_new_tokens=2,
        batch_size=8,
        device=device,
    )
    assert len(preds) == len(df), f"preds={len(preds)} vs rows={len(df)}"

    y_true = [norm_gold(x) for x in df["final_decision"].tolist()]
    y_pred = [p for p in preds]  # already normalized

    # Save predictions table
    out_preds = df[["id", "question", "contexts", "final_decision"]].copy()
    out_preds["prediction"] = y_pred
    out_preds.to_csv(os.path.join(save_dir, f"preds_{split_name}.csv"), index=False)

    # (Optional) quick debug:
    # print("y_true uniques:", sorted(set(y_true)))
    # print("y_pred uniques:", sorted(set(y_pred)))

    # Metrics + confusion matrix
    metrics = compute_metrics(y_true, y_pred)
    with open(os.path.join(save_dir, f"metrics_{split_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    cm_df = cm_table(y_true, y_pred)
    cm_df.to_csv(os.path.join(save_dir, f"cm_{split_name}.csv"))

    print(f"[{split_name}] n={len(df)} "
          f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} mcc={metrics['mcc']:.4f}")


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
        model, save_root = maybe_load_adapter(base_model, method, cfg.project.output_dir)
        model.to(device)
        model.eval()

        # Add this debug snippet
        print("[eval] active_adapter:", getattr(model, "active_adapter", None))
        print("[eval] peft_config keys:", list(getattr(model, "peft_config", {}).keys()))
        print()

        print(f"[eval] num_virtual_tokens: {cfg.peft.prompt_num_virtual_tokens if method == 'prompt_tuning' else cfg.peft.prefix_num_virtual_tokens}")
        print(f"[eval] lr_prompt: {cfg.train.lr_prompt}")
        print(f"[eval] lr_prefix: {cfg.train.lr_prefix}")
        print()
        os.makedirs(save_root, exist_ok=True)

        eval_split(cfg, "val",  val_df,  tok, model, save_root, device)
        eval_split(cfg, "test", test_df, tok, model, save_root, device)
        print()
        
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
