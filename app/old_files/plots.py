# app/plots.py
"""
Generate plots + a summary table from saved eval artifacts.

Reads (created by train.py and eval.py):
- outputs/base-eval/eval/ (actually base-eval/ has files directly)
- outputs/prompt_tuning-adapter/eval/
- outputs/prefix_tuning-adapter/eval/
- outputs/*/train_summary.json (for adapters)

Writes:
- outputs/viz/
    performance_bar_<split>.png
    latency_bar_<split>.png
    param_footprint_bar.png
    loss_curve.png
    cm_<method>_<split>.png
    summary_<split>.csv
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Config-like constants
# --------------------------
PROJECT_OUT = Path("outputs")
SPLIT = "test"   # "val" or "test"
VIZ_DIR = PROJECT_OUT / "viz"
METHODS = [
    ("base",            PROJECT_OUT / "base-eval"),
    ("prompt_tuning",   PROJECT_OUT / "prompt_tuning-adapter" / "eval"),
    ("prefix_tuning",   PROJECT_OUT / "prefix_tuning-adapter" / "eval"),
]

# --------------------------
# Helpers to load artifacts
# --------------------------
def read_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)

def safe_read_json(p: Path) -> dict:
    return read_json(p) if p.is_file() else {}

def method_pretty(name: str) -> str:
    return {
        "base": "Base",
        "prompt_tuning": "Prompt Tuning",
        "prefix_tuning": "Prefix Tuning",
    }.get(name, name)

def find_adapter_root(method: str) -> Path:
    if method == "prompt_tuning":
        return PROJECT_OUT / "prompt_tuning-adapter"
    if method == "prefix_tuning":
        return PROJECT_OUT / "prefix_tuning-adapter"
    return None

def load_train_summary(method: str) -> dict:
    root = find_adapter_root(method)
    if not root: 
        return {}
    p = root / "train_summary.json"
    return safe_read_json(p)

def count_total_params_from_any_adapter() -> int:
    """
    We didn't store total model params in train_summary.
    As a reasonable proxy, read one adapter's train_summary to get 'trainable_params'
    but we still need TOTAL. If not available anywhere, return 0 and plot will skip.
    """
    for m in ("prompt_tuning", "prefix_tuning"):
        d = load_train_summary(m)
        if d:
            # Try to infer total from well-known ratios if present, else 0.
            # (We didn't save total params explicitly; so we’ll skip if unknown.)
            return d.get("all_params", 0)  # only present if you added it later
    return 0

def load_metrics_for_method(method: str, base_path: Path, split: str) -> dict:
    # base-eval puts files directly in folder; adapters put them under /eval
    metrics = safe_read_json(base_path / f"metrics_{split}.json")
    return metrics

def load_cm_for_method(method: str, base_path: Path, split: str) -> pd.DataFrame:
    p = base_path / f"cm_{split}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p, index_col=0)

def ensure_vizdir():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Plotting helpers
# (matplotlib only, one chart per figure, no custom colors)
# --------------------------
def plot_performance_bar(rows: List[dict], split: str):
    names = [r["name"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    f1s  = [r["macro_f1"] for r in rows]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, f1s,  width, label="Macro-F1")
    ax.set_title(f"Performance ({split})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / f"performance_bar_{split}.png", dpi=160)
    plt.close(fig)

def plot_latency_bar(rows: List[dict], split: str):
    names = [r["name"] for r in rows]
    lmeans = [r.get("latency_mean_s", np.nan) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(names)), lmeans)
    ax.set_title(f"Latency (mean seconds per example) – {split}")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("seconds / example")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / f"latency_bar_{split}.png", dpi=160)
    plt.close(fig)

def plot_param_footprint_bar(tuning_rows: List[dict], total_params_hint: int = 0):
    """
    Show trainable params vs total (if we can infer total).
    For Base, trainable = 0 (no adapters).
    """
    names = [r["name"] for r in tuning_rows]
    trainables = [r.get("trainable_params", 0) for r in tuning_rows]
    totals = []

    # Prefer the per-method 'all_params' if present; otherwise fallback to hint; else skip.
    for r in tuning_rows:
        ap = r.get("all_params", 0)
        if ap:
            totals.append(ap)
        elif total_params_hint:
            totals.append(total_params_hint)
        else:
            totals.append(0)

    # If all totals are zero, skip plotting.
    if not any(totals):
        return

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, totals, width, label="Total params")
    ax.bar(x + width/2, trainables, width, label="Trainable params")
    ax.set_title("Parameter Footprint (adapters)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("parameters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "param_footprint_bar.png", dpi=160)
    plt.close(fig)

def plot_loss_curve():
    """
    Parse 'log_history' from each adapter's train_summary.json and plot smoothed loss.
    """
    curves = []
    for method in ("prompt_tuning", "prefix_tuning"):
        summ = load_train_summary(method)
        if not summ or "log_history" not in summ:
            continue
        # Extract (step, loss)
        xs, ys = [], []
        step = 0
        for rec in summ["log_history"]:
            if "loss" in rec:
                step += 1
                xs.append(step)
                ys.append(rec["loss"])
        if xs:
            curves.append((method_pretty(method), np.array(xs), np.array(ys)))

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, xs, ys in curves:
        # light smoothing
        if len(ys) >= 5:
            k = 5
            ys = np.convolve(ys, np.ones(k)/k, mode="valid")
            xs = xs[:len(ys)]
        ax.plot(xs, ys, label=name)
    ax.set_title("Training Loss (smoothed)")
    ax.set_xlabel("steps (proxy)")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "loss_curve.png", dpi=160)
    plt.close(fig)

def plot_confusion_matrix(cm_df: pd.DataFrame, method: str, split: str):
    if cm_df is None:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm_df.values, aspect="auto")
    ax.set_title(f"Confusion Matrix – {method_pretty(method)} ({split})")
    ax.set_xticks(range(cm_df.shape[1]))
    ax.set_yticks(range(cm_df.shape[0]))
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticklabels(cm_df.index)
    # annotate
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, int(cm_df.values[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / f"cm_{method}_{split}.png", dpi=160)
    plt.close(fig)

# --------------------------
# Main
# --------------------------
def main():
    ensure_vizdir()

    # Collect per-method metrics
    rows = []
    for method, base_path in METHODS:
        # For base-eval, metrics live directly in folder (no nested eval/)
        # For adapters, you've already pointed to .../eval/ above.
        m = load_metrics_for_method(method, base_path, SPLIT)
        if not m:
            # skip missing method silently
            continue
        rows.append({
            "method": method,
            "name": method_pretty(method),
            "accuracy": m.get("accuracy", np.nan),
            "macro_f1": m.get("macro_f1", np.nan),
            "mcc": m.get("mcc", np.nan),
            "latency_mean_s": m.get("latency_mean_s", np.nan),
            "latency_p95_s": m.get("latency_p95_s", np.nan),
        })

    # Performance bars
    if rows:
        plot_performance_bar(rows, SPLIT)
        plot_latency_bar(rows, SPLIT)

    # Param footprint bar (for adapters only)
    trows = []
    for method in ("prompt_tuning", "prefix_tuning"):
        summ = load_train_summary(method)
        if not summ:
            continue
        trows.append({
            "method": method,
            "name": method_pretty(method),
            "trainable_params": summ.get("trainable_params", 0),
            "all_params": summ.get("all_params", 0),  # likely absent unless you added it
        })
    plot_param_footprint_bar(trows, total_params_hint=count_total_params_from_any_adapter())

    # Loss curve
    plot_loss_curve()

    # Confusion matrices per method
    for method, base_path in METHODS:
        cm_df = load_cm_for_method(method, base_path, SPLIT)
        plot_confusion_matrix(cm_df, method, SPLIT)

    # Summary table
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(VIZ_DIR / f"summary_{SPLIT}.csv", index=False)

    print(f"✅ Plots and summary saved to: {VIZ_DIR.resolve()}")

if __name__ == "__main__":
    main()
