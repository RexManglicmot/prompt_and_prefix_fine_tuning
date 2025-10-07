# app/plots.py
"""
Generate plots + summary tables from saved eval artifacts.

Reads (created by train.py and eval.py):
- outputs/base-eval/
- outputs/prompt_tuning-adapter/eval/
- outputs/prefix_tuning-adapter/eval/
- outputs/*/train_summary.json (for adapters)

Writes:
- outputs/viz/
    performance_bar_<split>.png
    latency_bar_<split>.png
    param_footprint_bar.png
    loss_curve.png
    eval_loss_per_epoch.png
    eval_accuracy_per_epoch.png
    eval_macro_f1_per_epoch.png
    loss_train_vs_eval_four_lines.png
    cm_<method>_<split>.png
    summary_<split>.csv
    summary_<split>_long.csv
"""

import os
import json
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

def find_adapter_root(method: str) -> Path | None:
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

def load_metrics_for_method(base_path: Path, split: str) -> dict:
    return safe_read_json(base_path / f"metrics_{split}.json")

def load_cm_for_method(base_path: Path, split: str) -> pd.DataFrame | None:
    p = base_path / f"cm_{split}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p, index_col=0)

def ensure_vizdir():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Plotting helpers (matplotlib only)
# --------------------------
def _label_bars(ax, rects, fmt="{:.2f}", fontsize=9):
    """
    Put value labels on top of bars.
    fmt can be like "{:.2f}", "{:,.0f}", "{:.0%}", etc.
    """
    for r in rects:
        h = r.get_height()
        if isinstance(h, (float, np.floating)) and np.isnan(h):
            continue
        ax.annotate(
            fmt.format(h),
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3),  # 3 points upward offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

def plot_performance_bar(rows: List[dict], split: str):
    names = [r["name"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    f1s  = [r["macro_f1"] for r in rows]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, accs, width, label="Accuracy")
    rects2 = ax.bar(x + width/2, f1s,  width, label="Macro-F1")

    # Add a bit of headroom so labels don't clip
    vals = [v for v in accs + f1s if not (isinstance(v, float) and np.isnan(v))]
    if vals:
        ymax = max(vals)
        if ymax <= 1.0:
            ax.set_ylim(0, max(1.05, ymax * 1.10))  # for 0..1 metrics
        else:
            ax.set_ylim(0, ymax * 1.10)

    ax.set_title(f"Performance ({split})")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()

    # Numeric labels (decimal). If you prefer percent, switch to "{:.0%}".
    _label_bars(ax, rects1, fmt="{:.2f}")
    _label_bars(ax, rects2, fmt="{:.2f}")

    fig.savefig(VIZ_DIR / f"performance_bar_{split}.png", dpi=160)
    plt.close(fig)

def plot_latency_bar(rows: List[dict], split: str):
    names  = [r["name"] for r in rows]
    lmeans = [r.get("latency_mean_s", np.nan) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    rects = ax.bar(np.arange(len(names)), lmeans)

    # Headroom
    vals = [v for v in lmeans if not (isinstance(v, float) and np.isnan(v))]
    if vals:
        ymax = max(vals)
        ax.set_ylim(0, ymax * 1.10 if ymax > 0 else 1.0)

    ax.set_title(f"Latency (mean seconds per example) – {split}")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("seconds / example")
    fig.tight_layout()

    # Labels: more precision if values are sub-second
    fmt = "{:.3f}" if vals and max(vals) < 1 else "{:.2f}"
    _label_bars(ax, rects, fmt=fmt)

    fig.savefig(VIZ_DIR / f"latency_bar_{split}.png", dpi=160)
    plt.close(fig)

def plot_param_footprint_bar(tuning_rows: List[dict]):
    # If no trainable params recorded, skip
    if not tuning_rows:
        return
    names = [r["name"] for r in tuning_rows]
    trainables = [r.get("trainable_params", 0) for r in tuning_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    rects = ax.bar(np.arange(len(names)), trainables)

    # Headroom
    vals = [v for v in trainables if not (isinstance(v, float) and np.isnan(v))]
    if vals:
        ymax = max(vals)
        ax.set_ylim(0, ymax * 1.10 if ymax > 0 else 1.0)

    ax.set_title("Adapter Trainable Parameters")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("parameters")
    fig.tight_layout()

    # Labels: integer with thousands separator
    _label_bars(ax, rects, fmt="{:,.0f}")

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

def _extract_epoch_logs(method: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: epoch, eval_loss, eval_accuracy, eval_macro_f1
    pulled from train_summary.json -> log_history for the given adapter method.
    """
    summ = load_train_summary(method)
    if not summ or "log_history" not in summ:
        return pd.DataFrame(columns=["epoch","eval_loss","eval_accuracy","eval_macro_f1"])

    rows = []
    for rec in summ["log_history"]:
        if ("eval_loss" in rec) or ("eval_accuracy" in rec) or ("eval_macro_f1" in rec):
            rows.append({
                "epoch":         rec.get("epoch", None),
                "eval_loss":     rec.get("eval_loss", np.nan),
                "eval_accuracy": rec.get("eval_accuracy", np.nan),
                "eval_macro_f1": rec.get("eval_macro_f1", np.nan),
            })
    df = pd.DataFrame(rows).dropna(subset=["epoch"]).sort_values("epoch")
    return df

def plot_eval_loss_per_epoch():
    dfs = []
    for method in ("prompt_tuning", "prefix_tuning"):
        df = _extract_epoch_logs(method)
        if not df.empty:
            dfs.append((method_pretty(method), df))

    if not dfs:
        return

    fig, ax = plt.subplots(figsize=(8,5))
    for name, df in dfs:
        ax.plot(df["epoch"], df["eval_loss"], label=name)
    ax.set_title("Eval Loss per Epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("eval_loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "eval_loss_per_epoch.png", dpi=160)
    plt.close(fig)

def plot_eval_scores_per_epoch():
    dfs = []
    for method in ("prompt_tuning", "prefix_tuning"):
        df = _extract_epoch_logs(method)
        if not df.empty:
            dfs.append((method_pretty(method), df))

    if not dfs:
        return

    # Accuracy
    fig, ax = plt.subplots(figsize=(8,5))
    for name, df in dfs:
        ax.plot(df["epoch"], df["eval_accuracy"], label=f"{name} – Accuracy")
    ax.set_title("Eval Accuracy per Epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "eval_accuracy_per_epoch.png", dpi=160)
    plt.close(fig)

    # Macro-F1
    fig, ax = plt.subplots(figsize=(8,5))
    for name, df in dfs:
        ax.plot(df["epoch"], df["eval_macro_f1"], label=f"{name} – Macro-F1")
    ax.set_title("Eval Macro-F1 per Epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("macro_f1")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "eval_macro_f1_per_epoch.png", dpi=160)
    plt.close(fig)

def _extract_train_loss_per_epoch(method: str) -> pd.DataFrame:
    """
    From train_summary.json -> log_history, take the last 'loss' logged for each epoch.
    Returns columns: epoch, loss
    """
    summ = load_train_summary(method)
    if not summ or "log_history" not in summ:
        return pd.DataFrame(columns=["epoch", "loss"])

    rows = []
    for rec in summ["log_history"]:
        if "loss" in rec and "epoch" in rec:
            rows.append({"epoch": rec["epoch"], "loss": rec["loss"]})
    if not rows:
        return pd.DataFrame(columns=["epoch", "loss"])

    df = pd.DataFrame(rows).sort_values("epoch")
    df = df.groupby("epoch", as_index=False).last()  # keep last loss per epoch
    return df

def plot_train_eval_loss_four_lines():
    """
    Single figure with 4 lines:
      - Prompt Tuning: Train loss (dashed), Eval loss (solid)
      - Prefix Tuning: Train loss (dashed), Eval loss (solid)
    """
    series = []
    for method in ("prompt_tuning", "prefix_tuning"):
        df_train = _extract_train_loss_per_epoch(method)
        df_eval  = _extract_epoch_logs(method)
        if not df_train.empty or not df_eval.empty:
            series.append((method_pretty(method), df_train, df_eval))

    if not series:
        return

    fig, ax = plt.subplots(figsize=(8,5))
    for name, df_train, df_eval in series:
        if not df_train.empty:
            ax.plot(df_train["epoch"], df_train["loss"], linestyle="--", label=f"{name} – Train loss")
        if not df_eval.empty:
            ax.plot(df_eval["epoch"], df_eval["eval_loss"], label=f"{name} – Eval loss")

    ax.set_title("Train vs Eval Loss per Epoch (Prompt & Prefix)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "loss_train_vs_eval_four_lines.png", dpi=160)
    plt.close(fig)

def plot_confusion_matrix(cm_df: pd.DataFrame | None, method: str, split: str):
    if cm_df is None:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm_df.values, aspect="auto")
    ax.set_title(f"Confusion Matrix – {method_pretty(method)} ({split})")
    ax.set_xticks(range(cm_df.shape[1]))
    ax.set_yticks(range(cm_df.shape[0]))
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticklabels(cm_df.index)
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
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Collect per-method metrics (macro + latency)
    rows = []
    long_rows = []  # per-class precision/recall/f1/support (tidy)
    for method, base_path in METHODS:
        m = load_metrics_for_method(base_path, SPLIT)
        if not m:
            continue

        name = method_pretty(method)
        # macro row
        rows.append({
            "method": method,
            "name": name,
            "accuracy": m.get("accuracy", np.nan),
            "macro_f1": m.get("macro_f1", np.nan),
            "mcc": m.get("mcc", np.nan),
            "latency_mean_s": m.get("latency_mean_s", np.nan),
            "latency_p95_s": m.get("latency_p95_s", np.nan),
            # flatten per-class into wide columns
            "no_precision":  m.get("per_class", {}).get("no",  {}).get("precision", np.nan),
            "no_recall":     m.get("per_class", {}).get("no",  {}).get("recall",    np.nan),
            "no_f1":         m.get("per_class", {}).get("no",  {}).get("f1",        np.nan),
            "no_support":    m.get("per_class", {}).get("no",  {}).get("support",   np.nan),
            "yes_precision": m.get("per_class", {}).get("yes", {}).get("precision", np.nan),
            "yes_recall":    m.get("per_class", {}).get("yes", {}).get("recall",    np.nan),
            "yes_f1":        m.get("per_class", {}).get("yes", {}).get("f1",        np.nan),
            "yes_support":   m.get("per_class", {}).get("yes", {}).get("support",   np.nan),
        })

        # tidy per-class rows
        for label in ("no", "yes"):
            cls = m.get("per_class", {}).get(label, {})
            long_rows.append({
                "method": method,
                "name": name,
                "split": SPLIT,
                "class": label,
                "precision": cls.get("precision", np.nan),
                "recall":    cls.get("recall",    np.nan),
                "f1":        cls.get("f1",        np.nan),
                "support":   cls.get("support",   np.nan),
            })

    # Save tables (now include per-class precision/recall)
    if rows:
        pd.DataFrame(rows).to_csv(VIZ_DIR / f"summary_{SPLIT}.csv", index=False)
    if long_rows:
        pd.DataFrame(long_rows).to_csv(VIZ_DIR / f"summary_{SPLIT}_long.csv", index=False)

    # Plots
    if rows:
        plot_performance_bar(rows, SPLIT)
        plot_latency_bar(rows, SPLIT)

    # Param footprint bar (adapters only; uses train_summary.json trainable_params)
    trows = []
    for method in ("prompt_tuning", "prefix_tuning"):
        summ = load_train_summary(method)
        if not summ:
            continue
        trows.append({
            "method": method,
            "name": method_pretty(method),
            "trainable_params": summ.get("trainable_params", 0),
        })
    plot_param_footprint_bar(trows)

    # Loss curve (adapters only)
    plot_loss_curve()

    # Epoch-wise eval curves (requires compute_metrics in train.py)
    plot_eval_loss_per_epoch()
    plot_eval_scores_per_epoch()

    # Combined 4-line plot (train vs eval for both methods)
    plot_train_eval_loss_four_lines()

    # Confusion matrices per method
    for method, base_path in METHODS:
        cm_df = load_cm_for_method(base_path, SPLIT)
        plot_confusion_matrix(cm_df, method, SPLIT)

    print(f"✅ Plots and summary tables saved to: {VIZ_DIR.resolve()}")

if __name__ == "__main__":
    main()
