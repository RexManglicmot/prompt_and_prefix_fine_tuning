# app/stats_eval.py
# Minimal stats on top of existing eval outputs: Base vs Prompt/Prefix

import os, json
import numpy as np
import pandas as pd
from math import comb
from app.config import load_config  

def _exists(p): 
    # Tiny helper: True if file exists at path p.
    return os.path.isfile(p)

def _mcnemar_exact_two_sided(n01: int, n10: int) -> float:
    # Exact two-sided McNemar p-value.
    # n01: base wrong, tuned right (fixes)
    # n10: base right, tuned wrong (regressions)
    n = n01 + n10
    if n == 0: 
        return 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k+1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels) -> float:
    # Macro-F1 over the provided label order (e.g., ['no','yes']).
    f1s = []
    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab))
        fp = np.sum((y_true != lab) & (y_pred == lab))
        fn = np.sum((y_true == lab) & (y_pred != lab))
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1s.append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0)
    return float(np.mean(f1s))

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Simple accuracy over string labels.
    return float(np.mean(y_true == y_pred))

def _bootstrap_diff(y_true, y_base, y_tuned, *, B: int, seed: int, labels):
    # Paired bootstrap to get mean & 95% CI for:
    #   - Base/Tuned Accuracy and their difference
    #   - Base/Tuned Macro-F1 and their difference
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc_b = np.empty(B); acc_t = np.empty(B); acc_d = np.empty(B)
    f1_b  = np.empty(B); f1_t  = np.empty(B); f1_d  = np.empty(B)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt, yb, yu = y_true[idx], y_base[idx], y_tuned[idx]
        ab, at = _accuracy(yt, yb), _accuracy(yt, yu)
        fb, ft = _macro_f1(yt, yb, labels), _macro_f1(yt, yu, labels)
        acc_b[_], acc_t[_], acc_d[_] = ab, at, (at - ab)
        f1_b[_],  f1_t[_],  f1_d[_]  = fb, ft, (ft - fb)

    def ci(a):
        lo, hi = np.quantile(a, [0.025, 0.975])
        return float(a.mean()), float(lo), float(hi)

    return {
        "acc_base":  ci(acc_b),
        "acc_tuned": ci(acc_t),
        "acc_diff":  ci(acc_d),
        "f1_base":   ci(f1_b),
        "f1_tuned":  ci(f1_t),
        "f1_diff":   ci(f1_d),
    }

def _load_preds_csv(path):
    # Load a predictions CSV with columns: id, final_decision, prediction.
    # Normalize labels/preds to {'yes','no'} so comparisons are consistent.
    df = pd.read_csv(path, usecols=["id", "final_decision", "prediction"])

    def ny(x):
        s = str(x).strip().lower()
        if s.startswith("yes"): return "yes"
        if s.startswith("no"):  return "no"
        return "no"

    df["final_decision"] = df["final_decision"].map(
        lambda s: "yes" if str(s).strip().lower() == "yes" else "no"
    )
    df["prediction"] = df["prediction"].map(ny)
    return df

def compare_one(*, project_out: str, split: str, tuned_method: str, labels, B: int, seed: int):
    # For one split and one tuned method:
    #   - Align Base vs Tuned predictions on the same items (id, final_decision)
    #   - Compute McNemar stats + paired bootstrap CIs
    #   - Return a dict row for CSV; also print a concise summary
    
    base_dir  = os.path.join(project_out, "base-eval")
    tuned_dir = os.path.join(project_out, f"{tuned_method}-adapter", "eval")
    base_csv  = os.path.join(base_dir,  f"preds_{split}.csv")
    tuned_csv = os.path.join(tuned_dir, f"preds_{split}.csv")

    if not (_exists(base_csv) and _exists(tuned_csv)):
        print(f"[{split}][{tuned_method}] missing preds; skip")
        return None

    a = _load_preds_csv(base_csv).rename(columns={"prediction": "prediction_base"})
    b = _load_preds_csv(tuned_csv).rename(columns={"prediction": f"prediction_{tuned_method}"})
    m = a.merge(b, on=["id", "final_decision"], how="inner")
    if len(m) == 0:
        print(f"[{split}][{tuned_method}] no overlapping IDs; check files.")
        return None

    y  = m["final_decision"].values
    yb = m["prediction_base"].values
    yt = m[f"prediction_{tuned_method}"].values

    base_ok  = (yb == y)
    tuned_ok = (yt == y)
    n01 = int((~base_ok &  tuned_ok).sum())  # base wrong, tuned right
    n10 = int(( base_ok & ~tuned_ok).sum())  # base right, tuned wrong
    N   = len(m)
    p   = _mcnemar_exact_two_sided(n01, n10)
    delta_acc = (n01 - n10) / N

    boots = _bootstrap_diff(y_true=y, y_base=yb, y_tuned=yt, B=B, seed=seed, labels=labels)
    acc_b = boots["acc_base"];   acc_t = boots["acc_tuned"];  acc_d = boots["acc_diff"]
    f1_b  = boots["f1_base"];    f1_t  = boots["f1_tuned"];   f1_d  = boots["f1_diff"]

    print(f"\n[{split}][{tuned_method}] N={N}")
    print(f"  McNemar: n01={n01} (fixes), n10={n10} (regressions), Δacc={delta_acc:+.3f}, p={p:.4g}")
    print(f"  Accuracy:  Base={acc_b[0]:.3f} [{acc_b[1]:.3f},{acc_b[2]:.3f}]  "
          f"Tuned={acc_t[0]:.3f} [{acc_t[1]:.3f},{acc_t[2]:.3f}]  "
          f"Diff={acc_d[0]:+.3f} [{acc_d[1]:+.3f},{acc_d[2]:+.3f}]")
    print(f"  Macro-F1:  Base={f1_b[0]:.3f} [{f1_b[1]:.3f},{f1_b[2]:.3f}]  "
          f"Tuned={f1_t[0]:.3f} [{f1_t[1]:.3f},{f1_t[2]:.3f}]  "
          f"Diff={f1_d[0]:+.3f} [{f1_d[1]:+.3f},{f1_d[2]:+.3f}]")

    return {
        "split": split,
        "tuned_method": tuned_method,
        "N": N,
        "n01_base_wrong_tuned_right": n01,
        "n10_base_right_tuned_wrong": n10,
        "discordant_pairs": n01 + n10,
        "delta_accuracy": float(delta_acc),
        "mcnemar_p_two_sided": float(p),
        "acc_base_mean": acc_b[0],  "acc_base_lo": acc_b[1],  "acc_base_hi": acc_b[2],
        "acc_tuned_mean": acc_t[0], "acc_tuned_lo": acc_t[1], "acc_tuned_hi": acc_t[2],
        "acc_diff_mean": acc_d[0],  "acc_diff_lo": acc_d[1],  "acc_diff_hi": acc_d[2],
        "macro_f1_base_mean": f1_b[0],  "macro_f1_base_lo": f1_b[1],  "macro_f1_base_hi": f1_b[2],
        "macro_f1_tuned_mean": f1_t[0], "macro_f1_tuned_lo": f1_t[1], "macro_f1_tuned_hi": f1_t[2],
        "macro_f1_diff_mean": f1_d[0],  "macro_f1_diff_lo": f1_d[1],  "macro_f1_diff_hi": f1_d[2],
    }

def main():
    # Load config → pull everything from cfg.evaluation.stats → compute stats and save CSV.
    # Required keys in config.yaml:
    #   project.output_dir
    #   evaluation.stats.splits
    #   evaluation.stats.tuned_methods
    #   evaluation.stats.labels
    #   evaluation.stats.output_csv
    #   evaluation.stats.bootstrap.B
    #   evaluation.stats.bootstrap.seed
    
    # Load config
    cfg = load_config()

    # Pull all args from cfg (explicit; fail fast if missing)
    project_out   = cfg.project.output_dir
    stats_cfg     = cfg.evaluation.stats
    splits        = tuple(list(stats_cfg.splits))
    tuned_methods = tuple(list(stats_cfg.tuned_methods))
    labels        = list(stats_cfg.labels)
    output_csv    = stats_cfg.output_csv
    B             = int(stats_cfg.bootstrap.B)
    seed          = int(stats_cfg.bootstrap.seed)

    rows = []
    for split in splits:
        for method in tuned_methods:
            r = compare_one(
                project_out=project_out,
                split=split,
                tuned_method=method,
                labels=labels,
                B=B,
                seed=seed,
            )
            if r is not None:
                rows.append(r)

    if rows:
        df = pd.DataFrame(rows)
        out_dir = os.path.dirname(output_csv) or project_out
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n Saved CSV: {output_csv}")
    else:
        print("\n No results to save (missing files or zero overlap).")

if __name__ == "__main__":
    main()

# Run python3 -m app.stats_eval
# Was quick
# Next, is plots.py