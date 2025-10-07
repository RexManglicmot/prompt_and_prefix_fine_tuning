# app/dataset.py
"""
Dataset loader check for PubMedQA PEFT project.
- Loads train/val/test CSVs from config.yaml
- Verifies they exist
- Prints row counts and class balance
- Previews a formatted prompt from config.data.text_template
"""

import os
import sys
import pandas as pd
from app.config import load_config

REQUIRED_COLS = ["id", "question", "contexts", "final_decision"]

def describe_split(df: pd.DataFrame, name: str) -> None:
    n = len(df)
    yes = (df["final_decision"] == "yes").sum()
    no = (df["final_decision"] == "no").sum()
    print(f"\n[{name}] rows: {n} | yes: {yes} ({yes/n:.1%}) | no: {no} ({no/n:.1%})")

def build_prompt(cfg, row: pd.Series) -> str:
    tmpl = cfg.data.text_template
    return tmpl.format(question=row["question"], contexts=row["contexts"])

def preview_prompt(cfg, df: pd.DataFrame, n: int = 1) -> None:
    print("\n--- Prompt preview ---")
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        prompt = build_prompt(cfg, row)
        print(prompt)
        print(f'Label: {row["final_decision"]}')
        print("---")

def main():
    cfg = load_config()

    # Load the three splits directly
    paths = {
        "train": cfg.data.train_csv,
        "val":   cfg.data.val_csv,
        "test":  cfg.data.test_csv,
    }

    splits = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} split not found: {path}")
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} split missing columns: {missing}")
        splits[name] = df

    # Summaries
    for name, df in splits.items():
        describe_split(df, name)

    # Show a sample prompt from train
    preview_prompt(cfg, splits["train"].sample(1, random_state=cfg.project.seed))

    print("\n✅ Dataset loader check completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Dataset check failed: {e}")
        sys.exit(1)

# "-m" The -m flag tells Python to treat app/ as a package.
# This way, from app.config import load_config will work.

# Run python3 -m app.dataset