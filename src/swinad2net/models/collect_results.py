#!/usr/bin/env python3
"""Collects per-run summary.json files into a single comparison table.

Reads ``results/{dataset}_{task}/{model}/summary.json`` for every
(dataset, task, model) combination produced by ``run_experiments.py`` and
writes a flat CSV plus a Markdown table (ready to paste into the README)
comparing the paper replica against this project's tuned SwinAD2Net.

Usage:
    python -m src.swinad2net.models.collect_results --results-dir ./results
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def collect(results_dir: str) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "summary.json"))):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        agg = summary["aggregate"]
        rows.append({
            "dataset": summary["dataset"],
            "task": summary["task"],
            "model": summary["model"],
            "num_classes": summary["num_classes"],
            "num_samples": summary["num_samples"],
            "num_folds": summary["num_folds"],
            "params_M": round(summary["n_params"] / 1e6, 3),
            "accuracy_mean": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"],
            "precision_mean": agg["precision_mean"],
            "recall_mean": agg["recall_mean"],
            "specificity_mean": agg["specificity_mean"],
            "f1_mean": agg["f1_mean"],
            "f1_std": agg["f1_std"],
            "elapsed_minutes": round(summary["elapsed_seconds"] / 60, 1),
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| Dataset | Task | Model | Params (M) | Accuracy | Precision | Recall | Specificity | F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['task']} | {r['model']} | {r['params_M']:.2f} | "
            f"{r['accuracy_mean']*100:.2f}±{r['accuracy_std']*100:.2f}% | "
            f"{r['precision_mean']*100:.2f}% | {r['recall_mean']*100:.2f}% | "
            f"{r['specificity_mean']*100:.2f}% | {r['f1_mean']*100:.2f}±{r['f1_std']*100:.2f}% |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-md", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect(args.results_dir)
    if df.empty:
        print(f"No summary.json files found under {args.results_dir}")
        return

    out_csv = args.out_csv or os.path.join(args.results_dir, "comparison_table.csv")
    out_md = args.out_md or os.path.join(args.results_dir, "comparison_table.md")
    df.to_csv(out_csv, index=False)
    md = to_markdown(df)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print(md)
    print(f"\nSaved CSV to {out_csv}")
    print(f"Saved Markdown to {out_md}")


if __name__ == "__main__":
    main()
