#!/usr/bin/env python3
"""Sequential driver for the full paper-vs-ours comparison matrix.

Loops over (dataset, task) combinations and, for each one, trains the paper
replica (A2SDNet121) and this project's tuned SwinAD2Net variant -- always
one training run at a time (see ``train.py`` for why parallel GPU jobs are
avoided here). Every run's ``summary.json`` is later collected by
``collect_results.py`` into the comparison tables used in the README.

Usage:
    python -m src.swinad2net.models.run_experiments \
        --best-config ./results/hp_search/best_config.json \
        --output-dir ./results --folds 3 --epochs 60 --patience 12
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import torch

from . import train as train_module

# (dataset, task) combinations that make up the comparison tables.
# "combined" only supports binary because Herlev (7 classes) and SIPaKMeD
# (5 classes) do not share a class taxonomy.
EXPERIMENT_MATRIX: List[Tuple[str, str]] = [
    ("herlev", "binary"),
    ("herlev", "multiclass"),
    ("sipakmed", "binary"),
    ("sipakmed", "multiclass"),
    ("combined", "binary"),
]


def build_run_args(base: argparse.Namespace, model: str, dataset: str, task: str, best_cfg: dict) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base))
    args.model = model
    args.dataset = dataset
    args.task = task
    args.batch_size = None
    args.learning_rate = None
    args.weight_decay = None
    args.optimizer = None
    args.embed_dim = None
    args.growth_rate = None
    args.dilation_rates = None
    args.compression_rates = None
    args.dropout = None
    args.drop_path = None

    if model != "A2SDNet121" and best_cfg:
        args.embed_dim = best_cfg.get("embed_dim")
        args.growth_rate = best_cfg.get("growth_rate")
        args.dropout = best_cfg.get("dropout")
        args.learning_rate = best_cfg.get("learning_rate")
        if "dilation_rates" in best_cfg:
            args.dilation_rates = json.dumps(best_cfg["dilation_rates"])
    return args


def run(base: argparse.Namespace) -> None:
    best_cfg = {}
    best_model_name = "SwinAD2Net_ASPP_like_SwinResidual"
    if base.best_config:
        with open(base.best_config, "r", encoding="utf-8") as f:
            best = json.load(f)
        best_cfg = best["trial_config"]
        best_model_name = best_cfg["model"]
        print(f"Loaded tuned config from {base.best_config}: {best_cfg}")

    for dataset, task in EXPERIMENT_MATRIX:
        for model in ["A2SDNet121", best_model_name]:
            summary_path = os.path.join(base.output_dir, f"{dataset}_{task}", model, "summary.json")
            if os.path.exists(summary_path):
                print(f"\nSkipping {model} | dataset={dataset} | task={task} (summary.json already exists)")
                continue
            print(f"\n{'#' * 70}\n# {model} | dataset={dataset} | task={task}\n{'#' * 70}")
            args = build_run_args(base, model, dataset, task, best_cfg)
            train_module.run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best-config", type=str, default=None, help="Path to best_config.json from hyperparameter_search.py")
    parser.add_argument("--herlev-dir", type=str, default="./data/herlev")
    parser.add_argument("--sipakmed-dir", type=str, default="./data/sipakmed")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-checkpoints", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
