#!/usr/bin/env python3
"""Sequential hyperparameter search for the SwinAD2Net_* family.

Trains one candidate configuration at a time (never in parallel: see the
module docstring in ``train.py`` for why concurrent GPU jobs are avoided in
this project) using a single stratified train/val split and a short epoch
budget. This is meant to find a good-enough configuration quickly, not to
exhaustively tune the network -- the winning configuration is then trained
with the full protocol (k-fold CV, longer patience) via ``train.py``.

Usage:
    python -m src.swinad2net.models.hyperparameter_search \
        --dataset combined --task binary --trials 8 --epochs 18 \
        --output-dir ./results/hp_search
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from typing import Any, Dict, List

import numpy as np
import torch

from .train import build_model, load_dataframe, train_one_fold
from sklearn.model_selection import train_test_split

SEARCH_SPACE: Dict[str, List[Any]] = {
    "model": ["SwinAD2Net_ASPP_like", "SwinAD2Net_ASPP_like_SwinResidual"],
    "embed_dim": [64, 128],
    "growth_rate": [16, 32],
    "dropout": [0.0, 0.1, 0.2],
    "learning_rate": [5e-4, 1e-3],
    "dilation_rates": [[1, 2, 3], [3, 5]],
}


def sample_configs(n_trials: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(SEARCH_SPACE.keys())
    all_combos = list(itertools.product(*[SEARCH_SPACE[k] for k in keys]))
    rng.shuffle(all_combos)
    chosen = all_combos[:n_trials]
    return [dict(zip(keys, combo)) for combo in chosen]


def make_args(base_args: argparse.Namespace, trial_cfg: Dict[str, Any]) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.model = trial_cfg["model"]
    args.embed_dim = trial_cfg["embed_dim"]
    args.growth_rate = trial_cfg["growth_rate"]
    args.dropout = trial_cfg["dropout"]
    args.learning_rate = trial_cfg["learning_rate"]
    args.dilation_rates = json.dumps(trial_cfg["dilation_rates"])
    args.compression_rates = json.dumps([0.25, 0.25, 0.25])
    args.drop_path = 0.0
    args.batch_size = None
    args.weight_decay = None
    args.optimizer = None
    args.save_checkpoints = False
    return args


def run(base_args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(base_args.seed)
    np.random.seed(base_args.seed)

    df, num_classes, class_names = load_dataframe(base_args)
    labels = df["label"].values
    train_idx, val_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=base_args.seed, stratify=labels
    )

    trials = sample_configs(base_args.trials, base_args.seed)
    os.makedirs(base_args.output_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for i, trial_cfg in enumerate(trials, start=1):
        trial_dir = os.path.join(base_args.output_dir, f"trial_{i:02d}")
        args = make_args(base_args, trial_cfg)
        print(f"\n--- Trial {i}/{len(trials)}: {trial_cfg} ---")
        t0 = time.time()
        result = train_one_fold(args.model, df, train_idx, val_idx, num_classes, trial_dir, args)
        elapsed = time.time() - t0
        results.append({
            "trial": i,
            "trial_config": trial_cfg,
            "val_metrics": result["best_val_metrics"],
            "n_params": result["n_params"],
            "elapsed_seconds": elapsed,
        })
        with open(os.path.join(base_args.output_dir, "search_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    results.sort(key=lambda r: r["val_metrics"]["f1"], reverse=True)
    best = results[0]
    with open(os.path.join(base_args.output_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("=" * 70)
    print("Hyperparameter search finished. Ranking (by val F1):")
    for r in results:
        print(f"  trial {r['trial']:02d} f1={r['val_metrics']['f1']:.4f} acc={r['val_metrics']['accuracy']:.4f} "
              f"params={r['n_params']/1e6:.2f}M cfg={r['trial_config']}")
    print(f"\nBest config: {best['trial_config']}")
    print(f"Saved to {os.path.join(base_args.output_dir, 'best_config.json')}")
    print("=" * 70)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["herlev", "sipakmed", "combined"])
    parser.add_argument("--task", required=True, choices=["binary", "multiclass"])
    parser.add_argument("--herlev-dir", type=str, default="./data/herlev")
    parser.add_argument("--sipakmed-dir", type=str, default="./data/sipakmed")
    parser.add_argument("--output-dir", type=str, default="./results/hp_search")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
