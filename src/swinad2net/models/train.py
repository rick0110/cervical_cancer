#!/usr/bin/env python3
"""Single-model, single-dataset training with stratified k-fold cross-validation.

This is the canonical training entrypoint for the project. It trains exactly
one model configuration on one dataset/task combination per invocation and
logs everything needed to reproduce and compare runs later (metrics.json,
per-fold history, TensorBoard scalars). Run it once per (dataset, task,
model) combination you want to evaluate; see ``run_experiments.py`` for a
thin sequential driver that loops over the combinations used in the README
comparison tables.

Why not train several models at once in parallel processes/threads?
On a single GPU, concurrent training runs contend for the same compute and
memory bandwidth, so wall-clock time is dominated by context-switching and
memory thrashing rather than useful work -- several models trained "in
parallel" on one GPU end up slower in aggregate than the same models trained
one after another. This script is intentionally sequential; use a shell loop
or ``run_experiments.py`` to cover multiple configurations.

Usage example:
    python -m src.swinad2net.models.train \
        --model SwinAD2Net_ASPP_like --dataset herlev --task multiclass \
        --herlev-dir ./data/herlev --output-dir ./results --folds 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data_registry import DATASET_SPECS, build_combined_binary_dataframe, build_herlev_dataframe, build_sipakmed_dataframe
from .dataset import SimpleImageFolder
from .model import A2SDNet121, SwinAD2Net_ASPP_like, SwinAD2Net_ASPP_like_SwinResidual

MODEL_BUILDERS = {
    "A2SDNet121": A2SDNet121,
    "SwinAD2Net_ASPP_like": SwinAD2Net_ASPP_like,
    "SwinAD2Net_ASPP_like_SwinResidual": SwinAD2Net_ASPP_like_SwinResidual,
}

# Default hyperparameters. A2SDNet121 mirrors the protocol reported by
# Zhang et al. (2025); the SwinAD2Net_* defaults come from this project's
# own hyperparameter search (see hyperparameter_search.py / README).
DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A2SDNet121": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "SGD",
        "momentum": 0.9,
    },
    "SwinAD2Net_ASPP_like": {
        "embed_dim": 128,
        "growth_rate": 16,
        "dilation_rates": [3, 5],
        "compression_rates": [0.25, 0.25, 0.25],
        "drop_path": 0.0,
        "dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
    },
    "SwinAD2Net_ASPP_like_SwinResidual": {
        "embed_dim": 128,
        "growth_rate": 16,
        "dilation_rates": [3, 5],
        "compression_rates": [0.25, 0.25, 0.25],
        "drop_path": 0.0,
        "dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
    },
}

IMAGE_SIZE = 224


def load_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, int, List[str]]:
    if args.dataset == "herlev":
        df = build_herlev_dataframe(args.herlev_dir)
        if args.task == "binary":
            df["label"] = df["binary_label"]
            class_names = ["normal", "abnormal"]
        else:
            class_names = DATASET_SPECS["herlev"].class_names
    elif args.dataset == "sipakmed":
        df = build_sipakmed_dataframe(args.sipakmed_dir)
        if args.task == "binary":
            df["label"] = df["binary_label"]
            class_names = ["normal", "abnormal"]
        else:
            class_names = DATASET_SPECS["sipakmed"].class_names
    elif args.dataset == "combined":
        if args.task != "binary":
            raise ValueError("The combined dataset only supports the binary task "
                              "(Herlev and SIPaKMeD do not share a class taxonomy).")
        df = build_combined_binary_dataframe(args.herlev_dir, args.sipakmed_dir)
        class_names = ["normal", "abnormal"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    num_classes = df["label"].nunique()
    return df, num_classes, class_names


def build_model(model_name: str, num_classes: int, overrides: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    config = dict(DEFAULT_CONFIGS[model_name])
    config.update({k: v for k, v in overrides.items() if v is not None})

    if model_name == "A2SDNet121":
        model = A2SDNet121(num_classes=num_classes)
    else:
        builder = MODEL_BUILDERS[model_name]
        model = builder(
            num_classes=num_classes,
            embed_dim=config["embed_dim"],
            image_size=IMAGE_SIZE,
            patch_size_embed=4,
            growth_rate=config["growth_rate"],
            dilation_rates=config["dilation_rates"],
            compression_rates=config["compression_rates"],
            dropout=config["dropout"],
            drop_path=config["drop_path"],
        )
    return model, config


def build_transforms() -> Tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose(
        [
            T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            normalize,
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    name = config["optimizer"]
    if name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            momentum=config.get("momentum", 0.9),
        )
    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    raise ValueError(f"Unsupported optimizer: {name}")


def compute_specificity(cm: np.ndarray, positive_class: Optional[int] = None) -> float:
    """Specificity (TN / (TN + FP)) from a confusion matrix.

    For binary classification, macro-averaging specificity over both classes
    is mathematically identical to macro-averaged recall (each class's
    specificity equals the other class's recall when there are only two
    classes), which would make it a redundant metric. Pass
    ``positive_class`` for the binary case to instead report the standard
    clinical definition -- specificity with respect to a single designated
    positive ("abnormal") class, i.e. the true-negative rate of the negative
    class. When ``positive_class`` is None, macro-average one-vs-rest
    specificity is returned (used for multi-class tasks, where it does not
    degenerate to recall).
    """
    total = cm.sum()

    def _one_vs_rest(c: int) -> float:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        return tn / max(tn + fp, 1)

    if positive_class is not None:
        return float(_one_vs_rest(positive_class))

    specificities = [_one_vs_rest(c) for c in range(cm.shape[0])]
    return float(np.mean(specificities))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> Dict[str, Any]:
    model.eval()
    running_loss, total, correct = 0.0, 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    if num_classes == 2:
        # Standard clinical convention: label 1 ("abnormal") is the positive
        # class. Recall here is sensitivity; specificity is the true-negative
        # rate of the "normal" class. See compute_specificity() for why this
        # must NOT be macro-averaged over both classes for binary tasks.
        avg_kwargs: Dict[str, Any] = {"average": "binary", "pos_label": 1}
        specificity = compute_specificity(cm, positive_class=1)
    else:
        avg_kwargs = {"average": "macro"}
        specificity = compute_specificity(cm)

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0, **avg_kwargs),
        "recall": recall_score(all_labels, all_preds, zero_division=0, **avg_kwargs),
        "specificity": specificity,
        "f1": f1_score(all_labels, all_preds, zero_division=0, **avg_kwargs),
        "confusion_matrix": cm.tolist(),
    }


def train_one_fold(
    model_name: str,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
    fold_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    os.makedirs(fold_dir, exist_ok=True)
    device = torch.device(args.device)

    overrides = {
        "embed_dim": args.embed_dim,
        "growth_rate": args.growth_rate,
        "dilation_rates": json.loads(args.dilation_rates) if args.dilation_rates else None,
        "compression_rates": json.loads(args.compression_rates) if args.compression_rates else None,
        "drop_path": args.drop_path,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
    }
    model, config = build_model(model_name, num_classes, overrides)
    model = model.to(device)

    train_transform, eval_transform = build_transforms()
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(
        SimpleImageFolder(df=train_df, transform=train_transform, augment=False),
        batch_size=config["batch_size"], shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        SimpleImageFolder(df=val_df, transform=eval_transform, augment=False),
        batch_size=config["batch_size"], shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    use_amp = device.type == "cuda"
    writer = SummaryWriter(log_dir=os.path.join(fold_dir, "tensorboard"))

    best_f1 = -1.0
    best_state: Optional[Dict[str, Any]] = None
    best_val_metrics: Optional[Dict[str, Any]] = None
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, total, correct = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                loss.backward()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"], "val_f1": val_metrics["f1"],
        })

        improved = val_metrics["f1"] > best_f1
        if improved:
            best_f1 = val_metrics["f1"]
            best_val_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[{model_name}] fold={os.path.basename(fold_dir)} epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"(best_f1={best_f1:.4f}, patience={epochs_without_improvement}/{args.patience})"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (no F1 improvement in {args.patience} epochs).")
            break

    writer.close()
    assert best_state is not None and best_val_metrics is not None

    if args.save_checkpoints:
        torch.save({"model_state_dict": best_state, "config": config}, os.path.join(fold_dir, "best_model.pth"))

    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
    with open(os.path.join(fold_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {
        "fold_dir": fold_dir,
        "best_val_metrics": best_val_metrics,
        "epochs_trained": len(history),
        "config": config,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df, num_classes, class_names = load_dataframe(args)
    run_dir = os.path.join(
        args.output_dir, f"{args.dataset}_{args.task}", args.model,
    )
    os.makedirs(run_dir, exist_ok=True)

    labels = df["label"].values
    fold_results: List[Dict[str, Any]] = []

    if args.folds <= 1:
        train_idx, val_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=args.seed, stratify=labels
        )
        splits = [(train_idx, val_idx)]
    else:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.zeros(len(df)), labels))

    start = time.time()
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_dir = os.path.join(run_dir, f"fold_{fold_id}")
        result = train_one_fold(args.model, df, train_idx, val_idx, num_classes, fold_dir, args)
        fold_results.append(result)
    elapsed = time.time() - start

    metrics_keys = ["accuracy", "precision", "recall", "specificity", "f1", "loss"]
    aggregate = {}
    for key in metrics_keys:
        values = [r["best_val_metrics"][key] for r in fold_results]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "num_classes": num_classes,
        "class_names": class_names,
        "num_folds": len(splits),
        "num_samples": len(df),
        "n_params": fold_results[0]["n_params"],
        "config": fold_results[0]["config"],
        "elapsed_seconds": elapsed,
        "per_fold": [
            {"fold": i + 1, **r["best_val_metrics"], "epochs_trained": r["epochs_trained"]}
            for i, r in enumerate(fold_results)
        ],
        "aggregate": aggregate,
    }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"{args.model} on {args.dataset} ({args.task}) | folds={len(splits)} | elapsed={elapsed / 60:.1f} min")
    print(f"accuracy={aggregate['accuracy_mean']:.4f}±{aggregate['accuracy_std']:.4f} "
          f"f1={aggregate['f1_mean']:.4f}±{aggregate['f1_std']:.4f} "
          f"precision={aggregate['precision_mean']:.4f} recall={aggregate['recall_mean']:.4f} "
          f"specificity={aggregate['specificity_mean']:.4f}")
    print(f"Summary written to {os.path.join(run_dir, 'summary.json')}")
    print("=" * 70)

    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset", required=True, choices=["herlev", "sipakmed", "combined"])
    parser.add_argument("--task", required=True, choices=["binary", "multiclass"])
    parser.add_argument("--herlev-dir", type=str, default="./data/herlev")
    parser.add_argument("--sipakmed-dir", type=str, default="./data/sipakmed")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--folds", type=int, default=3, help="Use 1 for a single stratified 80/20 split.")
    parser.add_argument("--epochs", type=int, default=70, help="Max epochs per fold (early stopping usually stops sooner).")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-checkpoints", action="store_true")

    # Hyperparameter overrides (None keeps the model's DEFAULT_CONFIGS value).
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["SGD", "AdamW", None])
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--growth-rate", type=int, default=None)
    parser.add_argument("--dilation-rates", type=str, default=None, help="JSON list, e.g. '[1,2,3]'")
    parser.add_argument("--compression-rates", type=str, default=None, help="JSON list, e.g. '[0.25,0.25,0.25]'")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--drop-path", type=float, default=None)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
