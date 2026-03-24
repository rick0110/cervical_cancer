"""
Comparative training script: A2SDNet121 (paper protocol) vs SwinAD2Net_ASPP_like (Hyperband config).

This script trains both models in parallel using multiple CPU processes and saves:
- Per-epoch histories (CSV)
- Final metrics and configs (JSON)
- Validation predictions (CSV)
- Aggregated comparison tables (CSV/JSON/PKL)
- TensorBoard logs for each fold/model

Usage example:
    python -m src.swinad2net.models.script_compare_paper_vs_hyperband \
        --data-dir ./data \
        --output-dir ./comparison_runs \
        --folds 5 \
        --workers 2 \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from .dataset import SimpleImageFolder
from .model import A2SDNet121, SwinAD2Net_ASPP_like, SwinAD2Net_ASPP_like_SwinResidual


LABEL_MAP = {
    "koilocytotic": 0,
    "dyskeratotic": 1,
    "metaplastic": 2,
    "superficial": 3,
    "parabasal": 4,
}

A2SDNET121_PAPER_CONFIG = {
    "image_size": 224,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "SGD",
    "epochs": 300,
    "scheduler": "StepLR",
    "step_size": 30,
    "gamma": 0.1,
}

SWIN_HYPERBAND_CONFIG = {
    "embed_dim": 128,
    "image_size": 224,
    "patch_size_embed": 4,
    "growth_rate": 16,
    "dilation_rates": [3, 5],
    "compression_rates": [0.25, 0.25, 0.25],
    "drop_path": 0.0,
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "optimizer": "AdamW",
    "epochs": 300,
    "scheduler": "StepLR",
    "step_size": 30,
    "gamma": 0.1,
}


HISTOGRAM_LOG_EVERY = 5
CONFUSION_MATRIX_LOG_EVERY = 5


@dataclass
class TrainingJob:
    model_name: str
    fold_id: int
    train_indices: List[int]
    val_indices: List[int]
    run_dir: str
    data_dir: str
    device: str
    num_classes: int
    num_workers_loader: int


def load_dataset(data_dir: str, label_map: Dict[str, int]) -> pd.DataFrame:
    paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, file_name))

    df = pd.DataFrame({"path": paths})

    def extract_label(path: str) -> float:
        lower = path.lower()
        for cls_name, cls_id in label_map.items():
            if cls_name in lower:
                return float(cls_id)
        return np.nan

    df["label"] = df["path"].apply(extract_label)
    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def build_model(model_name: str, num_classes: int) -> Tuple[nn.Module, Dict[str, Any]]:
    if model_name == "A2SDNet121":
        model = A2SDNet121(num_classes=num_classes)
        config = dict(A2SDNET121_PAPER_CONFIG)
    elif model_name == "SwinAD2Net_ASPP_like":
        model = SwinAD2Net_ASPP_like(
            num_classes=num_classes,
            embed_dim=SWIN_HYPERBAND_CONFIG["embed_dim"],
            image_size=SWIN_HYPERBAND_CONFIG["image_size"],
            patch_size_embed=SWIN_HYPERBAND_CONFIG["patch_size_embed"],
            growth_rate=SWIN_HYPERBAND_CONFIG["growth_rate"],
            dilation_rates=SWIN_HYPERBAND_CONFIG["dilation_rates"],
            compression_rates=SWIN_HYPERBAND_CONFIG["compression_rates"],
            drop_path=SWIN_HYPERBAND_CONFIG["drop_path"],
            dropout=SWIN_HYPERBAND_CONFIG["dropout"],
        )
        config = dict(SWIN_HYPERBAND_CONFIG)
    elif model_name == "SwinAD2Net_ASPP_like_SwinResidual":
        model = SwinAD2Net_ASPP_like_SwinResidual(
            num_classes=num_classes,
            embed_dim=SWIN_HYPERBAND_CONFIG["embed_dim"],
            image_size=SWIN_HYPERBAND_CONFIG["image_size"],
            patch_size_embed=SWIN_HYPERBAND_CONFIG["patch_size_embed"],
            growth_rate=SWIN_HYPERBAND_CONFIG["growth_rate"],
            dilation_rates=SWIN_HYPERBAND_CONFIG["dilation_rates"],
            compression_rates=SWIN_HYPERBAND_CONFIG["compression_rates"],
            drop_path=SWIN_HYPERBAND_CONFIG["drop_path"],
            dropout=SWIN_HYPERBAND_CONFIG["dropout"],
        )
        config = dict(SWIN_HYPERBAND_CONFIG)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model, config


def build_transforms(model_name: str, image_size: int) -> Tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if model_name == "A2SDNet121":
        train_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = T.Compose(
            [
                T.Resize((image_size + 32, image_size + 32)),
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.RandomRotation(degrees=10),
                T.ToTensor(),
                normalize,
            ]
        )

    val_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


def train_and_evaluate(job: TrainingJob, full_df: pd.DataFrame) -> Dict[str, Any]:
    torch.manual_seed(42 + job.fold_id)
    np.random.seed(42 + job.fold_id)

    os.makedirs(job.run_dir, exist_ok=True)
    model_dir = os.path.join(job.run_dir, job.model_name, f"fold_{job.fold_id}")
    os.makedirs(model_dir, exist_ok=True)

    tb_dir = os.path.join(job.run_dir, "tensorboard", job.model_name, f"fold_{job.fold_id}")
    os.makedirs(tb_dir, exist_ok=True)

    train_df = full_df.iloc[job.train_indices].reset_index(drop=True)
    val_df = full_df.iloc[job.val_indices].reset_index(drop=True)

    model, train_config = build_model(job.model_name, job.num_classes)
    image_size = int(train_config["image_size"])

    train_transform, val_transform = build_transforms(job.model_name, image_size)

    train_dataset = SimpleImageFolder(df=train_df, transform=train_transform, augment=False)
    val_dataset = SimpleImageFolder(df=val_df, transform=val_transform, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=job.num_workers_loader,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=job.num_workers_loader,
        pin_memory=False,
    )

    device = torch.device(job.device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_name = str(train_config["optimizer"])
    if optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_config["step_size"]),
        gamma=float(train_config["gamma"]),
    )

    epochs = int(train_config["epochs"])
    use_amp = device.type == "cuda" and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None

    writer = SummaryWriter(log_dir=tb_dir)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "grad_norm": [],
        "param_norm": [],
        "gpu_mem_mb": [],
    }

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(model_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_running_loss = 0.0
        train_total = 0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        # Compute gradient and parameter norms for training diagnostics.
        total_grad_sq = 0.0
        total_param_sq = 0.0
        for _, param in model.named_parameters():
            if param.grad is not None:
                total_grad_sq += float(param.grad.detach().norm(2).item() ** 2)
            total_param_sq += float(param.detach().norm(2).item() ** 2)

        grad_norm = total_grad_sq ** 0.5
        param_norm = total_param_sq ** 0.5

        train_loss = train_running_loss / max(train_total, 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)

        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_correct = 0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = val_running_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        cm = confusion_matrix(
            all_labels,
            all_preds,
            labels=list(range(job.num_classes)),
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        gpu_mem_mb = 0.0
        if use_amp:
            gpu_mem_mb = float(torch.cuda.memory_allocated(device=device) / (1024 ** 2))

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)
        history["grad_norm"].append(grad_norm)
        history["param_norm"].append(param_norm)
        history["gpu_mem_mb"].append(gpu_mem_mb)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LearningRate", lr, epoch)
        writer.add_scalar("Diagnostics/GradNorm", grad_norm, epoch)
        writer.add_scalar("Diagnostics/ParamNorm", param_norm, epoch)
        writer.add_scalar("Diagnostics/GPUMemoryMB", gpu_mem_mb, epoch)

        should_log_hist = (epoch % HISTOGRAM_LOG_EVERY == 0) or epoch == 1 or epoch == epochs
        if should_log_hist:
            for name, param in model.named_parameters():
                tag = name.replace(".", "/")
                writer.add_histogram(f"Weights/{tag}", param.detach().cpu(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{tag}", param.grad.detach().cpu(), epoch)

        should_log_cm = (epoch % CONFUSION_MATRIX_LOG_EVERY == 0) or epoch == 1 or epoch == epochs
        if should_log_cm:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set_title(f"Confusion Matrix - {job.model_name} Fold {job.fold_id}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(job.num_classes))
            ax.set_yticks(range(job.num_classes))

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )

            fig.tight_layout()
            writer.add_figure("Validation/ConfusionMatrix", fig, global_step=epoch)
            plt.close(fig)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": train_config,
                },
                best_ckpt_path,
            )

    writer.close()

    final_ckpt_path = os.path.join(model_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": train_config,
        },
        final_ckpt_path,
    )

    final_metrics = {
        "val_accuracy": float(accuracy_score(all_labels, all_preds)),
        "val_precision": float(precision_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "val_recall": float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "val_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "val_loss": float(history["val_loss"][-1]),
        "best_val_acc_percent": float(best_val_acc),
    }

    history_path = os.path.join(model_dir, "history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    preds_path = os.path.join(model_dir, "val_predictions.csv")
    pd.DataFrame({"label": all_labels, "prediction": all_preds}).to_csv(preds_path, index=False)

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    return {
        "model_name": job.model_name,
        "fold_id": job.fold_id,
        "metrics": final_metrics,
        "history": history,
        "paths": {
            "history_csv": history_path,
            "predictions_csv": preds_path,
            "metrics_json": metrics_path,
            "config_json": config_path,
            "best_checkpoint": best_ckpt_path,
            "final_checkpoint": final_ckpt_path,
            "tensorboard_dir": tb_dir,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel comparison: A2SDNet121 vs SwinAD2Net_ASPP_like")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory with images")
    parser.add_argument("--output-dir", type=str, default="./comparison_runs", help="Output root directory")
    parser.add_argument("--folds", type=int, default=10, help="Number of stratified folds")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel processes")
    parser.add_argument("--loader-workers", type=int, default=0, help="DataLoader workers per process")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of output classes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    df = load_dataset(args.data_dir, LABEL_MAP)
    if len(df) == 0:
        raise RuntimeError(f"No images found in {args.data_dir}")

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    jobs: List[TrainingJob] = []
    model_names = ["A2SDNet121", "SwinAD2Net_ASPP_like", "SwinAD2Net_ASPP_like_SwinResidual"]

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(df, df["label"].values), start=1):
        for model_name in model_names:
            jobs.append(
                TrainingJob(
                    model_name=model_name,
                    fold_id=fold_idx,
                    train_indices=train_idx.tolist(),
                    val_indices=val_idx.tolist(),
                    run_dir=run_dir,
                    data_dir=args.data_dir,
                    device=args.device,
                    num_classes=args.num_classes,
                    num_workers_loader=args.loader_workers,
                )
            )

    print(f"Submitting {len(jobs)} jobs with {args.workers} workers")

    all_results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train_and_evaluate, job, df): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(
                    f"Completed {job.model_name} fold {job.fold_id} | "
                    f"val_acc={result['metrics']['val_accuracy']:.4f}"
                )
            except Exception as exc:
                print(f"Failed {job.model_name} fold {job.fold_id}: {exc}")

    if not all_results:
        raise RuntimeError("All jobs failed")

    rows: List[Dict[str, Any]] = []
    for item in all_results:
        metrics = item["metrics"]
        rows.append(
            {
                "model": item["model_name"],
                "fold": item["fold_id"],
                "val_accuracy": metrics["val_accuracy"],
                "val_precision": metrics["val_precision"],
                "val_recall": metrics["val_recall"],
                "val_f1": metrics["val_f1"],
                "val_loss": metrics["val_loss"],
                "best_val_acc_percent": metrics["best_val_acc_percent"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(drop=True)
    results_csv = os.path.join(run_dir, "comparison_per_fold.csv")
    results_df.to_csv(results_csv, index=False)

    summary_df = (
        results_df.groupby("model")
        .agg(
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            val_f1_mean=("val_f1", "mean"),
            val_f1_std=("val_f1", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
        )
        .reset_index()
    )
    summary_csv = os.path.join(run_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    all_results_json = os.path.join(run_dir, "all_results.json")
    with open(all_results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    all_histories_pkl = os.path.join(run_dir, "all_histories.pkl")
    with open(all_histories_pkl, "wb") as f:
        pickle.dump(all_results, f)

    print("=" * 70)
    print("Comparison finished")
    print(f"Run directory: {run_dir}")
    print(f"Per-fold metrics: {results_csv}")
    print(f"Summary metrics: {summary_csv}")
    print(f"TensorBoard root: {os.path.join(run_dir, 'tensorboard')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
