"""Single-run comparative training script.

This script compares A2SDNet121, SwinAD2Net_ASPP_like and
SwinAD2Net_ASPP_like_SwinResidual in a single experiment using:
- One stratified train/validation/test split (no cross-validation)
- Validation during training
- Test evaluation only after training is complete
- TensorBoard logging, including per-layer Lipschitz rates

Usage example:
    python -m src.swinad2net.models.script_compare_paper_vs_hyperband \
        --data-dir ./data \
        --output-dir ./comparison_runs \
        --workers 2 \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
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
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from .dataset import SimpleImageFolder
from .lipschitz_regularization import LipschitzRegularizer
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
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    run_dir: str
    data_dir: str
    device: str
    num_classes: int
    num_workers_loader: int
    seed: int


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


def stratified_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_ratio + val_ratio + test_ratio must be 1.0, got {total}")

    if min(train_ratio, val_ratio, test_ratio) <= 0.0:
        raise ValueError("train_ratio, val_ratio and test_ratio must be > 0")

    labels = df["label"].values
    all_indices = np.arange(len(df))

    train_idx, holdout_idx = train_test_split(
        all_indices,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=labels,
    )

    holdout_labels = labels[holdout_idx]
    val_share_in_holdout = val_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=(1.0 - val_share_in_holdout),
        random_state=seed,
        stratify=holdout_labels,
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


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

    eval_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            normalize,
        ]
    )

    return train_transform, eval_transform


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    optimizer_name = str(config["optimizer"])
    if optimizer_name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    if optimizer_name == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    split_name: str,
) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        f"{split_name}_accuracy": float(accuracy_score(all_labels, all_preds)),
        f"{split_name}_precision": float(
            precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        ),
        f"{split_name}_recall": float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)),
        f"{split_name}_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        f"{split_name}_loss": float(avg_loss),
        f"{split_name}_acc_percent": float(100.0 * acc),
        f"{split_name}_cm": cm.tolist(),
        f"{split_name}_labels": all_labels,
        f"{split_name}_preds": all_preds,
    }


def log_confusion_matrix(writer: SummaryWriter, cm: np.ndarray, num_classes: int, tag: str, epoch: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(tag)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))

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
    writer.add_figure(tag, fig, global_step=epoch)
    plt.close(fig)


def log_lipschitz_to_tensorboard(
    writer: SummaryWriter,
    regularizer: LipschitzRegularizer,
    model: nn.Module,
    epoch: int,
) -> Tuple[float, float]:
    layer_lips = regularizer.compute_layer_lipschitz(model)
    if not layer_lips:
        writer.add_scalar("Lipschitz/NetworkBound", 1.0, epoch)
        writer.add_scalar("Lipschitz/NetworkBoundLog10", 0.0, epoch)
        return 1.0, 0.0

    network_bound = 1.0
    for layer_name, lip_tensor in layer_lips.items():
        lip_value = float(lip_tensor.detach().item())
        network_bound *= max(lip_value, 1e-12)
        tag_name = layer_name.replace(".", "/")
        writer.add_scalar(f"Lipschitz/Layers/{tag_name}", lip_value, epoch)

    network_log10 = math.log10(max(network_bound, 1e-12))
    writer.add_scalar("Lipschitz/NetworkBound", network_bound, epoch)
    writer.add_scalar("Lipschitz/NetworkBoundLog10", network_log10, epoch)
    writer.add_scalar("Lipschitz/NumTrackedLayers", len(layer_lips), epoch)

    return network_bound, network_log10


def train_and_evaluate(job: TrainingJob, full_df: pd.DataFrame) -> Dict[str, Any]:
    torch.manual_seed(job.seed)
    np.random.seed(job.seed)

    os.makedirs(job.run_dir, exist_ok=True)
    model_dir = os.path.join(job.run_dir, job.model_name, "single_run")
    os.makedirs(model_dir, exist_ok=True)

    tb_dir = os.path.join(job.run_dir, "tensorboard", job.model_name, "single_run")
    os.makedirs(tb_dir, exist_ok=True)

    train_df = full_df.iloc[job.train_indices].reset_index(drop=True)
    val_df = full_df.iloc[job.val_indices].reset_index(drop=True)
    test_df = full_df.iloc[job.test_indices].reset_index(drop=True)

    model, train_config = build_model(job.model_name, job.num_classes)
    image_size = int(train_config["image_size"])
    train_transform, eval_transform = build_transforms(job.model_name, image_size)

    train_dataset = SimpleImageFolder(df=train_df, transform=train_transform, augment=False)
    val_dataset = SimpleImageFolder(df=val_df, transform=eval_transform, augment=False)
    test_dataset = SimpleImageFolder(df=test_df, transform=eval_transform, augment=False)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=job.num_workers_loader,
        pin_memory=False,
    )

    device = torch.device(job.device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, train_config)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_config["step_size"]),
        gamma=float(train_config["gamma"]),
    )

    use_amp = device.type == "cuda" and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None

    lipschitz_regularizer = LipschitzRegularizer(
        upper_bound=10.0,
        lower_bound=0.1,
        lambda_upper=0.0,
        lambda_lower=0.0,
        use_exact_svd=False,
        n_power_iterations=1,
    )

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
        "lipschitz_bound": [],
        "lipschitz_bound_log10": [],
    }

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(model_dir, "best_model.pth")
    epochs = int(train_config["epochs"])

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

        val_eval = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=job.num_classes,
            split_name="val",
        )
        val_loss = float(val_eval["val_loss"])
        val_acc = float(val_eval["val_acc_percent"])
        cm = np.array(val_eval["val_cm"])

        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])

        gpu_mem_mb = 0.0
        if use_amp:
            gpu_mem_mb = float(torch.cuda.memory_allocated(device=device) / (1024 ** 2))

        lip_bound, lip_bound_log10 = log_lipschitz_to_tensorboard(
            writer=writer,
            regularizer=lipschitz_regularizer,
            model=model,
            epoch=epoch,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)
        history["grad_norm"].append(grad_norm)
        history["param_norm"].append(param_norm)
        history["gpu_mem_mb"].append(gpu_mem_mb)
        history["lipschitz_bound"].append(lip_bound)
        history["lipschitz_bound_log10"].append(lip_bound_log10)

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
            log_confusion_matrix(
                writer=writer,
                cm=cm,
                num_classes=job.num_classes,
                tag="Validation/ConfusionMatrix",
                epoch=epoch,
            )

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

    if os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])

    val_eval_final = evaluate_model(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=job.num_classes,
        split_name="val",
    )

    test_eval_final = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=job.num_classes,
        split_name="test",
    )

    final_metrics = {
        "best_val_acc_percent": float(best_val_acc),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "val_accuracy": float(val_eval_final["val_accuracy"]),
        "val_precision": float(val_eval_final["val_precision"]),
        "val_recall": float(val_eval_final["val_recall"]),
        "val_f1": float(val_eval_final["val_f1"]),
        "val_loss": float(val_eval_final["val_loss"]),
        "test_accuracy": float(test_eval_final["test_accuracy"]),
        "test_precision": float(test_eval_final["test_precision"]),
        "test_recall": float(test_eval_final["test_recall"]),
        "test_f1": float(test_eval_final["test_f1"]),
        "test_loss": float(test_eval_final["test_loss"]),
    }

    history_path = os.path.join(model_dir, "history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    val_preds_path = os.path.join(model_dir, "val_predictions.csv")
    pd.DataFrame(
        {
            "label": val_eval_final["val_labels"],
            "prediction": val_eval_final["val_preds"],
        }
    ).to_csv(val_preds_path, index=False)

    test_preds_path = os.path.join(model_dir, "test_predictions.csv")
    pd.DataFrame(
        {
            "label": test_eval_final["test_labels"],
            "prediction": test_eval_final["test_preds"],
        }
    ).to_csv(test_preds_path, index=False)

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    split_path = os.path.join(model_dir, "split_indices.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_indices": job.train_indices,
                "val_indices": job.val_indices,
                "test_indices": job.test_indices,
            },
            f,
            indent=2,
        )

    return {
        "model_name": job.model_name,
        "metrics": final_metrics,
        "history": history,
        "paths": {
            "history_csv": history_path,
            "val_predictions_csv": val_preds_path,
            "test_predictions_csv": test_preds_path,
            "metrics_json": metrics_path,
            "config_json": config_path,
            "split_indices_json": split_path,
            "best_checkpoint": best_ckpt_path,
            "final_checkpoint": final_ckpt_path,
            "tensorboard_dir": tb_dir,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-run comparison: paper protocol vs hyperband config")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory with images")
    parser.add_argument("--output-dir", type=str, default="./comparison_runs", help="Output root directory")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel processes")
    parser.add_argument("--loader-workers", type=int, default=0, help="DataLoader workers per process")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of output classes")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    df = load_dataset(args.data_dir, LABEL_MAP)
    if len(df) == 0:
        raise RuntimeError(f"No images found in {args.data_dir}")

    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model_names = ["A2SDNet121", "SwinAD2Net_ASPP_like", "SwinAD2Net_ASPP_like_SwinResidual"]
    jobs: List[TrainingJob] = []

    for offset, model_name in enumerate(model_names):
        jobs.append(
            TrainingJob(
                model_name=model_name,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                run_dir=run_dir,
                data_dir=args.data_dir,
                device=args.device,
                num_classes=args.num_classes,
                num_workers_loader=args.loader_workers,
                seed=args.seed + offset,
            )
        )

    print(
        f"Submitting {len(jobs)} jobs with {args.workers} workers | "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )

    all_results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train_and_evaluate, job, df): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(
                    f"Completed {job.model_name} | "
                    f"val_acc={result['metrics']['val_accuracy']:.4f} "
                    f"test_acc={result['metrics']['test_accuracy']:.4f}"
                )
            except Exception as exc:
                print(f"Failed {job.model_name}: {exc}")

    if not all_results:
        raise RuntimeError("All jobs failed")

    rows: List[Dict[str, Any]] = []
    for item in all_results:
        metrics = item["metrics"]
        rows.append(
            {
                "model": item["model_name"],
                "train_samples": metrics["train_samples"],
                "val_samples": metrics["val_samples"],
                "test_samples": metrics["test_samples"],
                "val_accuracy": metrics["val_accuracy"],
                "val_precision": metrics["val_precision"],
                "val_recall": metrics["val_recall"],
                "val_f1": metrics["val_f1"],
                "val_loss": metrics["val_loss"],
                "test_accuracy": metrics["test_accuracy"],
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "test_f1": metrics["test_f1"],
                "test_loss": metrics["test_loss"],
                "best_val_acc_percent": metrics["best_val_acc_percent"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["model"]).reset_index(drop=True)
    results_csv = os.path.join(run_dir, "comparison_single_run.csv")
    results_df.to_csv(results_csv, index=False)

    summary_df = (
        results_df[["val_accuracy", "val_f1", "val_loss", "test_accuracy", "test_f1", "test_loss"]]
        .agg(["mean", "std"]) 
        .reset_index()
        .rename(columns={"index": "stat"})
    )
    summary_csv = os.path.join(run_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    split_metadata = {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    split_path = os.path.join(run_dir, "global_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2)

    all_results_json = os.path.join(run_dir, "all_results.json")
    with open(all_results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    all_histories_pkl = os.path.join(run_dir, "all_histories.pkl")
    with open(all_histories_pkl, "wb") as f:
        pickle.dump(all_results, f)

    print("=" * 70)
    print("Comparison finished (single split, no cross-validation)")
    print(f"Run directory: {run_dir}")
    print(f"Per-model metrics: {results_csv}")
    print(f"Summary metrics: {summary_csv}")
    print(f"Global split metadata: {split_path}")
    print(f"TensorBoard root: {os.path.join(run_dir, 'tensorboard')}")
    print("=" * 70)


if __name__ == "__main__":
    main()