#!/usr/bin/env python3
"""Attention comparison between the paper replica (A2SDNet121) and our SwinAD2Net.

A2SDNet121 has no explicit self-attention mechanism (only SE channel
recalibration), so Grad-CAM on the last spatial feature map before global
average pooling is used as a common, architecture-agnostic lens for both
networks: it answers "which image regions most influenced this
prediction?" for either model, which is the fair basis for comparison.

Grad-CAM was designed for CNNs, though, and is only an indirect proxy for
what a transformer's self-attention actually does. For SwinAD2Net we
therefore *also* compute a genuine self-attention rollout (Abnar &
Zuidema, 2020) over its last Swin stage -- see
``swin_attention_rollout.py`` -- which has no equivalent for A2SDNet121
since it has no self-attention to roll out.

For each sampled image this script renders a 4-panel figure (original |
A2SDNet121 Grad-CAM | SwinAD2Net Grad-CAM | SwinAD2Net attention rollout)
and reports quantitative "focus" scores -- the normalized Shannon entropy
of each heatmap (lower = more spatially concentrated / focused attention,
higher = more diffuse) -- averaged over the sampled images, so the
qualitative figures are backed by a number.

Usage:
    python -m src.swinad2net.interpretability.attention_compare \
        --dataset sipakmed --task multiclass \
        --results-dir ./results --output-dir ./results/attention_sipakmed_multiclass
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ..models.data_registry import DATASET_SPECS
from ..models.model import A2SDNet121, SwinAD2Net_ASPP_like, SwinAD2Net_ASPP_like_SwinResidual
from ..models.train import IMAGE_SIZE, build_transforms, load_dataframe, MODEL_BUILDERS
from .swin_attention_rollout import swin_stage4_rollout

TARGET_LAYER_NAME = {
    "A2SDNet121": "norm_final",
    "SwinAD2Net_ASPP_like": "swin_block4_2",
    "SwinAD2Net_ASPP_like_SwinResidual": "swin_block4_2",
}


def load_trained_model(run_dir: str, model_name: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    fold_dirs = sorted(glob.glob(os.path.join(run_dir, model_name, "fold_*")))
    if not fold_dirs:
        raise FileNotFoundError(f"No trained fold found under {os.path.join(run_dir, model_name)}")
    fold_dir = fold_dirs[0]
    ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), map_location=device)
    config = ckpt["config"]

    if model_name == "A2SDNet121":
        model = A2SDNet121(num_classes=num_classes)
    else:
        model = MODEL_BUILDERS[model_name](
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def cam_entropy(cam: np.ndarray) -> float:
    """Normalized Shannon entropy of a [0, 1] CAM heatmap treated as a distribution."""
    flat = cam.flatten().astype(np.float64)
    flat = np.clip(flat, 1e-12, None)
    flat = flat / flat.sum()
    entropy = -np.sum(flat * np.log(flat))
    max_entropy = np.log(flat.size)
    return float(entropy / max_entropy)


def sample_images(df, class_names: List[str], n_per_class: int, seed: int) -> List[Tuple[str, int, str]]:
    rng = np.random.default_rng(seed)
    samples = []
    for label_id, class_name in enumerate(class_names):
        subset = df[df["label"] == label_id]
        if subset.empty:
            continue
        chosen = subset.sample(n=min(n_per_class, len(subset)), random_state=seed)
        for _, row in chosen.iterrows():
            samples.append((row["path"], label_id, class_name))
    return samples


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    df, num_classes, class_names = load_dataframe(args)
    run_dir = os.path.join(args.results_dir, f"{args.dataset}_{args.task}")

    best_cfg_path = os.path.join(args.results_dir, "hp_search", "best_config.json")
    ours_model_name = "SwinAD2Net_ASPP_like_SwinResidual"
    if os.path.exists(best_cfg_path):
        with open(best_cfg_path, "r", encoding="utf-8") as f:
            ours_model_name = json.load(f)["trial_config"]["model"]

    model_paper = load_trained_model(run_dir, "A2SDNet121", num_classes, device)
    model_ours = load_trained_model(run_dir, ours_model_name, num_classes, device)

    layer_paper = dict(model_paper.named_modules())[TARGET_LAYER_NAME["A2SDNet121"]]
    layer_ours = dict(model_ours.named_modules())[TARGET_LAYER_NAME[ours_model_name]]

    cam_paper = GradCAM(model=model_paper, target_layers=[layer_paper])
    cam_ours = GradCAM(model=model_ours, target_layers=[layer_ours])

    _, eval_transform = build_transforms()
    samples = sample_images(df, class_names, args.samples_per_class, args.seed)

    entropies_paper: List[float] = []
    entropies_ours: List[float] = []
    entropies_rollout: List[float] = []
    fig_paths: List[str] = []

    for path, label_id, class_name in samples:
        image = Image.open(path).convert("RGB")
        input_tensor = eval_transform(image).unsqueeze(0).to(device)

        rgb_for_overlay = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0

        target = [ClassifierOutputTarget(label_id)]
        grayscale_cam_paper = cam_paper(input_tensor=input_tensor, targets=target)[0]
        grayscale_cam_ours = cam_ours(input_tensor=input_tensor, targets=target)[0]
        rollout_map = swin_stage4_rollout(model_ours, input_tensor, image_size=IMAGE_SIZE).numpy()

        entropies_paper.append(cam_entropy(grayscale_cam_paper))
        entropies_ours.append(cam_entropy(grayscale_cam_ours))
        entropies_rollout.append(cam_entropy(rollout_map))

        overlay_paper = show_cam_on_image(rgb_for_overlay, grayscale_cam_paper, use_rgb=True)
        overlay_ours = show_cam_on_image(rgb_for_overlay, grayscale_cam_ours, use_rgb=True)
        overlay_rollout = show_cam_on_image(rgb_for_overlay, rollout_map, use_rgb=True)

        fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
        axes[0].imshow(rgb_for_overlay)
        axes[0].set_title(f"{class_name}\n(original)", fontsize=9)
        axes[1].imshow(overlay_paper)
        axes[1].set_title("A2SDNet121 (paper)\nGrad-CAM", fontsize=9)
        axes[2].imshow(overlay_ours)
        axes[2].set_title(f"{ours_model_name}\n(ours) Grad-CAM", fontsize=9)
        axes[3].imshow(overlay_rollout)
        axes[3].set_title(f"{ours_model_name}\n(ours) attention rollout", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output_dir, f"{class_name}_{base}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        fig_paths.append(out_path)

    summary = {
        "dataset": args.dataset,
        "task": args.task,
        "ours_model": ours_model_name,
        "num_samples": len(samples),
        "mean_cam_entropy_paper_A2SDNet121": float(np.mean(entropies_paper)) if entropies_paper else None,
        "mean_cam_entropy_ours": float(np.mean(entropies_ours)) if entropies_ours else None,
        "mean_rollout_entropy_ours": float(np.mean(entropies_rollout)) if entropies_rollout else None,
        "note": (
            "Lower entropy means attention is more spatially concentrated "
            "(focused on a smaller region); higher entropy means it is more diffuse "
            "across the image. Grad-CAM entropies are comparable across both models; "
            "the attention-rollout entropy uses SwinAD2Net's own self-attention weights "
            "and has no A2SDNet121 equivalent (see swin_attention_rollout.py)."
        ),
        "figures": fig_paths,
    }
    with open(os.path.join(args.output_dir, "attention_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["herlev", "sipakmed", "combined"])
    parser.add_argument("--task", required=True, choices=["binary", "multiclass"])
    parser.add_argument("--herlev-dir", type=str, default="./data/herlev")
    parser.add_argument("--sipakmed-dir", type=str, default="./data/sipakmed")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--output-dir", type=str, default="./results/attention")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
