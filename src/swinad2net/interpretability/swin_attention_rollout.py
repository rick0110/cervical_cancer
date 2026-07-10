#!/usr/bin/env python3
"""Genuine Swin self-attention rollout for SwinAD2Net (Abnar & Zuidema, 2020).

Grad-CAM (used in ``attention_compare.py``) is architecture-agnostic but was
designed for CNNs: it explains *both* networks with the same tool, at the
cost of not using SwinAD2Net's actual self-attention weights at all. This
module instead reads the real window-attention matrices out of SwinAD2Net's
last stage and "rolls them out" across blocks, giving a saliency map that is
faithful to what the transformer half of the network actually attends to.
A2SDNet121 has no self-attention mechanism, so this method has no equivalent
for it -- it is a SwinAD2Net-only diagnostic, complementary to Grad-CAM.

Why stage 4 specifically: at the network's last stage the spatial resolution
(7x7) is not larger than the configured window size (7), so
``SwinTransformerBlock`` disables windowing entirely (see the
``make_windows`` check in ``layers.py``) and both blocks already compute a
single full 49x49 self-attention over the whole feature map -- i.e. stage 4
is already "global", so no cross-window bookkeeping is needed to roll it
out. This also matches the "last spatial layer before pooling" choice used
for Grad-CAM, keeping the two methods comparable.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def _compute_attention_weights(attn_module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Recomputes WindowMultiHeadAttention's softmax attention weights.

    Mirrors ``WindowMultiHeadAttention.forward`` up to (but not including)
    the call to ``F.scaled_dot_product_attention``, which fuses the softmax
    and never exposes the attention matrix itself.

    Returns a tensor of shape [batch, heads, tokens, tokens].
    """
    batch_size_windows, channels, height, width = input_tensor.shape
    tokens = height * width

    x = input_tensor.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
    qkv = attn_module.mapping_qkv(x)
    qkv = qkv.view(batch_size_windows, tokens, 3, attn_module.number_of_heads, channels // attn_module.number_of_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key, _value = qkv[0], qkv[1], qkv[2]

    query = F.normalize(query, p=2, dim=-1, eps=1e-06)
    key = F.normalize(key, p=2, dim=-1, eps=1e-06)
    tau = attn_module.tau.clamp(min=0.01)
    query = query / tau

    relative_position_bias = attn_module.meta_network(attn_module.relative_coordinates_log)
    relative_position_bias = relative_position_bias.permute(1, 0)
    relative_position_bias = relative_position_bias.reshape(
        attn_module.number_of_heads, attn_module.window_size * attn_module.window_size,
        attn_module.window_size * attn_module.window_size,
    )
    attn_bias = relative_position_bias.unsqueeze(0)

    attn_logits = query @ key.transpose(-2, -1) + attn_bias
    attn_weights = torch.softmax(attn_logits, dim=-1)
    return attn_weights


def _capture_stage4_inputs(model, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs a forward pass and captures the tensors fed into each stage-4
    window-attention module via forward pre-hooks."""
    captured: List[torch.Tensor] = []

    def make_hook():
        def hook(_module, args):
            captured.append(args[0].detach())
        return hook

    h1 = model.swin_block4_1.window_attention.register_forward_pre_hook(make_hook())
    h2 = model.swin_block4_2.window_attention.register_forward_pre_hook(make_hook())
    try:
        with torch.no_grad():
            model.eval()
            model(input_tensor)
    finally:
        h1.remove()
        h2.remove()

    if len(captured) != 2:
        raise RuntimeError(f"Expected to capture 2 stage-4 attention inputs, got {len(captured)}")
    return captured[0], captured[1]


def _rollout_step(attn: torch.Tensor) -> torch.Tensor:
    """Adds the residual-connection identity and re-normalizes rows to sum
    to 1, as in Abnar & Zuidema (2020)."""
    batch, tokens, _ = attn.shape
    identity = torch.eye(tokens, device=attn.device, dtype=attn.dtype).unsqueeze(0)
    attn_res = attn + identity
    attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
    return attn_res


@torch.no_grad()
def swin_stage4_rollout(model, input_tensor: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Computes the attention-rollout saliency map for SwinAD2Net's stage 4.

    Args:
        model: a SwinAD2Net_ASPP_like / SwinAD2Net_ASPP_like_SwinResidual instance.
        input_tensor: a single preprocessed image, shape [1, 3, image_size, image_size].
        image_size: spatial size to upsample the resulting map to.

    Returns:
        A [image_size, image_size] tensor in [0, 1], the rolled-out saliency
        for the global-average-pooled classifier output, high where the
        network's own self-attention routes the most information from.
    """
    model.eval()
    in1, in2 = _capture_stage4_inputs(model, input_tensor)

    attn1 = _compute_attention_weights(model.swin_block4_1.window_attention, in1)  # [B, heads, 49, 49]
    attn2 = _compute_attention_weights(model.swin_block4_2.window_attention, in2)

    # Average over heads (stage 4 uses a single head in this architecture,
    # but averaging keeps this general if that ever changes).
    attn1 = attn1.mean(dim=1)  # [B, 49, 49]
    attn2 = attn2.mean(dim=1)

    attn1 = _rollout_step(attn1)
    attn2 = _rollout_step(attn2)

    # block4_1 runs first, then block4_2: compose so that rollout[i, j] is
    # how much final output token i's representation traces back to
    # original stage-4-input token j.
    rollout = attn2 @ attn1  # [B, 49, 49]

    # Global average pooling reads all 49 output tokens with equal weight,
    # so the relevance of input token j is the mean over output tokens i.
    saliency = rollout.mean(dim=1)  # [B, 49]

    side = int(saliency.shape[-1] ** 0.5)
    saliency_map = saliency.view(-1, 1, side, side)
    saliency_map = F.interpolate(saliency_map, size=(image_size, image_size), mode="bilinear", align_corners=False)
    saliency_map = saliency_map[0, 0]

    saliency_map = saliency_map - saliency_map.min()
    denom = saliency_map.max().clamp(min=1e-12)
    saliency_map = saliency_map / denom
    return saliency_map.cpu()
