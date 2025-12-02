#!/usr/bin/env python3
"""
Unit tests for the main custom layer blocks used in SwinAD2Net.
Each test checks if the outputs of the blocks have the expected shape and number of channels.
"""

import math

import torch

from src.models.layers import (
    AtrousDenseBlock,
    AtrousDenseBlock_ASPP_like,
    TransitionLayer,
    SwinTransformerBlock,
    Adb_SE_Transition,
    Adb_SE_Transition_ASPP_like,
    PatchEmb,
    bchw_to_bhwc,
    bhwc_to_bchw,
    unfold,
    fold,
)



def test_atrous_dense_block_shape_and_channels():
    """
    Tests if AtrousDenseBlock returns the correct shape and correctly calculates the number of output channels.
    """
    x = torch.randn(2, 16, 32, 32)
    block = AtrousDenseBlock(in_channels=16, growth_rate=8, dilation_rates=[1, 2])

    y = block(x)
    # out_channels attribute should match the actual shape
    assert y.shape[0] == 2
    assert y.shape[2:] == (32, 32)
    assert y.shape[1] == block.out_channels
    assert block.out_channels == 16 + 8 * 2  # in_channels + growth_rate * number of branches



def test_transition_layer_downsamples_and_compresses():
    """
    Tests if TransitionLayer correctly halves the spatial resolution and compresses the channels.
    """
    x = torch.randn(2, 32, 32, 32)
    layer = TransitionLayer(in_channels=32, theta=0.5, p=0.0)

    y = layer(x)
    # spatial /2, channels reduced by theta
    assert y.shape == (2, layer.out_channels, 16, 16)
    assert layer.out_channels == int(32 * 0.5)



def test_swin_transformer_block_forward():
    """
    Tests if SwinTransformerBlock returns the same shape as the input, using a small resolution for faster testing.
    """
    x = torch.randn(1, 16, 8, 8)
    block = SwinTransformerBlock(
        in_channels=16,
        input_resolution=(8, 8),
        number_of_heads=2,
        window_size=4,
        shift_size=2,
        dropout=0.0,
        dropout_attention=0.0,
        dropout_path=0.0,
    )

    y = block(x)
    assert y.shape == x.shape


def test_AtrousDenseBlock_ASPP_like_forward():
    """
    Tests if AtrousDenseBlock_ASPP_like returns the correct shape and correctly calculates the number of output channels.
    """
    x = torch.randn(2, 32, 56, 56)
    block = AtrousDenseBlock_ASPP_like(in_channels=32, growth_rate=8, dilation_rates=[1, 2, 3])

    y = block(x)
    # out_channels attribute should match the actual shape
    assert y.shape[0] == 2
    assert y.shape[2:] == (56, 56)
    assert y.shape[1] == block.out_channels
    assert block.out_channels == 32 + 8 * 3  # in_channels + growth_rate * number of branches


def test_adb_se_transition_chain_shapes():
    """Ensures Adb_SE_Transition applies ADB, SE, and transition while keeping expected dimensions."""
    x = torch.randn(2, 16, 32, 32)
    module = Adb_SE_Transition(in_channels=16, growth_rate=8, theta=0.5, p_transition=0.0, dilation_rates=[1, 2])

    y = module(x)
    expected_channels = int(math.floor(0.5 * (16 + 8 * 2)))
    assert module.adb.out_channels == 16 + 8 * 2
    assert module.out_channels == expected_channels
    assert y.shape == (2, expected_channels, 16, 16)


def test_adb_se_transition_aspp_like_chain_shapes():
    """Checks Adb_SE_Transition_ASPP_like runs all branches in parallel and downsamples correctly."""
    x = torch.randn(1, 24, 40, 40)
    module = Adb_SE_Transition_ASPP_like(
        in_channels=24,
        growth_rate=6,
        theta=0.5,
        p_transition=0.0,
        dilation_rates=[1, 2, 3, 4],
    )

    y = module(x)
    expected_channels = int(math.floor(0.5 * (24 + 6 * 4)))
    assert module.adb.out_channels == 24 + 6 * 4
    assert module.out_channels == expected_channels
    assert y.shape == (1, expected_channels, 20, 20)


def test_patch_emb_normalizes_each_patch():
    """PatchEmb should reduce spatial dimensions by patch size and apply LayerNorm per patch."""
    layer = PatchEmb(patch_size=4, in_channels=3, embed_dim=6)
    x = torch.randn(2, 3, 8, 8)

    y = layer(x)
    assert y.shape == (2, 6, 2, 2)
    patches = y.permute(0, 2, 3, 1).reshape(-1, 6)
    means = patches.mean(dim=1)
    variances = patches.var(dim=1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(variances, torch.ones_like(variances), atol=1e-4)


def test_bchw_to_bhwc_round_trip():
    """bchw_to_bhwc followed by bhwc_to_bchw should restore the original tensor exactly."""
    x = torch.randn(2, 5, 4, 3)
    converted = bchw_to_bhwc(x)
    restored = bhwc_to_bchw(converted)
    assert torch.equal(restored, x)


def test_unfold_and_fold_are_inverses():
    """The custom unfold and fold helpers should form an inverse pair for non-overlapping windows."""
    x = torch.randn(2, 4, 8, 8)
    window_size = 4
    windows = unfold(x, window_size)
    expected_windows = 2 * (8 // window_size) * (8 // window_size)
    assert windows.shape == (expected_windows, 4, window_size, window_size)
    reconstructed = fold(windows, window_size, height=8, width=8)
    assert torch.allclose(reconstructed, x)


def test_swin_transformer_block_update_resolution():
    """Updating the resolution of a SwinTransformerBlock should adjust window handling without changing outputs."""
    block = SwinTransformerBlock(
        in_channels=16,
        input_resolution=(8, 8),
        number_of_heads=4,
        window_size=4,
        shift_size=0,
        dropout=0.0,
        dropout_attention=0.0,
        dropout_path=0.0,
    )
    block.update_resolution(new_window_size=2, new_input_resolution=(8, 8))
    assert block.window_size == 2
    assert block.window_attention.window_size == 2
    assert block.make_windows is True

    x = torch.randn(1, 16, 8, 8)
    y = block(x)
    assert y.shape == x.shape

