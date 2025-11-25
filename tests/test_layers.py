import torch

from src.models.layers import (
    AtrousDenseBlock,
    TransitionLayer,
    SwinTransformerBlock,
)


def test_atrous_dense_block_shape_and_channels():
    x = torch.randn(2, 16, 32, 32)
    block = AtrousDenseBlock(in_channels=16, growth_rate=8, dilation_rates=[1, 2])

    y = block(x)
    # out_channels attribute deve bater com o shape real
    assert y.shape[0] == 2
    assert y.shape[2:] == (32, 32)
    assert y.shape[1] == block.out_channels


def test_transition_layer_downsamples_and_compresses():
    x = torch.randn(2, 32, 32, 32)
    layer = TransitionLayer(in_channels=32, theta=0.5, p=0.0)

    y = layer(x)
    # spatial /2, channels reduzidos por theta
    assert y.shape == (2, layer.out_channels, 16, 16)
    assert layer.out_channels == int(32 * 0.5)


def test_swin_transformer_block_forward():
    # usa resolução pequena para ficar rápido
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

