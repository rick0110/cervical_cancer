import torch
from src.models.simple_cnn import SimpleCNN


def test_forward_shape():
    model = SimpleCNN(num_classes=3)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out.shape == (4, 3)
