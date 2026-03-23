import torch
import pytest

from src.swinad2net.models.model import SwinAD2Net, SwinAD2Net_ASPP_like, A2SDNet121

class TestSwinAD2Net:
    """Tests for the SwinAD2Net model."""

    def test_forward_and_num_classes(self):
        """Test forward pass produces correct output shape for given num_classes."""
        x = torch.randn(4, 3, 224, 224)
        model = SwinAD2Net(
            num_classes=3, 
            embed_dim=32, 
            image_size=224, 
            patch_size_embed=4, 
            growth_rate=8
        )
        y = model(x)

        assert y.shape == (4, 3), f"Expected output shape (4, 3), got {y.shape}"

    def test_different_num_classes(self):
        """Test that model correctly handles different number of output classes."""
        x = torch.randn(2, 3, 224, 224)
        
        for num_classes in [2, 5, 10]:
            model = SwinAD2Net(
                num_classes=num_classes,
                embed_dim=32,
                image_size=224,
                patch_size_embed=4,
                growth_rate=8
            )
            y = model(x)
            assert y.shape == (2, num_classes), f"Failed for num_classes={num_classes}"

    def test_batch_size_one(self):
        """Test that model handles batch size of 1."""
        x = torch.randn(1, 3, 224, 224)
        model = SwinAD2Net(
            num_classes=2,
            embed_dim=32,
            image_size=224,
            patch_size_embed=4,
            growth_rate=8
        )
        y = model(x)

        assert y.shape == (1, 2), f"Expected output shape (1, 2), got {y.shape}"

    def test_eval_mode(self):
        """Test that model works correctly in eval mode."""
        x = torch.randn(2, 3, 224, 224)
        model = SwinAD2Net(
            num_classes=2,
            embed_dim=32,
            image_size=224,
            patch_size_embed=4,
            growth_rate=8
        )
        model.eval()
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 2), f"Expected output shape (2, 2), got {y.shape}"

    def test_output_requires_grad_in_train_mode(self):
        """Test that output tensor requires grad when model is in training mode."""
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        model = SwinAD2Net(
            num_classes=2,
            embed_dim=32,
            image_size=224,
            patch_size_embed=4,
            growth_rate=8
        )
        model.train()
        y = model(x)
        
        assert y.requires_grad, "Output should require gradients in train mode"


class TestSwinAD2NetASPPLike:
    """Tests for the SwinAD2Net_ASPP_like model."""

    def test_forward_and_num_classes(self):
        """Test forward pass produces correct output shape for given num_classes."""
        x = torch.randn(4, 3, 224, 224)
        model = SwinAD2Net_ASPP_like(
            num_classes=3,
            embed_dim=32,
            image_size=224,
            patch_size_embed=4,
            growth_rate=8
        )
        y = model(x)

        assert y.shape == (4, 3), f"Expected output shape (4, 3), got {y.shape}"

    def test_different_num_classes(self):
        """Test that model correctly handles different number of output classes."""
        x = torch.randn(2, 3, 224, 224)
        
        for num_classes in [2, 5]:
            model = SwinAD2Net_ASPP_like(
                num_classes=num_classes,
                embed_dim=32,
                image_size=224,
                patch_size_embed=4,
                growth_rate=8
            )
            y = model(x)
            assert y.shape == (2, num_classes), f"Failed for num_classes={num_classes}"

    def test_eval_mode(self):
        """Test that model works correctly in eval mode."""
        x = torch.randn(2, 3, 224, 224)
        model = SwinAD2Net_ASPP_like(
            num_classes=2,
            embed_dim=32,
            image_size=224,
            patch_size_embed=4,
            growth_rate=8
        )
        model.eval()
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 2), f"Expected output shape (2, 2), got {y.shape}"


class TestA2SDNet121:
    """Tests for the A2SDNet121 model."""

    def test_forward_shape(self):
        """Test forward pass produces logits with the expected class dimension."""
        x = torch.randn(2, 3, 224, 224)
        model = A2SDNet121(num_classes=5)
        y = model(x)
        assert y.shape == (2, 5), f"Expected output shape (2, 5), got {y.shape}"

