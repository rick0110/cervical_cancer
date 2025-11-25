import torch

from src.models.model import PatchEmb, Adb_SE_Transition, SwinAD2Net


def test_patch_emb_keeps_spatial_divides_by_patch():
    x = torch.randn(2, 3, 224, 224)
    patch = PatchEmb(patch_size=4, in_chans=3, embed_dim=64)

    y = patch(x)
    assert y.shape[0] == 2
    assert y.shape[1] == 64
    assert y.shape[2:] == (224 // 4, 224 // 4)


def test_adb_se_transition_channels_and_downsample():
    x = torch.randn(2, 32, 56, 56)
    mod = Adb_SE_Transition(in_channels=32, growth_rate=16, theta=0.5, p_transition=0.0)

    y = mod(x)
    assert y.shape[0] == 2
    # deve reduzir H,W pela metade
    assert y.shape[2:] == (28, 28)
    assert y.shape[1] == mod.out_channels


def test_swinad2net_forward_and_num_classes_cpu():
    x = torch.randn(4, 3, 224, 224)
    model = SwinAD2Net(num_classes=3, embed_dim=32, image_size=224, patch_size_embed=4, growth_rate=8)
    y = model(x)

    assert y.shape == (4, 3)

