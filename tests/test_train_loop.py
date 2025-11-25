import os

import pandas as pd
import torch

from src.models.train import train_swinad2net


def _make_dummy_image(path: str, size=(64, 64)):
    from PIL import Image

    img = Image.new("RGB", size, (0, 255, 0))
    img.save(path)


def test_train_swinad2net_one_epoch_cpu(tmp_path, monkeypatch):
    # cria 4 imagens e um pequeno dataframe
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    paths = []
    labels = []
    for i in range(4):
        p = img_dir / f"img_{i}.bmp"
        _make_dummy_image(str(p))
        paths.append(str(p))
        labels.append(i % 2)

    train_df = pd.DataFrame({"path": paths, "label": labels})

    # força device CPU
    model, history, scores, preds = train_swinad2net(
        train_df=train_df,
        val_df=None,
        num_classes=2,
        image_size=64,
        embed_dim=16,
        growth_rate=8,
        patch_size_embed=4,
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        device="cpu",
    )

    # verificações básicas
    assert any("loss_train" in k for k in history.keys())
    assert os.path.isdir(tmp_path / "ckpt")
    assert os.path.isdir(tmp_path / "logs")

