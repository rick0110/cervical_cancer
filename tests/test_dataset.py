import os
import tempfile

import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.dataset import (
    RandomImageDataset,
    SimpleImageFolder,
    prepare_bmp_only,
    augment_data_prepared,
)


def _make_dummy_image(path: str, size=(32, 32), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    img.save(path)


def test_random_image_dataset_requires_df_for_getitem():
    ds = RandomImageDataset(length=5)
    assert len(ds) == 5
    with pytest.raises(ValueError):
        _ = ds[0]


def test_random_image_dataset_with_df(tmp_path):
    img_path = tmp_path / "img.bmp"
    _make_dummy_image(str(img_path))

    df = pd.DataFrame({"path": [str(img_path)], "label": [1]})
    ds = RandomImageDataset(df=df)

    assert len(ds) == 1
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 3
    assert y == 1


def test_simple_image_folder_basic(tmp_path):
    img1 = tmp_path / "a.bmp"
    img2 = tmp_path / "b.bmp"
    _make_dummy_image(str(img1))
    _make_dummy_image(str(img2))

    df = pd.DataFrame({"path": [str(img1), str(img2)], "label": [0, 1]})
    ds = SimpleImageFolder(df=df)

    assert len(ds) == 2
    x0, y0 = ds[0]
    x1, y1 = ds[1]

    assert isinstance(x0, torch.Tensor)
    assert x0.shape[1:] == (224, 224)
    assert {y0, y1} == {0, 1}
    assert ds.num_classes == 2


def test_simple_image_folder_invalid_df():
    with pytest.raises(ValueError):
        SimpleImageFolder(df="not a df")  # type: ignore


def test_prepare_bmp_only_and_safety(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    bmp = src / "img1.BMP"
    txt = src / "note.txt"
    _make_dummy_image(str(bmp))
    txt.write_text("hello")

    copied = prepare_bmp_only(str(src), str(dst))
    assert copied == 1
    assert os.path.exists(dst / "img1.BMP")
    assert not os.path.exists(dst / "note.txt")

    # dst inside src should raise
    inner_dst = src / "sub"
    inner_dst.mkdir()
    with pytest.raises(ValueError):
        prepare_bmp_only(str(src), str(inner_dst))


def test_augment_data_prepared(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    img = data_dir / "img.bmp"
    _make_dummy_image(str(img))

    generated = augment_data_prepared(str(data_dir), augmentations_per_image=2)
    assert generated == 2

    # Não deve tentar re‑augmentar arquivos já augmentados
    generated2 = augment_data_prepared(str(data_dir), augmentations_per_image=1)
    assert generated2 == 1  # apenas a original é usada

