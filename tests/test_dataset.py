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
    """Helper function to create a dummy image for testing."""
    img = Image.new("RGB", size, color)
    img.save(path)


# ============================================================================
# RandomImageDataset Tests
# ============================================================================

def test_random_image_dataset_requires_df_for_init():
    """Test that RandomImageDataset raises ValueError when df is not provided."""
    with pytest.raises(ValueError, match="requires a DataFrame"):
        RandomImageDataset(length=5)


def test_random_image_dataset_with_df(tmp_path):
    """Test RandomImageDataset initialization and basic functionality with DataFrame."""
    img_path = tmp_path / "img.bmp"
    _make_dummy_image(str(img_path))

    # Use correct column name 'paths' to match the default parameter
    df = pd.DataFrame({"paths": [str(img_path)], "label": [1]})
    ds = RandomImageDataset(df=df)

    assert len(ds) == 1
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 3  # RGB channels
    assert y == 1


def test_random_image_dataset_custom_columns(tmp_path):
    """Test RandomImageDataset with custom column names."""
    img1 = tmp_path / "img1.bmp"
    img2 = tmp_path / "img2.bmp"
    _make_dummy_image(str(img1), color=(255, 0, 0))
    _make_dummy_image(str(img2), color=(0, 255, 0))

    df = pd.DataFrame({
        "image_path": [str(img1), str(img2)],
        "class": [0, 1]
    })
    ds = RandomImageDataset(df=df, path_col="image_path", label_col="class")

    assert len(ds) == 2
    x0, y0 = ds[0]
    x1, y1 = ds[1]
    
    assert isinstance(x0, torch.Tensor)
    assert isinstance(x1, torch.Tensor)
    assert y0 == 0
    assert y1 == 1


def test_random_image_dataset_invalid_df():
    """Test RandomImageDataset with invalid DataFrame input."""
    with pytest.raises(ValueError, match="must be a pandas.DataFrame"):
        RandomImageDataset(df="not a df")  # type: ignore


def test_random_image_dataset_missing_columns(tmp_path):
    """Test RandomImageDataset raises error when required columns are missing."""
    img_path = tmp_path / "img.bmp"
    _make_dummy_image(str(img_path))

    # Missing 'paths' column
    df = pd.DataFrame({"wrong_col": [str(img_path)], "label": [1]})
    with pytest.raises(ValueError, match="Path column 'paths' not found"):
        RandomImageDataset(df=df)

    # Missing 'label' column
    df = pd.DataFrame({"paths": [str(img_path)], "wrong_col": [1]})
    with pytest.raises(ValueError, match="Label column 'label' not found"):
        RandomImageDataset(df=df)


def test_random_image_dataset_with_transform(tmp_path):
    """Test RandomImageDataset with custom transform."""
    import torchvision.transforms as T
    
    img_path = tmp_path / "img.bmp"
    _make_dummy_image(str(img_path), size=(100, 100))

    df = pd.DataFrame({"paths": [str(img_path)], "label": [0]})
    
    transform = T.Compose([
        T.Resize((50, 50)),
        T.ToTensor()
    ])
    
    ds = RandomImageDataset(df=df, transform=transform)
    x, y = ds[0]
    
    assert x.shape == (3, 50, 50)  # 3 channels, 50x50 after resize
    assert y == 0


# ============================================================================
# SimpleImageFolder Tests
# ============================================================================

def test_simple_image_folder_basic(tmp_path):
    """Test SimpleImageFolder with basic image loading."""
    img1 = tmp_path / "a.bmp"
    img2 = tmp_path / "b.bmp"
    _make_dummy_image(str(img1), color=(255, 0, 0))
    _make_dummy_image(str(img2), color=(0, 255, 0))

    df = pd.DataFrame({"path": [str(img1), str(img2)], "label": [0, 1]})
    ds = SimpleImageFolder(df=df)

    assert len(ds) == 2
    x0, y0 = ds[0]
    x1, y1 = ds[1]

    assert isinstance(x0, torch.Tensor)
    assert isinstance(x1, torch.Tensor)
    assert x0.shape == (3, 224, 224)  # Default resize to 224x224
    assert x1.shape == (3, 224, 224)
    assert {y0, y1} == {0, 1}
    assert ds.num_classes == 2


def test_simple_image_folder_invalid_df():
    """Test SimpleImageFolder with invalid DataFrame type."""
    with pytest.raises(ValueError, match="must be a pandas.DataFrame"):
        SimpleImageFolder(df="not a df")  # type: ignore


def test_simple_image_folder_missing_columns(tmp_path):
    """Test SimpleImageFolder raises error when required columns are missing."""
    img_path = tmp_path / "img.bmp"
    _make_dummy_image(str(img_path))

    # Missing 'path' column
    df = pd.DataFrame({"wrong_col": [str(img_path)], "label": [1]})
    with pytest.raises(ValueError, match="Column 'path' not found"):
        SimpleImageFolder(df=df)

    # Missing 'label' column
    df = pd.DataFrame({"path": [str(img_path)], "wrong_col": [1]})
    with pytest.raises(ValueError, match="Column 'label' not found"):
        SimpleImageFolder(df=df)


def test_simple_image_folder_multiple_classes(tmp_path):
    """Test SimpleImageFolder correctly counts multiple classes."""
    images = []
    labels = []
    for i in range(5):
        img_path = tmp_path / f"img_{i}.bmp"
        _make_dummy_image(str(img_path))
        images.append(str(img_path))
        labels.append(i % 3)  # 3 classes: 0, 1, 2

    df = pd.DataFrame({"path": images, "label": labels})
    ds = SimpleImageFolder(df=df)

    assert len(ds) == 5
    assert ds.num_classes == 3


def test_simple_image_folder_custom_columns(tmp_path):
    """Test SimpleImageFolder with custom column names."""
    img_path = tmp_path / "img.bmp"
    _make_dummy_image(str(img_path))

    df = pd.DataFrame({"img_path": [str(img_path)], "class_id": [5]})
    ds = SimpleImageFolder(df=df, path_col="img_path", label_col="class_id")

    assert len(ds) == 1
    x, y = ds[0]
    assert y == 5


# ============================================================================
# prepare_bmp_only Tests
# ============================================================================

def test_prepare_bmp_only_basic(tmp_path):
    """Test prepare_bmp_only copies only BMP files."""
    src_dir = tmp_path / "source"
    dst_dir = tmp_path / "dest"
    src_dir.mkdir()

    # Create test files
    bmp1 = src_dir / "image1.bmp"
    bmp2 = src_dir / "image2.BMP"  # Test case insensitivity
    jpg = src_dir / "image.jpg"
    txt = src_dir / "readme.txt"

    _make_dummy_image(str(bmp1))
    _make_dummy_image(str(bmp2))
    _make_dummy_image(str(jpg))
    txt.write_text("test")

    copied = prepare_bmp_only(str(src_dir), str(dst_dir))

    assert copied == 2
    assert (dst_dir / "image1.bmp").exists()
    assert (dst_dir / "image2.BMP").exists()
    assert not (dst_dir / "image.jpg").exists()
    assert not (dst_dir / "readme.txt").exists()


def test_prepare_bmp_only_nested_structure(tmp_path):
    """Test prepare_bmp_only preserves directory structure."""
    src_dir = tmp_path / "source"
    dst_dir = tmp_path / "dest"
    
    subdir1 = src_dir / "class1"
    subdir2 = src_dir / "class2"
    subdir1.mkdir(parents=True)
    subdir2.mkdir(parents=True)

    img1 = subdir1 / "img1.bmp"
    img2 = subdir2 / "img2.bmp"
    _make_dummy_image(str(img1))
    _make_dummy_image(str(img2))

    copied = prepare_bmp_only(str(src_dir), str(dst_dir))

    assert copied == 2
    assert (dst_dir / "class1" / "img1.bmp").exists()
    assert (dst_dir / "class2" / "img2.bmp").exists()


def test_prepare_bmp_only_invalid_src(tmp_path):
    """Test prepare_bmp_only with invalid source directory."""
    with pytest.raises(ValueError, match="does not exist or is not a directory"):
        prepare_bmp_only("/nonexistent/path", str(tmp_path / "dest"))


def test_prepare_bmp_only_same_directory(tmp_path):
    """Test prepare_bmp_only rejects same source and destination."""
    src_dir = tmp_path / "source"
    src_dir.mkdir()

    with pytest.raises(ValueError, match="cannot be the same"):
        prepare_bmp_only(str(src_dir), str(src_dir))


def test_prepare_bmp_only_nested_destination(tmp_path):
    """Test prepare_bmp_only rejects destination inside source."""
    src_dir = tmp_path / "source"
    dst_dir = src_dir / "inside"
    src_dir.mkdir()

    with pytest.raises(ValueError, match="cannot be the same.*inside"):
        prepare_bmp_only(str(src_dir), str(dst_dir))


def test_prepare_bmp_only_empty_directory(tmp_path):
    """Test prepare_bmp_only with empty source directory."""
    src_dir = tmp_path / "source"
    dst_dir = tmp_path / "dest"
    src_dir.mkdir()

    copied = prepare_bmp_only(str(src_dir), str(dst_dir))

    assert copied == 0


# ============================================================================
# augment_data_prepared Tests
# ============================================================================

def test_augment_data_prepared_directory_mode(tmp_path):
    """Test augment_data_prepared in directory mode."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    img = data_dir / "img.bmp"
    _make_dummy_image(str(img))

    generated = augment_data_prepared(str(data_dir), augmentations_per_image=2)
    assert generated == 2

    # Check that augmented files exist
    assert (data_dir / "img_aug1.bmp").exists()
    assert (data_dir / "img_aug2.bmp").exists()


def test_augment_data_prepared_no_reaugmentation(tmp_path):
    """Test that already augmented files are not re-augmented."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    img = data_dir / "img.bmp"
    _make_dummy_image(str(img))

    # First augmentation
    generated = augment_data_prepared(str(data_dir), augmentations_per_image=2)
    assert generated == 2

    # Second augmentation - should only process the original
    generated2 = augment_data_prepared(str(data_dir), augmentations_per_image=1)
    assert generated2 == 1  # Only 1 from the original image


def test_augment_data_prepared_dataframe_mode(tmp_path):
    """Test augment_data_prepared in DataFrame mode."""
    img1 = tmp_path / "img1.bmp"
    img2 = tmp_path / "img2.bmp"
    _make_dummy_image(str(img1))
    _make_dummy_image(str(img2))

    df = pd.DataFrame({
        "path": [str(img1), str(img2)],
        "label": [0, 1]
    })

    result = augment_data_prepared(df_paths=df, augmentations_per_image=2)

    # Should return tuple with (generated_count, paths_list, updated_df)
    assert isinstance(result, tuple)
    generated, paths_aug, df_updated = result

    assert generated == 4  # 2 images × 2 augmentations
    assert len(paths_aug) == 4
    assert len(df_updated) == 6  # 2 original + 4 augmented
def test_augment_data_prepared_dataframe_no_reaugmentation(tmp_path):
    """Test DataFrame mode doesn't re-augment existing augmented files."""
    img = tmp_path / "img.bmp"
    _make_dummy_image(str(img))

    df = pd.DataFrame({"path": [str(img)], "label": [0]})

    # First augmentation
    generated1, paths1, df1 = augment_data_prepared(df_paths=df, augmentations_per_image=2)
    assert generated1 == 2

    # Try to augment again with the updated dataframe (includes augmented images)
    generated2, paths2, df2 = augment_data_prepared(df_paths=df1, augmentations_per_image=1)
    # Should only process the original, not the _aug files
    assert generated2 == 1


def test_augment_data_prepared_invalid_directory():
    """Test augment_data_prepared with invalid directory."""
    with pytest.raises(ValueError, match="does not exist"):
        augment_data_prepared(data_dir="/nonexistent/path")


def test_augment_data_prepared_preserves_structure(tmp_path):
    """Test augment_data_prepared preserves subdirectory structure."""
    data_dir = tmp_path / "data"
    subdir = data_dir / "class1"
    subdir.mkdir(parents=True)

    img = subdir / "img.bmp"
    _make_dummy_image(str(img))

    generated = augment_data_prepared(str(data_dir), augmentations_per_image=3)
    
    assert generated == 3
    assert (subdir / "img_aug1.bmp").exists()
    assert (subdir / "img_aug2.bmp").exists()
    assert (subdir / "img_aug3.bmp").exists()


def test_augment_data_prepared_empty_dataframe():
    """Test augment_data_prepared with empty DataFrame."""
    df = pd.DataFrame({"path": [], "label": []})
    result = augment_data_prepared(df_paths=df, augmentations_per_image=2)
    
    generated, paths_aug, df_updated = result
    assert generated == 0
    assert len(paths_aug) == 0
    assert len(df_updated) == 0

