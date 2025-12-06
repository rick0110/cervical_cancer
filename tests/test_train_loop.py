import os

import pandas as pd
import torch
import pytest

from src.swinad2net.models.train import train_swinad2net
from src.swinad2net.models.model import SwinAD2Net


def _make_dummy_image(path: str, size=(224, 224)):
    """Create a dummy RGB image for testing."""
    from PIL import Image

    img = Image.new("RGB", size, (0, 255, 0))
    img.save(path)


@pytest.fixture
def train_data_fixture(tmp_path):
    """Fixture to create dummy training data."""
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
    return train_df, tmp_path


@pytest.fixture
def train_val_data_fixture(tmp_path):
    """Fixture to create dummy training and validation data."""
    train_dir = tmp_path / "train_imgs"
    train_dir.mkdir()
    val_dir = tmp_path / "val_imgs"
    val_dir.mkdir()

    # Training data
    train_paths = []
    train_labels = []
    for i in range(4):
        p = train_dir / f"train_img_{i}.bmp"
        _make_dummy_image(str(p))
        train_paths.append(str(p))
        train_labels.append(i % 2)
    train_df = pd.DataFrame({"path": train_paths, "label": train_labels})

    # Validation data
    val_paths = []
    val_labels = []
    for i in range(2):
        p = val_dir / f"val_img_{i}.bmp"
        _make_dummy_image(str(p))
        val_paths.append(str(p))
        val_labels.append(i % 2)
    val_df = pd.DataFrame({"path": val_paths, "label": val_labels})

    return train_df, val_df, tmp_path


class TestTrainSwinAD2Net:
    """Tests for the train_swinad2net function."""

    def test_one_epoch_cpu_train_only(self, train_data_fixture):
        """Test training for one epoch on CPU without validation."""
        train_df, tmp_path = train_data_fixture

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=None,
            model=SwinAD2Net(num_classes=2,
                            embed_dim=32,
                            image_size=224,
                            patch_size_embed=4,
                            growth_rate=8),
            num_classes=2,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # Verify history contains training metrics
        assert "loss_train" in history, "History should contain 'loss_train'"
        assert "acc_train" in history, "History should contain 'acc_train'"
        
        # Verify directories were created
        assert os.path.isdir(tmp_path / "ckpt"), "Checkpoint directory should exist"
        assert os.path.isdir(tmp_path / "logs"), "Logs directory should exist"
        
        # Verify final model was saved
        assert os.path.isfile(tmp_path / "ckpt" / "final_model.pth"), "Final model should be saved"
        
        # Verify model is returned
        assert isinstance(model, SwinAD2Net), "Should return a SwinAD2Net model"
        
        # Scores and predictions should be empty without validation
        assert scores == {}, "Scores should be empty without validation"
        assert preds == {}, "Predictions should be empty without validation"

    def test_one_epoch_with_validation(self, train_val_data_fixture):
        """Test training for one epoch with validation data."""
        train_df, val_df, tmp_path = train_val_data_fixture

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=val_df,
            model=SwinAD2Net(num_classes=2,
                            embed_dim=32,
                            image_size=224,
                            patch_size_embed=4,
                            growth_rate=8),
            num_classes=2,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # Verify history contains both train and val metrics
        assert len(history["loss_train"]) == 1, "Should have 1 epoch of train loss"
        assert len(history["loss_val"]) == 1, "Should have 1 epoch of val loss"
        assert len(history["acc_train"]) == 1, "Should have 1 epoch of train accuracy"
        assert len(history["acc_val"]) == 1, "Should have 1 epoch of val accuracy"

        # Verify scores are computed
        assert "val_accuracy" in scores, "Should compute validation accuracy"
        assert "val_recall" in scores, "Should compute validation recall"
        assert "val_precision" in scores, "Should compute validation precision"
        assert "val_f1" in scores, "Should compute validation F1"

        # Verify predictions are returned
        assert "val_labels" in preds, "Should return validation labels"
        assert "val_predictions" in preds, "Should return validation predictions"
        assert len(preds["val_labels"]) == len(val_df), "Should have predictions for all val samples"

    def test_checkpoint_saved_at_intervals(self, train_data_fixture):
        """Test that checkpoints are saved at specified intervals (every 10 epochs)."""
        train_df, tmp_path = train_data_fixture

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=None,
            num_classes=2,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=10,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # Checkpoint at epoch 10 should exist
        checkpoint_path = tmp_path / "ckpt" / "checkpoint_epoch_10.pth"
        assert os.path.isfile(checkpoint_path), "Checkpoint at epoch 10 should be saved"

    def test_best_model_saved_on_validation_improvement(self, train_val_data_fixture):
        """Test that best model is saved when validation accuracy improves."""
        train_df, val_df, tmp_path = train_val_data_fixture

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=val_df,
            num_classes=2,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # Best model should be saved
        best_model_path = tmp_path / "ckpt" / "best_model.pth"
        assert os.path.isfile(best_model_path), "Best model should be saved"

        # Verify checkpoint contains expected keys
        checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint, "Checkpoint should contain model_state_dict"
        assert "epoch" in checkpoint, "Checkpoint should contain epoch"
        assert "val_acc" in checkpoint, "Checkpoint should contain val_acc"

    def test_model_in_eval_mode_after_training(self, train_data_fixture):
        """Test that training runs and model can be set to eval mode."""
        train_df, tmp_path = train_data_fixture

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=None,
            num_classes=2,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # Model should be usable in eval mode
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 2), f"Output shape should be (1, 2), got {output.shape}"

    def test_different_num_classes(self, train_data_fixture):
        """Test training with different number of classes."""
        train_df, tmp_path = train_data_fixture
        
        # Modify labels for 3 classes
        train_df["label"] = [0, 1, 2, 0]

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=None,
            num_classes=3,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # Verify model output has correct number of classes
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 3), f"Output shape should be (1, 3), got {output.shape}"

    def test_tensorboard_logs_created(self, train_data_fixture):
        """Test that TensorBoard log files are created."""
        train_df, tmp_path = train_data_fixture

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=None,
            num_classes=2,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # TensorBoard events file should be created
        log_files = list((tmp_path / "logs").iterdir())
        assert len(log_files) > 0, "TensorBoard log files should be created"

    def test_history_values_are_valid(self, train_val_data_fixture):
        """Test that history values are valid numbers."""
        train_df, val_df, tmp_path = train_val_data_fixture

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=val_df,
            num_classes=2,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=2,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # All history values should be valid floats
        for key in ["loss_train", "loss_val", "acc_train", "acc_val"]:
            for val in history[key]:
                assert isinstance(val, (int, float)), f"{key} should contain numbers"
                assert val >= 0, f"{key} values should be non-negative"

        # Accuracy should be between 0 and 100
        for val in history["acc_train"] + history["acc_val"]:
            assert 0 <= val <= 100, "Accuracy should be between 0 and 100"

    def test_scores_values_are_valid(self, train_val_data_fixture):
        """Test that score metrics are valid values between 0 and 1."""
        train_df, val_df, tmp_path = train_val_data_fixture

        model, history, scores, preds = train_swinad2net(
            train_df=train_df,
            val_df=val_df,
            num_classes=2,
            image_size=224,
            embed_dim=32,
            growth_rate=8,
            patch_size_embed=4,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # All score values should be between 0 and 1
        for key in ["val_accuracy", "val_recall", "val_precision", "val_f1"]:
            assert 0 <= scores[key] <= 1, f"{key} should be between 0 and 1"

