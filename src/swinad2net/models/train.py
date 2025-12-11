"""Training utilities for SwinAD2Net.

This module provides a convenience function `train_swinad2net` to train the
SwinAD2Net image classification model using Pandas DataFrames for training
and optional validation. The training procedure includes data augmentation,
TensorBoard logging, learning-rate scheduling, checkpointing and final
metric computation (when a validation DataFrame is provided).

Typical usage:
    from src.models.train import train_swinad2net
    model, history, scores, predictions = train_swinad2net(train_df, val_df)

The module is intentionally lightweight and relies on `torch`, `torchvision`
and a user-provided DataFrame that contains columns `path` and `label`.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from .model import SwinAD2Net
from .dataset import SimpleImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.amp import GradScaler, autocast

class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss does not improve.
    """
    def __init__(self, patience: int = 20, verbose: bool = False, delta: float = 0.001, path: str = 'checkpoint_in_best_early_stop.pth'):
        """
        Args:
            - patience (int): How long to wait after last time validation loss improved.
            - verbose (bool): If True, prints a message for each validation loss improvement.
            - delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            - path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = float('-inf')
        self.val_loss_min = float('inf')
        self.best_state_dict = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        """
        Call method to check if early stopping condition is met.
        
        Args:
            - val_loss (float): Current validation loss.
            - model (nn.Module): Model to save if validation loss improves.
        """
        score = -val_loss

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_state_dict = model.state_dict()

    def save_checkpoint(self):
        """
        Saves model when validation loss decreases.
        
        Args:
            - model (nn.Module): Model to save.
        """
        torch.save(self.best_state_dict, self.path)
        if self.verbose:
            print(f'The best model saved by early stopping!')

def train_swinad2net(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    model: Optional[SwinAD2Net] = None,
    image_size: int = 224,
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "runs",
    device: str = "cuda",
    state_dict=None,
    epoch_stopped: int = None

):
    """Train a SwinAD2Net image classification model.

    This function performs a complete training loop for the `SwinAD2Net` model
    using training and optional validation data provided as Pandas DataFrames.
    It builds datasets with simple torchvision transforms, runs the training
    loop with an AdamW optimizer and cosine annealing scheduler, logs metrics
    to TensorBoard, saves periodic checkpoints and records the best validation
    model based on validation accuracy.

    Args:
        train_df (pd.DataFrame): DataFrame with columns `path` (image filesystem
            path) and `label` (integer class label) for training.
        val_df (Optional[pd.DataFrame]): Validation DataFrame with the same
            format as `train_df`. If provided, validation is evaluated every
            epoch and metrics are returned.
        model (Optional[SwinAD2Net]): Optional SwinAD2Net model instance.
        num_classes (int): Number of classification output classes.
        image_size (int): Input image size (height and width) used by the
            torchvision transforms.
        batch_size (int): Training batch size.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Initial learning rate for the optimizer.
        weight_decay (float): L2 weight decay for optimizer regularization.
        patience (int): patience for early stopping. It indicates after how many epoches
        without improvement in loss in validation the training will be stopped.
        checkpoint_dir (str): Directory where checkpoints and best model are
            saved. Created if it does not exist.
        log_dir (str): Directory where TensorBoard logs are written.
        device (str): PyTorch device string, e.g. `'cuda'` or `'cpu'`.
        state_dict (optional): Optional state dictionary used to initialize the
            model parameters before training (useful for fine-tuning).

    Returns:
        tuple: A 4-tuple `(model, history, scores, predictions_dict)` where:
            - `model` (torch.nn.Module): The trained model (on `device`).
            - `history` (dict): Training history containing lists for
              `'loss_train'`, `'acc_train'`, `'loss_val'`, `'acc_val'`.
            - `scores` (dict): Validation metrics (`val_accuracy`,
              `val_recall`, `val_precision`, `val_f1`) if `val_df` was provided,
              otherwise an empty dict.
            - `predictions_dict` (dict): If validation was performed, contains
              `'val_labels'` and `'val_predictions'` lists; otherwise empty.

    Notes:
        - Checkpoints are saved every 10 epochs as `checkpoint_epoch_{epoch}.pth`.
        - The best validation model is saved as `best_model.pth` inside
          `checkpoint_dir` when validation accuracy improves.
        - TensorBoard logs are written to `log_dir`. Run
          `tensorboard --logdir {log_dir}` to visualize training progress.
    """

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(checkpoint_dir, 'checkpoint_in_best_early_stop.pth'))
    use_amp = device.startswith('cuda') and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None
    DTYPE = torch.bfloat16 # is only used in amp optimization with nvidia gpu
    num_classes = model.num_classes

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=False
            )
            print("torch.backends.cuda.sdp_kernel: trying to enable FlashAttention.")
        except AttributeError:
            print("sdp_kernel is not available")
    else:
        print("CUDA not detected — SDPA/FlashAttention will not be used.")

    print(f"\n{'='*60}")
    print(f"Training: ")
    print(f"Device: {device} | Classes: {num_classes} | Epochs: {num_epochs}")
    print(f"{'='*60}\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    import torchvision.transforms as T
    
    transform_train = T.Compose([
        T.Resize([image_size, image_size]), 
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = T.Compose([
        T.Resize([image_size, image_size]),  
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SimpleImageFolder(df=train_df, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    val_loader = None
    if val_df is not None:
        val_dataset = SimpleImageFolder(df=val_df, transform=transform_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")
    else:
        print(f"Train: {len(train_dataset)}\n")
    
    print("Creating model SwinAD2Net...")
    model = model.to(device)
    if state_dict:
        model.load_state_dict(state_dict=state_dict)

    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group.get('lr', learning_rate)
    warmup_epochs = 15
    def warmup_lr(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr, last_epoch = epoch_stopped if epoch_stopped is not None else -1)
    cosine_annealing_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs, last_epoch=epoch_stopped if epoch_stopped is not None else -1)

    writer = SummaryWriter(log_dir=log_dir)
    best_val_acc = 0.0
    history = {'loss_train': [], 'acc_train': [], 'loss_val': [], 'acc_val': []}

    for epoch in range(epoch_stopped + 1 if epoch_stopped is not None else 1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc="Train", leave=False)
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type=device, dtype=DTYPE):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
            
            if batch_idx % 10 == 0:
                global_step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('Train/Loss_step', loss.item(), global_step)
        
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
        writer.add_scalar('Train/Accuracy_epoch', train_acc, epoch)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        if val_loader is not None:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Val", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = 100. * val_correct / val_total
            writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
            writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            history['loss_train'].append(train_loss)
            history['acc_train'].append(train_acc)
            history['loss_val'].append(val_loss)
            history['acc_val'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                           'val_acc': val_acc}, os.path.join(checkpoint_dir, "best_model.pth"))
                print(f"✓ better model saved! (Val Acc: {val_acc:.2f}%)")
        
        #scheduler.step()
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_annealing_scheduler.step()
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                      os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
            print(f"✓ Checkpoint salvo: epoch_{epoch}")

        early_stopping(val_loss if val_loader is not None else train_loss, model)
        if early_stopping.early_stop:
            if input("Early stopping triggered. Stop training? (y/n): ").lower() == 'y':
                print("Early stopping activated. Ending training.")
                early_stopping.save_checkpoint()
                break
    
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, "final_model.pth"))
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Treinamento concluído! | Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print(f"{'='*60}\n")

    scores = {}
    predictions_dict = {}  
    
    if val_df is not None and val_loader is not None:
        val_targets = val_df['label'].values
        val_predictions = [] 
        
        model.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(val_loader, desc="Predicting", leave=False):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())

        scores = {
            'val_accuracy': accuracy_score(val_targets, val_predictions),
            'val_recall': recall_score(val_targets, val_predictions, average='weighted'),
            'val_precision': precision_score(val_targets, val_predictions, average='weighted'),
            'val_f1': f1_score(val_targets, val_predictions, average='weighted')
        }

        predictions_dict = {
            'val_labels': val_targets.tolist(),
            'val_predictions': val_predictions
        }

        print(f"\n{'='*60}")
        print("Métricas de Validação:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
        print(f"{'='*60}\n")

    return model, history, scores, predictions_dict


