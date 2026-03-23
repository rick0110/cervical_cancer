"""
Hyperband Parallel Training Script with Lipschitz Regularization.

This script implements:
1. Parallel training across multiple CPU/GPU cores
2. Hyperband for efficient hyperparameter optimization
3. Adaptive early stopping for poor-performing models
4. Lipschitz regularization to control model stability
5. Real-time comparison of results

Usage:
    python -m src.swinad2net.models.script_hyperband_train

Configuration can be modified in the CONFIG section below.
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm

# Local imports
from .model import SwinAD2Net, SwinAD2Net_ASPP_like, Densenet121, A2SDNet121
from .dataset import SimpleImageFolder
from .lipschitz_regularization import LipschitzRegularizer, compute_exact_spectral_norm
from .hyperband_scheduler import HyperbandScheduler, AdaptiveEarlyStopping, Trial, TrialStatus
from .parallel_training import ParallelTrainingManager, TrainingJob, TrainingResult, SharedResultsManager


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data settings
    'data_dir': './data',
    'image_size': 224,
    
    # Hyperband settings
    'max_budget': 100,  # Maximum epochs per config
    'eta': 3,  # Reduction factor
    'metric_to_optimize': 'val_loss',
    'optimization_mode': 'min',  # 'min' or 'max'
    
    # Parallel training
    'num_workers': max(1, cpu_count() - 2),  # Leave some cores free
    
    # Lipschitz regularization
    'lipschitz_upper_bound': 10.0,
    'lipschitz_lower_bound': 0.1,
    'lipschitz_lambda_upper': 0.01,
    'lipschitz_lambda_lower': 0.001,
    
    # Early stopping
    'patience': 15,
    'min_delta': 0.001,
    'divergence_threshold': 5.0,
    
    # Checkpointing
    'checkpoint_dir': './hyperband_checkpoints',
    'log_dir': './hyperband_runs',
    
    # Label mapping
    'label_map': {
        'koilocytotic': 0,
        'dyskeratotic': 1,
        'metaplastic': 2,
        'superficial': 3,
        'parabasal': 4
    },
    'num_classes': 5,
}

# Model-specific default configurations
# Use these when a hyperparameter is not specified in the search
MODEL_DEFAULTS = {
    'SwinAD2Net_ASPP_like': {
        'embed_dim': 128,
        'image_size': 224,
        'patch_size_embed': 4,
        'growth_rate': 16,
        'dilation_rates': [3, 5],
        'compression_rates': [0.25, 0.25, 0.25],
        'drop_path': 0.0,
        'dropout': 0.0,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'optimizer': 'AdamW',
    },
    'SwinAD2Net': {
        'embed_dim': 128,
        'image_size': 224,
        'patch_size_embed': 4,
        'growth_rate': 32,
        'dilation_rates': [1, 2, 3],
        'compression_rates': [0.5, 0.5, 0.5],
        'drop_path': 0.0,
        'dropout': 0.0,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'optimizer': 'AdamW',
    },
    'Densenet121': {
        'embed_dim': 64,  # Not used but kept for compatibility
        'image_size': 192,
        'patch_size_embed': 4,  # Not used
        'growth_rate': 32,
        'dilation_rates': [1, 2, 3],  # Not used
        'compression_rates': [0.5, 0.5, 0.5],  # Not used
        'drop_path': 0.0,
        'dropout': 0.0,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'optimizer': 'AdamW',
    },
    'A2SDNet121': {
        'embed_dim': 64,  # Not used but kept for compatibility
        'image_size': 224,
        'patch_size_embed': 4,  # Not used
        'growth_rate': 32,
        'dilation_rates': [1, 2, 3],  # Last three dense layers use atrous rates
        'compression_rates': [0.5, 0.5, 0.5],  # Not used
        'drop_path': 0.0,
        'dropout': 0.0,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 8,
        'optimizer': 'SGD',
    },
}

# Hyperparameter search space
# Set to None or remove key to use MODEL_DEFAULTS value
# Use a list for categorical, tuple (min, max, scale) for continuous
HYPERPARAMETER_SPACE = {
    # Model architecture
    'model_type': ['SwinAD2Net_ASPP_like', 'Densenet121', 'A2SDNet121'],
    
    # Architecture hyperparameters (set to None to use model default)
    'embed_dim': [64, 96, 128],           # Embedding dimension
    'image_size': [192, 224, 256],         # Input image size
    'patch_size_embed': [4],               # Patch size for embedding (usually fixed)
    'growth_rate': [16, 24, 32],           # Growth rate for dense blocks
    'dilation_rates': [[3, 5], [2, 4, 6], [1, 2, 3]],  # Atrous convolution rates
    'compression_rates': [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],  # Transition compression
    'drop_path': [0.0, 0.1, 0.2],          # Stochastic depth drop rate
    'dropout': [0.0, 0.1, 0.2, 0.3],       # Dropout rate
    
    # Training hyperparameters  
    'learning_rate': (1e-5, 1e-2, 'log'),  # Log scale search
    'weight_decay': (1e-6, 1e-2, 'log'),   # L2 regularization
    'batch_size': [16, 32, 64],            # Batch size
    
    # Lipschitz regularization
    'lip_lambda_upper': (0.001, 0.1, 'log'),   # Penalty for large Lipschitz
    'lip_lambda_lower': (0.0001, 0.01, 'log'), # Penalty for small Lipschitz
    
    # Optimizer
    'optimizer': ['AdamW', 'SGD'],
}


def get_config_with_defaults(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Merge config with model-specific defaults.
    User config takes precedence over defaults.
    
    Args:
        config: User-provided configuration
        model_type: Type of model ('SwinAD2Net_ASPP_like', 'Densenet121', etc.)
        
    Returns:
        Complete configuration with defaults filled in
    """
    defaults = MODEL_DEFAULTS.get(model_type, MODEL_DEFAULTS['SwinAD2Net_ASPP_like'])
    merged = defaults.copy()
    
    # Override with user config (only non-None values)
    for key, value in config.items():
        if value is not None:
            merged[key] = value
            
    return merged


def create_custom_config(
    model_type: str = 'SwinAD2Net_ASPP_like',
    embed_dim: Optional[int] = None,
    image_size: Optional[int] = None,
    patch_size_embed: Optional[int] = None,
    growth_rate: Optional[int] = None,
    dilation_rates: Optional[List[int]] = None,
    compression_rates: Optional[List[float]] = None,
    drop_path: Optional[float] = None,
    dropout: Optional[float] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    batch_size: Optional[int] = None,
    optimizer: Optional[str] = None,
    lip_lambda_upper: Optional[float] = None,
    lip_lambda_lower: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create a custom configuration for training.
    Parameters not specified will use model-specific defaults.
    
    Args:
        model_type: 'SwinAD2Net_ASPP_like', 'SwinAD2Net', or 'Densenet121'
        embed_dim: Embedding dimension (default: 128 for Swin, 64 for DenseNet)
        image_size: Input image size (default: 224 for Swin, 192 for DenseNet)
        patch_size_embed: Patch size for embedding (default: 4)
        growth_rate: Growth rate for dense blocks (default: 16-32)
        dilation_rates: List of dilation rates for atrous conv (default: [3, 5])
        compression_rates: Compression rates for transitions (default: [0.25, 0.25, 0.25])
        drop_path: Stochastic depth drop rate (default: 0.0)
        dropout: Dropout rate (default: 0.0)
        learning_rate: Learning rate (default: 1e-3 for Swin, 1e-4 for DenseNet)
        weight_decay: Weight decay (default: 1e-4)
        batch_size: Batch size (default: 32)
        optimizer: 'AdamW' or 'SGD' (default: 'AdamW')
        lip_lambda_upper: Lipschitz upper penalty weight
        lip_lambda_lower: Lipschitz lower penalty weight
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = create_custom_config(
        ...     model_type='SwinAD2Net_ASPP_like',
        ...     embed_dim=128,
        ...     image_size=224,
        ...     dilation_rates=[3, 5],
        ...     learning_rate=1e-3
        ... )
    """
    config = {'model_type': model_type}
    
    # Only add non-None values
    if embed_dim is not None:
        config['embed_dim'] = embed_dim
    if image_size is not None:
        config['image_size'] = image_size
    if patch_size_embed is not None:
        config['patch_size_embed'] = patch_size_embed
    if growth_rate is not None:
        config['growth_rate'] = growth_rate
    if dilation_rates is not None:
        config['dilation_rates'] = dilation_rates
    if compression_rates is not None:
        config['compression_rates'] = compression_rates
    if drop_path is not None:
        config['drop_path'] = drop_path
    if dropout is not None:
        config['dropout'] = dropout
    if learning_rate is not None:
        config['learning_rate'] = learning_rate
    if weight_decay is not None:
        config['weight_decay'] = weight_decay
    if batch_size is not None:
        config['batch_size'] = batch_size
    if optimizer is not None:
        config['optimizer'] = optimizer
    if lip_lambda_upper is not None:
        config['lip_lambda_upper'] = lip_lambda_upper
    if lip_lambda_lower is not None:
        config['lip_lambda_lower'] = lip_lambda_lower
        
    return config


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(data_dir: str, label_map: Dict[str, int]) -> pd.DataFrame:
    """Load dataset and create DataFrame with paths and labels."""
    paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(root, file))
    
    df = pd.DataFrame({'path': paths})
    
    def get_label(path):
        for key, val in label_map.items():
            if key in path.lower():
                return val
        return np.nan
    
    df['label'] = df['path'].apply(get_label)
    df = df.dropna().reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    
    return df


# ============================================================================
# TRAINING FUNCTION (called by workers)
# ============================================================================

def train_single_config(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
    epochs: int,
    device: str,
    job_id: int,
    checkpoint_path: Optional[str] = None,
    intermediate_callback: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a single model configuration.
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        config: Hyperparameter configuration
        epochs: Number of epochs to train
        device: Device to use
        job_id: Job identifier
        checkpoint_path: Path to resume from
        intermediate_callback: Callback for reporting intermediate results
        
    Returns:
        Dict with metrics, history, and checkpoint_path
    """
    import torchvision.transforms as T
    
    # Get model type first to apply correct defaults
    model_type = config.get('model_type', 'SwinAD2Net_ASPP_like')
    
    # Merge with model-specific defaults
    full_config = get_config_with_defaults(config, model_type)
    
    # Extract hyperparameters (now with proper defaults applied)
    embed_dim = full_config.get('embed_dim')
    image_size = full_config.get('image_size')
    patch_size_embed = full_config.get('patch_size_embed')
    growth_rate = full_config.get('growth_rate')
    dilation_rates = full_config.get('dilation_rates')
    compression_rates = full_config.get('compression_rates')
    dropout = full_config.get('dropout')
    drop_path = full_config.get('drop_path')
    learning_rate = full_config.get('learning_rate')
    weight_decay = full_config.get('weight_decay')
    batch_size = full_config.get('batch_size')
    optimizer_type = full_config.get('optimizer')
    num_classes = config.get('num_classes', 5)
    
    # Log configuration
    print(f"\n{'='*50}")
    print(f"Job {job_id} - Configuration:")
    print(f"  Model: {model_type}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  image_size: {image_size}")
    print(f"  patch_size_embed: {patch_size_embed}")
    print(f"  growth_rate: {growth_rate}")
    print(f"  dilation_rates: {dilation_rates}")
    print(f"  compression_rates: {compression_rates}")
    print(f"  drop_path: {drop_path}")
    print(f"  dropout: {dropout}")
    print(f"  learning_rate: {learning_rate:.2e}")
    print(f"  weight_decay: {weight_decay:.2e}")
    print(f"  batch_size: {batch_size}")
    print(f"  optimizer: {optimizer_type}")
    print(f"{'='*50}")
    
    # Lipschitz settings
    lip_lambda_upper = config.get('lip_lambda_upper', 0.01)
    lip_lambda_lower = config.get('lip_lambda_lower', 0.001)
    lip_upper_bound = config.get('lipschitz_upper_bound', 10.0)
    lip_lower_bound = config.get('lipschitz_lower_bound', 0.1)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(CONFIG['checkpoint_dir'], f'job_{job_id}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup transforms
    transform_train = T.Compose([
        T.Resize([image_size + 32, image_size + 32]),
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = T.Compose([
        T.Resize([image_size, image_size]),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataloaders
    train_dataset = SimpleImageFolder(df=train_df, transform=transform_train)
    val_dataset = SimpleImageFolder(df=val_df, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0, pin_memory=False)
    
    # Create model
    if model_type == 'SwinAD2Net_ASPP_like':
        model = SwinAD2Net_ASPP_like(
            num_classes=num_classes,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size_embed=patch_size_embed,
            growth_rate=growth_rate,
            dilation_rates=dilation_rates,
            compression_rates=compression_rates,
            drop_path=drop_path,
            dropout=dropout
        )
    elif model_type == 'A2SDNet121':
        model = A2SDNet121(num_classes=num_classes)
    elif model_type == 'Densenet121':
        model = Densenet121(num_classes=num_classes)
    else:
        model = SwinAD2Net(
            num_classes=num_classes,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size_embed=patch_size_embed,
            growth_rate=growth_rate,
            dilation_rates=dilation_rates
        )
    
    model = model.to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Job {job_id}: Resumed from epoch {start_epoch}")
    
    # Setup Lipschitz regularizer
    lipschitz_regularizer = LipschitzRegularizer(
        upper_bound=lip_upper_bound,
        lower_bound=lip_lower_bound,
        lambda_upper=lip_lambda_upper,
        lambda_lower=lip_lambda_lower,
        use_exact_svd=False,
        n_power_iterations=1
    )
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                               weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision
    use_amp = device.startswith('cuda') and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None
    DTYPE = torch.bfloat16
    
    # Early stopping
    early_stopping = AdaptiveEarlyStopping(
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta'],
        divergence_threshold=CONFIG['divergence_threshold'],
        mode='min'
    )
    
    # Training history
    history = {
        'loss_train': [], 'acc_train': [],
        'loss_val': [], 'acc_val': [],
        'lip_loss': [], 'lip_bound': []
    }
    
    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        running_lip_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with autocast(device_type='cuda', dtype=DTYPE):
                    outputs = model(inputs)
                    ce_loss = criterion(outputs, labels)
                    
                    # Compute Lipschitz regularization
                    lip_loss, _ = lipschitz_regularizer.compute_regularization_loss(model)
                    loss = ce_loss + lip_loss
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                ce_loss = criterion(outputs, labels)
                lip_loss, _ = lipschitz_regularizer.compute_regularization_loss(model)
                loss = ce_loss + lip_loss
                loss.backward()
                optimizer.step()
            
            running_loss += ce_loss.item() * inputs.size(0)
            running_lip_loss += lip_loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        avg_lip_loss = running_lip_loss / total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        # Compute Lipschitz bound
        lip_bound = lipschitz_regularizer.get_network_lipschitz_bound(model)
        
        # Update history
        history['loss_train'].append(train_loss)
        history['acc_train'].append(train_acc)
        history['loss_val'].append(val_loss)
        history['acc_val'].append(val_acc)
        history['lip_loss'].append(avg_lip_loss)
        history['lip_bound'].append(lip_bound)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, best_checkpoint_path)
        
        # Report intermediate results if callback provided
        reference_best = None
        if intermediate_callback:
            reference_best = intermediate_callback(
                epoch,
                {
                    'val_loss': val_loss,
                    'val_accuracy': val_acc / 100,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc / 100,
                    'lip_loss': avg_lip_loss,
                    'lip_bound': lip_bound
                }
            )
        
        # Check early stopping
        should_stop, reason = early_stopping.check(val_loss, reference_best)
        if should_stop:
            print(f"Job {job_id}: Early stopping at epoch {epoch} - {reason}")
            break
        
        scheduler.step()
        
        # Progress logging (every 10 epochs)
        if epoch % 10 == 0:
            print(f"Job {job_id} Epoch {epoch}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%, Lip: {lip_bound:.2e}")
    
    # Final validation with full metrics
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    final_metrics = {
        'val_accuracy': accuracy_score(all_labels, all_preds),
        'val_recall': recall_score(all_labels, all_preds, average='weighted'),
        'val_precision': precision_score(all_labels, all_preds, average='weighted'),
        'val_f1': f1_score(all_labels, all_preds, average='weighted'),
        'val_loss': history['loss_val'][-1] if history['loss_val'] else float('inf'),
        'lip_bound': lip_bound
    }
    
    return {
        'metrics': final_metrics,
        'history': history,
        'checkpoint_path': best_checkpoint_path
    }


# ============================================================================
# HYPERBAND TRAINING LOOP
# ============================================================================

def run_hyperband_training(
    df: pd.DataFrame,
    config_space: Dict[str, Any],
    max_budget: int = 100,
    eta: int = 3,
    num_workers: int = 4,
    device_list: Optional[List[str]] = None
):
    """
    Run Hyperband hyperparameter optimization with parallel training.
    
    Args:
        df: Full dataset DataFrame
        config_space: Hyperparameter search space
        max_budget: Maximum epochs per configuration
        eta: Reduction factor
        num_workers: Number of parallel workers
        device_list: List of devices for workers
    """
    
    # Split data
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"Starting Hyperband Training")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Max budget: {max_budget} epochs")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}\n")
    
    # Initialize scheduler
    scheduler = HyperbandScheduler(
        max_budget=max_budget,
        eta=eta,
        metric='val_loss',
        mode='min',
        checkpoint_dir=CONFIG['checkpoint_dir']
    )
    
    # Initialize brackets with configurations
    scheduler._init_brackets(config_space)
    
    # Device setup
    if device_list is None:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_list = [f"cuda:{i % gpu_count}" for i in range(num_workers)]
        else:
            device_list = ["cpu"] * num_workers
    
    # Run training for each bracket
    all_results = []
    
    while not scheduler.is_complete():
        # Get pending trials
        pending_trials = scheduler.get_next_trials()
        
        if not pending_trials:
            # Check if any bracket can advance
            scheduler.save_state()
            time.sleep(1)
            continue
        
        print(f"\n{'='*40}")
        print(f"Running {len(pending_trials)} trials")
        print(f"{'='*40}")
        
        # Group trials by budget for efficient batching
        trials_by_budget = {}
        for trial in pending_trials:
            budget = trial.budget
            if budget not in trials_by_budget:
                trials_by_budget[budget] = []
            trials_by_budget[budget].append(trial)
        
        # Process each budget group
        for budget, trials in trials_by_budget.items():
            print(f"\nTraining {len(trials)} configs for {budget} epochs")
            
            # Train in parallel batches
            batch_size = min(len(trials), num_workers)
            
            for i in range(0, len(trials), batch_size):
                batch = trials[i:i+batch_size]
                
                # Use multiprocessing for parallel training
                from concurrent.futures import ProcessPoolExecutor, as_completed
                
                futures = {}
                
                # Note: For GPU training, we run sequentially to avoid memory issues
                # For CPU training, we can parallelize
                if all(d == 'cpu' for d in device_list):
                    with ProcessPoolExecutor(max_workers=batch_size) as executor:
                        for j, trial in enumerate(batch):
                            device = device_list[j % len(device_list)]
                            
                            # Prepare config
                            trial_config = trial.config.copy()
                            trial_config['image_size'] = CONFIG['image_size']
                            trial_config['num_classes'] = CONFIG['num_classes']
                            trial_config['lipschitz_upper_bound'] = CONFIG['lipschitz_upper_bound']
                            trial_config['lipschitz_lower_bound'] = CONFIG['lipschitz_lower_bound']
                            
                            future = executor.submit(
                                train_single_config,
                                train_df=train_df,
                                val_df=val_df,
                                config=trial_config,
                                epochs=budget,
                                device=device,
                                job_id=trial.trial_id,
                                checkpoint_path=trial.state_dict_path
                            )
                            futures[future] = trial
                            trial.status = TrialStatus.RUNNING
                            trial.start_time = datetime.now()
                        
                        # Collect results
                        for future in as_completed(futures):
                            trial = futures[future]
                            try:
                                result = future.result()
                                scheduler.report_trial_result(
                                    trial,
                                    result['metrics'],
                                    result['checkpoint_path']
                                )
                                all_results.append({
                                    'trial_id': trial.trial_id,
                                    'config': trial.config,
                                    'metrics': result['metrics'],
                                    'history': result['history']
                                })
                                print(f"Trial {trial.trial_id} completed: "
                                      f"val_loss={result['metrics']['val_loss']:.4f}, "
                                      f"val_acc={result['metrics']['val_accuracy']:.4f}")
                            except Exception as e:
                                print(f"Trial {trial.trial_id} failed: {e}")
                                scheduler.report_trial_failure(trial, str(e))
                else:
                    # Sequential GPU training
                    for j, trial in enumerate(batch):
                        device = device_list[j % len(device_list)]
                        
                        trial_config = trial.config.copy()
                        trial_config['image_size'] = CONFIG['image_size']
                        trial_config['num_classes'] = CONFIG['num_classes']
                        trial_config['lipschitz_upper_bound'] = CONFIG['lipschitz_upper_bound']
                        trial_config['lipschitz_lower_bound'] = CONFIG['lipschitz_lower_bound']
                        
                        trial.status = TrialStatus.RUNNING
                        trial.start_time = datetime.now()
                        
                        try:
                            result = train_single_config(
                                train_df=train_df,
                                val_df=val_df,
                                config=trial_config,
                                epochs=budget,
                                device=device,
                                job_id=trial.trial_id,
                                checkpoint_path=trial.state_dict_path
                            )
                            scheduler.report_trial_result(
                                trial,
                                result['metrics'],
                                result['checkpoint_path']
                            )
                            all_results.append({
                                'trial_id': trial.trial_id,
                                'config': trial.config,
                                'metrics': result['metrics'],
                                'history': result['history']
                            })
                            print(f"Trial {trial.trial_id} completed: "
                                  f"val_loss={result['metrics']['val_loss']:.4f}, "
                                  f"val_acc={result['metrics']['val_accuracy']:.4f}")
                        except Exception as e:
                            print(f"Trial {trial.trial_id} failed: {e}")
                            import traceback
                            traceback.print_exc()
                            scheduler.report_trial_failure(trial, str(e))
        
        # Print progress
        print(scheduler.get_progress_summary())
        scheduler.save_state()
    
    # Final results
    print(f"\n{'='*60}")
    print("HYPERBAND OPTIMIZATION COMPLETE")
    print(f"{'='*60}\n")
    
    try:
        best_config, best_metric, best_trial = scheduler.get_best_config()
        print("Best Configuration:")
        print(f"  Metric (val_loss): {best_metric:.4f}")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
        print(f"\nBest model saved at: {best_trial.state_dict_path}")
    except:
        print("No valid results found")
    
    # Save all results
    results_path = os.path.join(CONFIG['checkpoint_dir'], 'all_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to: {results_path}")
    
    return scheduler, all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hyperband Parallel Training')
    parser.add_argument('--data-dir', type=str, default=CONFIG['data_dir'],
                       help='Directory containing training data')
    parser.add_argument('--max-budget', type=int, default=CONFIG['max_budget'],
                       help='Maximum epochs per configuration')
    parser.add_argument('--eta', type=int, default=3,
                       help='Hyperband reduction factor')
    parser.add_argument('--workers', type=int, default=CONFIG['num_workers'],
                       help='Number of parallel workers')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from saved state')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['data_dir'] = args.data_dir
    CONFIG['max_budget'] = args.max_budget
    CONFIG['num_workers'] = args.workers
    
    # Load dataset
    print("Loading dataset...")
    df = load_dataset(CONFIG['data_dir'], CONFIG['label_map'])
    
    if len(df) == 0:
        print(f"ERROR: No images found in {CONFIG['data_dir']}")
        print("Please ensure the data directory contains images with class names in the path")
        return
    
    print(f"Loaded {len(df)} images")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Setup devices
    device_list = None
    if args.device:
        device_list = [args.device] * args.workers
    
    # Run Hyperband
    scheduler, results = run_hyperband_training(
        df=df,
        config_space=HYPERPARAMETER_SPACE,
        max_budget=args.max_budget,
        eta=args.eta,
        num_workers=args.workers,
        device_list=device_list
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
