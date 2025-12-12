from .train import *
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
import pickle
from .dataset import augment_data_prepared
from .model import SwinAD2Net, SwinAD2Net_ASPP_like, Densenet121
from typing import Optional, Tuple, List, Dict

maps = {
    'koilocytotic': 0,
    'dyskeratotic': 1,
    'metaplastic': 2,
    'superficial': 3,
    'parabasal': 4
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

paths = []
for root, dirs, files in os.walk('./data'):
    for file in files:
        if file.lower().endswith('.bmp'):
            paths.append(os.path.join(root, file))


df = pd.DataFrame({'path': paths})

def label_map_from_path(path):
    # use global maps
    for key in maps.keys():
        if key in path.lower():
            return maps[key]
    return np.nan

df['label'] = df['path'].apply(label_map_from_path)
df = df.sample(frac=1, random_state=847).dropna().reset_index(drop=True)

print(f'The model is training with {len(df)} data')
print(f'Class distribution:\n{df["label"].value_counts()}')

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

models = {}  #-> model[ state_dict, history{loss_train, acc_train, loss_val, acc_val}, scores{val_acc, val_recall, val_precision}, predictions{val_predictions, val_targets} ]
fold = 1

for train_index, val_index in k_fold.split(df):
    df_train = df.iloc[train_index]
    df_val = df.iloc[val_index]

    model, history, scores, predictions = train_swinad2net(
        train_df=df_train,
        val_df=df_val,
        model=SwinAD2Net_ASPP_like(num_classes=5,
                        embed_dim=128,
                        image_size=224,
                        patch_size_embed=4,
                        growth_rate=32,
                        dilation_rates=[1, 2, 3, 4],
                        compression_rates=[0.25, 0.25, 0.25, 0.25],
                        drop_path=0.2,
                        dropout=0.7
                        ),
        batch_size=32,
        num_epochs=400,
        patience=40,
        learning_rate=1e-3,
        weight_decay=1e-3,
        checkpoint_dir=f'./src/swinad2net/models/checkpoints/kfold_experiment_ASPP_like_64_embed/fold_{fold}',
        device=device,
        log_dir=f'./src/swinad2net/models/SwinAD2Net_ASPP_like/foldruns/fold_{fold}',
        state_dict=None,
        epoch_stopped=None,
        optimizer="AdamW"
    )
    models[f'fold_{fold}'] = [model.state_dict(), history, scores, predictions]

    os.makedirs(f'./src/swinad2net/models/checkpoints/kfold_experiment_ASPP_like_64_embed', exist_ok=True)
    with open(f'./src/swinad2net/models/checkpoints/kfold_experiment_ASPP_like_64_embed/kfold_results.pkl', 'wb') as f:
        pickle.dump(models, f)

    fold += 1


print(f"\n{'='*60}")
print(f"K-Fold Training Complete!")
print(f"Results were saved in: ./src/swinad2net/models/checkpoints/kfold_experiment_ASPP_like_128_embed/kfold_results.pkl")
print(f"Total folds: {len(models)}")
print(f"{'='*60}\n")

print("Summary of Metrics by Fold:")
print(f"{'='*60}")
for fold_name, (state_dict, history, scores, predictions) in models.items():
    print(f"\n{fold_name}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
print(f"\n{'='*60}\n")


k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

models = {}  #-> model[ state_dict, history{loss_train, acc_train, loss_val, acc_val}, scores{val_acc, val_recall, val_precision}, predictions{val_predictions, val_targets} ]
fold = 1

for train_index, val_index in k_fold.split(df):
    df_train = df.iloc[train_index]
    df_val = df.iloc[val_index]

    model, history, scores, predictions = train_swinad2net(
        train_df=df_train,
        val_df=df_val,
        model=Densenet121(num_classes=5),
        image_size=192,
        patience=40,
        batch_size=32,
        num_epochs=400,
        learning_rate=1e-4,
        checkpoint_dir=f'./src/swinad2net/models/checkpoints/kfold_experiment_DenseNet121/fold_{fold}',
        device=device,
        log_dir=f'./src/swinad2net/models/DenseNet121/foldruns/fold_{fold}',
        state_dict=None,
        epoch_stopped=None,
        optimizer="AdamW"
    )
    models[f'fold_{fold}'] = [model.state_dict(), history, scores, predictions]

    os.makedirs('./src/swinad2net/models/checkpoints/kfold_experiment_DenseNet121', exist_ok=True)
    with open('./src/swinad2net/models/checkpoints/kfold_experiment_DenseNet121/kfold_results.pkl', 'wb') as f:
        pickle.dump(models, f)

    fold += 1

print(f"\n{'='*60}")
print(f"K-Fold Training Complete!")
print(f"Results were saved in: ./src/swinad2net/models/checkpoints/kfold_experiment_DenseNet121/kfold_results.pkl")
print(f"Total folds: {len(models)}")
print(f"{'='*60}\n")

print("Summary of Metrics by Fold:")
print(f"{'='*60}")
for fold_name, (state_dict, history, scores, predictions) in models.items():
    print(f"\n{fold_name}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
print(f"\n{'='*60}\n")
