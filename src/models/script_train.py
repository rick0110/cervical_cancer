from train import *
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
import pickle

maps = {
    'carcinoma': 1,
    'dysplastic': 1,
    'koilocytotic': 1,
    'dyskeratotic': 1,
    'metaplastic': 0,
    'columnar': 0,
    'normal-intermediate': 0,
    'superficial': 0,
    'parabasal': 0,

}

device = "cuda" if torch.cuda.is_available() else "cpu"
paths = []
for root, dirs, files in os.walk('./../../data'):
    for file in files:
        if file.lower().endswith('.bmp'):
            paths.append(os.path.abspath(os.path.join(root, file)))

df = pd.DataFrame({'path': paths})

def label_map_from_path(path):
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

model_state_dict = None # torch.load('./checkpoints/kfold_experiment/checkpoint_epoch_50.pth')['model_state_dict']

for train_index, val_index in k_fold.split(df):
    df_train = df.iloc[train_index]
    df_val = df.iloc[val_index]

    model, history, scores, predictions = train_swinad2net(
        train_df=df_train,
        val_df=df_val,
        num_classes=2,
        image_size=224,
        embed_dim=128,
        growth_rate=32,
        dilation_rates=[1, 2, 3],
        batch_size=32,
        num_epochs=200,
        learning_rate=1e-3,
        checkpoint_dir='./checkpoints_256_embed/kfold_experiment',
        device=device,
        log_dir='runs',
        state_dict=model_state_dict
    )
    models[f'fold_{fold}'] = [model.state_dict(), history, scores, predictions]
    fold += 1

os.makedirs('./checkpoints/kfold_experiment', exist_ok=True)
with open('./checkpoints/kfold_experiment/kfold_results.pkl', 'wb') as f:
    pickle.dump(models, f)

print(f"\n{'='*60}")
print(f"K-Fold Training Complete!")
print(f"Results were saved in: ./checkpoints/kfold_experiment/kfold_results.pkl")
print(f"Total folds: {len(models)}")
print(f"{'='*60}\n")

print("Summary of Metrics by Fold:")
print(f"{'='*60}")
for fold_name, (state_dict, history, scores, predictions) in models.items():
    print(f"\n{fold_name}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
print(f"\n{'='*60}\n")
