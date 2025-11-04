from train import *
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold

maps = {
    'carcinoma': 1,
    'dysplastic': 1,
    'columnar': 0,
    'intermediate': 0,
    'superficiel': 0
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = []
for root, dirs, files in os.walk('./../data/data_prepared'):
    for file in files:
        paths.append(os.path.abspath(os.path.join(root, file)))

df = pd.DataFrame({'path': paths})

def label_map_from_path(path):
    for key in maps.keys():
        if key in path.lower():
            return maps[key]
    return np.nan

df['label'] = df['path'].apply(label_map_from_path)

df_train, df_val = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=342)

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

models = {}  #-> model[ state_dict, history{loss_train, acc_train, loss_val, acc_val}, scores[val_acc, val_recall, val_precision], predictions[val_predictions, val_targets] ]
fold = 1

for train_index, val_index in k_fold.split(df):
    df_train = df.iloc[train_index]
    df_val = df.iloc[val_index]

    model, history, scores, predictions = train_swinad2net(
        train_df=df_train,
        val_df=df_val,
        num_classes=2,
        image_size=(224, 224),
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-3,
        checkpoint_dir='./checkpoints/kfold_experiment'
    )
    
    models[f'fold_{fold}'] = [model.state_dict(), ]
    fold += 1
