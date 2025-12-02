#!/usr/bin/env python3

# ============================================================================
# This is a test script to train the SwinAD2Net model on the SipakMed dataset.
# without k-fold cross-validation.
# It performs data loading, augmentation, model training, and saves training history and graphs.
# ============================================================================

from .train import *
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle
from dataset import augment_data_prepared
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


maps = {
    'koilocytotic': 0,
    'dyskeratotic': 1,
    'metaplastic': 2,
    'superficial': 3,
    'parabasal': 4,

}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')
paths = []
for root, dirs, files in os.walk('./../../../data'):
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


df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

_, paths_aug, df_train = augment_data_prepared(data_dir=None, df_paths=df_train, augmentations_per_image=5)

print(f'Generated data {_}')
print(f'The model is training with {len(df_train)} data')
print(f'Class distribution:\n{df_train["label"].value_counts()}')


model, history, scores, predictions = train_swinad2net(
    train_df=df_train,
    val_df=df_val,
    num_classes=5,
    image_size=224,
    embed_dim=128,
    growth_rate=32,
    dilation_rates=[1, 2, 3],
    batch_size=32,
    num_epochs=700,
    learning_rate=1e-3,
    checkpoint_dir='./checkpoints_256_embed/kfold_experiment',
    device=device,
    log_dir='runs',
)

for path in paths_aug:
    os.remove(path)

# salvar history, scores e predictions em pickle
os.makedirs('./history_and_graphs', exist_ok=True)
with open('./history_and_graphs/history.pkl', 'wb') as f:
    pickle.dump(history, f)
with open('./history_and_graphs/scores.pkl', 'wb') as f:
    pickle.dump(scores, f)
with open('./history_and_graphs/predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)

# assumir que history é um dict com chaves simples como abaixo
train_loss = history.get('train_loss', [])
val_loss = history.get('val_loss', [])
train_acc = history.get('train_acc', [])
val_acc = history.get('val_acc', [])

epochs = range(1, len(train_loss) + 1) if train_loss else range(1, len(val_loss) + 1)

# gráfico de loss
plt.figure()
if train_loss:
    plt.plot(epochs, train_loss, label='Train Loss')
if val_loss:
    plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./history_and_graphs/loss.png')
plt.close()

# gráfico de acurácia
if train_acc or val_acc:
    plt.figure()
    if train_acc:
        plt.plot(epochs, train_acc, label='Train Acc')
    if val_acc:
        plt.plot(epochs, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./history_and_graphs/accuracy.png')
    plt.close()

# assumir que predictions é um dict simples com chaves 'y_true_train', 'y_prob_train', 'y_true_val', 'y_prob_val'
y_true_train = np.array(predictions.get('y_true_train', []))
y_prob_train = np.array(predictions.get('y_prob_train', []))
y_true_val = np.array(predictions.get('y_true_val', []))
y_prob_val = np.array(predictions.get('y_prob_val', []))

# matrizes de confusão
if y_true_train.size and y_prob_train.size:
    y_pred_train = (y_prob_train >= 0.5).astype(int)
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    plt.figure()
    plt.imshow(cm_train, cmap='Blues')
    plt.title('Matriz de Confusão - Treino')
    plt.colorbar()
    plt.xlabel('Predito')
    plt.ylabel('Real')
    for (i, j), v in np.ndenumerate(cm_train):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig('./history_and_graphs/confusion_train.png')
    plt.close()

if y_true_val.size and y_prob_val.size:
    y_pred_val = (y_prob_val >= 0.5).astype(int)
    cm_val = confusion_matrix(y_true_val, y_pred_val)
    plt.figure()
    plt.imshow(cm_val, cmap='Blues')
    plt.title('Matriz de Confusão - Validação')
    plt.colorbar()
    plt.xlabel('Predito')
    plt.ylabel('Real')
    for (i, j), v in np.ndenumerate(cm_val):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig('./history_and_graphs/confusion_val.png')
    plt.close()

# curva ROC
if y_true_train.size and y_prob_train.size:
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_prob_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    plt.figure()
    plt.plot(fpr_train, tpr_train, label=f'Treino (AUC = {roc_auc_train:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - Treino')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('./history_and_graphs/roc_train.png')
    plt.close()

if y_true_val.size and y_prob_val.size:
    fpr_val, tpr_val, _ = roc_curve(y_true_val, y_prob_val)
    roc_auc_val = auc(fpr_val, tpr_val)
    plt.figure()
    plt.plot(fpr_val, tpr_val, label=f'Validação (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - Validação')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('./history_and_graphs/roc_val.png')
    plt.close()


