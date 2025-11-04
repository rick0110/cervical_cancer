import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from model import SwinAD2Net
from dataset import SimpleImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def train_swinad2net(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    num_classes: int = 2,
    image_size: int = 224,
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "runs",
    device: str = "cuda"
):
    """
    Função simples de treinamento para o modelo SwinAD2Net.
    
    Args:
        train_df: DataFrame com colunas 'path' (caminho das imagens) e 'label' (rótulos numéricos)
        val_df: DataFrame de validação (opcional)
        num_classes: número de classes para classificação
        image_size: tamanho das imagens (altura e largura)
        batch_size: tamanho do batch
        num_epochs: número de épocas de treinamento
        learning_rate: taxa de aprendizado inicial
        weight_decay: weight decay para regularização L2
        checkpoint_dir: diretório para salvar checkpoints
        log_dir: diretório para logs do TensorBoard
        device: 'cuda' ou 'cpu'
    
    Returns:
        Tupla (modelo treinado, dicionário de métricas)
    """
    """
    Função simples de treinamento para o modelo SwinAD2Net.
    
    Args:
        train_df: DataFrame com colunas 'path' (caminho das imagens) e 'label' (rótulos numéricos)
        val_df: DataFrame de validação (opcional)
        num_classes: número de classes para classificação
        image_size: tamanho das imagens (altura e largura)
        batch_size: tamanho do batch
        num_epochs: número de épocas de treinamento
        learning_rate: taxa de aprendizado inicial
        weight_decay: weight decay para regularização L2
        checkpoint_dir: diretório para salvar checkpoints
        log_dir: diretório para logs do TensorBoard
        device: 'cuda' ou 'cpu'
    
    Returns:
        Modelo treinado
    """
    
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"\n{'='*60}")
    print(f"Treinamento do SwinAD2Net")
    print(f"Device: {device} | Classes: {num_classes} | Epochs: {num_epochs}")
    print(f"{'='*60}\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    import torchvision.transforms as T
    
    transform_train = T.Compose([
        T.Resize([image_size, image_size]),  # Usar lista ao invés de tupla
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = T.Compose([
        T.Resize([image_size, image_size]),  # Usar lista ao invés de tupla
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SimpleImageFolder(df=train_df, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = None
    if val_df is not None:
        val_dataset = SimpleImageFolder(df=val_df, transform=transform_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")
    else:
        print(f"Train: {len(train_dataset)}\n")
    
    print("Criando modelo SwinAD2Net...")
    model = SwinAD2Net(num_classes=num_classes, image_size=image_size).to(device)
    print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    writer = SummaryWriter(log_dir=log_dir)
    best_val_acc = 0.0
    history = {'loss_train': [], 'acc_train': [], 'loss_val': [], 'acc_val': []}
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nÉpoca {epoch}/{num_epochs}")
        print("-" * 40)
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc="Train", leave=False)
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
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
                print(f"✓ Melhor modelo salvo! (Val Acc: {val_acc:.2f}%)")
        
        scheduler.step()
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                      os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
            print(f"✓ Checkpoint salvo: epoch_{epoch}")
    
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, "final_model.pth"))
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Treinamento concluído! | Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print(f"{'='*60}\n")

    # Calcular métricas apenas se houver validação
    scores = {}
    predictions_dict = {}  # Inicializar vazio para evitar erros quando val_df=None
    
    if val_df is not None and val_loader is not None:
        val_targets = val_df['label'].values
        val_predictions = []  # CORRIGIDO: typo val_predinctions
        
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
            'val_labels': val_targets.tolist(),  # Converter para lista para serialização
            'val_predictions': val_predictions
        }

        print(f"\n{'='*60}")
        print("Métricas de Validação:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
        print(f"{'='*60}\n")

    return model, history, scores, predictions_dict


if __name__ == '__main__':
    # Exemplo de uso com DataFrame
    import pandas as pd
    
    # Criar DataFrame de exemplo (substitua pelos seus dados)
    train_data = {
        'path': [
            'data_prepared/class_0/img1.BMP',
            'data_prepared/class_0/img2.BMP',
            'data_prepared/class_1/img3.BMP',
        ],
        'label': [0, 0, 1]
    }
    train_df = pd.DataFrame(train_data)
    
    # Treinar modelo
    model = train_swinad2net(
        train_df=train_df,
        val_df=None,
        num_classes=2,
        image_size=224,
        batch_size=16,
        num_epochs=50,
        learning_rate=1e-3
    )

