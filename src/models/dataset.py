from typing import Optional, Tuple, List
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import random


class RandomImageDataset(Dataset):
    """Dataset que gera imagens aleatórias para testes e desenvolvimento.

    Comportamento novo: se um `DataFrame` for passado via `df`, o dataset usa os
    caminhos e labels do DataFrame em vez de gerar imagens aleatórias.

    Args:
        length: comprimento do dataset (usado apenas quando df is None)
        image_size: tamanho das imagens aleatórias
        num_classes: número de classes (apenas para geração aleatória)
        transform: transformações aplicadas à imagem
        df: pandas DataFrame com colunas contendo caminhos e labels. Se fornecido,
            o dataset usa os caminhos do DataFrame. Espera colunas 'path' e 'label'
            por padrão ou use `path_col`/`label_col`.
        path_col, label_col: nomes das colunas no DataFrame
    """

    def __init__(self,
                 length: int = 1000,
                 image_size: Tuple[int, int] = (64, 64),
                 num_classes: int = 2,
                 transform: Optional[T.Compose] = None,
                 df: Optional[pd.DataFrame] = None,
                 path_col: str = 'path',
                 label_col: str = 'label'):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform or T.Compose([T.ToTensor()])

        self.samples: List[tuple] = []
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise ValueError('df deve ser um pandas.DataFrame')
            if path_col not in df.columns:
                raise ValueError(f"Coluna de caminho '{path_col}' não encontrada no DataFrame")
            if label_col not in df.columns:
                df = df.copy()
                df[label_col] = 0

            for _, row in df.iterrows():
                self.samples.append((str(row[path_col]), int(row[label_col])))

    def __len__(self):
        if self.samples:
            return len(self.samples)
        return self.length

    def __getitem__(self, idx):
        if self.samples:
            path, label = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        else:
            raise ValueError("Dataset foi inicializado sem DataFrame de amostras.")


class SimpleImageFolder(Dataset):
    """Dataset que recebe um DataFrame com caminhos de imagens e rótulos numéricos.
    
    O DataFrame deve conter colunas para path (caminho da imagem) e label (rótulo numérico).
    Todos os dados são armazenados em self.data para acesso rápido.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 transform: Optional[T.Compose] = None,
                 path_col: str = 'path',
                 label_col: str = 'label'):
        """
        Inicializa o dataset a partir de um DataFrame.
        
        Args:
            df: DataFrame com caminhos das imagens e labels
            transform: transformações a aplicar nas imagens
            path_col: nome da coluna com os caminhos das imagens
            label_col: nome da coluna com os labels numéricos
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df deve ser um pandas.DataFrame')
        
        if path_col not in df.columns:
            raise ValueError(f"Coluna '{path_col}' não encontrada no DataFrame")
        
        if label_col not in df.columns:
            raise ValueError(f"Coluna '{label_col}' não encontrada no DataFrame")
        
        self.data = df[[path_col, label_col]].copy()
        self.data.columns = ['path', 'label']  # padroniza nomes
        self.data['label'] = self.data['label'].astype(int)  # garante que label é int

        self.transform = transform or T.Compose([T.Resize((224, 224)), T.ToTensor()])

        self.num_classes = self.data['label'].nunique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retorna uma tupla (imagem_tensor, label) para o índice dado.
        """
        row = self.data.iloc[idx]
        img_path = row['path']
        label = row['label']
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def prepare_bmp_only(src_dir: str, dst_dir: str = "data_prepared") -> int:
    if not os.path.isdir(src_dir):
        raise ValueError(f"src_dir não existe ou não é um diretório: {src_dir}")

    abs_src = os.path.abspath(src_dir)
    abs_dst = os.path.abspath(dst_dir)
    if abs_src == abs_dst or abs_dst.startswith(abs_src + os.sep):
        raise ValueError("dst_dir não pode ser o mesmo que src_dir nem estar dentro de src_dir")

    copied = 0
    for dirpath, dirnames, filenames in os.walk(src_dir):
        rel_dir = os.path.relpath(dirpath, src_dir)
        if rel_dir == '.':
            rel_dir = ''
        target_dir = os.path.join(dst_dir, rel_dir) if rel_dir else dst_dir

        for fname in filenames:
            if fname.lower().endswith('.bmp'):
                os.makedirs(target_dir, exist_ok=True)
                src_path = os.path.join(dirpath, fname)
                dst_path = os.path.join(target_dir, fname)
                try:
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                except Exception as e:
                    # se houver erro ao copiar, continua e registra em stderr
                    print(f"Erro copiando {src_path} -> {dst_path}: {e}")

    return copied


def augment_data_prepared(data_dir: str = None, 
                            df_paths = None,
                          augmentations_per_image: int = 3) -> int:

    generated = 0
    if data_dir:
        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir não existe: {data_dir}")
    
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if fname.lower().endswith('.bmp') and '_aug' not in fname:
                    src_path = os.path.join(root, fname)
                    try:
                        img = Image.open(src_path).convert('RGB')
                        base_name, ext = os.path.splitext(fname)

                        for i in range(augmentations_per_image):
                            aug_img = apply_random_augmentation(img)
                            aug_filename = f"{base_name}_aug{i+1}{ext}"
                            aug_path = os.path.join(root, aug_filename)
                            aug_img.save(aug_path)
                            generated += 1

                    except Exception as e:
                        print(f"Erro processando {src_path}: {e}")
                        continue
    elif df_paths is not None and len(df_paths) > 0:
        paths_aug = []
        labels_aug = []
        for _, row in df_paths.iterrows():
            src_path = str(row['path'])
            label = int(row['label'])
            if src_path.lower().endswith('.bmp') and '_aug' not in src_path:
                try:
                    img = Image.open(src_path).convert('RGB')
                    base_name, ext = os.path.splitext(os.path.basename(src_path))

                    for i in range(augmentations_per_image):
                        aug_img = apply_random_augmentation(img)
                        aug_filename = f"{base_name}_aug{i+1}{ext}"
                        aug_path = os.path.join(os.path.dirname(src_path), aug_filename)
                        aug_img.save(aug_path)
                        paths_aug.append(aug_path)
                        labels_aug.append(label)
                        generated += 1

                except Exception as e:
                    print(f"Erro processando {src_path}: {e}")

        df_aug = pd.DataFrame({'path': paths_aug, 'label': labels_aug})
        df_paths = pd.concat([df_paths.reset_index(drop=True), df_aug], ignore_index=True)
        return generated, paths_aug, df_paths


    return generated


def apply_random_augmentation(img: Image.Image) -> Image.Image:
    """Aplica transformações aleatórias a uma imagem PIL.
    
    Transformações possíveis:
    - Rotação aleatória (-30 a 30 graus)
    - Flip horizontal (50% chance)
    - Flip vertical (50% chance)
    - Ajuste de brilho (0.8 a 1.2)
    - Ajuste de contraste (0.8 a 1.2)
    - Zoom aleatório (0.9 a 1.1)
    
    Retorna nova imagem PIL.
    """
    from PIL import ImageEnhance
    angle = random.uniform(-30, 30)
    img = img.rotate(angle, fillcolor=(0, 0, 0))
    
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    if random.random() > 0.3:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    if random.random() > 0.3:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    if random.random() > 0.5:
        w, h = img.size
        zoom_factor = random.uniform(0.9, 1.1)
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        if zoom_factor > 1.0:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
        else:
            from PIL import ImageOps
            delta_w = w - new_w
            delta_h = h - new_h
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            img = ImageOps.expand(img, padding, fill=(0, 0, 0))
    
    return img



if __name__ == '__main__':
    augment_data_prepared("./../data", 7)

