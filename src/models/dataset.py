#!/usr/bin/env python3

# ============================================================================
# This module defines dataset classes and utility functions for image datasets.
# It includes a RandomImageDataset for generating random samples of images and a
# SimpleImageFolder for loading images from a DataFrame. This also provides 
# functions to prepare and augment image data.

# - RandomImageDataset: Generates random samples of images for testing.
# - SimpleImageFolder: Loads images and labels from a DataFrame with image paths and labels.
# - augment_data_prepared: Augments images by applying random transformations.
# - augment_data_prepared: Augments images by applying random transformations.

# ============================================================================

from typing import Optional, Tuple, List
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import random


class RandomImageDataset(Dataset):
    """Dataset that select a random sample of images for testing and development.

    Args:
        length: length of the dataset (used only when df is None)
        image_size: size of the random images
        num_classes: number of classes (only for random generation)
        transform: transformations applied to the image
        df: pandas DataFrame with columns containing paths and labels. If provided,
            the dataset uses the paths from the DataFrame. Expects 'path' and 'label'
            columns by default or use `path_col`/`label_col`.
        path_col, label_col: column names in the DataFrame
    """

    def __init__(self,
                 length: int = 1000,
                 image_size: Tuple[int, int] = (224, 224),
                 num_classes: int = 2,
                 transform: Optional[T.Compose] = None,
                 df: Optional[pd.DataFrame] = None,
                 path_col: str = 'paths',
                 label_col: str = 'label'):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform or T.Compose([T.ToTensor()])

        self.samples: List[tuple] = []
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise ValueError('df must be a pandas.DataFrame')
            if path_col not in df.columns:
                raise ValueError(f"Path column '{path_col}' not found in DataFrame")
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in DataFrame")

            for _, row in df.iterrows():
                self.samples.append((str(row[path_col]), int(row[label_col])))
        else:
            raise ValueError("RandomImageDataset requires a DataFrame 'df' to initialize samples.")

    def __len__(self):
        """Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        if self.samples:
            return len(self.samples)
        return self.length

    def __getitem__(self, idx):
        """Returns a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: A tuple containing (image_tensor, label).
            
        Raises:
            ValueError: If the dataset was not initialized with a DataFrame.
        """
        if self.samples:
            path, label = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        else:
            raise ValueError("Dataset without samples DataFrame.")


class SimpleImageFolder(Dataset):
    """Dataset that receives a DataFrame with image paths and numeric labels.
    
    The DataFrame must contain columns for path (image path) and label (numeric label).
    All data is stored in self.data for fast access.

    Args:
        df: DataFrame with image paths and labels
        transform: transformations to apply to the images
        path_col: name of the column with image paths
        label_col: name of the column with numeric labels
    """

    def __init__(self,
                 df: pd.DataFrame,
                 transform: Optional[T.Compose] = None,
                 path_col: str = 'path',
                 label_col: str = 'label'):
        """Initializes the dataset from a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with image paths and labels.
            transform (Optional[T.Compose]): Transformations to apply to the images.
            path_col (str): Name of the column with image paths. Default is 'path'.
            label_col (str): Name of the column with numeric labels. Default is 'label'.
            
        Raises:
            ValueError: If df is not a pandas DataFrame or if specified columns are not found.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be a pandas.DataFrame')
        
        if path_col not in df.columns:
            raise ValueError(f"Column '{path_col}' not found in DataFrame")
        
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in DataFrame")
        
        self.data = df[[path_col, label_col]].copy()
        self.data.columns = ['path', 'label']  
        self.data['label'] = self.data['label'].astype(int) 

        self.transform = transform or T.Compose([T.Resize((224, 224)), T.ToTensor()])

        self.num_classes = self.data['label'].nunique()

    def __len__(self):
        """Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: A tuple containing (image_tensor, label) where image_tensor is a
                   transformed PIL Image converted to tensor and label is an integer.
        """
        row = self.data.iloc[idx]
        img_path = row['path']
        label = row['label']
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def prepare_bmp_only(src_dir: str, dst_dir: str = "data_prepared") -> int:
    """Copies only .bmp files from source directory to destination directory, preserving structure.
    
    This function recursively walks through the source directory, copying only .bmp files
    to the destination directory while maintaining the original directory structure.
    
    Args:
        src_dir (str): Source directory path containing .bmp files.
        dst_dir (str): Destination directory path. Default is 'data_prepared'.
        
    Returns:
        int: Number of files successfully copied.
        
    Raises:
        ValueError: If src_dir does not exist, is not a directory, or if dst_dir is
                    the same as or inside src_dir.
    """
    if not os.path.isdir(src_dir):
        raise ValueError(f"src_dir does not exist or is not a directory: {src_dir}")

    abs_src = os.path.abspath(src_dir)
    abs_dst = os.path.abspath(dst_dir)
    if abs_src == abs_dst or abs_dst.startswith(abs_src + os.sep):
        raise ValueError("dst_dir cannot be the same as src_dir nor be inside src_dir")

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
                    # if there is an error copying, continue and log to stderr
                    print(f"Error copying {src_path} -> {dst_path}: {e}")

    return copied


def augment_data_prepared(data_dir: str = None, 
                            df_paths: pd.DataFrame = None,
                          augmentations_per_image: int = 3):
    """Augments data by creating variations of existing images using random transformations.
    
    This function can work in two modes:
    1. Directory mode: Augments all .bmp images in the specified directory.
    2. DataFrame mode: Augments images specified in a DataFrame with 'path' and 'label' columns.
    
    Only non-augmented .bmp files (without '_aug' in filename) are processed to avoid
    recursively augmenting already augmented images.
    
    Args:
        data_dir (str, optional): Directory containing images to augment. If None, df_paths must be provided.
        df_paths (pd.DataFrame, optional): DataFrame with 'path' and 'label' columns containing
                                           image paths and labels to augment. If None, data_dir must be provided.
        augmentations_per_image (int): Number of augmented versions to create per image. Default is 3.
        
    Returns:
        int: Number of generated augmented images (when data_dir is provided).
        tuple: If df_paths is provided, returns (int, List[str], pd.DataFrame) containing:
               - Number of generated images
               - List of paths to generated images
               - Updated DataFrame with original and augmented images
        
    Raises:
        ValueError: If data_dir is provided but does not exist.
    """
    generated = 0
    if data_dir:
        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir does not exist: {data_dir}")
    
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                # Process only .bmp files that are not already augmented
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
                        print(f"Error processing {src_path}: {e}")
                        continue
        return generated
    elif df_paths is not None:
        paths_aug = []
        labels_aug = []
        for _, row in df_paths.iterrows():
            src_path = str(row['path'])
            label = int(row['label'])
            filename = os.path.basename(src_path)
            # Process only .bmp files that are not already augmented
            if filename.lower().endswith('.bmp') and '_aug' not in filename:
                try:
                    img = Image.open(src_path).convert('RGB')
                    base_name, ext = os.path.splitext(filename)

                    for i in range(augmentations_per_image):
                        aug_img = apply_random_augmentation(img)
                        aug_filename = f"{base_name}_aug{i+1}{ext}"
                        aug_path = os.path.join(os.path.dirname(src_path), aug_filename)
                        aug_img.save(aug_path)
                        paths_aug.append(aug_path)
                        labels_aug.append(label)
                        generated += 1

                except Exception as e:
                    print(f"Error processing {src_path}: {e}")

        df_aug = pd.DataFrame({'path': paths_aug, 'label': labels_aug})
        df_paths = pd.concat([df_paths.reset_index(drop=True), df_aug], ignore_index=True)
        return generated, paths_aug, df_paths
    
    raise ValueError("Either data_dir or df_paths must be provided")


def apply_random_augmentation(img: Image.Image) -> Image.Image:
    """Applies random transformations to a PIL image for data augmentation.
    
    This function applies a combination of random transformations to create
    variations of the input image. The transformations are applied sequentially
    with varying probabilities.
    
    Possible transformations:
    - Random rotation: -30 to 30 degrees (always applied)
    - Horizontal flip: 50% chance
    - Vertical flip: 50% chance
    - Brightness adjustment: 70% chance, factor range 0.8 to 1.2
    - Contrast adjustment: 70% chance, factor range 0.8 to 1.2
    - Random zoom: 50% chance, zoom factor range 0.9 to 1.1
      - Zoomed in images are center-cropped
      - Zoomed out images are padded with black pixels
    
    Args:
        img (Image.Image): Input PIL Image to be augmented.
        
    Returns:
        Image.Image: New PIL Image with random transformations applied.
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
            # Crop center if zoomed in
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
        else:
            # Pad with black if zoomed out
            from PIL import ImageOps
            delta_w = w - new_w
            delta_h = h - new_h
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            img = ImageOps.expand(img, padding, fill=(0, 0, 0))
    
    return img



if __name__ == '__main__':
    augment_data_prepared("./../data", 7)

