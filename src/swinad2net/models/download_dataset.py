#!/usr/bin/env python3

# ============================================================================
# This script downloads and extracts the SIPAKMED dataset.
# It creates a temporary directory for downloads and cleans up after extraction.
# ============================================================================

import requests
import os
import tqdm
import shutil
import py7zr
from typing import Optional, Tuple, List, Dict

def download_dataset(url: str, save_dir: str, compact_format: str = "7z") -> None:
    """
    Downloads and extracts a dataset from a given URL.
    Warning: A directory named 'temporary_downloads' will be created in the current working directory. This directory will be deleted after extraction.
    
    Args:
        url (str): The URL to download the dataset from.
        save_dir (str): The directory to save the extracted dataset.
        compact_format (str): The compression format of the dataset. Currently only '7z' is supported.
    
    Returns:
        None
    """
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_length_bytes = int(response.headers.get('content-length', 0))
        print(f"Downloading dataset with {total_length_bytes} bytes from {url}.")
        if os.path.exists('./temporary_downloads'):
            raise FileExistsError("The directory './temporary_downloads' already exists. Please remove it before downloading.")
        os.mkdir('./temporary_downloads')
        with tqdm.tqdm(desc="Downloading", total=total_length_bytes, unit='B', unit_scale=True) as pbar:
            with open('./temporary_downloads/dataset', 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        shutil.rmtree('./temporary_downloads', ignore_errors=True)
        return
    
    print(f"Download complete. Extracting dataset {compact_format}...")
    if compact_format == "7z":
        try:
            with py7zr.SevenZipFile('./temporary_downloads/dataset', mode='r') as z:
                z.extractall(path=save_dir)
            print(f"Dataset extracted to {save_dir}.")
        except Exception as e:
            print(f"Error extracting dataset: {e}")
        finally:
            shutil.rmtree('./temporary_downloads', ignore_errors=True)
        shutil.rmtree('./temporary_downloads', ignore_errors=True)
        return


if __name__ == "__main__":
    dict_urls = {
        "superficial-intermediate": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Superficial-Intermediate.7z', '7z'),
        "parabasal": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Parabasal.7z', '7z'),
        "koilocytotic": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Koilocytotic.7z', '7z'),
        "metaplastic": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Metaplastic.7z', '7z'),
        "dyskeratotic": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Dyskeratotic.7z', '7z')
    }
    for class_name, (url, compact_format) in dict_urls.items():
        os.makedirs('./../../../data', exist_ok=True)
        save_directory = os.path.join('./../../../data', class_name)
        download_dataset(url, save_directory)








    