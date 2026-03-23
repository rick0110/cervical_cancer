#!/usr/bin/env python3

# ============================================================================
# This script downloads and extracts the SIPAKMED dataset in MULTITHREAD.
# It creates unique temporary directories for concurrent downloads and cleans up.
# ============================================================================

import requests
import os
import tqdm
import shutil
import py7zr
import tarfile
import zipfile
import concurrent.futures
from typing import Optional, Tuple, List, Dict

def download_dataset(url: str, save_dir: str, class_name: str, position: int, compact_format: str = "7z") -> None:
    """
    Downloads and extracts a dataset from a given URL.
    
    Args:
        url (str): The URL to download the dataset from.
        save_dir (str): The directory to save the extracted dataset.
        class_name (str): Identifier used to create a unique temporary directory for thread safety.
        position (int): Position for the tqdm progress bar to avoid terminal clutter.
        compact_format (str): The compression format of the dataset.
    """
    
    # Cria um diretório temporário ÚNICO para esta thread
    temp_dir = f'./temp_downloads_{class_name}'
    temp_file = os.path.join(temp_dir, 'dataset.tmp')
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_length_bytes = int(response.headers.get('content-length', 0))
        
        # Garante que o diretório temporário exista e esteja limpo
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        with tqdm.tqdm(desc=f"[{class_name}]", total=total_length_bytes, unit='B', unit_scale=True, position=position, leave=True) as pbar:
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        print(f"\nError downloading {class_name}: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    
    # Extração
    if compact_format == "7z":
        try:
            with py7zr.SevenZipFile(temp_file, mode='r') as z:
                z.extractall(path=save_dir)
        except Exception as e:
            print(f"\nError extracting {class_name}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    elif compact_format == "tar":
        try:
            with tarfile.open(temp_file, 'r') as tar:
                tar.extractall(path=save_dir)
        except Exception as e:
            print(f"\nError extracting {class_name}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    elif compact_format == "zip":
        try:
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
        except Exception as e:
            print(f"\nError extracting {class_name}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

def process_item(index: int, class_name: str, url: str, compact_format: str):
    """Função auxiliar para organizar os argumentos para a thread."""
    os.makedirs('./data', exist_ok=True)
    save_directory = os.path.join('./data', class_name)
    download_dataset(url, save_directory, class_name, position=index, compact_format=compact_format)


if __name__ == "__main__":
    dict_urls = {
        "superficial-intermediate": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Superficial-Intermediate.7z', '7z'),
        "parabasal": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Parabasal.7z', '7z'),
        "koilocytotic": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Koilocytotic.7z', '7z'),
        "metaplastic": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Metaplastic.7z', '7z'),
        "dyskeratotic": ('https://www.cs.uoi.gr/~marina/SIPAKMED/im_Dyskeratotic.7z', '7z')
    }
    
    print("Iniciando downloads simultâneos...")
    
    # Usando ThreadPoolExecutor para baixar tudo simultaneamente
    # max_workers = número de classes que temos, assim todas baixam juntas
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(dict_urls)) as executor:
        futures = []
        for index, (class_name, (url, compact_format)) in enumerate(dict_urls.items()):
            # Submete a tarefa para a pool de threads
            future = executor.submit(process_item, index, class_name, url, compact_format)
            futures.append(future)
        
        # Aguarda todas as tarefas terminarem
        concurrent.futures.wait(futures)
        
    # Imprime quebra de linhas para limpar o terminal após as barras do tqdm
    print("\n" * len(dict_urls))
    print("Todos os downloads e extrações foram concluídos com sucesso!")


    