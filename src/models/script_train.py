# from train import *
import os
import pandas as pd
import numpy as np
maps = {
    'carcinoma': 1,
    'dysplastic': 1,
    'columnar': 0,
    'intermediate': 0,
    'superficiel': 0
}
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
for i in df.values:
    print(i)


