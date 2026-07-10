"""Dataset registry for cervical cytology experiments.

Builds pandas DataFrames (image path + labels) for the two public datasets used
in this project:

- Herlev / MDE-Lab Pap Smear Collection (917 single-cell images, 7 classes).
- SIPaKMeD (4049 single-cell CROPPED images, 5 classes).

Each DataFrame exposes both the fine-grained multi-class label and a shared
binary label (0 = normal, 1 = abnormal) so the two datasets can be merged into
a single binary-classification pool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

HERLEV_CLASSES: Dict[str, int] = {
    "normal_superficiel": 0,
    "normal_intermediate": 1,
    "normal_columnar": 2,
    "light_dysplastic": 3,
    "moderate_dysplastic": 4,
    "severe_dysplastic": 5,
    "carcinoma_in_situ": 6,
}
HERLEV_NORMAL_CLASSES = {"normal_superficiel", "normal_intermediate", "normal_columnar"}
HERLEV_CLASS_NAMES: List[str] = [k for k, _ in sorted(HERLEV_CLASSES.items(), key=lambda kv: kv[1])]

SIPAKMED_CLASSES: Dict[str, int] = {
    "im_Superficial-Intermediate": 0,
    "im_Parabasal": 1,
    "im_Koilocytotic": 2,
    "im_Dyskeratotic": 3,
    "im_Metaplastic": 4,
}
SIPAKMED_NORMAL_CLASSES = {"im_Superficial-Intermediate", "im_Parabasal"}
SIPAKMED_CLASS_NAMES: List[str] = [k.replace("im_", "") for k, _ in sorted(SIPAKMED_CLASSES.items(), key=lambda kv: kv[1])]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    class_map: Dict[str, int]
    normal_classes: set
    class_names: List[str]


def build_herlev_dataframe(root: str) -> pd.DataFrame:
    """Scans the Herlev/MDE-Lab directory tree and returns path/label rows.

    Only the original ``*.BMP`` images are kept; the ``*-d.bmp`` files shipped
    alongside them are nuclei-boundary annotations, not classification inputs.
    """
    rows = []
    for class_name, class_id in HERLEV_CLASSES.items():
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(".bmp"):
                continue
            if fname.lower().endswith("-d.bmp"):
                continue
            rows.append(
                {
                    "path": os.path.join(class_dir, fname),
                    "label": class_id,
                    "class_name": class_name,
                    "binary_label": 0 if class_name in HERLEV_NORMAL_CLASSES else 1,
                    "dataset": "herlev",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No Herlev images found under {root}")
    return df


def build_sipakmed_dataframe(root: str) -> pd.DataFrame:
    """Scans the SIPaKMeD directory tree and returns path/label rows for the
    single-cell CROPPED images (the standard classification split)."""
    rows = []
    for class_name, class_id in SIPAKMED_CLASSES.items():
        class_dir = os.path.join(root, class_name, "CROPPED")
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(".bmp"):
                continue
            rows.append(
                {
                    "path": os.path.join(class_dir, fname),
                    "label": class_id,
                    "class_name": class_name.replace("im_", ""),
                    "binary_label": 0 if class_name in SIPAKMED_NORMAL_CLASSES else 1,
                    "dataset": "sipakmed",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No SIPaKMeD images found under {root}")
    return df


def build_combined_binary_dataframe(herlev_root: str, sipakmed_root: str) -> pd.DataFrame:
    """Merges both datasets into a single binary (normal vs. abnormal) pool.

    The two datasets do not share a class taxonomy (7 vs. 5 fine-grained
    classes), so ``label`` here is set to the shared ``binary_label``.
    """
    herlev_df = build_herlev_dataframe(herlev_root)
    sipakmed_df = build_sipakmed_dataframe(sipakmed_root)
    combined = pd.concat([herlev_df, sipakmed_df], ignore_index=True)
    combined["label"] = combined["binary_label"]
    return combined


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "herlev": DatasetSpec("herlev", HERLEV_CLASSES, HERLEV_NORMAL_CLASSES, HERLEV_CLASS_NAMES),
    "sipakmed": DatasetSpec("sipakmed", SIPAKMED_CLASSES, SIPAKMED_NORMAL_CLASSES, SIPAKMED_CLASS_NAMES),
    "combined": DatasetSpec("combined", {"normal": 0, "abnormal": 1}, {"normal"}, ["normal", "abnormal"]),
}
