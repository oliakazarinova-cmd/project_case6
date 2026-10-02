"""Raw input schema shared by training and inference."""

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "N_Days",
    "Age",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Tryglicerides",
    "Platelets",
    "Prothrombin",
    "Stage",
]
CATEGORICAL = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
FEATURES = NUMERIC + CATEGORICAL
CLASSES = {"C", "CL", "D"}


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select expected features; reject missing columns and malformed numbers."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input dataset is empty")
    result = frame.loc[:, FEATURES].copy()
    for column in NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if np.isinf(result[NUMERIC].to_numpy(dtype=float)).any():
        raise ValueError("Numeric features contain infinity")
    return result


def load_training(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load original class labels without relying on preprocessed CSV files."""
    frame = pd.read_csv(path)
    if "Status" not in frame:
        raise ValueError("Training data must contain Status")
    target = frame["Status"]
    if target.isna().any() or set(target.unique()) != CLASSES:
        raise ValueError("Status must contain all three classes C, CL, D and no nulls")
    return features(frame), target
