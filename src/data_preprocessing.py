"""
Data Preprocessing Module
-------------------------
This module is responsible for:

1. Loading the dataset
2. Splitting features and target
3. Scaling features
4. Splitting into train and test sets
5. Saving the scaler
"""

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    SCALER_PATH,
    RANDOM_STATE,
    TEST_SIZE
)


def load_data():
    """
    Load the raw dataset.
    """
    return pd.read_csv(RAW_DATA_PATH)


def split_features_target(df):
    """
    Separate features and target.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


def scale_features(X):
    """
    Scale the feature matrix.
    """
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)

    return X_scaled, scaler


def split_data(X, y):
    """
    Split into train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )