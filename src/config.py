"""
Project Configuration
---------------------
This module stores all project constants and file paths.
"""

from pathlib import Path

# ======================================================
# Base Directory
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ======================================================
# Project Directories
# ======================================================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"

OUTPUTS_DIR = BASE_DIR / "outputs"

PLOTS_DIR = OUTPUTS_DIR / "plots"

METRICS_DIR = OUTPUTS_DIR / "metrics"

# ======================================================
# Model Parameters
# ======================================================

RANDOM_STATE = 42

TEST_SIZE = 0.2

SVM_KERNEL = "rbf"

DT_MAX_DEPTH = 5

# ======================================================
# File Paths
# ======================================================

RAW_DATA_PATH = RAW_DATA_DIR / "breast_cancer.csv"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "breast_cancer_processed.csv"

SCALER_PATH = MODELS_DIR / "scaler.pkl"

SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"

DT_MODEL_PATH = MODELS_DIR / "decision_tree_model.pkl"

MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"

FEATURE_IMPORTANCE_PLOT = PLOTS_DIR / "feature_importance.png"