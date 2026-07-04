"""
Model Evaluation Module
-----------------------
This module provides reusable functions to:
1. Evaluate classification models
2. Generate classification reports
3. Plot confusion matrices
4. Compare multiple models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.config import MODEL_COMPARISON_PATH


def evaluate_model(y_true, y_pred):
    """
    Compute evaluation metrics for a classification model.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

    return metrics


def print_classification_report(y_true, y_pred):
    """
    Print the classification report.
    """
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plot a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


def compare_models(svm_metrics, dt_metrics):
    """
    Compare SVM and Decision Tree metrics.
    """

    comparison = pd.DataFrame({
        "Metric": list(svm_metrics.keys()),
        "SVM": list(svm_metrics.values()),
        "Decision Tree": list(dt_metrics.values())
    })

    return comparison


def save_comparison(comparison):
    """
    Save model comparison results.
    """
    comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False
    )