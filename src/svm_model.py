"""
Support Vector Machine (SVM) Module
-----------------------------------
This module provides functions to:
1. Train an SVM model
2. Make predictions
3. Save the trained model
4. Load a saved model
"""

import joblib

from sklearn.svm import SVC

from src.config import (
    RANDOM_STATE,
    SVM_KERNEL,
    SVM_MODEL_PATH
)


def train_svm(X_train, y_train):
    """
    Train the Support Vector Machine model.
    """
    model = SVC(
        kernel=SVM_KERNEL,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    return model


def predict_svm(model, X_test):
    """
    Make predictions using the trained SVM model.
    """
    return model.predict(X_test)


def save_svm_model(model):
    """
    Save the trained SVM model.
    """
    joblib.dump(model, SVM_MODEL_PATH)


def load_svm_model():
    """
    Load the trained SVM model.
    """
    return joblib.load(SVM_MODEL_PATH)