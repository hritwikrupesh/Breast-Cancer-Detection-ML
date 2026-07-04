"""
Decision Tree Module
--------------------
This module provides functions to:
1. Train a Decision Tree model
2. Make predictions
3. Save the trained model
4. Load a saved model
"""

import joblib

from sklearn.tree import DecisionTreeClassifier

from src.config import (
    RANDOM_STATE,
    DT_MAX_DEPTH,
    DT_MODEL_PATH
)


def train_decision_tree(X_train, y_train):
    """
    Train the Decision Tree model.
    """
    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=DT_MAX_DEPTH
    )

    model.fit(X_train, y_train)

    return model


def predict_decision_tree(model, X_test):
    """
    Make predictions using the trained Decision Tree.
    """
    return model.predict(X_test)


def save_decision_tree(model):
    """
    Save the trained Decision Tree model.
    """
    joblib.dump(model, DT_MODEL_PATH)


def load_decision_tree():
    """
    Load the trained Decision Tree model.
    """
    return joblib.load(DT_MODEL_PATH)