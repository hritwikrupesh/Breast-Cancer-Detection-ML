"""
Prediction Module
-----------------
This module loads the trained model and scaler
to make predictions on new patient data.
"""

import joblib
import pandas as pd

from src.config import (
    SCALER_PATH,
    SVM_MODEL_PATH
)


def load_scaler():
    """Load the saved StandardScaler."""
    return joblib.load(SCALER_PATH)


def load_model():
    """Load the trained SVM model."""
    return joblib.load(SVM_MODEL_PATH)


def predict(sample):
    """
    Predict whether a tumor is Benign or Malignant.

    Parameters
    ----------
    sample : pandas.DataFrame or dict
        Input features for one patient.

    Returns
    -------
    int
        0 -> Malignant
        1 -> Benign
    """

    scaler = load_scaler()
    model = load_model()

    # Convert dictionary to DataFrame
    if isinstance(sample, dict):
        sample = pd.DataFrame([sample])

    # Ensure input is a DataFrame
    if not isinstance(sample, pd.DataFrame):
        raise ValueError(
            "Input must be a pandas DataFrame or a dictionary."
        )

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    return prediction[0]


def predict_label(sample):
    """
    Return prediction as text.
    """

    prediction = predict(sample)

    if prediction == 1:
        return "Benign"

    return "Malignant"