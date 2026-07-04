"""
Prediction Module
-----------------
Loads the trained model and scaler to make predictions.
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
    Predict breast cancer diagnosis.

    Returns
    -------
    dict
        {
            "prediction": "Benign" or "Malignant",
            "class": 0 or 1,
            "confidence": xx.xx
        }
    """

    scaler = load_scaler()
    model = load_model()

    if isinstance(sample, dict):
        sample = pd.DataFrame([sample])

    if not isinstance(sample, pd.DataFrame):
        raise ValueError(
            "Input must be a pandas DataFrame or dictionary."
        )

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    probabilities = model.predict_proba(sample_scaled)[0]

    confidence = probabilities[prediction] * 100

    label = "Benign" if prediction == 1 else "Malignant"

    return {
    "prediction": label,
    "class": int(prediction),
    "confidence": confidence,
    "benign_probability": probabilities[1] * 100,
    "malignant_probability": probabilities[0] * 100
}