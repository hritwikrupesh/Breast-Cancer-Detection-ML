from src.predict import predict_label
import pandas as pd

df = pd.read_csv("data/processed/breast_cancer_processed.csv")

sample = df.drop("target", axis=1).iloc[[0]]

print("Prediction:", predict_label(sample))