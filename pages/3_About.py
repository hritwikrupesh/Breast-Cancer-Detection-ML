import streamlit as st

st.title("ℹ️ About This Project")

st.markdown("""
## 🏥 Project Overview

Breast cancer is one of the most common cancers worldwide.
Early detection significantly improves treatment outcomes.

This application uses **Machine Learning** to predict whether a
breast tumor is **Benign** or **Malignant** using diagnostic measurements.

The model was trained using the
**Wisconsin Breast Cancer Diagnostic Dataset**
from Scikit-Learn.
""")

st.markdown("---")

st.subheader("🎯 Project Objective")

st.write("""
The objective of this project is to build a complete
Machine Learning pipeline that:

- Understands the dataset
- Performs data preprocessing
- Builds classification models
- Evaluates model performance
- Deploys the best model using Streamlit
""")

st.markdown("---")

st.subheader("🤖 Machine Learning Models")

col1, col2 = st.columns(2)

with col1:
    st.success("""
Support Vector Machine (SVM)

✔ Accuracy : 98.25%

✔ Selected for Deployment
""")

with col2:
    st.info("""
Decision Tree

✔ Accuracy : 92.10%

✔ Used for Comparison
""")

st.markdown("---")

st.subheader("⚙️ Technologies Used")

st.write("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
""")

st.markdown("---")

st.subheader("📂 Project Workflow")

st.write("""
1. Data Understanding

2. Data Preprocessing

3. Exploratory Data Analysis

4. Feature Scaling

5. Model Training

6. Model Evaluation

7. Model Comparison

8. Model Deployment
""")

st.markdown("---")

st.subheader("📈 Model Performance")

st.table({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Support Vector Machine": ["98.25%", "98.61%", "98.61%", "98.61%"],
    "Decision Tree": ["92.10%", "95.65%", "91.67%", "93.62%"]
})

st.markdown("---")

st.subheader("🚀 Future Improvements")

st.write("""
- Hyperparameter Optimization
- Cross Validation
- Deep Learning Models
- Explainable AI (SHAP/LIME)
- Batch Prediction using CSV Upload
- Cloud Deployment
""")

st.markdown("---")

st.caption(
    "Developed as an end-to-end Machine Learning project using Streamlit."
)