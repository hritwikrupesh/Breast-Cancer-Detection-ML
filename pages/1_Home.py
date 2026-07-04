import streamlit as st

st.title("🏥 Breast Cancer Detection System")

st.markdown(
    """
    ### Early Detection of Breast Cancer using Machine Learning

    This web application predicts whether a breast tumor is **Benign** or **Malignant**
    using a trained **Support Vector Machine (SVM)** model.

    It demonstrates a complete Machine Learning pipeline from data preprocessing
    to model deployment using Streamlit.
    """
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Information")

    st.success("Best Model : Support Vector Machine (SVM)")
    st.success("Accuracy : 98.25%")
    st.success("Precision : 98.61%")
    st.success("Recall : 98.61%")
    st.success("F1 Score : 98.61%")

with col2:
    st.subheader("📁 Dataset")

    st.info("""
Wisconsin Breast Cancer Diagnostic Dataset

• 569 Samples

• 30 Features

• Binary Classification

• Target Classes:
    - Benign
    - Malignant
""")

st.markdown("---")

st.subheader("⚙️ Technologies Used")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.metric("Python", "3.12")

with tech2:
    st.metric("Scikit-Learn", "ML")

with tech3:
    st.metric("Pandas", "Data")

with tech4:
    st.metric("Streamlit", "UI")

st.markdown("---")

st.subheader("🚀 Machine Learning Workflow")

st.markdown("""
1. Data Understanding

2. Data Preprocessing

3. Exploratory Data Analysis (EDA)

4. Model Building

5. Model Evaluation

6. Model Deployment using Streamlit
""")