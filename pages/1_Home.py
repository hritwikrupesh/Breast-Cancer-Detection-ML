import streamlit as st
import pandas as pd

st.title("🏥 Breast Cancer Detection System")

st.markdown(
    """
    ### Early Detection of Breast Cancer using Machine Learning

    This application predicts whether a breast tumor is **Benign** or **Malignant**
    using a trained **Support Vector Machine (SVM)** model.

    The goal is to assist in early diagnosis using machine learning techniques.
    """
)

st.markdown("---")

# ==========================
# Dashboard Metrics
# ==========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Dataset", "569")

with col2:
    st.metric("🧬 Features", "30")

with col3:
    st.metric("🤖 Best Model", "SVM")

with col4:
    st.metric("🎯 Accuracy", "98.25%")

st.markdown("---")

# ==========================
# Model Comparison
# ==========================

st.subheader("📈 Model Performance Comparison")

comparison = pd.DataFrame(
    {
        "Model": ["Support Vector Machine", "Decision Tree"],
        "Accuracy": [98.25, 92.10]
    }
)

st.bar_chart(
    comparison.set_index("Model")
)

st.markdown("---")

# ==========================
# Technologies
# ==========================

st.subheader("⚙️ Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.success("🐍 Python")

with tech2:
    st.success("📊 Pandas")

with tech3:
    st.success("🤖 Scikit-Learn")

with tech4:
    st.success("🎨 Streamlit")

st.markdown("---")

# ==========================
# Workflow
# ==========================

st.subheader("🚀 Machine Learning Workflow")

workflow = [
    "📂 Data Understanding",
    "🧹 Data Preprocessing",
    "📊 Exploratory Data Analysis",
    "🤖 Model Training",
    "📈 Model Evaluation",
    "🚀 Streamlit Deployment"
]

for step in workflow:
    st.write(step)