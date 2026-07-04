import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Breast Cancer Detection System")

st.markdown("---")

st.header("Welcome!")

st.write(
    """
    This application predicts whether a breast tumor is **Benign** or **Malignant**
    using a trained **Support Vector Machine (SVM)** model.

    👈 Use the navigation menu on the left to explore the application.
    """
)

st.info("Choose a page from the sidebar to get started.")