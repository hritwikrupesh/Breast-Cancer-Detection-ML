import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🏥",
    layout="wide"
)

# ==========================
# Sidebar
# ==========================

st.sidebar.title("🏥 Breast Cancer Detection")

st.sidebar.markdown("---")

st.sidebar.success("🤖 Best Model")
st.sidebar.write("Support Vector Machine (SVM)")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Project Statistics")

st.sidebar.metric("Accuracy", "98.25%")
st.sidebar.metric("Dataset", "569 Samples")
st.sidebar.metric("Features", "30")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Technologies")

st.sidebar.markdown("""
- 🐍 Python
- 📊 Pandas
- 🤖 Scikit-Learn
- 📈 Matplotlib
- 🎨 Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Developed By**

    Hritwik Rupesh Gollu

    B.Tech CSE

    End-to-End Machine Learning Project
    """
)

# ==========================
# Main Page
# ==========================

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