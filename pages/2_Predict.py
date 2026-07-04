import streamlit as st
import pandas as pd

from src.predict import predict_label

st.title("🔬 Breast Cancer Prediction")

st.markdown(
    """
    Enter the patient's tumor measurements below and click **Predict**.
    """
)

st.markdown("---")

# ======================================================
# Mean Features
# ======================================================

st.subheader("Mean Features")

col1, col2, col3 = st.columns(3)

with col1:
    mean_radius = st.number_input("Mean Radius", value=14.12)
    mean_area = st.number_input("Mean Area", value=654.89)
    mean_compactness = st.number_input("Mean Compactness", value=0.10)
    mean_concave_points = st.number_input("Mean Concave Points", value=0.05)

with col2:
    mean_texture = st.number_input("Mean Texture", value=19.29)
    mean_smoothness = st.number_input("Mean Smoothness", value=0.09)
    mean_concavity = st.number_input("Mean Concavity", value=0.08)
    mean_symmetry = st.number_input("Mean Symmetry", value=0.18)

with col3:
    mean_perimeter = st.number_input("Mean Perimeter", value=91.97)
    mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.06)

st.markdown("---")

# ======================================================
# Error Features
# ======================================================

st.subheader("Error Features")

col1, col2, col3 = st.columns(3)

with col1:
    radius_error = st.number_input("Radius Error", value=0.40)
    area_error = st.number_input("Area Error", value=40.33)
    compactness_error = st.number_input("Compactness Error", value=0.02)
    concave_points_error = st.number_input("Concave Points Error", value=0.01)

with col2:
    texture_error = st.number_input("Texture Error", value=1.21)
    smoothness_error = st.number_input("Smoothness Error", value=0.007)
    concavity_error = st.number_input("Concavity Error", value=0.03)
    symmetry_error = st.number_input("Symmetry Error", value=0.02)

with col3:
    perimeter_error = st.number_input("Perimeter Error", value=2.86)
    fractal_dimension_error = st.number_input("Fractal Dimension Error", value=0.003)

st.markdown("---")

# ======================================================
# Worst Features
# ======================================================

st.subheader("Worst Features")

col1, col2, col3 = st.columns(3)

with col1:
    worst_radius = st.number_input("Worst Radius", value=16.27)
    worst_area = st.number_input("Worst Area", value=880.58)
    worst_compactness = st.number_input("Worst Compactness", value=0.25)
    worst_concave_points = st.number_input("Worst Concave Points", value=0.11)

with col2:
    worst_texture = st.number_input("Worst Texture", value=25.67)
    worst_smoothness = st.number_input("Worst Smoothness", value=0.13)
    worst_concavity = st.number_input("Worst Concavity", value=0.27)
    worst_symmetry = st.number_input("Worst Symmetry", value=0.29)

with col3:
    worst_perimeter = st.number_input("Worst Perimeter", value=107.26)
    worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=0.08)

st.markdown("---")

predict_button = st.button(
    "Predict Breast Cancer",
    use_container_width=True
)

if predict_button:

    patient_data = pd.DataFrame([{

        # Mean Features
        "mean radius": mean_radius,
        "mean texture": mean_texture,
        "mean perimeter": mean_perimeter,
        "mean area": mean_area,
        "mean smoothness": mean_smoothness,
        "mean compactness": mean_compactness,
        "mean concavity": mean_concavity,
        "mean concave points": mean_concave_points,
        "mean symmetry": mean_symmetry,
        "mean fractal dimension": mean_fractal_dimension,

        # Error Features
        "radius error": radius_error,
        "texture error": texture_error,
        "perimeter error": perimeter_error,
        "area error": area_error,
        "smoothness error": smoothness_error,
        "compactness error": compactness_error,
        "concavity error": concavity_error,
        "concave points error": concave_points_error,
        "symmetry error": symmetry_error,
        "fractal dimension error": fractal_dimension_error,

        # Worst Features
        "worst radius": worst_radius,
        "worst texture": worst_texture,
        "worst perimeter": worst_perimeter,
        "worst area": worst_area,
        "worst smoothness": worst_smoothness,
        "worst compactness": worst_compactness,
        "worst concavity": worst_concavity,
        "worst concave points": worst_concave_points,
        "worst symmetry": worst_symmetry,
        "worst fractal dimension": worst_fractal_dimension

    }])

    prediction = predict_label(patient_data)

    st.markdown("---")

    st.subheader("📋 Prediction Report")

    col1, col2 = st.columns([2, 1])

    with col1:

        if prediction == "Benign":

            st.success("### 🟢 Diagnosis : BENIGN")

            st.info("""
    The model predicts that the tumor is **Benign**.

    This means the tumor is **unlikely to be cancerous** based on the input measurements.

    **Recommendation:**
    Please consult a qualified healthcare professional for a complete medical evaluation.
    """)

        else:

            st.error("### 🔴 Diagnosis : MALIGNANT")

            st.warning("""
    The model predicts that the tumor is **Malignant**.

    This means the tumor **may be cancerous** based on the input measurements.

    **Recommendation:**
    Please consult an oncologist or healthcare professional immediately for further diagnosis.
    """)

    with col2:

        st.metric(
            label="Model",
            value="SVM"
        )

        st.metric(
            label="Accuracy",
            value="98.25%"
        )

        st.metric(
            label="Dataset",
            value="569 Samples"
        )


    st.markdown("---")

    with st.expander("📄 View Entered Patient Measurements"):

        st.dataframe(
            patient_data.T,
            use_container_width=True
        )

    st.markdown("---")

    st.caption(
        "Breast Cancer Detection using Machine Learning | Streamlit Application"
    )