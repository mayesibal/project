
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Page settings
st.set_page_config(page_title="Student Performance Prediction")
st.title("🎓 Student Performance Prediction")
st.markdown("Enter the student's academic information to predict their final GPA and risk status.")

# Input fields
admission = st.number_input("Admission Test Score (0–100)", min_value=0.0, max_value=100.0, step=0.1)
past_gpa = st.number_input("Past Academic GPA (0.00–4.00)", min_value=0.0, max_value=4.0, step=0.01)
first_year_gpa = st.number_input("First-Year GPA (0.00–4.00)", min_value=0.0, max_value=4.0, step=0.01)

# Predict button
if st.button("Predict Performance"):
    if admission == 0 or past_gpa == 0 or first_year_gpa == 0:
        st.warning("⚠️ Please enter all 3 values.")
    else:
        X = np.array([[admission, past_gpa, first_year_gpa]])
        predicted_gpa = model.predict(X)[0]

        # Risk threshold logic
        if predicted_gpa < 2.15:
            status = "At Risk"
            st.error(f"Predicted GPA: {predicted_gpa:.2f}  \nStatus: **{status}**")
        else:
            status = "Not at Risk"
            st.success(f"Predicted GPA: {predicted_gpa:.2f}  \nStatus: **{status}**")
