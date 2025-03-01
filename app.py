import streamlit as st
import numpy as np
import pickle

# Load trained model using pickle
with open("models/rondomforest.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("🎓 Student Dropout Prediction App")

# Input fields for user data
st.write("Enter student details below:")

# Define feature inputs
features = [
    "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (grade)",
    "Course",
    "Curricular units 2nd sem (evaluations)",
    "Tuition fees up to date",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 2nd sem (enrolled)",
    "Age at enrollment"
]

# Collect input values
input_data = []
for feature in features:
    value = st.number_input(f"{feature}:", min_value=0.0, max_value=10000.0, step=0.1)
    input_data.append(value)

# Predict button
if st.button("Predict Dropout Status"):
    input_array = np.array([input_data]).reshape(1, -1)
    prediction = model.predict(input_array)
    
    # Map prediction to readable format
    prediction_map = {0: "Enrolled", 1: "Dropout", 2: "Graduate"}
    
    st.success(f"🎯 Prediction: **{prediction_map[prediction[0]]}**")
