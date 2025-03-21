import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load the model, scaler and feature names
with open('log_reg_model.pkl', 'rb') as f:
  model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
  scaler = pickle.load(f)

# App title
st.title("Diabetic Retinopathy Prediction")

# Input features with appropriate defaults
age = st.number_input("Age", min_value=0, max_value=120, value=60)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, max_value=160, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=140, value=80)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=200, value=100)

# Empty DataFrame with the features expected by the model
input_data = pd.DataFrame(
  [{'age': age, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp, 'cholesterol': cholesterol}]
                          )

def predict_prognosis(age, systolic_bp, diastolic_bp, cholesterol):
  input_data = pd.DataFrame([{
    'age': float(age),
    'systolic_bp': float(systolic_bp),
    'diastolic_bp': float(diastolic_bp),
    'cholesterol': float(cholesterol)}])

  #Scale input data
  input_data_scaled = scaler.transform(input_data)

  # Make prediction
  prognosis = model.predict(input_data_scaled)[0]
  return "Positive" if prognosis == 1 else "Negative"

if st.button("Predict"):
    result = predict_prognosis(age, systolic_bp, diastolic_bp, cholesterol)
    st.write(f"Prediction: {result}")
