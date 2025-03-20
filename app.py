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
with open('feature_names.pkl', 'rb') as f:
  feature_names = pickle.load(f)

# App title
st.title("Diabetic Retinopathy Prediction")

# Input features with appropriate defaults
age = st.number_input("Age", min_value=0, max_value=120, value=60)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, max_value=160, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=140, value=80)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=200, value=100)

def predict_prognosis():
  # Create a dictionary with the input data
  input_data_dict = {
    'Age': [age],
    'Systolic Blood Pressure': [systolic_bp],
    'Diastolic Blood Pressure': [diastolic_bp],
    'Cholesterol': [cholesterol]
  }

  # Create the input DataFrame with the correct feature order
  input_data = pd.DataFrame(input_data_dict, columns=feature_names)

  #Scale input data
  input_data_scaled = scaler.transform(input_data)

  # Make prediction
  prognosis = model.predict(input_data_scaled)[0]
  return prognosis

def print_prognosis(prognosis):
  if prognosis == 1:
    st.write("This patient is predicted to have diabetic retinopathy.")
  else:
    st.write("It is predicted that this passenger does not have diabetic retinopathy.")

if st.button("Predict"):
  predict_prognosis()
  print_prognosis(prognosis)

