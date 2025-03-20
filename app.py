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
  #Convert to list of strings
  feature_names = list(feature_names.astype(str))

# App title
st.title("Diabetic Retinopathy Prediction")

# Input features with appropriate defaults
age = st.number_input("Age", min_value=0, max_value=120, value=60)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, max_value=160, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=140, value=80)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=200, value=100)

# Empty DataFrame with the features expected by the model
input_data = pd.DataFrame(
  columns=feature_names,
  index=[0], dtype=float).fillna(0) 

#Initialize prognosis
prognosis = None

def predict_prognosis():
  input_data['age'] = age
  input_data['systolic_bp'] = systolic_bp
  input_data['diastolic_bp'] = diastolic_bp
  input_data['cholesterol'] = cholesterol

  #Scale input data
  input_data_scaled = scaler.transform(input_data)

  # Make prediction
  prognosis = model.predict(input_data_scaled)[0]
  return prognosis

def print_prognosis(prognosis):
  if prognosis == 1:
    st.write("This patient is predicted to have diabetic retinopathy.")
  elif prognosis == 0 :
    st.write("It is predicted that this passenger does not have diabetic retinopathy.")
  else:
    st.write("Error in prognosis. Retry. ")

if st.button("Predict"):
  prognosis = predict_prognosis()
  print_prognosis(prognosis)
