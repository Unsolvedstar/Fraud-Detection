import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below:")

# Input fields
inputs = []
for i in range(1, 16):
    val = st.number_input(f"V{i}", value=0.0, format="%.4f")
    inputs.append(val)

amount = st.number_input("Transaction Amount", value=100.0)
time = st.number_input("Time (seconds since start)", value=50000.0)

inputs.append(amount)
inputs.append(time)

# Convert to NumPy array and scale
input_array = np.array(inputs).reshape(1, -1)
scaled_input = scaler.transform(input_array)

# Predict
if st.button("Detect Fraud"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - proba:.2f})")
