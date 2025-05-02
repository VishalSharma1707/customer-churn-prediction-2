
import streamlit as st
import requests

st.title("Customer Churn Prediction")

fields = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

input_data = {}
for field in fields:
    if field in ["MonthlyCharges", "TotalCharges"]:
        input_data[field] = st.number_input(f"{field}", step=0.01)
    else:
        input_data[field] = st.number_input(f"{field}", step=1)

if st.button("Predict"):
    url = "http://127.0.0.1:8000/predict/"
    response = requests.post(url, json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Churn Probability: {result['churn_probability']}")
        st.info("Likely to Churn" if result['churn_prediction'] == 1 else "Not Likely to Churn")
    else:
        st.error(f"Prediction failed: {response.status_code}")
