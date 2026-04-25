import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Risk Prediction")
st.write("Predict customer churn probability and assign risk category.")

# Load saved model files
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")
st.write(features)

st.success("Model loaded successfully!")

st.sidebar.header("Enter Customer Details")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
age = st.sidebar.number_input("Age", 18, 100, 40)
tenure = st.sidebar.number_input("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.number_input("Number of Products", 1, 4, 2)
has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

if st.sidebar.button("Predict Churn Risk"):


    input_data = pd.DataFrame([{
    "Year": 2020,  # FIX

    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary,

    "BalanceSalaryRatio": balance / (salary + 1),
    "ProductDensity": num_products / (tenure + 1),
    "EngagementProductInteraction": is_active * num_products,
    "AgeTenureInteraction": age * tenure,

    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain": 1 if geography == "Spain" else 0,
    "Gender_Male": 1 if gender == "Male" else 0
}])


    input_data = input_data[features]

    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1]

    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{probability:.2%}")
    st.metric("Risk Category", risk)