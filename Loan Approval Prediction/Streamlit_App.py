 
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Loan Approval Predictor")
model = joblib.load("loan_approval_model.pkl")  # saved pipeline

# Input widgets (mirror features used)
gender = st.selectbox("Gender", ["Male","Female"])
married = st.selectbox("Married", ["Yes","No"])
dependents = st.selectbox("Dependents", ["0","1","2","3+"])
education = st.selectbox("Education", ["Graduate","Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes","No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (thousands)", min_value=0)
loan_term = st.number_input("Loan Amount Term (days/months?)", min_value=0, value=360)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])

# derived features must match training preprocessing
total_income = applicant_income + coapplicant_income
emi = (loan_amount*1000) / (loan_term + 1e-6)
income_to_emi = total_income / (emi + 1e-6)

input_dict = {
    "Gender":[gender],
    "Married":[ 'Yes' if married=='Yes' else 'No'],
    "Dependents":[dependents],
    "Education":[education],
    "Self_Employed":[self_employed],
    "ApplicantIncome":[applicant_income],
    "CoapplicantIncome":[coapplicant_income],
    "LoanAmount":[loan_amount],
    "Loan_Amount_Term":[loan_term],
    "Credit_History":[credit_history],
    "Property_Area":[property_area],
    "Total_Income":[total_income],
    "EMI":[emi],
    "Income_to_EMI":[income_to_emi]
}

input_df = pd.DataFrame(input_dict)

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0,1]
    pred = model.predict(input_df)[0]
    st.write(f"Probability of approval: {prob:.2f}")
    st.write("Prediction:", "Approved ✅" if pred==1 else "Rejected ❌")
