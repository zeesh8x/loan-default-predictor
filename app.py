import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/best_model.pkl")

# -------------------- Title and Introduction --------------------
st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("💳 Credit Card Default Prediction App")
st.markdown("""
Use this tool to predict the likelihood of a customer defaulting on their credit card payment next month.  
Fill in the customer's financial and payment history details below.
""")

# -------------------- Mappings --------------------
sex_map = {"Male": 1, "Female": 2}
education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}
marriage_map = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}
payment_status_map = {
    "-2 (No Consumption)": -2,
    "-1 (Paid in Full)": -1,
    "0 (Use Revolving Credit)": 0,
    "1 Month Delay": 1,
    "2 Months Delay": 2,
    "3 Months Delay": 3,
    "4 Months Delay": 4,
    "5 Months Delay": 5,
    "6 Months Delay": 6,
    "7 Months Delay": 7,
    "8 Months Delay": 8
}

# -------------------- Input Form --------------------
with st.form("prediction_form"):
    st.header("📌 Demographic Information")
    LIMIT_BAL = st.number_input("Credit Limit (NT dollar)", value=20000, step=1000)
    SEX = st.selectbox("Gender", list(sex_map.keys()))
    EDUCATION = st.selectbox("Education Level", list(education_map.keys()))
    MARRIAGE = st.selectbox("Marital Status", list(marriage_map.keys()))
    AGE = st.number_input("Age", min_value=18, max_value=100, value=30)

    st.header("🕒 Repayment History (Most Recent First)")
    PAY_0 = st.selectbox("September Payment Status", list(payment_status_map.keys()))
    PAY_2 = st.selectbox("August Payment Status", list(payment_status_map.keys()))
    PAY_3 = st.selectbox("July Payment Status", list(payment_status_map.keys()))
    PAY_4 = st.selectbox("June Payment Status", list(payment_status_map.keys()))
    PAY_5 = st.selectbox("May Payment Status", list(payment_status_map.keys()))
    PAY_6 = st.selectbox("April Payment Status", list(payment_status_map.keys()))

    st.header("📄 Past Bill Amounts (NT dollars)")
    BILL_AMT1 = st.number_input("September Bill Amount", value=5000)
    BILL_AMT2 = st.number_input("August Bill Amount", value=5000)
    BILL_AMT3 = st.number_input("July Bill Amount", value=5000)
    BILL_AMT4 = st.number_input("June Bill Amount", value=5000)
    BILL_AMT5 = st.number_input("May Bill Amount", value=5000)
    BILL_AMT6 = st.number_input("April Bill Amount", value=5000)

    st.header("💸 Previous Payments (NT dollars)")
    PAY_AMT1 = st.number_input("September Payment", value=2000)
    PAY_AMT2 = st.number_input("August Payment", value=2000)
    PAY_AMT3 = st.number_input("July Payment", value=2000)
    PAY_AMT4 = st.number_input("June Payment", value=2000)
    PAY_AMT5 = st.number_input("May Payment", value=2000)
    PAY_AMT6 = st.number_input("April Payment", value=2000)

    submitted = st.form_submit_button("Predict")

# -------------------- Prediction --------------------
if submitted:
    input_features = np.array([[
        LIMIT_BAL,
        sex_map[SEX],
        education_map[EDUCATION],
        marriage_map[MARRIAGE],
        AGE,
        payment_status_map[PAY_0],
        payment_status_map[PAY_2],
        payment_status_map[PAY_3],
        payment_status_map[PAY_4],
        payment_status_map[PAY_5],
        payment_status_map[PAY_6],
        BILL_AMT1,
        BILL_AMT2,
        BILL_AMT3,
        BILL_AMT4,
        BILL_AMT5,
        BILL_AMT6,
        PAY_AMT1,
        PAY_AMT2,
        PAY_AMT3,
        PAY_AMT4,
        PAY_AMT5,
        PAY_AMT6
    ]])

    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    st.subheader("✅ Prediction Result")
    st.markdown(f"**Prediction:** {'🔴 Will Default' if prediction == 1 else '🟢 Will Not Default'}")
    st.markdown(f"**Probability of Default:** `{probability:.2%}`")
