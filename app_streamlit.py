import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import boto3
import os
import traceback
import json
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "sagemaker-churned-v1")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features):
    runtime = get_runtime_client()

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(features), 
    )

    return json.loads(response["Body"].read().decode("utf-8"))


class CreditInferencePipeline:
    def __init__(self):
        
            self.feature_columns = [
            'Month', 'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
            'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
            'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
            'Credit_Utilization_Ratio', 'Payment_of_Min_Amount',
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
            'Credit_History_Age_Months', 'Spending_Level', 'Payment_Size',
            'Loan_Count', 'Loan_Auto_Loan', 'Loan_Credit_Builder_Loan',
            'Loan_Debt_Consolidation_Loan', 'Loan_Home_Equity_Loan',
            'Loan_Mortgage_Loan', 'Loan_Not_Specified', 'Loan_Payday_Loan',
            'Loan_Personal_Loan', 'Loan_Student_Loan', 'Occ_Architect',
            'Occ_Developer', 'Occ_Doctor', 'Occ_Engineer', 'Occ_Entrepreneur',
            'Occ_Journalist', 'Occ_Lawyer', 'Occ_Manager', 'Occ_Mechanic',
            'Occ_Media_Manager', 'Occ_Musician', 'Occ_Scientist', 'Occ_Teacher',
            'Occ_Unknown', 'Occ_Writer'
        ]

    def predict(self, raw_input_dict):
        df = pd.DataFrame([raw_input_dict])
        
        # Month mapping
        month_map = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 
                     'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11}
        df['Month'] = df['Month'].map(month_map).fillna(1)
        
        # Payment Behaviour Extraction custom OHE
        pb = str(raw_input_dict.get('Payment_Behaviour', ''))
        df['Spending_Level'] = 1 if "High_spent" in pb else (0 if "Low_spent" in pb else -1) 
        df['Payment_Size'] = 2 if "Large" in pb else (1 if "Medium" in pb else (0 if "Small" in pb else -1))
        
        # Mapping
        credit_mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
        df['Credit_Mix'] = df['Credit_Mix'].map(credit_mix_map).fillna(1)
        
        poma_map = {'No': 0, 'Yes': 1, 'Not Mentioned': 2}
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map(poma_map).fillna(2)

        # FE Loan COunt
        loans_selected = raw_input_dict.get('Type_of_Loan', [])
        df['Loan_Count'] = len(loans_selected)
        
        # Fixing Loan Feature
        loan_features = [c for c in self.feature_columns if c.startswith('Loan_') and c != 'Loan_Count']
        for lf in loan_features:
            clean_loan_name = lf.replace('Loan_', '').replace('_', ' ')
            df[lf] = 1 if clean_loan_name in loans_selected else 0

        # Custom OHE
        occ_selected = raw_input_dict.get('Occupation', 'Unknown')
        occ_features = [c for c in self.feature_columns if c.startswith('Occ_')]
        for of in occ_features:
            clean_occ_name = of.replace('Occ_', '')
            df[of] = 1 if clean_occ_name == occ_selected else 0

        # Reindexing
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        payload = df.to_dict(orient="records")[0]
        
        try:
            response = invoke_endpoint(payload)
            prediction_class = response.get("credit_score_prediction", "unknown")
            probability_class = response.get("probabilities", [0, 0, 0])

            prob_dict = {
                "Good": probability_class[0],
                "Standard": probability_class[1],
                "Poor": probability_class[2]
            }
            return prediction_class, prob_dict
        
        except Exception as e:
            st.error(f"Gagal menghubungi AWS Endpoint: {e}")
            traceback.format_exc()
            st.stop()

def main():
    st.set_page_config(page_title="Credit Scoring System", layout="centered")

    st.title("Credit Performance Evaluation System")
    st.subheader("Smart Internal Institution App to Classify Customer Credit Performance. ")
    st.markdown('---')

    inference = CreditInferencePipeline()

    st.header("Input Customer Data:")
    with st.form(key="Customer_Form"):
        st.subheader("Demographic & Core Features")
        
        month = st.selectbox("Current Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", ['Scientist', 'Teacher', 'Engineer', 'Developer', 'Doctor', 'Lawyer', 'Manager', 'Entrepreneur', 'Architect', 'Writer', 'Media_Manager', 'Mechanic', 'Journalist', 'Musician', 'Unknown'])
        
        st.subheader("Financial Metrics")
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        monthly_inhand_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0, value=4000)
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0, value=1500)
        total_emi_per_month = st.number_input("Total Equated Monthly Installment (EMI) ($)", min_value=0, value=500)
        amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0, value=200)
        monthly_balance = st.number_input("Monthly Balance ($)", value=1800)
        
        st.subheader("Credit Behavior Attributes")
        num_bank_accounts = st.slider("Number of Bank Accounts", 0, 10, 2)
        num_credit_card = st.slider("Number of Credit Cards", 0, 12, 3)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0, value=10)
        num_of_loan = st.number_input("Number of Existing Loans", min_value=0, value=2)
        
        
        type_of_loan = st.multiselect("Type of Loan", ['Auto Loan', 'Credit Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan', 'Personal Loan', 'Student Loan', 'Payday_Loan', 'Not Specified'])
        
        delay_from_due_date = st.number_input("Average Days Delay from Due Date", min_value=0, value=5)
        num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=2)
        changed_credit_limit = st.number_input("Changed Credit Limit Percentage", value=10)
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=1)
        credit_mix = st.selectbox("Credit Mix", ['Good', 'Standard', 'Bad'])
        credit_utilization_ratio = st.slider("Credit Utilization Ratio (%)", 0, 100, 30)
        credit_history_age_months = st.number_input("Credit History Age (In Months)", min_value=0, value=48)
        payment_of_min_amount = st.selectbox("Payment of Minimum Amount Only?", ['No', 'Yes', 'Not Mentioned'])
        payment_behaviour = st.selectbox("Payment Behaviour", ['High_spent_Large_value_payments', 'High_spent_Medium_value_payments', 'High_spent_Small_value_payments', 'Low_spent_Large_value_payments', 'Low_spent_Medium_value_payments', 'Low_spent_Small_value_payments', 'Unknown'])

        submit_button = st.form_submit_button(label="Evaluate Credit Performance")
        
    if submit_button:
        raw_input = {
            'Month': month, 'Age': age, 'Occupation': occupation, 'Annual_Income': annual_income,
            'Monthly_Inhand_Salary': monthly_inhand_salary, 'Outstanding_Debt': outstanding_debt,
            'Total_EMI_per_month': total_emi_per_month, 'Amount_invested_monthly': amount_invested_monthly,
            'Monthly_Balance': monthly_balance, 'Num_Bank_Accounts': num_bank_accounts,
            'Num_Credit_Card': num_credit_card, 'Interest_Rate': interest_rate, 'Num_of_Loan': num_of_loan,
            'Type_of_Loan': type_of_loan, 'Delay_from_due_date': delay_from_due_date,
            'Num_of_Delayed_Payment': num_of_delayed_payment, 'Changed_Credit_Limit': changed_credit_limit,
            'Num_Credit_Inquiries': num_credit_inquiries, 'Credit_Mix': credit_mix,
            'Credit_Utilization_Ratio': credit_utilization_ratio, 'Credit_History_Age_Months': credit_history_age_months,
            'Payment_of_Min_Amount': payment_of_min_amount, 'Payment_Behaviour': payment_behaviour
        }
        
        with st.spinner("Processing credit risk classification..."):
            class_prediction, probability_prediction = inference.predict(raw_input)
            
        st.subheader("Evaluation Result")
        
        if class_prediction in [0, "0", "Good", "good"]:
            st.success("### Category: **GOOD CREDIT PERFORMANCE**")
        elif class_prediction in [1, "1", "Standard", "standard"]:
            st.warning("### Category: **STANDARD CREDIT PERFORMANCE**")
        else:
            st.error("### Category: **POOR CREDIT PERFORMANCE (HIGH RISK)**")
            
        st.write("Model Confidence Score:")
        
        prob_data = pd.Series(probability_prediction)
        st.bar_chart(prob_data)
        

if __name__ == "__main__":
    main()
