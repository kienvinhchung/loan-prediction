

import streamlit as st
import joblib
import pandas as pd


Model = joblib.load("loan_approval_model.pkl")
Scaler = joblib.load("rb_scaler.pkl")


numerical_cols = ['annual_income', 'loan_amount', 'loan_term', 'credit_score', 
                  'residential_av', 'commercial_av', 'luxury_av', 'bank_av']


def predict(dependents, education , self_employed, annual_income, loan_amount, loan_term, credit_score, 
            residential_av, commercial_av, luxury_av, bank_av):
    test_df = pd.DataFrame(columns = ['dependents', 'education', 'self_employed', 'annual_income', 
                                      'loan_amount', 'loan_term', 'credit_score', 'residential_av', 
                                      'commercial_av', 'luxury_av', 'bank_av'])
    test_df.at[0,'dependents'] = dependents
    test_df.at[0,'education'] = education
    test_df.at[0,'self_employed'] = self_employed
    test_df.at[0,'annual_income'] = float(annual_income)
    test_df.at[0,'loan_amount'] = float(loan_amount)
    test_df.at[0,'loan_term'] = float(loan_term) * 365
    test_df.at[0,'credit_score'] = float(credit_score)
    test_df.at[0,'residential_av'] = float(residential_av)
    test_df.at[0,'commercial_av'] = float(commercial_av)
    test_df.at[0,'luxury_av'] = float(luxury_av)
    test_df.at[0,'bank_av'] = float(bank_av)
    
    temp_df = pd.DataFrame(Scaler.transform(test_df[numerical_cols]),columns=numerical_cols)
    test_df.drop(numerical_cols, axis=1, inplace=True)
    test_df = pd.concat([test_df, temp_df], axis=1)
    test_df.dependents = test_df.dependents.astype(int)
    test_df.education = test_df.education.astype(int)
    test_df.self_employed = test_df.self_employed.astype(int)
    print(test_df.dtypes)
    result = Model.predict(test_df)[0]
    return result


# Streamlit app
st.title("Loan Approval Prediction")
st.header("Enter Your Information")
st.text("Currency: Inidian Rupee. Symbol: ₹")
dependents = st.slider("Number of dependents", min_value=0, max_value=5, value=0, step=1)
education = int(st.selectbox("Education (1=Yes, 0=No)" ,[0 , 1]))
self_employed = int(st.selectbox("Self Employment (1=Yes/Self employed, 0=No/Employed)" , [0 , 1]))
annual_income = st.slider("Annual Income (₹)", min_value=0, max_value=9900000, value=200000, step=10000)
loan_amount = st.slider("Loan Amount (₹)" , min_value=100000, max_value=39500000, value=300000, step=300000)
loan_term = st.slider("Loan Term (in Year)", min_value=2, max_value=20, value=2, step=1)
credit_score = st.slider("Cibil Credit Score" , min_value=300, max_value=900, value=300, step=1)
residential_av = st.slider("Residential Assets Value in (₹)", min_value=0, max_value=29100000, value=0, step=10000)
commercial_av = st.slider("Commerical Assets Value (₹)", min_value=0, max_value=19400000, value=0, step=10000)
luxury_av = st.slider("Luxury Assets Value (₹)", min_value=0, max_value=39200000, value=0, step=10000)
bank_av = st.slider("Bank Asset Value (₹)", min_value=0, max_value=14700000, value=0, step=10000)


input_data = pd.DataFrame({
    'dependents': [dependents],
    'education': [education],
    'self_employed': [self_employed],
    'annual_income': [annual_income],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'credit_score': [credit_score],
    'residential_av': [residential_av],
    'commercial_av': [commercial_av],
    'luxury_av': [luxury_av],
    'bank_av': [bank_av]
})


# Predict loan approval
if st.button("Predict"):
        result = predict(dependents, 
                         education, 
                         self_employed, 
                         annual_income, 
                         loan_amount, 
                         loan_term, 
                         credit_score, 
                         residential_av, 
                         commercial_av, 
                         luxury_av, 
                         bank_av)
        label = ["rejected","approved"]
        st.markdown("## The application is **{}**.".format(label[result]))
