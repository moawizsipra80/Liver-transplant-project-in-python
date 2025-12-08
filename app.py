import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_objects():
    clf = pickle.load(open('clf.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le_sex = pickle.load(open('le_sex.pkl', 'rb'))
    le_abo = pickle.load(open('le_abo.pkl', 'rb'))
    feature_cols = pickle.load(open('feature_cols.pkl', 'rb'))
    return clf, scaler, le_sex, le_abo, feature_cols

clf, scaler, le_sex, le_abo, feature_cols = load_objects()

st.title("🏥 Liver Transplant Death Risk Predictor")
st.write("This app uses a Random Forest model trained on liver transplant waitlist data to estimate death risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=90, value=50)
    year = st.number_input("Listing year", min_value=1985, max_value=2025, value=1995)
    futime = st.number_input("Follow-up time (days)", min_value=0, max_value=2000, value=200)

with col2:
    sex = st.selectbox("Sex", ['m', 'f'])
    abo = st.selectbox("Blood group", ['A', 'B', 'AB', 'O'])

if st.button("Predict death risk"):
    # encode
    sex_enc = le_sex.transform([sex])[0]
    abo_enc = le_abo.transform([abo])[0]

    # row
    row = pd.DataFrame([{
        'age': age,
        'year': year,
        'futime': futime,
        'sex_enc': sex_enc,
        'abo_enc': abo_enc
    }])

    # scale
    row_scaled = scaler.transform(row[feature_cols])

    # predict
    pred = clf.predict(row_scaled)[0]
    proba = clf.predict_proba(row_scaled)[0][1]

    st.subheader("Result")
    st.write(f"**Estimated death probability:** {proba:.1%}")
    if pred == 1:
        st.error("Model suggests: HIGH RISK (death).")
    else:
        st.success("Model suggests: LOW RISK (no death).")
