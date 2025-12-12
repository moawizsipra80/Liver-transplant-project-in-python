import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------
# Basic styling (background + fonts)
# -----------------------
st.set_page_config(
    page_title="TransplantCare – Waitlist Risk",
    page_icon="🩺",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;  /* dark navy */
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Load saved objects
# -----------------------
@st.cache_resource
def load_objects():
    clf = pickle.load(open('clf.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le_sex = pickle.load(open('le_sex.pkl', 'rb'))
    le_abo = pickle.load(open('le_abo.pkl', 'rb'))
    feature_cols = pickle.load(open('feature_cols.pkl', 'rb'))
    return clf, scaler, le_sex, le_abo, feature_cols

clf, scaler, le_sex, le_abo, feature_cols = load_objects()

# -----------------------
# Header
# -----------------------
st.markdown("## 🩺 TransplantCare – Waitlist Death Risk")
st.caption(
    "Interactive tool using a Random Forest model trained on liver transplant waitlist data "
    "to estimate death risk for educational purposes only."
)

st.markdown("---")
st.markdown("### Enter patient details")

# -----------------------
# Input form
# -----------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=50)
    year = st.number_input("Listing year", min_value=1985, max_value=2025, value=1995)
    futime = st.number_input("Follow-up time (days)", min_value=0, max_value=2000, value=200)

with col2:
    sex = st.selectbox("Sex", ['m', 'f'])
    abo = st.selectbox("Blood group", ['A', 'B', 'AB', 'O'])

st.markdown("")
predict_btn = st.button("🔮 Predict death risk")

# -----------------------
# Prediction
# -----------------------
if predict_btn:
    # encode
    sex_enc = le_sex.transform([sex])[0]
    abo_enc = le_abo.transform([abo])[0]

    # row (same feature order as training)
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

    st.markdown("---")
    st.markdown("### Result")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric("Estimated death probability", f"{proba:.1%}")
    with mcol2:
        label = "HIGH RISK (death)" if pred == 1 else "LOW RISK (no death)"
        if pred == 1:
            st.error(label)
        else:
            st.success(label)

    st.caption(
        "Note: This model is built on historical waitlist data and is intended for study/demo use, "
        "not for real medical decision‑making."
    )
