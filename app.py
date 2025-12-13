import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Basic styling (background + fonts)
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
st.markdown("## TransplantCare – Waitlist Death Risk")
st.caption(
    "Interactive tool using a Random Forest model trained on liver transplant waitlist data "
    "to estimate death risk for educational purposes only. "
    "Made by Muhammad Moawiz Sipra (Roll no: BSDSF24A036)."
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

# -----------------------
# EDA / Graphs section
# -----------------------
st.markdown("---")
st.markdown("### 📊 Explore dataset (EDA)")

# CSV ko load karo (app ke same folder se)
@st.cache_data
def load_data():
    # Streamlit app ke folder me transplant.csv rakho
    df = pd.read_csv("transplant.csv")
    return df

try:
    df = load_data()
    show_raw = st.checkbox("Show first 10 rows of raw data")
    if show_raw:
        st.dataframe(df.head(10))

    sns.set(style="whitegrid")

    # 1) Age distribution
    st.markdown("#### Age distribution")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.histplot(df["age"].dropna(), bins=20, kde=True, ax=ax1, color="#38bdf8")
    ax1.set_xlabel("Age (years)")
    st.pyplot(fig1)

    # 2) Follow‑up time distribution
    st.markdown("#### Follow-up time (days)")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(df["futime"], bins=30, kde=True, ax=ax2, color="#22c55e")
    ax2.set_xlabel("Follow-up time (days)")
    st.pyplot(fig2)

    # 3) Event counts (death / ltx / censored / withdraw)
    st.markdown("#### Outcome / event counts")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x="event", order=df["event"].value_counts().index, ax=ax3, palette="mako")
    ax3.set_xlabel("Event")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

    # 4) Sex vs event
    st.markdown("#### Sex vs event")
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x="sex", hue="event", ax=ax4, palette="Set2")
    ax4.set_xlabel("Sex")
    st.pyplot(fig4)

    # 5) Blood group vs event
    st.markdown("#### Blood group vs event")
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x="abo", hue="event", ax=ax5, palette="Set3")
    ax5.set_xlabel("Blood group")
    st.pyplot(fig5)

except FileNotFoundError:
    st.warning(
        "For EDA graphs, please upload **transplant.csv** to the same directory as `app.py` "
        "in your GitHub repository."
    )
