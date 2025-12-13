tum mujhe file do proper
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots

import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

st.set_page_config(
    page_title="TransplantCare – Advanced Waitlist Risk Analyzer",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3, h4 {
        color: #0f766e;
        font-weight: 600;
    }
    p, label, span, .stMarkdown, .stText, .stCaption, .stDataFrame, .stMetric {
        color: #334155 !important;
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #334155 !important;
        font-size: 0.95rem;
    }
    .stRadio > label {
        font-weight: 500;
        color: #0f766e;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 14px rgba(15, 118, 110, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #115e59 0%, #0f766e 100%);
        box-shadow: 0 6px 20px rgba(15, 118, 110, 0.4);
        transform: translateY(-2px);
    }
    .stMetric > div > div > div {
        color: #0f766e !important;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div > div > div {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        color: #334155 !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
        color: white;
    }
    .plotly-graph-div {
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_objects():
    clf = pickle.load(open("clf.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le_sex = pickle.load(open("le_sex.pkl", "rb"))
    le_abo = pickle.load(open("le_abo.pkl", "rb"))
    feature_cols = pickle.load(open("feature_cols.pkl", "rb"))
    explainer = shap.TreeExplainer(clf)  # [web:345]
    return clf, scaler, le_sex, le_abo, feature_cols, explainer

clf, scaler, le_sex, le_abo, feature_cols, explainer = load_objects()
@st.cache_data
def load_data():
    df = pd.read_csv("transplant.csv")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    if "abo" in df.columns:
        df["abo"] = df["abo"].astype(str).str.strip().str.upper()
    return df

def compute_population_stats(df, clf, scaler, le_sex, le_abo, feature_cols):
    df_prep = df.copy()
    if "sex" in df_prep.columns:
        df_prep = df_prep[df_prep["sex"].isin(le_sex.classes_)]
    if "abo" in df_prep.columns:
        df_prep = df_prep[df_prep["abo"].isin(le_abo.classes_)]

    df_prep["sex_enc"] = le_sex.transform(df_prep["sex"])
    df_prep["abo_enc"] = le_abo.transform(df_prep["abo"])

    X = df_prep[feature_cols]
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]
    df_prep["death_proba"] = probs
    df_prep["risk_percentile"] = df_prep["death_proba"].rank(pct=True) * 100

    overall_death_rate = df_prep["death_proba"].mean() * 100
    death_by_age = (
        df_prep.groupby("age_group")["death_proba"].mean() * 100
        if "age_group" in df_prep.columns
        else pd.Series(dtype=float)
    )
    death_by_sex = df_prep.groupby("sex")["death_proba"].mean() * 100
    death_by_abo = df_prep.groupby("abo")["death_proba"].mean() * 100

    return df_prep, overall_death_rate, death_by_age, death_by_sex, death_by_abo

st.markdown(
    """
    <div class="main-header">
        <h1 style='margin: 0; font-size: 2.5rem;'>🩺 TransplantCare</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
            Advanced AI-Powered Liver Transplant Waitlist Risk Analyzer
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #64748b; max-width: 800px; margin: 0 auto;'>
            Leveraging Random Forest ML with SHAP explanations, interactive survival curves, and synthetic clinical features for educational insights.<br>
            <em>Demo only – not for clinical use.</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.sidebar.title(" Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        " Risk Prediction & Insights",
        " EDA – Basic Distributions",
        " EDA – Advanced Clinical Features",
        " Model Performance & Explanations",
        " Survival Analysis",
    ],
)

# =====================================================
# 1) Risk Prediction & Insights
# =====================================================
if page == " Risk Prediction & Insights":
    st.markdown("### Enter Patient Profile")
    tab1, tab2 = st.tabs(["Basic Inputs", "Advanced Clinical Inputs"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 1, 100, 50)
            year = st.number_input("Listing Year", 1985, 2025, 1995)
            futime = st.number_input("Follow-up Time (days)", 0, 2000, 200)
        with col2:
            sex = st.selectbox("Sex", ["m", "f"])
            abo = st.selectbox("Blood Group", ["A", "B", "AB", "O"])

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            bmi = st.slider("BMI (kg/m²)", 15.0, 45.0, 25.0)
            meld_self = st.slider("MELD-like Score", 6, 40, 20)
            sodium = st.slider("Serum Sodium (mmol/L)", 120, 145, 135)
        with col4:
            bilirubin = st.slider("Bilirubin (mg/dL)", 0.5, 30.0, 5.0)
            creatinine = st.slider("Creatinine (mg/dL)", 0.5, 4.0, 1.2)
            inr = st.slider("INR", 1.0, 4.0, 1.5)
            albumin = st.slider("Albumin (g/dL)", 2.0, 4.5, 3.2)
        col5, col6 = st.columns(2)
        with col5:
            ascites = st.checkbox("Presence of Ascites")
            encephalopathy = st.checkbox("Hepatic Encephalopathy")
            diabetes = st.checkbox("Diabetes")
        with col6:
            hypertension = st.checkbox("Hypertension")
            smoker = st.checkbox("Current/Former Smoker")

    st.markdown("---")
    predict_btn = st.button(" Generate Advanced Risk Report", use_container_width=True)

    if predict_btn:
        sex_norm = sex.lower().strip()
        abo_norm = abo.upper().strip()
        sex_enc = le_sex.transform([sex_norm])[0]
        abo_enc = le_abo.transform([abo_norm])[0]

        row = pd.DataFrame(
            [
                {
                    "age": age,
                    "year": year,
                    "futime": futime,
                    "sex_enc": sex_enc,
                    "abo_enc": abo_enc,
                }
            ]
        )
        row_scaled = scaler.transform(row[feature_cols])
        pred = clf.predict(row_scaled)[0]
        proba = float(clf.predict_proba(row_scaled)[0][1])

        df_all = load_data()
        df_prep, overall_death_rate, death_by_age, death_by_sex, death_by_abo = compute_population_stats(
            df_all, clf, scaler, le_sex, le_abo, feature_cols
        )

        meld_adjust = (meld_self - 20) / 20 * 0.1
        bmi_adjust = -0.05 if bmi > 30 else 0
        sodium_adjust = -0.1 if sodium < 130 else 0
        adjusted_proba = float(np.clip(proba + meld_adjust + bmi_adjust + sodium_adjust, 0, 1))

        patient_probas = df_prep["death_proba"].values
        risk_percentile = np.sum(adjusted_proba > patient_probas) / len(patient_probas) * 100

        # -------- SHAP BLOCK (fully flattened) --------
        shap_raw = explainer.shap_values(row_scaled)
        shap_arr = np.array(shap_raw[-1]) if isinstance(shap_raw, list) else np.array(shap_raw)
        shap_flat = shap_arr.reshape(-1)          # flatten to 1D [web:346]
        shap_vec = shap_flat[: len(feature_cols)] # match feature count
        shap_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "shap_value": shap_vec,
            }
        )
        shap_df = shap_df.reindex(
            shap_df["shap_value"].abs().sort_values(ascending=False).index
        )
        # ----------------------------------------------

        st.markdown("###  Risk Assessment Report")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Base Death Probability", f"{proba*100:.1f}%")
        with m2:
            st.metric("Adjusted Risk", f"{adjusted_proba*100:.1f}%")
        with m3:
            label = "HIGH RISK " if pred == 1 else "LOW RISK "
            st.metric("Predicted Outcome", label)
        with m4:
            st.metric(
                "Risk Percentile",
                f"{risk_percentile:.0f}%",
                delta=f"vs {overall_death_rate:.1f}% avg",
            )

        if adjusted_proba < 0.15:
            txt = "Low risk: Monitor routinely."
            icon = ""
        elif adjusted_proba < 0.35:
            txt = "Moderate risk: Consider expedited evaluation."
            icon = ""
        else:
            txt = "High risk: Urgent intervention recommended."
            icon = ""
        st.markdown(f"**{icon} Interpretation:** {txt}")

        st.markdown("###  Your Risk vs Population")
        c1, c2 = st.columns(2)
        with c1:
            sex_mean = float(death_by_sex.get(sex_norm, overall_death_rate))
            abo_mean = float(death_by_abo.get(abo_norm, overall_death_rate))
            comp_df = pd.DataFrame(
                {
                    "Group": ["Your Risk", "Overall Avg", "By Sex", "By ABO"],
                    "Rate (%)": [adjusted_proba * 100, overall_death_rate, sex_mean, abo_mean],
                }
            )
            fig_comp = px.bar(
                comp_df,
                x="Group",
                y="Rate (%)",
                title="Risk Benchmarks",
                color="Group",
                color_discrete_sequence=["#ef4444", "#10b981", "#3b82f6", "#f59e0b"],
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with c2:
            st.markdown("###  Feature Impact (SHAP)")
            fig_shap = px.bar(
                shap_df.head(5),
                x="shap_value",
                y="feature",
                orientation="h",
                title="Top Contributors to Risk",
                color="shap_value",
                color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("### 🩺 Clinical Feature Insights")
        clinical_df = pd.DataFrame(
            {
                "Feature": ["BMI", "MELD", "Sodium", "Bilirubin", "Creatinine", "INR", "Albumin"],
                "Value": [bmi, meld_self, sodium, bilirubin, creatinine, inr, albumin],
                "Status": [
                    "Obese" if bmi > 30 else "Normal",
                    "High" if meld_self > 25 else "Moderate",
                    "Low" if sodium < 130 else "Normal",
                    "Elevated" if bilirubin > 2 else "Normal",
                    "High" if creatinine > 1.5 else "Normal",
                    "Elevated" if inr > 1.5 else "Normal",
                    "Low" if albumin < 3.5 else "Normal",
                ],
            }
        )
        st.dataframe(clinical_df, use_container_width=True)
        st.caption("*Adjustments are heuristic for demo; real models would integrate all features.*")
