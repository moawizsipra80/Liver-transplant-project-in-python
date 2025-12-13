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
import google.generativeai as genai  # Gemini

# -----------------------
# Gemini config
# -----------------------
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    # sirf warning; app phir bhi chale
    st.sidebar.warning("GEMINI_API_KEY secret not set; chat page will be disabled.")

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="TransplantCare – Advanced Waitlist Risk Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Global styling
# -----------------------
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

# -----------------------
# Load ML objects
# -----------------------
@st.cache_resource
def load_objects():
    clf = pickle.load(open("clf.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le_sex = pickle.load(open("le_sex.pkl", "rb"))
    le_abo = pickle.load(open("le_abo.pkl", "rb"))
    feature_cols = pickle.load(open("feature_cols.pkl", "rb"))
    explainer = shap.TreeExplainer(clf)
    return clf, scaler, le_sex, le_abo, feature_cols, explainer

clf, scaler, le_sex, le_abo, feature_cols, explainer = load_objects()

# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("transplant.csv")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    if "abo" in df.columns:
        df["abo"] = df["abo"].astype(str).str.strip().str.upper()
    return df

# -----------------------
# Population stats
# -----------------------
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

# -----------------------
# Gemini helper
# -----------------------
def ask_gemini_about_data(question: str, df: pd.DataFrame) -> str:
    if not question.strip():
        return "Please type a question first."
    if "GEMINI_API_KEY" not in st.secrets:
        return "Gemini API key is not configured in Streamlit secrets."

    df_sample = df.sample(min(len(df), 200), random_state=0)
    cols = [c for c in df_sample.columns if c not in ["sex_enc", "abo_enc"]]
    df_sample = df_sample[cols]
    sample_text = df_sample.to_csv(index=False)[:6000]

    prompt = f"""
You are a data analyst for a liver transplant waitlist dataset.

Here is a CSV sample from the dataset:

{sample_text}

Answer the user's question using only information consistent with this dataset sample.
Explain briefly (3-6 sentences) in simple language.

User question: {question}
"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(prompt)
    return resp.text.strip()

# -----------------------
# Header
# -----------------------
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

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        " Risk Prediction & Insights",
        " EDA – Basic Distributions",
        " EDA – Advanced Clinical Features",
        " Model Performance & Explanations",
        " Survival Analysis",
        " Chat with Dataset",
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
        shap_flat = shap_arr.reshape(-1)
        shap_vec = shap_flat[: len(feature_cols)]
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
            label = "HIGH RISK" if pred == 1 else "LOW RISK"
            st.metric("Predicted Outcome", label)
        with m4:
            st.metric(
                "Risk Percentile",
                f"{risk_percentile:.0f}%",
                delta=f"vs {overall_death_rate:.1f}% avg",
            )

        if adjusted_proba < 0.15:
            txt = "Low risk: Monitor routinely."
        elif adjusted_proba < 0.35:
            txt = "Moderate risk: Consider expedited evaluation."
        else:
            txt = "High risk: Urgent intervention recommended."
        st.markdown(f"**Interpretation:** {txt}")

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

# =====================================================
# 2) EDA – Basic Distributions
# =====================================================
elif page == " EDA – Basic Distributions":
    st.markdown("### 📊 Interactive Dataset Exploration (Basic)")
    try:
        df = load_data()
        if st.checkbox("Show first 20 rows"):
            st.dataframe(df.head(20), use_container_width=True)

        tab1, tab2 = st.tabs(["Demographics", "Outcomes"])
        with tab1:
            a, b = st.columns(2)
            with a:
                fig_age = px.histogram(
                    df,
                    x="age",
                    nbins=20,
                    marginal="rug",
                    title="Age Distribution",
                    color_discrete_sequence=["#38bdf8"],
                )
                fig_age.update_layout(height=400)
                st.plotly_chart(fig_age, use_container_width=True)
            with b:
                fig_ft = px.histogram(
                    df,
                    x="futime",
                    nbins=30,
                    marginal="box",
                    title="Follow-up Time Distribution",
                    color_discrete_sequence=["#22c55e"],
                )
                fig_ft.update_layout(height=400)
                st.plotly_chart(fig_ft, use_container_width=True)

        with tab2:
            c, d = st.columns(2)
            with c:
                fig_sex = px.histogram(
                    df,
                    x="sex",
                    color="event",
                    title="Sex vs Event Outcomes",
                    barmode="group",
                    color_discrete_map={
                        "death": "#ef4444",
                        "ltx": "#10b981",
                        "censored": "#f59e0b",
                        "withdraw": "#8b5cf6",
                    },
                )
                st.plotly_chart(fig_sex, use_container_width=True)
            with d:
                fig_abo = px.histogram(
                    df,
                    x="abo",
                    color="event",
                    title="Blood Group vs Event Outcomes",
                    barmode="group",
                    color_discrete_map={
                        "death": "#ef4444",
                        "ltx": "#10b981",
                        "censored": "#f59e0b",
                        "withdraw": "#8b5cf6",
                    },
                )
                st.plotly_chart(fig_abo, use_container_width=True)

        st.markdown("### 🔗 Correlation Heatmap (Numeric Features)")
        num_cols = df.select_dtypes(include=[np.number]).columns
        fig_corr = px.imshow(
            df[num_cols].corr(),
            title="Feature Correlations",
            aspect="auto",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    except FileNotFoundError:
        st.warning("Upload transplant.csv to view EDA.")

# =====================================================
# 3) EDA – Advanced Clinical Features
# =====================================================
elif page == " EDA – Advanced Clinical Features":
    st.markdown("### 🧬 Deep Dive: Synthetic Clinical Risk Factors")
    try:
        df = load_data()
        needed_cols = [
            "age_group",
            "bmi",
            "meld_score",
            "sodium",
            "bilirubin",
            "creatinine",
            "inr",
            "albumin",
            "ascites",
            "encephalopathy",
            "diabetes",
            "hypertension",
            "smoker",
            "center_region",
            "is_death",
        ]
        available = [c for c in needed_cols if c in df.columns]
        if len(available) < len(needed_cols) * 0.5:
            st.error("Enriched CSV with synthetic features required.")
        else:
            st.markdown("#### 🔍 Dynamic Filters")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                age_group_sel = st.multiselect(
                    "Age Group",
                    sorted(df["age_group"].dropna().unique()),
                    default=sorted(df["age_group"].dropna().unique()),
                )
            with c2:
                region_sel = st.multiselect(
                    "Center Region",
                    sorted(df["center_region"].dropna().unique()),
                    default=sorted(df["center_region"].dropna().unique()),
                )
            with c3:
                event_sel = st.multiselect(
                    "Event Type",
                    sorted(df["event"].dropna().unique()),
                    default=sorted(df["event"].dropna().unique()),
                )
            with c4:
                death_only = st.checkbox("Filter to Death Cases Only")

            df_f = df.copy()
            if age_group_sel:
                df_f = df_f[df_f["age_group"].isin(age_group_sel)]
            if region_sel:
                df_f = df_f[df_f["center_region"].isin(region_sel)]
            if event_sel:
                df_f = df_f[df_f["event"].isin(event_sel)]
            if death_only:
                df_f = df_f[df_f["is_death"] == 1]

            st.markdown("#### 🚀 One-Click Analyses")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                btn_meld = st.button("📊 MELD vs Outcomes")
            with b2:
                btn_bmi = st.button("📈 BMI & Comorbidities")
            with b3:
                btn_region = st.button("🌍 Regional Variations")
            with b4:
                btn_labs = st.button("🧪 Lab Correlations")

            a, b = st.columns(2)
            with a:
                if "meld_score" in df_f.columns:
                    fig_meld = px.histogram(
                        df_f,
                        x="meld_score",
                        color="is_death",
                        title="MELD Score Distribution by Death Risk",
                        nbins=25,
                        marginal="violin",
                        color_discrete_map={0: "#10b981", 1: "#ef4444"},
                    )
                    st.plotly_chart(fig_meld, use_container_width=True)
            with b:
                if "sodium" in df_f.columns and "meld_score" in df_f.columns:
                    fig_scatter = px.scatter(
                        df_f.sample(min(len(df_f), 500)),
                        x="sodium",
                        y="meld_score",
                        color="event",
                        title="Sodium vs MELD (Colored by Event)",
                        hover_data=["age", "sex"],
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

            if btn_meld:
                fig_box = px.box(
                    df_f,
                    x="event",
                    y="meld_score",
                    title="MELD by Event Type",
                    color="event",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            if btn_bmi:
                fig_bmi = px.density_contour(
                    df_f,
                    x="bmi",
                    y="age",
                    color="event",
                    title="BMI Density by Age & Event",
                )
                st.plotly_chart(fig_bmi, use_container_width=True)

            if btn_region and "center_region" in df_f.columns:
                rate = df_f.groupby("center_region")["is_death"].mean().reset_index()
                fig_region = px.bar(
                    rate,
                    x="center_region",
                    y="is_death",
                    title="Death Rate by Region",
                    color="is_death",
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig_region, use_container_width=True)

            if btn_labs:
                lab_cols = ["bilirubin", "creatinine", "inr", "albumin"]
                labs = [c for c in lab_cols if c in df_f.columns]
                if labs:
                    fig_labs = px.imshow(
                        df_f[labs + ["is_death"]].corr(),
                        title="Lab Correlations with Death Risk",
                        color_continuous_scale="RdBu_r",
                    )
                    st.plotly_chart(fig_labs, use_container_width=True)

            st.markdown("### 🤰 Comorbidity Overview")
            if all(c in df_f.columns for c in ["diabetes", "hypertension", "smoker", "ascites"]):
                comm_df = (
                    df_f[["diabetes", "hypertension", "smoker", "ascites"]]
                    .sum()
                    .reset_index()
                )
                comm_df.columns = ["Condition", "Count"]
                fig_pie = px.pie(
                    comm_df,
                    values="Count",
                    names="Condition",
                    title="Prevalence of Key Comorbidities",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data: {e}. Ensure enriched CSV is uploaded.")

# =====================================================
# 4) Model Performance & Explanations
# =====================================================
elif page == " Model Performance & Explanations":
    st.markdown("### 🤖 Random Forest Model Deep Dive")
    try:
        df = load_data()
        df_prep, _, _, _, _ = compute_population_stats(df, clf, scaler, le_sex, le_abo, feature_cols)
        X = df_prep[feature_cols]
        y = (
            df_prep["is_death"]
            if "is_death" in df_prep.columns
            else (df_prep["event"] == "death").astype(int)
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        t1, t2 = st.tabs(["Metrics", "Visualizations"])
        with t1:
            st.markdown("#### 📊 Key Performance Metrics")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Accuracy", f"{(y_pred == y_test).mean():.2%}")
            rep = classification_report(y_test, y_pred, output_dict=True)
            with c2:
                st.metric("Precision (Death)", f"{rep['1']['precision']:.2%}")
            with c3:
                st.metric("Recall (Death)", f"{rep['1']['recall']:.2%}")
            with c4:
                st.metric("AUC-ROC", f"{roc_auc:.3f}")
            st.text(classification_report(y_test, y_pred))

        with t2:
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            fig_roc = px.area(
                x=fpr,
                y=tpr,
                title=f"ROC Curve (AUC = {roc_auc:.3f})",
                labels=dict(x="False Positive Rate", y="True Positive Rate"),
            )
            fig_roc.add_shape(
                type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1
            )
            st.plotly_chart(fig_roc, use_container_width=True)

            importance = pd.DataFrame(
                {"feature": feature_cols, "importance": clf.feature_importances_}
            ).sort_values("importance", ascending=False)
            fig_imp = px.bar(
                importance,
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance",
                color="importance",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"Error evaluating model: {e}")

# =====================================================
# 5) Survival Analysis
# =====================================================
elif page == " Survival Analysis":
    st.markdown("### 📉 Time-to-Event Analysis")
    try:
        df = load_data()
        if "futime" not in df.columns or "event" not in df.columns:
            st.error("Dataset missing futime or event columns.")
        else:
            df["surv_event"] = (df["event"] == "death").astype(int)
            stratify_by = st.selectbox(
                "Stratify KM by", ["None", "sex", "abo", "age_group", "center_region"]
            )
            kmf = KaplanMeierFitter()
            plt.figure(figsize=(6, 4))

            if stratify_by == "None":
                kmf.fit(df["futime"], event_observed=df["surv_event"])
                kmf.plot_survival_function()
                plt.title("Overall Waitlist Survival")
            else:
                groups = df[stratify_by].dropna().unique()
                colors = px.colors.qualitative.Set1
                for i, g in enumerate(groups):
                    mask = df[stratify_by] == g
                    kmf.fit(
                        df.loc[mask, "futime"],
                        event_observed=df.loc[mask, "surv_event"],
                        label=str(g),
                    )
                    kmf.plot_survival_function(color=colors[i % len(colors)])
                plt.title(f"Survival by {stratify_by.capitalize()}")

            plt.ylabel("Survival Probability")
            plt.xlabel("Time (Days)")
            st.pyplot(plt.gcf())

            if stratify_by != "None":
                groups = df[stratify_by].dropna().unique()
                if len(groups) >= 2:
                    res = logrank_test(
                        df[df[stratify_by] == groups[0]]["futime"],
                        df[df[stratify_by] == groups[1]]["futime"],
                        event_observed_A=df[df[stratify_by] == groups[0]]["surv_event"],
                        event_observed_B=df[df[stratify_by] == groups[1]]["surv_event"],
                    )
                    st.markdown(
                        f"**Log-Rank Test p-value:** {res.p_value:.4f} "
                        f"({'Significant' if res.p_value < 0.05 else 'Not Significant'})"
                    )

    except Exception as e:
        st.error(f"Error in survival analysis: {e}")

# =====================================================
# 6) Chat with Dataset (Gemini)
# =====================================================
elif page == " Chat with Dataset":
    st.markdown("### 🤖 Ask questions about this dataset/model")
    st.write(
        "Type natural-language questions about the waitlist data or model behaviour. "
        "The Gemini assistant will use a sample of the dataset to answer."
    )

    df = load_data()
    user_q = st.text_area(
        "Your question",
        value="How does death risk differ between males and females in this dataset?",
        height=80,
    )

    if st.button("Ask Gemini", use_container_width=True):
        with st.spinner("Analyzing dataset and generating answer..."):
            answer = ask_gemini_about_data(user_q, df)
        st.markdown("#### Answer")
        st.write(answer)

    st.caption(
        "Note: The assistant only sees a sampled subset of the dataset for privacy and performance, "
        "so answers are approximate and for educational use."
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>"
    "© 2025 TransplantCare Demo | Educational Purposes Only"
    "</div>",
    unsafe_allow_html=True,
)
