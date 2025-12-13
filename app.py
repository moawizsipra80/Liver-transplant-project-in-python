import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="TransplantCare – Waitlist Risk",
    page_icon="🩺",
    layout="wide"
)

# -----------------------
# Global styling (white background + clear text + strong buttons)
# -----------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;  /* pure white */
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1100px;
        margin: 0 auto;
    }
    h1, h2, h3, h4 {
        color: #111827;  /* very dark gray */
        font-weight: 700;
    }
    p, label, span, .stMarkdown, .stText, .stCaption, .stDataFrame, .stMetric {
        color: #111827 !important;
        font-size: 0.95rem;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f3f4f6;
    }
    section[data-testid="stSidebar"] * {
        color: #111827 !important;
        font-size: 0.95rem;
    }
    .stRadio > label {
        font-weight: 600;
    }
    /* Buttons */
    .stButton > button {
        background-color: #0f766e;
        color: #ffffff;
        border-radius: 999px;
        padding: 0.5rem 1.4rem;
        border: none;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stButton > button:hover {
        background-color: #115e59;
        border: none;
    }
    /* Checkboxes, multiselect labels */
    .stCheckbox > label, .stMultiSelect > label {
        font-weight: 600;
        color: #111827 !important;
    }
    .stMarkdown h4, .stMarkdown h3 {
        margin-top: 0.75rem;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)  # [web:336][web:333][web:340]

# -----------------------
# Load saved ML objects
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
# Load dataset for EDA
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("transplant.csv")
    return df

# -----------------------
# Header + layout
# -----------------------
st.markdown("## TransplantCare – Waitlist Death Risk")
st.caption(
    "Interactive tool using a Random Forest model trained on liver transplant waitlist data "
    "to estimate death risk for educational purposes only. "
    "Made by Muhammad Moawiz Sipra (Roll no: BSDSF24A036)."
)

st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Risk Prediction", "EDA – Basic", "EDA – Advanced features"]
)  # [web:334]

# -----------------------
# 1) Risk prediction page
# -----------------------
if page == "Risk Prediction":
    st.markdown("### Enter patient details")

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

    if predict_btn:
        # encode
        sex_enc = le_sex.transform([sex])[0]
        abo_enc = le_abo.transform([abo])[0]

        # row same order as training
        row = pd.DataFrame([{
            'age': age,
            'year': year,
            'futime': futime,
            'sex_enc': sex_enc,
            'abo_enc': abo_enc
        }])

        row_scaled = scaler.transform(row[feature_cols])

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
# 2) EDA – Basic
# -----------------------
elif page == "EDA – Basic":
    st.markdown("### 📊 Explore dataset (basic EDA)")

    try:
        df = load_data()
        show_raw = st.checkbox("Show first 10 rows of raw data")
        if show_raw:
            st.dataframe(df.head(10))

        sns.set(style="whitegrid")

        st.markdown("#### Age distribution")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.histplot(df["age"].dropna(), bins=20, kde=True, ax=ax1, color="#38bdf8")
        ax1.set_xlabel("Age (years)")
        st.pyplot(fig1)

        st.markdown("#### Follow-up time (days)")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.histplot(df["futime"], bins=30, kde=True, ax=ax2, color="#22c55e")
        ax2.set_xlabel("Follow-up time (days)")
        st.pyplot(fig2)

        st.markdown("#### Outcome / event counts")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="event", order=df["event"].value_counts().index, ax=ax3, palette="mako")
        ax3.set_xlabel("Event")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)

        st.markdown("#### Sex vs event")
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="sex", hue="event", ax=ax4, palette="Set2")
        ax4.set_xlabel("Sex")
        st.pyplot(fig4)

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

# -----------------------
# 3) EDA – Advanced features
# -----------------------
elif page == "EDA – Advanced features":
    st.markdown("### 🧬 Advanced features (synthetic clinical variables)")
    st.caption(
        "These extra features are synthetic but inspired by real liver waitlist risk factors "
        "like MELD score, sodium, bilirubin, creatinine, albumin, ascites, encephalopathy and center region."
    )  # [web:311][web:324]

    try:
        df = load_data()

        needed_cols = [
            "age_group", "bmi", "meld_score", "sodium", "bilirubin",
            "creatinine", "inr", "albumin", "ascites", "encephalopathy",
            "diabetes", "hypertension", "smoker", "center_region", "is_death"
        ]
        available = [c for c in needed_cols if c in df.columns]

        if len(available) == 0:
            st.error("New synthetic feature columns not found in transplant.csv. Please upload the enriched file.")
        else:
            # Filters
            st.markdown("#### Filters")
            c1, c2, c3 = st.columns(3)
            with c1:
                age_group_sel = st.multiselect(
                    "Age group",
                    options=sorted(df["age_group"].dropna().unique()),
                    default=sorted(df["age_group"].dropna().unique())
                ) if "age_group" in df.columns else []
            with c2:
                region_sel = st.multiselect(
                    "Center region",
                    options=sorted(df["center_region"].dropna().unique()) if "center_region" in df.columns else [],
                    default=sorted(df["center_region"].dropna().unique()) if "center_region" in df.columns else []
                ) if "center_region" in df.columns else []
            with c3:
                event_sel = st.multiselect(
                    "Event type",
                    options=sorted(df["event"].dropna().unique()),
                    default=sorted(df["event"].dropna().unique())
                )

            df_f = df.copy()
            if "age_group" in df.columns and age_group_sel:
                df_f = df_f[df_f["age_group"].isin(age_group_sel)]
            if "center_region" in df.columns and region_sel:
                df_f = df_f[df_f["center_region"].isin(region_sel)]
            if event_sel:
                df_f = df_f[df_f["event"].isin(event_sel)]

            sns.set(style="whitegrid")

            # Quick buttons
            st.markdown("#### Quick analysis buttons")
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                btn_meld = st.button("MELD vs death")
            with bcol2:
                btn_bmi = st.button("BMI distribution")
            with bcol3:
                btn_region = st.button("Region-wise death rate")

            # Main plots
            st.markdown("#### MELD score distribution")
            if "meld_score" in df_f.columns:
                fig_m, ax_m = plt.subplots(figsize=(5, 3))
                sns.histplot(df_f["meld_score"], bins=25, kde=True, ax=ax_m, color="#6366f1")
                ax_m.set_xlabel("MELD score (synthetic)")
                st.pyplot(fig_m)

            st.markdown("#### Sodium vs MELD (scatter)")
            if "sodium" in df_f.columns and "meld_score" in df_f.columns:
                fig_sc, ax_sc = plt.subplots(figsize=(5, 3))
                sns.scatterplot(
                    data=df_f.sample(min(len(df_f), 300), random_state=0),
                    x="sodium", y="meld_score",
                    hue="event", alpha=0.8, ax=ax_sc
                )
                ax_sc.set_xlabel("Sodium (mmol/L)")
                ax_sc.set_ylabel("MELD score")
                st.pyplot(fig_sc)

            if btn_meld and "is_death" in df_f.columns and "meld_score" in df_f.columns:
                st.markdown("#### MELD score by outcome")
                fig_b, ax_b = plt.subplots(figsize=(5, 3))
                sns.boxplot(data=df_f, x="event", y="meld_score", ax=ax_b)
                ax_b.set_xlabel("Event")
                ax_b.set_ylabel("MELD score")
                st.pyplot(fig_b)

            if btn_bmi and "bmi" in df_f.columns:
                st.markdown("#### BMI distribution by event")
                fig_bmi, ax_bmi = plt.subplots(figsize=(5, 3))
                sns.kdeplot(
                    data=df_f, x="bmi", hue="event",
                    fill=True, common_norm=False, alpha=0.4, ax=ax_bmi
                )
                ax_bmi.set_xlabel("BMI")
                st.pyplot(fig_bmi)

            if btn_region and "center_region" in df_f.columns and "is_death" in df_f.columns:
                st.markdown("#### Approx death rate per region")
                rate = (
                    df_f.groupby("center_region")["is_death"]
                    .mean()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                fig_r, ax_r = plt.subplots(figsize=(5, 3))
                sns.barplot(data=rate, x="center_region", y="is_death", ax=ax_r, color="#f97316")
                ax_r.set_ylabel("Death rate")
                st.pyplot(fig_r)

    except FileNotFoundError:
        st.warning(
            "For advanced EDA, please upload the enriched **transplant.csv** (with synthetic features) "
            "to the same directory as `app.py`."
        )
