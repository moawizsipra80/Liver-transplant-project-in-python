import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap  # For model explanations (assume installed)
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import lifelines  # For survival analysis (assume installed)
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

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
# Global styling (enhanced for modern look: gradients, shadows, animations)
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
    /* Sidebar enhancements */
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
    /* Buttons (glowing effect) */
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
    /* Metrics cards */
    .stMetric > div > div > div {
        color: #0f766e !important;
        font-size: 1.2rem;
        font-weight: 700;
    }
    /* Inputs */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div > div > div {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        color: #334155 !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
        color: white;
    }
    /* Animations for plots */
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
# Load saved ML objects (enhanced with SHAP explainer)
# -----------------------
@st.cache_resource
def load_objects():
    clf = pickle.load(open('clf.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le_sex = pickle.load(open('le_sex.pkl', 'rb'))
    le_abo = pickle.load(open('le_abo.pkl', 'rb'))
    feature_cols = pickle.load(open('feature_cols.pkl', 'rb'))
    # SHAP explainer for advanced explanations
    explainer = shap.TreeExplainer(clf)
    return clf, scaler, le_sex, le_abo, feature_cols, explainer

clf, scaler, le_sex, le_abo, feature_cols, explainer = load_objects()

# -----------------------
# Load dataset for EDA and evaluations
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("transplant.csv")
    # Ensure enriched columns are present
    return df

# -----------------------
# Compute population statistics for comparisons
# -----------------------
@st.cache_data
def compute_population_stats(df, clf, scaler, le_sex, le_abo, feature_cols):
    # Prepare full dataset for predictions
    df_prep = df.copy()
    df_prep['sex_enc'] = le_sex.transform(df_prep['sex'])
    df_prep['abo_enc'] = le_abo.transform(df_prep['abo'])
    X = df_prep[feature_cols]
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]
    df_prep['death_proba'] = probs
    df_prep['risk_percentile'] = df_prep['death_proba'].rank(pct=True) * 100
    
    # Overall death rate
    overall_death_rate = df['is_death'].mean() * 100 if 'is_death' in df.columns else 0
    
    # Stratified rates
    death_by_age = df_prep.groupby('age_group')['death_proba'].mean() * 100 if 'age_group' in df.columns else pd.Series()
    death_by_sex = df_prep.groupby('sex')['death_proba'].mean() * 100
    death_by_abo = df_prep.groupby('abo')['death_proba'].mean() * 100
    
    return df_prep, overall_death_rate, death_by_age, death_by_sex, death_by_abo

# -----------------------
# Header + layout
# -----------------------
st.markdown(
    """
    <div class="main-header">
        <h1 style='margin: 0; font-size: 2.5rem;'>🩺 TransplantCare</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Advanced AI-Powered Liver Transplant Waitlist Risk Analyzer</p>
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
# Sidebar navigation (enhanced with icons and sub-options)
# -----------------------
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "🔮 Risk Prediction & Insights",
        "📊 EDA – Basic Distributions",
        "🧬 EDA – Advanced Clinical Features",
        "📈 Model Performance & Explanations",
        "📉 Survival Analysis"
    ]
)

# -----------------------
# 1) Enhanced Risk Prediction with SHAP, Percentiles, and Advanced Inputs
# -----------------------
if page == "🔮 Risk Prediction & Insights":
    st.markdown("### Enter Patient Profile")
    tab1, tab2 = st.tabs(["Basic Inputs", "Advanced Clinical Inputs"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=100, value=50)
            year = st.number_input("Listing Year", min_value=1985, max_value=2025, value=1995)
            futime = st.number_input("Follow-up Time (days)", min_value=0, max_value=2000, value=200)
        with col2:
            sex = st.selectbox("Sex", ['m', 'f'])
            abo = st.selectbox("Blood Group", ['A', 'B', 'AB', 'O'])
    
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
        # Binary comorbidities
        col5, col6 = st.columns(2)
        with col5:
            ascites = st.checkbox("Presence of Ascites")
            encephalopathy = st.checkbox("Hepatic Encephalopathy")
            diabetes = st.checkbox("Diabetes")
        with col6:
            hypertension = st.checkbox("Hypertension")
            smoker = st.checkbox("Current/Former Smoker")
    
    st.markdown("---")
    predict_btn = st.button("🔮 Generate Advanced Risk Report", use_container_width=True)
    
    if predict_btn:
        # Basic prediction
        sex_enc = le_sex.transform([sex])[0]
        abo_enc = le_abo.transform([abo])[0]
        row = pd.DataFrame([{
            'age': age, 'year': year, 'futime': futime,
            'sex_enc': sex_enc, 'abo_enc': abo_enc
        }])
        row_scaled = scaler.transform(row[feature_cols])
        pred = clf.predict(row_scaled)[0]
        proba = float(clf.predict_proba(row_scaled)[0][1])
        
        # Load data for comparisons
        df_prep, overall_death_rate, death_by_age, death_by_sex, death_by_abo = compute_population_stats(load_data(), clf, scaler, le_sex, le_abo, feature_cols)
        
        # Risk adjustments based on advanced features (heuristic for demo)
        meld_adjust = (meld_self - 20) / 20 * 0.1  # +/- 10% based on MELD deviation
        bmi_adjust = -0.05 if bmi > 30 else 0  # Penalty for obesity
        sodium_adjust = -0.1 if sodium < 130 else 0  # Hyponatremia penalty
        adjusted_proba = np.clip(proba + meld_adjust + bmi_adjust + sodium_adjust, 0, 1)
        
        # Percentile
        patient_probas = df_prep['death_proba'].values
        risk_percentile = np.sum(adjusted_proba > patient_probas) / len(patient_probas) * 100
        
        # SHAP explanation
        shap_values = explainer.shap_values(row_scaled)[1]
        shap_df = pd.DataFrame({'feature': feature_cols, 'shap_value': shap_values[0]})
        shap_df = shap_df.reindex(shap_df['shap_value'].abs().sort_values(ascending=False).index)
        
        # Results layout
        st.markdown("### 📋 Risk Assessment Report")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Base Death Probability", f"{proba*100:.1f}%")
        with mcol2:
            st.metric("Adjusted Risk (w/ Clinical Features)", f"{adjusted_proba*100:.1f}%")
        with mcol3:
            label = "HIGH RISK 🚨" if pred == 1 else "LOW RISK ✅"
            color = "inverse" if pred == 1 else "normal"
            st.metric("Predicted Outcome", label, delta=None, delta_color=color)
        with mcol4:
            st.metric("Risk Percentile", f"{risk_percentile:.0f}%", delta=f"vs {overall_death_rate:.1f}% avg")
        
        # Risk interpretation
        if adjusted_proba < 0.15:
            risk_text = "Low risk: Monitor routinely."
            icon = "🟢"
        elif adjusted_proba < 0.35:
            risk_text = "Moderate risk: Consider expedited evaluation."
            icon = "🟡"
        else:
            risk_text = "High risk: Urgent intervention recommended."
            icon = "🔴"
        st.markdown(f"**{icon} Interpretation:** {risk_text}")
        
        # Stratified comparisons
        st.markdown("### 📊 Your Risk vs Population")
        c1, c2 = st.columns(2)
        with c1:
            fig_comp = px.bar(
                pd.DataFrame({
                    'Group': ['Your Risk', 'Overall Avg', 'By Sex', 'By ABO'],
                    'Rate (%)': [adjusted_proba*100, overall_death_rate, death_by_sex[sex], death_by_abo[abo]]
                }),
                x='Group', y='Rate (%)', title="Risk Benchmarks",
                color='Group', color_discrete_sequence=['#ef4444', '#10b981', '#3b82f6', '#f59e0b']
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        # SHAP Waterfall
        with c2:
            st.markdown("### 🔍 Feature Impact (SHAP)")
            fig_shap = px.bar(shap_df.head(5), x='shap_value', y='feature', 
                              orientation='h', title="Top Contributors to Risk",
                              color='shap_value', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_shap, use_container_width=True)
        
        # Advanced clinical summary
        st.markdown("### 🩺 Clinical Feature Insights")
        clinical_df = pd.DataFrame({
            'Feature': ['BMI', 'MELD', 'Sodium', 'Bilirubin', 'Creatinine', 'INR', 'Albumin'],
            'Value': [bmi, meld_self, sodium, bilirubin, creatinine, inr, albumin],
            'Status': ['Obese' if bmi > 30 else 'Normal', 
                       'High' if meld_self > 25 else 'Moderate', 
                       'Low' if sodium < 130 else 'Normal',
                       'Elevated' if bilirubin > 2 else 'Normal',
                       'High' if creatinine > 1.5 else 'Normal',
                       'Elevated' if inr > 1.5 else 'Normal',
                       'Low' if albumin < 3.5 else 'Normal']
        })
        st.dataframe(clinical_df, use_container_width=True)
        
        st.caption("*Adjustments are heuristic for demo; real models would integrate all features.*")

# -----------------------
# 2) Enhanced EDA – Basic with Interactive Plots
# -----------------------
elif page == "📊 EDA – Basic Distributions":
    st.markdown("### 📊 Interactive Dataset Exploration (Basic)")
    st.write(
        "Dive into core distributions with Plotly interactivity: zoom, hover for details, and filter dynamically."
    )
    
    try:
        df = load_data()
        show_raw = st.checkbox("👁️ Show Raw Data Preview (First 20 Rows)")
        if show_raw:
            st.dataframe(df.head(20), use_container_width=True)
        
        # Interactive tabs for plots
        tab_basic1, tab_basic2 = st.tabs(["Demographics", "Outcomes"])
        
        with tab_basic1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_age = px.histogram(df, x="age", nbins=20, marginal="rug", 
                                       title="Age Distribution", color_discrete_sequence=["#38bdf8"])
                fig_age.update_layout(height=400)
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col_b:
                fig_futime = px.histogram(df, x="futime", nbins=30, marginal="box", 
                                          title="Follow-up Time Distribution", color_discrete_sequence=["#22c55e"])
                fig_futime.update_layout(height=400)
                st.plotly_chart(fig_futime, use_container_width=True)
        
        with tab_basic2:
            col_c, col_d = st.columns(2)
            with col_c:
                fig_sex_event = px.histogram(df, x="sex", color="event", 
                                             title="Sex vs Event Outcomes", barmode="group",
                                             color_discrete_map={"death": "#ef4444", "ltx": "#10b981", "censored": "#f59e0b", "withdraw": "#8b5cf6"})
                st.plotly_chart(fig_sex_event, use_container_width=True)
            
            with col_d:
                fig_abo_event = px.histogram(df, x="abo", color="event", 
                                             title="Blood Group vs Event Outcomes", barmode="group",
                                             color_discrete_map={"death": "#ef4444", "ltx": "#10b981", "censored": "#f59e0b", "withdraw": "#8b5cf6"})
                st.plotly_chart(fig_abo_event, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔗 Correlation Heatmap (Numeric Features)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        fig_corr = px.imshow(df[numeric_cols].corr(), title="Feature Correlations", aspect="auto",
                             color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Upload **transplant.csv** to view EDA.")

# -----------------------
# 3) Enhanced EDA – Advanced Features with Filters & More Plots
# -----------------------
elif page == "🧬 EDA – Advanced Clinical Features":
    st.markdown("### 🧬 Deep Dive: Synthetic Clinical Risk Factors")
    st.caption(
        "Explore engineered features like MELD, labs, comorbidities. Use filters for stratified analysis."
    )
    
    try:
        df = load_data()
        needed_cols = [
            "age_group", "bmi", "meld_score", "sodium", "bilirubin",
            "creatinine", "inr", "albumin", "ascites", "encephalopathy",
            "diabetes", "hypertension", "smoker", "center_region", "is_death"
        ]
        available = [c for c in needed_cols if c in df.columns]
        if len(available) < len(needed_cols) * 0.5:
            st.error("Enriched CSV with synthetic features required. Use the Python script to generate.")
            st.stop()
        
        # Advanced filters
        st.markdown("#### 🔍 Dynamic Filters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age_group_sel = st.multiselect(
                "Age Group", options=sorted(df["age_group"].dropna().unique()),
                default=sorted(df["age_group"].dropna().unique())
            )
        with c2:
            region_sel = st.multiselect(
                "Center Region", options=sorted(df["center_region"].dropna().unique()),
                default=sorted(df["center_region"].dropna().unique())
            )
        with c3:
            event_sel = st.multiselect(
                "Event Type", options=sorted(df["event"].dropna().unique()),
                default=sorted(df["event"].dropna().unique())
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
        
        # Quick analysis buttons (enhanced)
        st.markdown("#### 🚀 One-Click Analyses")
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            btn_meld = st.button("📊 MELD vs Outcomes")
        with bcol2:
            btn_bmi = st.button("📈 BMI & Comorbidities")
        with bcol3:
            btn_region = st.button("🌍 Regional Variations")
        with bcol4:
            btn_labs = st.button("🧪 Lab Correlations")
        
        # Default plots
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            if "meld_score" in df_f.columns:
                fig_meld = px.histogram(df_f, x="meld_score", color="is_death", 
                                        title="MELD Score Distribution by Death Risk",
                                        nbins=25, marginal="violin",
                                        color_discrete_map={0: "#10b981", 1: "#ef4444"})
                st.plotly_chart(fig_meld, use_container_width=True)
        
        with col_adv2:
            if "sodium" in df_f.columns and "meld_score" in df_f.columns:
                fig_scatter = px.scatter(df_f.sample(min(len(df_f), 500)), 
                                         x="sodium", y="meld_score", color="event",
                                         title="Sodium vs MELD (Colored by Event)",
                                         hover_data=["age", "sex"])
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Button-triggered advanced plots
        if btn_meld:
            fig_box = px.box(df_f, x="event", y="meld_score", 
                             title="MELD by Event Type", color="event")
            st.plotly_chart(fig_box, use_container_width=True)
        
        if btn_bmi:
            fig_bmi_kde = px.density_contour(df_f, x="bmi", y="age", color="event",
                                             title="BMI Density by Age & Event")
            st.plotly_chart(fig_bmi_kde, use_container_width=True)
        
        if btn_region:
            if "center_region" in df_f.columns:
                rate = df_f.groupby("center_region")["is_death"].mean().reset_index()
                fig_region = px.bar(rate, x="center_region", y="is_death", 
                                    title="Death Rate by Region", color="is_death",
                                    color_continuous_scale="Reds")
                st.plotly_chart(fig_region, use_container_width=True)
        
        if btn_labs:
            lab_cols = ["bilirubin", "creatinine", "inr", "albumin"]
            avail_labs = [c for c in lab_cols if c in df_f.columns]
            if avail_labs:
                fig_labs = px.imshow(df_f[avail_labs + ["is_death"]].corr(), 
                                     title="Lab Correlations with Death Risk",
                                     color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_labs, use_container_width=True)
        
        # Comorbidity pie
        st.markdown("### 🤰 Comorbidity Overview")
        if all(c in df_f.columns for c in ["diabetes", "hypertension", "smoker", "ascites"]):
            comm_df = df_f[["diabetes", "hypertension", "smoker", "ascites"]].sum().reset_index()
            comm_df.columns = ["Condition", "Count"]
            fig_pie = px.pie(comm_df, values="Count", names="Condition", 
                             title="Prevalence of Key Comorbidities")
            st.plotly_chart(fig_pie, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading data: {e}. Ensure enriched CSV is uploaded.")

# -----------------------
# 4) New Page: Model Performance & Explanations
# -----------------------
elif page == "📈 Model Performance & Explanations":
    st.markdown("### 🤖 Random Forest Model Deep Dive")
    st.write("Evaluate model accuracy, feature importance, and bias checks using the full dataset.")
    
    try:
        df = load_data()
        df_prep, _, _, _, _ = compute_population_stats(df, clf, scaler, le_sex, le_abo, feature_cols)
        
        # Prepare test set (80/20 split for demo)
        X = df_prep[feature_cols]
        y = df_prep['is_death'] if 'is_death' in df_prep.columns else (df_prep['event'] == 'death').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        tab_perf1, tab_perf2 = st.tabs(["Metrics", "Visualizations"])
        
        with tab_perf1:
            st.markdown("#### 📊 Key Performance Metrics")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                accuracy = (y_pred == y_test).mean()
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col_m2:
                precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
                st.metric("Precision (Death)", f"{precision:.2%}")
            with col_m3:
                recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
                st.metric("Recall (Death)", f"{recall:.2%}")
            with col_m4:
                st.metric("AUC-ROC", f"{roc_auc:.3f}")
            
            st.text(classification_report(y_test, y_pred))
        
        with tab_perf2:
            # Confusion Matrix
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix",
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # ROC Curve
            fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.3f})",
                              labels=dict(x="False Positive Rate", y="True Positive Rate"))
            fig_roc.add_shape(type="line", line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Feature Importance
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            fig_imp = px.bar(importance, x='importance', y='feature', 
                             title="Feature Importance", orientation='h',
                             color='importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error evaluating model: {e}")

# -----------------------
# 5) New Page: Survival Analysis with Kaplan-Meier
# -----------------------
elif page == "📉 Survival Analysis":
    st.markdown("### 📉 Time-to-Event Analysis")
    st.write("Kaplan-Meier curves for waitlist survival, stratified by key factors. Log-rank tests for significance.")
    
    try:
        df = load_data()
        # Ensure required columns
        if 'futime' not in df.columns or 'event' not in df.columns:
            st.error("Dataset missing futime or event columns.")
            st.stop()
        
        # Prepare for KM: event indicator (death=1, others=0 for survival)
        df['surv_event'] = (df['event'] == 'death').astype(int)
        
        # Filters
        stratify_by = st.selectbox("Stratify KM by", ["None", "sex", "abo", "age_group", "center_region"])
        
        # Kaplan-Meier
        kmf = KaplanMeierFitter()
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        if stratify_by == "None":
            kmf.fit(df['futime'], event_observed=df['surv_event'])
            kmf.plot_survival_function()
            plt.title("Overall Waitlist Survival")
        else:
            groups = df[stratify_by].unique()
            colors = px.colors.qualitative.Set1
            for i, group in enumerate(groups):
                mask = df[stratify_by] == group
                kmf.fit(df.loc[mask, 'futime'], event_observed=df.loc[mask, 'surv_event'], label=group)
                kmf.plot_survival_function(color=colors[i % len(colors)])
                plt.title(f"Survival by {stratify_by.capitalize()}")
                plt.ylabel("Survival Probability")
                plt.xlabel("Time (Days)")
        
        st.pyplot(plt.gcf())
        
        # Log-rank test if stratified
        if stratify_by != "None":
            results = logrank_test(df[df[stratify_by] == groups[0]]['futime'], 
                                   df[df[stratify_by] == groups[1]]['futime'],
                                   event_observed_A=df[df[stratify_by] == groups[0]]['surv_event'],
                                   event_observed_B=df[df[stratify_by] == groups[1]]['surv_event'])
            st.markdown(f"**Log-Rank Test p-value:** {results.p_value:.4f} ({'Significant' if results.p_value < 0.05 else 'Not Significant'})")
    
    except Exception as e:
        st.error(f"Error in survival analysis: {e}. Install lifelines if needed.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>"
    "© 2025 TransplantCare Demo | Built with ❤️ using Streamlit & Scikit-learn | Educational Purposes Only"
    "</div>",
    unsafe_allow_html=True,
)
