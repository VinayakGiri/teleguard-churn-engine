"""
dashboard/app.py
----------------
TeleGuard — AI Churn Intelligence Engine
Streamlit dashboard for /Users/vinayakgiri/telecom_churn/teleguard-churn-engine

Compatible with artefacts saved by src/model.py:
  - models/churn_model.pkl   (joblib)
  - models/scaler.pkl        (joblib)
  - models/feature_names.pkl (joblib list)
  - models/top_features.json (JSON dict)
  - models/model_results.json (JSON list, keys: model / accuracy / auc)

Run from project root:
    streamlit run dashboard/app.py
"""

# ---------------------------------------------------------------------------
# Path Setup — ensures src.gemini_insights resolves from any working directory
# ---------------------------------------------------------------------------

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Standard Imports
# ---------------------------------------------------------------------------

import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# joblib — matches how src/model.py saves artefacts
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

import pickle  # fallback

# Local Gemini AI module
try:
    from src.gemini_insights import get_retention_strategy, get_customer_risk_explanation
    GEMINI_AVAILABLE = True
except Exception as e:
    GEMINI_AVAILABLE = False
    GEMINI_IMPORT_ERROR = str(e)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TeleGuard — Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0e1117; }

    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1b2a 100%);
        border: 1px solid #00c9a7;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .main-header h1 { color: #00c9a7; font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p  { color: #8899aa; margin: 4px 0 0; font-size: 0.9rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #151b27);
        border: 1px solid #2a3348;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .label { color: #8899aa; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #f0f4f8; font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
    .metric-card .delta { font-size: 0.75rem; margin-top: 2px; }

    .section-title {
        color: #00c9a7;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border-left: 3px solid #00c9a7;
        padding-left: 10px;
        margin: 24px 0 12px;
    }

    div[data-baseweb="tab-list"] { border-bottom: 2px solid #2a3348; }
    button[data-baseweb="tab"]   { color: #8899aa; font-weight: 500; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #00c9a7; border-bottom: 2px solid #00c9a7; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths — relative to project root
# ---------------------------------------------------------------------------

MODELS_DIR         = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH         = os.path.join(MODELS_DIR, "churn_model.pkl")
SCALER_PATH        = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")   # joblib pickle list
TOP_FEATURES_PATH  = os.path.join(MODELS_DIR, "top_features.json")
MODEL_RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.json")

# ---------------------------------------------------------------------------
# Cached artefact loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading TeleGuard Intelligence Engine…")
def load_artefacts():
    """Loads all model artefacts once and caches them in memory."""
    a = {"model": None, "scaler": None, "feature_names": None,
         "top_features": None, "model_results": None, "errors": []}

    # joblib .pkl files
    for key, path in [("model", MODEL_PATH), ("scaler", SCALER_PATH),
                      ("feature_names", FEATURE_NAMES_PATH)]:
        if os.path.exists(path):
            try:
                a[key] = joblib.load(path) if JOBLIB_AVAILABLE else pickle.load(open(path, "rb"))
            except Exception as e:
                a["errors"].append(f"Could not load {os.path.basename(path)}: {e}")
        else:
            a["errors"].append(f"Missing: {path}")

    # JSON files
    for key, path in [("top_features", TOP_FEATURES_PATH),
                      ("model_results", MODEL_RESULTS_PATH)]:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    a[key] = json.load(f)
            except Exception as e:
                a["errors"].append(f"Could not load {os.path.basename(path)}: {e}")
        else:
            a["errors"].append(f"Missing: {path}")

    return a

# ---------------------------------------------------------------------------
# Helper: preprocess an uploaded DataFrame to match training feature space
# ---------------------------------------------------------------------------

def preprocess_df(df: pd.DataFrame, feature_names: list, scaler) -> np.ndarray:
    """
    Mirrors exactly what src/preprocess.py does so uploaded CSVs score correctly.
    Steps: drop ID/Churn → TotalCharges numeric → binary encode → get_dummies → align → scale.
    """
    df = df.copy()

    # Drop columns that must not be fed to the model
    for col in ["customerID", "Churn"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Numeric coercion for TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Binary columns — match preprocess.py exactly
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    # One-hot encode categoricals (drop_first=True matches training)
    cat_cols = ["gender", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaymentMethod"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to training feature set
    df = df.reindex(columns=feature_names, fill_value=0).fillna(0)

    # Scale and sanitise
    scaled = scaler.transform(df)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def risk_label(p: float) -> str:
    return "High" if p >= 0.7 else ("Medium" if p >= 0.4 else "Low")

def risk_color(p: float) -> str:
    return "#ff4d6d" if p >= 0.7 else ("#ff9f40" if p >= 0.4 else "#00c9a7")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>📡 TeleGuard — AI Churn Intelligence Engine</h1>
    <p>Built by Vinayak Giri &nbsp;·&nbsp; Powered by Scikit-learn + Gemini 1.5 Flash</p>
</div>
""", unsafe_allow_html=True)

# Load artefacts
art = load_artefacts()
model         = art["model"]
scaler        = art["scaler"]
feature_names = art["feature_names"]
top_features  = art["top_features"]
model_results = art["model_results"]

if art["errors"]:
    with st.expander("⚠️ Some model files are missing", expanded=True):
        for err in art["errors"]:
            st.warning(err)
        st.code("cd /Users/vinayakgiri/telecom_churn/teleguard-churn-engine\n"
                "python3 src/preprocess.py\npython3 src/model.py")

model_ready = all(x is not None for x in [model, scaler, feature_names])

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_batch, tab_single, tab_perf = st.tabs([
    "📂  Batch Prediction", "👤  Single Customer", "📊  Model Performance"
])

# ===========================================================================
# TAB 1 — BATCH PREDICTION
# ===========================================================================

with tab_batch:
    st.markdown('<p class="section-title">Upload Customer Dataset</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload telco_raw.csv (or any CSV with the standard Telco Churn columns)",
        type=["csv"],
        help="Upload data/telco_raw.csv from your project folder."
    )

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.markdown(f"**{len(raw_df):,} records loaded.** Preview:")
        st.dataframe(raw_df.head(5), use_container_width=True)

        if not model_ready:
            st.error("❌ Model not loaded. Run `python3 src/preprocess.py` then `python3 src/model.py` first.")
        else:
            with st.spinner("Scoring all customers…"):
                try:
                    X = preprocess_df(raw_df, feature_names, scaler)
                    probs = model.predict_proba(X)[:, 1]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            # ---- Metrics ----
            st.markdown('<p class="section-title">Key Metrics</p>', unsafe_allow_html=True)
            total       = len(probs)
            high_risk   = int((probs >= 0.7).sum())
            avg_score   = probs.mean() * 100
            avg_monthly = float(raw_df["MonthlyCharges"].mean()) if "MonthlyCharges" in raw_df.columns else 65.0
            rev_risk    = high_risk * avg_monthly * 83  # USD → INR

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="label">Total Customers</div>'
                            f'<div class="value">{total:,}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="label">High Risk ≥70%</div>'
                            f'<div class="value" style="color:#ff4d6d">{high_risk:,}</div>'
                            f'<div class="delta" style="color:#ff4d6d">{high_risk/total*100:.1f}% of base</div>'
                            f'</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="label">Avg Churn Score</div>'
                            f'<div class="value">{avg_score:.1f}%</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="label">Revenue at Risk</div>'
                            f'<div class="value" style="color:#ff4d6d">₹{rev_risk:,.0f}</div>'
                            f'<div class="delta" style="color:#8899aa">monthly exposure</div>'
                            f'</div>', unsafe_allow_html=True)

            # ---- Histogram ----
            st.markdown('<p class="section-title">Churn Probability Distribution</p>', unsafe_allow_html=True)
            hist_df = pd.DataFrame({"Churn Probability": probs,
                                    "Risk Level": [risk_label(p) for p in probs]})
            fig = px.histogram(hist_df, x="Churn Probability", color="Risk Level", nbins=40,
                               color_discrete_map={"Low": "#00c9a7", "Medium": "#ff9f40", "High": "#ff4d6d"},
                               category_orders={"Risk Level": ["Low", "Medium", "High"]})
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="#c9d6df", xaxis=dict(gridcolor="#2a3348"),
                              yaxis=dict(gridcolor="#2a3348"))
            st.plotly_chart(fig, use_container_width=True)

            # ---- Top 20 at-risk ----
            st.markdown('<p class="section-title">Top 20 Highest-Risk Customers</p>', unsafe_allow_html=True)
            res = raw_df.copy()
            res["Churn Probability"] = probs
            res["Risk Level"] = [risk_label(p) for p in probs]
            display_cols = [c for c in ["customerID", "tenure", "MonthlyCharges", "Contract",
                                         "InternetService", "Churn Probability", "Risk Level"] if c in res.columns]
            top20 = res.sort_values("Churn Probability", ascending=False).head(20)[display_cols].reset_index(drop=True)
            top20["Churn Probability"] = top20["Churn Probability"].map("{:.1%}".format)
            st.dataframe(top20, use_container_width=True)

            # ---- AI Strategy ----
            st.markdown('<p class="section-title">AI-Powered Retention Strategy</p>', unsafe_allow_html=True)
            if not GEMINI_AVAILABLE:
                st.warning(f"Gemini not available: {GEMINI_IMPORT_ERROR}")
            else:
                if st.button("🤖 Generate AI Retention Strategy", use_container_width=True):
                    if top_features is None:
                        st.error("top_features.json not found.")
                    else:
                        with st.spinner("Consulting Gemini 1.5 Flash…"):
                            strategy = get_retention_strategy(
                                top_features=top_features,
                                churn_rate=float(probs.mean()),
                                high_risk_count=high_risk,
                                avg_revenue=avg_monthly,
                            )
                        st.info(f"**Gemini Retention Strategy**\n\n{strategy}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;border:2px dashed #2a3348;border-radius:12px;color:#8899aa;">
            <h3 style="color:#8899aa;">📂 Upload telco_raw.csv to begin</h3>
            <p>Find it at <code>data/telco_raw.csv</code> inside your project folder.</p>
        </div>""", unsafe_allow_html=True)

# ===========================================================================
# TAB 2 — SINGLE CUSTOMER
# ===========================================================================

with tab_single:
    st.markdown('<p class="section-title">Enter Customer Profile</p>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    with col_b:
        contract    = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with col_c:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        payment      = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Score This Customer", use_container_width=True, type="primary"):
        if not model_ready:
            st.error("❌ Model not loaded.")
        else:
            # Build a raw-format row matching telco_raw.csv columns
            row = pd.DataFrame([{
                "tenure": tenure, "MonthlyCharges": monthly_charges,
                "TotalCharges": tenure * monthly_charges,
                "gender": "Male", "SeniorCitizen": 0,
                "Partner": "No", "Dependents": "No",
                "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": internet, "OnlineSecurity": "No",
                "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": tech_support, "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": contract,
                "PaperlessBilling": "Yes", "PaymentMethod": payment,
            }])
            with st.spinner("Calculating risk…"):
                try:
                    X = preprocess_df(row, feature_names, scaler)
                    prob = float(model.predict_proba(X)[0, 1])
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                    st.stop()

            # Risk display
            st.markdown("---")
            label = risk_label(prob)
            color = risk_color(prob)
            emoji = "🔴" if label == "High" else ("🟠" if label == "Medium" else "🟢")

            rc, bc = st.columns([1, 2])
            with rc:
                st.markdown(f"""
                <div class="metric-card" style="border-color:{color};">
                    <div class="label">Churn Probability</div>
                    <div class="value" style="color:{color};">{prob*100:.1f}%</div>
                    <div class="delta" style="color:{color};">{emoji} {label} Risk</div>
                </div>""", unsafe_allow_html=True)
            with bc:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Risk Score: {prob*100:.1f}%**")
                st.progress(prob)
                st.caption("🟢 Low < 40%  ·  🟠 Medium 40–70%  ·  🔴 High > 70%")

            if GEMINI_AVAILABLE:
                with st.spinner("Consulting Gemini…"):
                    exp = get_customer_risk_explanation(
                        customer_data={"Tenure": f"{tenure} months",
                                       "Monthly Charges": f"${monthly_charges}",
                                       "Contract": contract, "Internet": internet,
                                       "Tech Support": tech_support, "Payment": payment},
                        churn_probability=prob)
                st.info(f"**Gemini Analysis**\n\n{exp}")
            else:
                st.warning(f"Gemini not available: {GEMINI_IMPORT_ERROR}")

# ===========================================================================
# TAB 3 — MODEL PERFORMANCE
# ===========================================================================

with tab_perf:
    st.markdown('<p class="section-title">Model Comparison</p>', unsafe_allow_html=True)

    if model_results is None:
        st.error("model_results.json not found. Run `python3 src/model.py`.")
    else:
        perf_df = pd.DataFrame(model_results)
        # Normalise column names (model.py uses lowercase keys: model, accuracy, auc)
        perf_df.columns = [c.title() for c in perf_df.columns]
        if "Accuracy" in perf_df.columns:
            # Already stored as 0-1 decimal; convert to percentage
            if perf_df["Accuracy"].max() <= 1.0:
                perf_df["Accuracy (%)"] = (perf_df["Accuracy"] * 100).round(2)
                perf_df.drop(columns=["Accuracy"], inplace=True)
            else:
                perf_df.rename(columns={"Accuracy": "Accuracy (%)"}, inplace=True)
        if "Auc" in perf_df.columns:
            perf_df.rename(columns={"Auc": "AUC Score"}, inplace=True)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-title">Top Churn Drivers — Feature Importance</p>', unsafe_allow_html=True)

    if top_features is None:
        st.error("top_features.json not found. Run `python3 src/model.py`.")
    else:
        feat_df = (pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance"])
                   .sort_values("Importance", ascending=True))
        feat_df["Feature"] = feat_df["Feature"].str.replace("_", " ", regex=False)

        fig2 = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
            marker=dict(color=feat_df["Importance"], colorscale="Teal", showscale=True),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig2.update_layout(
            title=dict(text="Top Churn Drivers — Feature Importance",
                       font=dict(color="#c9d6df", size=14)),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#c9d6df",
            xaxis=dict(title="Importance Score", gridcolor="#2a3348"),
            yaxis=dict(gridcolor="#2a3348", automargin=True),
            height=400, margin=dict(l=0, r=20, t=50, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    if GEMINI_AVAILABLE:
        st.success("✅ Gemini 1.5 Flash connected — AI insights enabled.")
    else:
        st.warning(f"⚠️ Gemini disabled. Error: {GEMINI_IMPORT_ERROR}")
