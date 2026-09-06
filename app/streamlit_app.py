"""
streamlit_app.py
Production-grade interactive web application for Customer Segmentation & Marketing Response Prediction.
Combines K-Means clustering, 2D PCA, supervised classification, and actionable recommendation engine.
"""

import sys
import os

# Ensure the root project directory is in sys.path regardless of current working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def get_path(rel_path: str) -> str:
    """Returns absolute path relative to project root."""
    return os.path.join(PROJECT_ROOT, rel_path)

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Internal modules
from src.preprocessing import load_raw_data, clean_data, validate_schema
from src.feature_engineering import engineer_features
from src.clustering import (
    evaluate_optimal_k, fit_clustering_pipeline, predict_cluster, CLUSTERING_FEATURES
)
from src.prediction import (
    train_and_evaluate_models, predict_customer_response
)
from src.recommendations import generate_recommendation

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Intelligence Hub",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Theme State Management
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# 3. Dynamic Design System CSS
THEME_VARS = {
    "bg": "#09090b" if IS_DARK else "#ffffff",
    "bg_subtle": "#121217" if IS_DARK else "#f8fafc",
    "card": "#18181b" if IS_DARK else "#ffffff",
    "card_hover": "#222226" if IS_DARK else "#f1f5f9",
    "border": "#27272a" if IS_DARK else "#e2e8f0",
    "border_subtle": "#1f1f23" if IS_DARK else "#f1f5f9",
    "text": "#f8fafc" if IS_DARK else "#0f172a",
    "text_muted": "#94a3b8" if IS_DARK else "#475569",
    "text_dim": "#64748b" if IS_DARK else "#64748b",
    "accent": "#3b82f6" if IS_DARK else "#2563eb",
    "accent_bg": "rgba(59, 130, 246, 0.15)" if IS_DARK else "rgba(37, 99, 235, 0.08)",
    "green": "#10b981" if IS_DARK else "#059669",
    "green_bg": "rgba(16, 185, 129, 0.15)" if IS_DARK else "rgba(5, 150, 105, 0.08)",
    "red": "#f43f5e" if IS_DARK else "#e11d48",
    "red_bg": "rgba(244, 63, 94, 0.15)" if IS_DARK else "rgba(225, 29, 72, 0.08)",
    "amber": "#f59e0b" if IS_DARK else "#d97706",
    "amber_bg": "rgba(245, 158, 11, 0.15)" if IS_DARK else "rgba(217, 119, 6, 0.08)",
    "shadow": "0 4px 12px rgba(0, 0, 0, 0.4)" if IS_DARK else "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Completely remove Streamlit Chrome: Header, Deploy Button, Toolbar, Footer */
header[data-testid="stHeader"],
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton,
[data-testid="stAppDeployButton"],
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: {THEME_VARS["bg"]} !important;
    color: {THEME_VARS["text"]} !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.block-container {{
    padding: 1.25rem 2rem 2.5rem !important;
    max-width: 1440px !important;
}}

/* Global text readability */
p, span, label, div, h1, h2, h3, h4, h5, h6 {{
    color: {THEME_VARS["text"]};
}}

/* Form labels */
label[data-testid="stWidgetLabel"] p, label p, div[data-testid="stWidgetLabel"] p {{
    color: {THEME_VARS["text"]} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}}

/* Input controls and selectboxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div,
input, select, textarea {{
    background-color: {THEME_VARS["card"]} !important;
    color: {THEME_VARS["text"]} !important;
    border-color: {THEME_VARS["border"]} !important;
}}

div[data-baseweb="select"] span {{
    color: {THEME_VARS["text"]} !important;
}}

/* Dropdown popover listbox options */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {{
    background-color: {THEME_VARS["card"]} !important;
    color: {THEME_VARS["text"]} !important;
    border: 1px solid {THEME_VARS["border"]} !important;
}}

li[role="option"] {{
    background-color: {THEME_VARS["card"]} !important;
    color: {THEME_VARS["text"]} !important;
}}

li[role="option"]:hover,
li[aria-selected="true"] {{
    background-color: {THEME_VARS["card_hover"]} !important;
    color: {THEME_VARS["accent"]} !important;
}}

/* Number input step buttons (+ / -) */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {{
    background-color: {THEME_VARS["bg_subtle"]} !important;
    color: {THEME_VARS["text"]} !important;
    border-color: {THEME_VARS["border"]} !important;
}}

/* Slider styling */
div[data-testid="stSlider"] div {{
    color: {THEME_VARS["text"]} !important;
}}

/* Secondary & Primary buttons */
button[data-testid="baseButton-secondary"] {{
    background-color: {THEME_VARS["card"]} !important;
    color: {THEME_VARS["text"]} !important;
    border: 1px solid {THEME_VARS["border"]} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
button[data-testid="baseButton-secondary"]:hover {{
    border-color: {THEME_VARS["accent"]} !important;
    color: {THEME_VARS["accent"]} !important;
}}

button[data-testid="baseButton-primary"] {{
    background-color: {THEME_VARS["accent"]} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}

/* Brand Header */
.brand-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem 1.5rem;
    background: {THEME_VARS["card"]};
    border: 1px solid {THEME_VARS["border"]};
    border-radius: 12px;
    margin-bottom: 1.25rem;
    box-shadow: {THEME_VARS["shadow"]};
}}
.brand-title {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {THEME_VARS["text"]};
    letter-spacing: -0.02em;
}}
.brand-tagline {{
    font-size: 0.8rem;
    color: {THEME_VARS["text_muted"]};
    margin-top: 3px;
}}

/* KPI Metric Cards */
.metric-card {{
    background: {THEME_VARS["card"]};
    border: 1px solid {THEME_VARS["border"]};
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    box-shadow: {THEME_VARS["shadow"]};
    transition: transform 0.2s ease, border-color 0.2s ease;
}}
.metric-card:hover {{
    border-color: {THEME_VARS["accent"]};
    transform: translateY(-2px);
}}
.metric-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {THEME_VARS["text_muted"]};
    font-weight: 600;
}}
.metric-value {{
    font-size: 1.7rem;
    font-weight: 700;
    color: {THEME_VARS["text"]};
    letter-spacing: -0.03em;
    margin: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
}}
.metric-sub {{
    font-size: 0.75rem;
    color: {THEME_VARS["text_dim"]};
    display: flex;
    align-items: center;
    gap: 4px;
}}

/* Pill Tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: {THEME_VARS["text_muted"]} !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.1rem !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {THEME_VARS["text"]} !important;
    background: {THEME_VARS["card"]} !important;
    border-color: {THEME_VARS["border"]} !important;
    box-shadow: {THEME_VARS["shadow"]} !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 6px !important;
    background: {THEME_VARS["bg_subtle"]} !important;
    border: 1px solid {THEME_VARS["border"]} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin-bottom: 1.25rem !important;
}}

/* Chart Wrappers */
.chart-wrap {{
    background: {THEME_VARS["card"]};
    border: 1px solid {THEME_VARS["border"]};
    border-radius: 12px;
    padding: 1.1rem 1.25rem 0.6rem;
    box-shadow: {THEME_VARS["shadow"]};
    margin-bottom: 1rem;
}}
.chart-title {{
    font-size: 0.92rem;
    font-weight: 600;
    color: {THEME_VARS["text"]};
    letter-spacing: -0.01em;
}}
.chart-subtitle {{
    font-size: 0.74rem;
    color: {THEME_VARS["text_muted"]};
    margin-bottom: 0.75rem;
}}

/* Badges */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}}
.badge-green {{ color: {THEME_VARS["green"]}; background: {THEME_VARS["green_bg"]}; }}
.badge-red {{ color: {THEME_VARS["red"]}; background: {THEME_VARS["red_bg"]}; }}
.badge-amber {{ color: {THEME_VARS["amber"]}; background: {THEME_VARS["amber_bg"]}; }}
.badge-blue {{ color: {THEME_VARS["accent"]}; background: {THEME_VARS["accent_bg"]}; }}

/* Strategy Box */
.strategy-box {{
    background: {THEME_VARS["card"]};
    border: 1px solid {THEME_VARS["border"]};
    border-left: 4px solid {THEME_VARS["accent"]};
    border-radius: 10px;
    padding: 1.25rem;
    margin-top: 1rem;
    box-shadow: {THEME_VARS["shadow"]};
}}

/* Custom Tables */
.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.82rem;
}}
.data-table th {{
    text-align: left;
    padding: 0.7rem 0.85rem;
    color: {THEME_VARS["text_muted"]};
    font-weight: 600;
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid {THEME_VARS["border"]};
    background: {THEME_VARS["bg_subtle"]};
}}
.data-table td {{
    padding: 0.75rem 0.85rem;
    color: {THEME_VARS["text"]};
    border-bottom: 1px solid {THEME_VARS["border_subtle"]};
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 4. Plotly Chart Theme Layout (Dynamically adapted to current theme)
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, -apple-system, sans-serif", color=THEME_VARS["text_muted"], size=11),
    margin=dict(l=20, r=20, t=25, b=20),
    xaxis=dict(
        gridcolor=THEME_VARS["border_subtle"],
        zerolinecolor=THEME_VARS["border"],
        tickfont=dict(size=10, color=THEME_VARS["text_muted"]),
    ),
    yaxis=dict(
        gridcolor=THEME_VARS["border_subtle"],
        zerolinecolor=THEME_VARS["border"],
        tickfont=dict(size=10, color=THEME_VARS["text_muted"]),
    ),
    colorway=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#f43f5e"]
)

# 5. Model & Data Loading Functions
@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Loads processed dataset if available, otherwise cleans and engineers from raw data."""
    processed_path = get_path("models/processed_customers.csv")
    raw_path = get_path("data/customers.csv")

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        return df

    if os.path.exists(raw_path):
        raw = load_raw_data(raw_path)
        cleaned, _ = clean_data(raw)
        engineered = engineer_features(cleaned)
        km_path = get_path("models/clustering_model.pkl")
        sc_path = get_path("models/scaler.pkl")
        pca_path = get_path("models/pca_model.pkl")
        prof_path = get_path("models/cluster_profiles.json")
        if os.path.exists(km_path) and os.path.exists(sc_path):
            km = joblib.load(km_path)
            sc = joblib.load(sc_path)
            pca = joblib.load(pca_path)
            X_scaled = sc.transform(engineered[CLUSTERING_FEATURES])
            engineered["Cluster"] = km.predict(X_scaled)
            pca_coords = pca.transform(X_scaled)
            engineered["PCA1"] = pca_coords[:, 0]
            engineered["PCA2"] = pca_coords[:, 1]
            if os.path.exists(prof_path):
                with open(prof_path) as f:
                    profs = json.load(f)
                engineered["Cluster_Name"] = engineered["Cluster"].astype(str).map(
                    lambda c: profs.get(c, {}).get("name", f"Cluster {c}")
                )
        return engineered

    return pd.DataFrame()

@st.cache_resource
def load_models():
    """Loads all serialized models, preprocessors, and evaluation metrics."""
    artifacts = {}
    paths = {
        "scaler": get_path("models/scaler.pkl"),
        "kmeans": get_path("models/clustering_model.pkl"),
        "pca": get_path("models/pca_model.pkl"),
        "response_model": get_path("models/response_model.pkl"),
        "preprocessor": get_path("models/preprocessor.pkl"),
    }
    for key, path in paths.items():
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            artifacts[key] = None

    prof_path = get_path("models/cluster_profiles.json")
    if os.path.exists(prof_path):
        with open(prof_path) as f:
            artifacts["profiles"] = {int(k): v for k, v in json.load(f).items()}
    else:
        artifacts["profiles"] = None

    comp_path = get_path("models/model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            artifacts["comparison"] = json.load(f)
    else:
        artifacts["comparison"] = None

    k_path = get_path("models/k_evaluation.json")
    if os.path.exists(k_path):
        with open(k_path) as f:
            artifacts["k_eval"] = json.load(f)
    else:
        artifacts["k_eval"] = None

    return artifacts


# 6. Session State for Live Data & Models
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Default Benchmark Dataset (customers.csv)"

if "active_df" not in st.session_state:
    st.session_state.active_df = get_dataset()

if "active_models" not in st.session_state:
    st.session_state.active_models = load_models()

df = st.session_state.active_df
models = st.session_state.active_models

# Fallback check if models exist
if df.empty or models.get("kmeans") is None:
    st.warning("Models or datasets not yet trained. Click below to run the initial training pipeline.")
    if st.button("Train Machine Learning Models Now", type="primary"):
        with st.spinner("Executing end-to-end training pipeline..."):
            import subprocess
            res = subprocess.run([sys.executable, get_path("train.py")], cwd=PROJECT_ROOT, capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Models trained successfully!")
                st.session_state.active_df = get_dataset()
                st.session_state.active_models = load_models()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(f"Training failed: {res.stderr}")
    st.stop()


# 7. Brand Header with Theme Toggle & Dataset Status
head_col1, head_col2 = st.columns([7, 3])
with head_col1:
    st.markdown(f"""
    <div class="brand-container">
        <div>
            <div class="brand-title">
                Customer Segmentation & Marketing Propensity Engine
            </div>
            <div class="brand-tagline">
                Unsupervised K-Means Behavioral Clustering • Supervised Campaign Response Prediction • Personalized Strategy
            </div>
            <div style="margin-top: 6px;">
                <span class="badge badge-blue">Active Dataset: {st.session_state.dataset_name}</span>
                <span class="badge badge-green" style="margin-left: 6px;">Currency: INR (₹)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with head_col2:
    theme_btn_text = "Light Mode" if IS_DARK else "Dark Mode"
    t_c1, t_c2 = st.columns([1, 1])
    with t_c1:
        if st.button(theme_btn_text, use_container_width=True):
            toggle_theme()
            st.rerun()
    with t_c2:
        if "Uploaded:" in st.session_state.dataset_name:
            if st.button("Reset Dataset", use_container_width=True, help="Switch back to original benchmark dataset"):
                st.session_state.active_df = get_dataset()
                st.session_state.active_models = load_models()
                st.session_state.dataset_name = "Default Benchmark Dataset (customers.csv)"
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()


# 8. Navigation Tabs
tabs = st.tabs([
    "Dashboard",
    "Customer Segmentation",
    "Segment Analysis",
    "Campaign Prediction",
    "Customer Profile",
    "Data Upload",
    "Methodology & Audit"
])


# ==============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# ==============================================================================
with tabs[0]:
    total_cust = len(df)
    n_segments = df["Cluster"].nunique() if "Cluster" in df.columns else 4
    avg_spending = df["Total_Spending"].mean()
    overall_response = (df["Response"].mean() * 100) if "Response" in df.columns else 14.9

    # Identify High-Value Cluster count
    high_value_mask = df["Cluster_Name"].str.contains("High-Value|Elite", case=False, na=False)
    high_val_count = high_value_mask.sum()
    high_val_pct = (high_val_count / total_cust) * 100 if total_cust > 0 else 0

    # 5 KPI Metric Cards (using INR ₹)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total_cust:,}</div>
            <div class="metric-sub">Validated & De-duplicated</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Segments</div>
            <div class="metric-value">{n_segments}</div>
            <div class="metric-sub">Optimal K via Silhouette</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Customer Spending</div>
            <div class="metric-value">₹{avg_spending:,.0f}</div>
            <div class="metric-sub">Across 6 Product Categories</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Campaign Response Rate</div>
            <div class="metric-value">{overall_response:.1f}%</div>
            <div class="metric-sub">Historical Conversion Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High-Value Segment</div>
            <div class="metric-value">{high_val_count:,}</div>
            <div class="metric-sub"><span class="badge badge-green">{high_val_pct:.1f}% of Cohort</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Visual Row 1: Segment Distribution & Response Propensity by Segment
    row1_c1, row1_c2 = st.columns([1, 1])

    with row1_c1:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Customer Distribution by Behavioral Segment</div>
            <div class="chart-subtitle">Breakdown of customer population across K-Means identified clusters</div>
        """, unsafe_allow_html=True)
        seg_counts = df["Cluster_Name"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_donut = px.pie(
            seg_counts, values="Count", names="Segment", hole=0.55,
            color_discrete_sequence=PLOT_LAYOUT["colorway"]
        )
        fig_donut.update_layout(PLOT_LAYOUT, showlegend=True, height=330, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_c2:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Marketing Campaign Response Rate by Segment</div>
            <div class="chart-subtitle">Historical campaign conversion percentage per customer cluster</div>
        """, unsafe_allow_html=True)
        resp_by_seg = df.groupby("Cluster_Name")["Response"].mean().reset_index()
        resp_by_seg["Response_Pct"] = resp_by_seg["Response"] * 100
        resp_by_seg = resp_by_seg.sort_values(by="Response_Pct", ascending=True)

        fig_resp = px.bar(
            resp_by_seg, x="Response_Pct", y="Cluster_Name", orientation="h",
            labels={"Response_Pct": "Response Rate (%)", "Cluster_Name": "Segment"},
            color="Response_Pct",
            color_continuous_scale="Blues" if not IS_DARK else "tealgrn"
        )
        fig_resp.update_layout(PLOT_LAYOUT, showlegend=False, height=330, coloraxis_showscale=False)
        fig_resp.add_vline(x=overall_response, line_dash="dash", line_color=THEME_VARS["red"],
                           annotation_text=f"Avg ({overall_response:.1f}%)", annotation_position="top right")
        st.plotly_chart(fig_resp, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Segment Benchmark Summary Table (using ₹)
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Customer Segment Performance Benchmarks</div>
        <div class="chart-subtitle">Key demographic, channel, and spend metrics across all customer clusters</div>
    """, unsafe_allow_html=True)

    benchmark_cols = [
        "Cluster_Name", "Income", "Total_Spending", "Total_Purchases",
        "NumDealsPurchases", "NumWebPurchases", "NumStorePurchases", "Recency", "Response"
    ]
    summary_df = df[benchmark_cols].groupby("Cluster_Name").mean().reset_index()
    summary_df["Response"] = summary_df["Response"] * 100
    summary_df = summary_df.rename(columns={
        "Cluster_Name": "Segment",
        "Income": "Avg Income (₹)",
        "Total_Spending": "Avg Spend (₹)",
        "Total_Purchases": "Avg Purchases",
        "NumDealsPurchases": "Deals Bought",
        "NumWebPurchases": "Web Orders",
        "NumStorePurchases": "Store Orders",
        "Recency": "Recency (Days)",
        "Response": "Response Rate (%)"
    })

    headers_html = "".join(f"<th>{c}</th>" for c in summary_df.columns)
    rows_html = ""
    for _, row in summary_df.iterrows():
        cells = f"<td><b>{row['Segment']}</b></td>"
        cells += f"<td>₹{row['Avg Income (₹)']:,.0f}</td>"
        cells += f"<td><b>₹{row['Avg Spend (₹)']:,.0f}</b></td>"
        cells += f"<td>{row['Avg Purchases']:.1f}</td>"
        cells += f"<td>{row['Deals Bought']:.1f}</td>"
        cells += f"<td>{row['Web Orders']:.1f}</td>"
        cells += f"<td>{row['Store Orders']:.1f}</td>"
        cells += f"<td>{row['Recency (Days)']:.1f} d</td>"
        rate_val = row['Response Rate (%)']
        badge_cls = "badge-green" if rate_val >= overall_response else "badge-amber"
        cells += f"<td><span class='badge {badge_cls}'>{rate_val:.1f}%</span></td>"
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <table class="data-table">
        <thead><tr>{headers_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: CUSTOMER SEGMENTATION (MODEL 1: K-MEANS & PCA)
# ==============================================================================
with tabs[1]:
    st.markdown("### Model 1: K-Means Clustering & 2D PCA Space")
    st.markdown(
        "K-Means groups customers based on demographic, purchasing volume, channel preferences, and recency. "
        "StandardScaler normalizes all feature distributions to eliminate scale variance."
    )

    # 1. Optimal K Evaluation
    k_eval = models.get("k_eval")
    if k_eval:
        k_c1, k_c2 = st.columns([1, 1])
        with k_c1:
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">The Elbow Method (Within-Cluster Sum of Squares)</div>
                <div class="chart-subtitle">Evaluates diminishing returns in inertia reduction as K increases</div>
            """, unsafe_allow_html=True)
            fig_elbow = px.line(
                x=k_eval["k_values"], y=k_eval["inertias"], markers=True,
                labels={"x": "Number of Clusters (K)", "y": "Inertia (WCSS)"},
            )
            fig_elbow.update_traces(line_color=THEME_VARS["accent"], marker=dict(size=8))
            fig_elbow.update_layout(PLOT_LAYOUT, height=280)
            st.plotly_chart(fig_elbow, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with k_c2:
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Silhouette Score vs Number of Clusters (K)</div>
                <div class="chart-subtitle">Measures intra-cluster cohesion versus inter-cluster separation</div>
            """, unsafe_allow_html=True)
            fig_sil = px.line(
                x=k_eval["k_values"], y=k_eval["silhouette_scores"], markers=True,
                labels={"x": "Number of Clusters (K)", "y": "Silhouette Score"},
            )
            best_k = k_eval["recommended_k_silhouette"]
            fig_sil.update_traces(line_color=THEME_VARS["green"], marker=dict(size=8))
            fig_sil.add_vline(x=best_k, line_dash="dash", line_color=THEME_VARS["amber"],
                              annotation_text=f"Max Silhouette (K={best_k})", annotation_position="top left")
            fig_sil.update_layout(PLOT_LAYOUT, height=280)
            st.plotly_chart(fig_sil, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    # 2. Interactive PCA Scatter Plot
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    filter_c1, filter_c2 = st.columns([3, 1])
    with filter_c1:
        selected_clusters = st.multiselect(
            "Filter Segments to Display:",
            options=df["Cluster_Name"].unique().tolist(),
            default=df["Cluster_Name"].unique().tolist()
        )
    with filter_c2:
        pca_sample_size = st.slider("Max Points to Plot (Performance):", 500, len(df), min(2200, len(df)), step=250)

    plot_df = df[df["Cluster_Name"].isin(selected_clusters)].sample(
        min(pca_sample_size, len(df[df["Cluster_Name"].isin(selected_clusters)])), random_state=42
    )

    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Interactive 2D PCA Customer Projection</div>
        <div class="chart-subtitle">Feature space dimensionally reduced to PC1 & PC2 — hover over any customer point to view dossier</div>
    """, unsafe_allow_html=True)

    fig_pca = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color="Cluster_Name",
        hover_data={
            "Id": True,
            "Income": ":₹,.0f",
            "Total_Spending": ":₹,.0f",
            "Total_Purchases": True,
            "Recency": True,
            "PCA1": False,
            "PCA2": False,
            "Cluster_Name": True
        },
        labels={"PCA1": "Principal Component 1 (Purchasing Power & Volume)",
                "PCA2": "Principal Component 2 (Deals & Channel Preference)"},
        color_discrete_sequence=PLOT_LAYOUT["colorway"],
        opacity=0.82
    )
    fig_pca.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(255,255,255,0.4)")))
    fig_pca.update_layout(PLOT_LAYOUT, height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": True})
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Dynamic Cluster Dossiers / Narrative Cards (with ₹)
    st.markdown("#### Dynamic Segment Profiles & Strategic Personas")
    profiles = models.get("profiles", {})
    if profiles:
        prof_cols = st.columns(len(profiles))
        for idx, (c_id, p_data) in enumerate(profiles.items()):
            with prof_cols[idx]:
                st.markdown(f"""
                <div class="metric-card" style="height: 100%;">
                    <div class="badge badge-blue">Cluster {c_id}</div>
                    <div style="font-size: 1.05rem; font-weight: 700; margin: 6px 0 2px; color: {THEME_VARS['text']};">
                        {p_data['name']}
                    </div>
                    <div style="font-size: 0.76rem; color: {THEME_VARS['text_muted']}; line-height: 1.4; margin-bottom: 12px;">
                        {p_data['description']}
                    </div>
                    <hr style="border: none; border-top: 1px solid {THEME_VARS['border_subtle']}; margin: 8px 0;" />
                    <div style="font-size: 0.78rem; display: flex; justify-content: space-between; padding: 2px 0;">
                        <span style="color: {THEME_VARS['text_muted']};">Share:</span>
                        <b>{p_data['pct']}% ({p_data['count']})</b>
                    </div>
                    <div style="font-size: 0.78rem; display: flex; justify-content: space-between; padding: 2px 0;">
                        <span style="color: {THEME_VARS['text_muted']};">Avg Spend:</span>
                        <b style="color: {THEME_VARS['green']};">₹{p_data['spending']:,.0f}</b>
                    </div>
                    <div style="font-size: 0.78rem; display: flex; justify-content: space-between; padding: 2px 0;">
                        <span style="color: {THEME_VARS['text_muted']};">Avg Income:</span>
                        <b>₹{p_data['income']:,.0f}</b>
                    </div>
                    <div style="font-size: 0.78rem; display: flex; justify-content: space-between; padding: 2px 0;">
                        <span style="color: {THEME_VARS['text_muted']};">Deal Orders:</span>
                        <b>{p_data['deal_purchases']:.1f} ({p_data['deal_ratio']*100:.0f}%)</b>
                    </div>
                    <div style="font-size: 0.78rem; display: flex; justify-content: space-between; padding: 2px 0;">
                        <span style="color: {THEME_VARS['text_muted']};">Response Rate:</span>
                        <b style="color: {THEME_VARS['accent']};">{p_data['response_rate']:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: SEGMENT DEEP-DIVE ANALYSIS
# ==============================================================================
with tabs[2]:
    st.markdown("### Cross-Segment Behavioral Analysis")
    st.markdown(
        "Examine how customer cohorts differ across product spending categories, transaction channels, and recency."
    )

    # 1. Product Category Spending Breakdown (₹)
    spend_cats = ["MntWines", "MntMeatProducts", "MntGoldProds", "MntFishProducts", "MntSweetProducts", "MntFruits"]
    available_spend = [c for c in spend_cats if c in df.columns]

    spend_by_seg = df.groupby("Cluster_Name")[available_spend].mean().reset_index()
    melted_spend = pd.melt(
        spend_by_seg, id_vars=["Cluster_Name"], value_vars=available_spend,
        var_name="Category", value_name="Avg_Spend"
    )
    cat_names = {
        "MntWines": "Wines", "MntMeatProducts": "Meat", "MntGoldProds": "Gold Luxury",
        "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntFruits": "Fruits"
    }
    melted_spend["Category"] = melted_spend["Category"].map(cat_names)

    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Average Spending by Product Category Across Segments (₹)</div>
        <div class="chart-subtitle">Reveals product affinity differences across cohorts</div>
    """, unsafe_allow_html=True)
    fig_spend_cat = px.bar(
        melted_spend, x="Cluster_Name", y="Avg_Spend", color="Category",
        barmode="stack", labels={"Cluster_Name": "Segment", "Avg_Spend": "Average Spend (₹)"},
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_spend_cat.update_layout(PLOT_LAYOUT, height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_spend_cat, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Purchase Channels Breakdown & Income Distribution
    deep_c1, deep_c2 = st.columns([1, 1])

    with deep_c1:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Purchasing Channel Preferences by Segment</div>
            <div class="chart-subtitle">Average orders completed across Web, Store, Catalog, and Deals</div>
        """, unsafe_allow_html=True)
        channel_cols = ["NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases", "NumDealsPurchases"]
        avail_channels = [c for c in channel_cols if c in df.columns]
        chan_df = df.groupby("Cluster_Name")[avail_channels].mean().reset_index()
        chan_melted = pd.melt(chan_df, id_vars=["Cluster_Name"], value_vars=avail_channels,
                              var_name="Channel", value_name="Avg_Orders")
        chan_map = {
            "NumWebPurchases": "Web Orders", "NumStorePurchases": "Store Orders",
            "NumCatalogPurchases": "Catalog Orders", "NumDealsPurchases": "Deal Orders"
        }
        chan_melted["Channel"] = chan_melted["Channel"].map(chan_map)

        fig_chan = px.bar(
            chan_melted, x="Cluster_Name", y="Avg_Orders", color="Channel", barmode="group",
            labels={"Cluster_Name": "Segment", "Avg_Orders": "Average Number of Orders"},
            color_discrete_sequence=PLOT_LAYOUT["colorway"]
        )
        fig_chan.update_layout(PLOT_LAYOUT, height=330, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_chan, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with deep_c2:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Annual Income Distribution by Segment (₹)</div>
            <div class="chart-subtitle">Box plot depicting median income, IQR, and earnings dispersion</div>
        """, unsafe_allow_html=True)
        fig_inc = px.box(
            df, x="Cluster_Name", y="Income", color="Cluster_Name",
            labels={"Cluster_Name": "Segment", "Income": "Annual Income (₹)"},
            color_discrete_sequence=PLOT_LAYOUT["colorway"]
        )
        fig_inc.update_layout(PLOT_LAYOUT, showlegend=False, height=330)
        st.plotly_chart(fig_inc, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Recency & Children Analysis
    deep_c3, deep_c4 = st.columns([1, 1])
    with deep_c3:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Customer Recency (Days Since Last Order)</div>
            <div class="chart-subtitle">Lower values indicate recently active customers with higher engagement</div>
        """, unsafe_allow_html=True)
        fig_rec = px.violin(
            df, x="Cluster_Name", y="Recency", color="Cluster_Name", box=True,
            labels={"Cluster_Name": "Segment", "Recency": "Days Since Last Purchase"},
            color_discrete_sequence=PLOT_LAYOUT["colorway"]
        )
        fig_rec.update_layout(PLOT_LAYOUT, showlegend=False, height=300)
        st.plotly_chart(fig_rec, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with deep_c4:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Average Household Size & Children by Segment</div>
            <div class="chart-subtitle">Helps tailor family bundles versus individual luxury promotions</div>
        """, unsafe_allow_html=True)
        kids_df = df.groupby("Cluster_Name")[["Kidhome", "Teenhome", "Total_Children"]].mean().reset_index()
        fig_kids = px.bar(
            kids_df, x="Cluster_Name", y=["Kidhome", "Teenhome"],
            labels={"Cluster_Name": "Segment", "value": "Average Count", "variable": "Child Category"},
            color_discrete_sequence=["#38bdf8", "#818cf8"], barmode="stack"
        )
        fig_kids.update_layout(PLOT_LAYOUT, height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_kids, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 4: CAMPAIGN RESPONSE PREDICTION (MODEL 2 & RECOMMENDATION ENGINE)
# ==============================================================================
with tabs[3]:
    st.markdown("### What-If Campaign Simulator: Predict Segment & Response")
    st.markdown(
        "Enter customer demographic and shopping traits to dynamically predict **Segment** (via K-Means), "
        "**Campaign Response Probability** (via Supervised Classifier), and receive an actionable personalized marketing plan."
    )

    with st.form("simulation_form"):
        col_form1, col_form2, col_form3 = st.columns(3)

        with col_form1:
            st.markdown("##### Demographics")
            f_age = st.slider("Customer Age:", 18, 85, 42)
            f_edu = st.selectbox("Education Level:", ["Graduation", "Master", "PhD", "Basic", "2n Cycle"], index=0)
            f_marital = st.selectbox("Marital Status:", ["Married", "Together", "Single", "Divorced", "Widow"], index=0)
            f_income = st.number_input("Annual Income (₹):", min_value=5000, max_value=2000000, value=58000, step=2500)
            f_kids = st.slider("Young Children (Kidhome):", 0, 3, 0)
            f_teens = st.slider("Teens at Home (Teenhome):", 0, 3, 1)

        with col_form2:
            st.markdown("##### Product Spending (₹)")
            f_wines = st.number_input("Spent on Wines (₹):", 0, 50000, 350, step=25)
            f_meat = st.number_input("Spent on Meats (₹):", 0, 50000, 180, step=25)
            f_fish = st.number_input("Spent on Fish (₹):", 0, 20000, 40, step=10)
            f_sweets = st.number_input("Spent on Sweets (₹):", 0, 20000, 30, step=10)
            f_gold = st.number_input("Spent on Gold (₹):", 0, 20000, 50, step=10)
            f_fruits = st.number_input("Spent on Fruits (₹):", 0, 20000, 25, step=5)

        with col_form3:
            st.markdown("##### Channels & Engagement")
            f_recency = st.slider("Recency (Days since last order):", 0, 100, 28)
            f_web_p = st.number_input("Web Purchases:", 0, 50, 6)
            f_store_p = st.number_input("Store Purchases:", 0, 50, 7)
            f_cat_p = st.number_input("Catalog Purchases:", 0, 50, 3)
            f_deal_p = st.number_input("Deal Purchases:", 0, 50, 2)
            f_web_visits = st.slider("Web Visits / Month:", 0, 20, 5)
            f_prev_cmp = st.slider("Past Campaigns Accepted (0 to 5):", 0, 5, 0)

        predict_btn = st.form_submit_button("Run Dual-Model Prediction & Generate Strategy", type="primary", use_container_width=True)

    if predict_btn:
        tot_spend = f_wines + f_meat + f_fish + f_sweets + f_gold + f_fruits
        tot_purch = f_web_p + f_store_p + f_cat_p + f_deal_p
        tot_kids = f_kids + f_teens
        safe_purch = max(1, tot_purch)

        cust_dict = {
            "Year_Birth": 2014 - f_age,
            "Customer_Age": f_age,
            "Education": f_edu,
            "Marital_Status": f_marital,
            "Income": float(f_income),
            "Kidhome": f_kids,
            "Teenhome": f_teens,
            "Total_Children": tot_kids,
            "Has_Children": 1 if tot_kids > 0 else 0,
            "Customer_Tenure": 450,
            "Recency": f_recency,
            "MntWines": f_wines,
            "MntMeatProducts": f_meat,
            "MntFishProducts": f_fish,
            "MntSweetProducts": f_sweets,
            "MntGoldProds": f_gold,
            "MntFruits": f_fruits,
            "Total_Spending": tot_spend,
            "Total_Purchases": tot_purch,
            "NumWebPurchases": f_web_p,
            "NumStorePurchases": f_store_p,
            "NumCatalogPurchases": f_cat_p,
            "NumDealsPurchases": f_deal_p,
            "NumWebVisitsMonth": f_web_visits,
            "Web_Purchase_Ratio": round(f_web_p / safe_purch, 3),
            "Store_Purchase_Ratio": round(f_store_p / safe_purch, 3),
            "Catalog_Purchase_Ratio": round(f_cat_p / safe_purch, 3),
            "Deal_Purchase_Ratio": round(f_deal_p / safe_purch, 3),
            "Avg_Spending_Per_Purchase": round(tot_spend / safe_purch, 2),
            "Previous_Campaigns_Accepted": f_prev_cmp,
            "Has_Accepted_Previous_Cmp": 1 if f_prev_cmp > 0 else 0,
        }

        # Step 1: Model 1 Clustering
        cluster_res = predict_cluster(
            cust_dict, models["kmeans"], models["scaler"],
            models["pca"], models["profiles"]
        )
        cust_dict["Cluster"] = cluster_res["cluster_id"]
        assigned_segment = cluster_res["cluster_name"]

        # Step 2: Model 2 Response Prediction
        resp_model = models["response_model"]
        preprocessor = models["preprocessor"]
        comp_data = models.get("comparison", {})
        all_features = comp_data.get("feature_names", {}).get("all", [])

        resp_pred = predict_customer_response(
            cust_dict, resp_model, preprocessor, all_features, threshold=0.45
        )

        # Step 3: Recommendation Engine
        rec = generate_recommendation(
            cluster_name=assigned_segment,
            response_probability=resp_pred["probability"],
            is_responsive=resp_pred["prediction"] == 1,
            customer_data=cust_dict
        )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Prediction Results & Actionable Decision")

        out_c1, out_c2, out_c3 = st.columns(3)
        with out_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Assigned Segment (Model 1)</div>
                <div class="metric-value" style="font-size: 1.35rem; color: {THEME_VARS['accent']};">
                    {assigned_segment}
                </div>
                <div class="metric-sub">{cluster_res['description'][:85]}...</div>
            </div>
            """, unsafe_allow_html=True)

        with out_c2:
            prob_val = resp_pred["percentage"]
            color_badge = THEME_VARS["green"] if prob_val >= 45 else THEME_VARS["amber"]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Response Propensity (Model 2)</div>
                <div class="metric-value" style="color: {color_badge};">
                    {prob_val}%
                </div>
                <div class="metric-sub">Decision Threshold: 45%</div>
            </div>
            """, unsafe_allow_html=True)

        with out_c3:
            badge_type = "badge-green" if resp_pred["prediction"] == 1 else "badge-amber"
            status_text = "LIKELY TO RESPOND" if resp_pred["prediction"] == 1 else "UNLIKELY TO RESPOND"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Campaign Outcome</div>
                <div class="metric-value" style="font-size: 1.3rem;">
                    <span class="badge {badge_type}">{status_text}</span>
                </div>
                <div class="metric-sub">Primary Affinity: {rec['affinity_category']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="strategy-box">
            <div style="font-size: 0.76rem; text-transform: uppercase; color: {THEME_VARS['text_muted']}; font-weight: 700; letter-spacing: 0.05em;">
                Personalized Marketing Action Playbook
            </div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {THEME_VARS['text']}; margin: 4px 0 10px;">
                {rec['primary_strategy']}
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 12px;">
                <div>
                    <span style="font-size: 0.75rem; color: {THEME_VARS['text_muted']};">Campaign Offer:</span><br/>
                    <b>{rec['campaign_offer']}</b>
                </div>
                <div>
                    <span style="font-size: 0.75rem; color: {THEME_VARS['text_muted']};">Delivery Channel:</span><br/>
                    <b>{rec['recommended_channel']}</b>
                </div>
                <div>
                    <span style="font-size: 0.75rem; color: {THEME_VARS['text_muted']};">Messaging Tone:</span><br/>
                    <b>{rec['messaging_tone']}</b>
                </div>
                <div>
                    <span style="font-size: 0.75rem; color: {THEME_VARS['text_muted']};">Expected ROI Tier:</span><br/>
                    <b style="color: {THEME_VARS['green']};">{rec['expected_roi']}</b>
                </div>
            </div>
            <div style="font-size: 0.8rem; font-weight: 600; color: {THEME_VARS['text']}; margin-top: 8px;">Action Checklist:</div>
            <ul style="font-size: 0.8rem; color: {THEME_VARS['text_muted']}; margin-bottom: 0; padding-left: 20px;">
                {"".join(f"<li>{item}</li>" for item in rec['action_items'])}
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 5: CUSTOMER 360° PROFILE LOOKUP
# ==============================================================================
with tabs[4]:
    st.markdown("### Customer 360° Profile Dossier")
    st.markdown("Lookup any individual customer by their unique ID to view their transaction history, segment, and predicted campaign propensity.")

    if "Id" in df.columns:
        all_ids = df["Id"].tolist()
        c_search1, c_search2 = st.columns([2, 2])
        with c_search1:
            selected_id = st.selectbox("Select Customer ID:", options=all_ids, index=0)
        with c_search2:
            st.markdown(f"<div style='margin-top: 28px;'><span class='badge badge-blue'>Loaded {len(all_ids):,} customer records</span></div>", unsafe_allow_html=True)

        cust_row = df[df["Id"] == selected_id].iloc[0]

        # Customer Header Dossier (with ₹)
        cust_c1, cust_c2, cust_c3, cust_c4 = st.columns(4)
        with cust_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Demographics</div>
                <div class="metric-value" style="font-size: 1.25rem;">Age {int(cust_row.get('Customer_Age', 45))}</div>
                <div class="metric-sub">{cust_row.get('Education', 'N/A')} • {cust_row.get('Marital_Status', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with cust_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Lifetime Spending</div>
                <div class="metric-value">₹{cust_row.get('Total_Spending', 0):,.0f}</div>
                <div class="metric-sub">{cust_row.get('Total_Purchases', 0)} Total Purchases</div>
            </div>
            """, unsafe_allow_html=True)
        with cust_c3:
            rec_val = cust_row.get('Recency', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recency</div>
                <div class="metric-value">{rec_val} Days</div>
                <div class="metric-sub">Annual Income: ₹{cust_row.get('Income', 0):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with cust_c4:
            actual_resp = cust_row.get('Response', 0)
            resp_tag = "badge-green" if actual_resp == 1 else "badge-amber"
            resp_txt = "Responded" if actual_resp == 1 else "No Response"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Segment Tag</div>
                <div class="metric-value" style="font-size: 1.1rem; color: {THEME_VARS['accent']};">{cust_row.get('Cluster_Name', 'Cluster')}</div>
                <div class="metric-sub">Past Response: <span class="badge {resp_tag}">{resp_txt}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        doss_c1, doss_c2 = st.columns([1, 1])
        with doss_c1:
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Customer Product Spend Basket (₹)</div>
                <div class="chart-subtitle">Breakdown of purchases by category for Customer ID: """ + str(selected_id) + """</div>
            """, unsafe_allow_html=True)
            cust_spend_dict = {
                "Wines": cust_row.get("MntWines", 0),
                "Meat": cust_row.get("MntMeatProducts", 0),
                "Gold": cust_row.get("MntGoldProds", 0),
                "Fish": cust_row.get("MntFishProducts", 0),
                "Sweets": cust_row.get("MntSweetProducts", 0),
                "Fruits": cust_row.get("MntFruits", 0),
            }
            spend_df = pd.DataFrame(list(cust_spend_dict.items()), columns=["Category", "Spend"])
            fig_cust_spend = px.pie(spend_df, values="Spend", names="Category", hole=0.5,
                                    color_discrete_sequence=PLOT_LAYOUT["colorway"])
            fig_cust_spend.update_layout(PLOT_LAYOUT, height=260, margin=dict(t=5, b=5, l=5, r=5))
            st.plotly_chart(fig_cust_spend, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with doss_c2:
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Channel Activity Split</div>
                <div class="chart-subtitle">Order volume completed per retail channel</div>
            """, unsafe_allow_html=True)
            cust_chan_dict = {
                "Web Purchases": cust_row.get("NumWebPurchases", 0),
                "Store Purchases": cust_row.get("NumStorePurchases", 0),
                "Catalog Purchases": cust_row.get("NumCatalogPurchases", 0),
                "Deal Purchases": cust_row.get("NumDealsPurchases", 0),
            }
            chan_pie_df = pd.DataFrame(list(cust_chan_dict.items()), columns=["Channel", "Orders"])
            fig_cust_chan = px.bar(chan_pie_df, x="Orders", y="Channel", orientation="h",
                                   color="Channel", color_discrete_sequence=PLOT_LAYOUT["colorway"])
            fig_cust_chan.update_layout(PLOT_LAYOUT, height=260, showlegend=False, margin=dict(t=5, b=5, l=5, r=5))
            st.plotly_chart(fig_cust_chan, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # Single customer response prediction
        cust_dict = cust_row.to_dict()
        resp_model = models["response_model"]
        preprocessor = models["preprocessor"]
        comp_data = models.get("comparison", {})
        all_features = comp_data.get("feature_names", {}).get("all", [])

        pred_resp = predict_customer_response(cust_dict, resp_model, preprocessor, all_features)
        rec = generate_recommendation(
            cluster_name=cust_row.get("Cluster_Name", "General"),
            response_probability=pred_resp["probability"],
            is_responsive=pred_resp["prediction"] == 1,
            customer_data=cust_dict
        )

        st.markdown(f"""
        <div class="strategy-box">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: {THEME_VARS['text_muted']}; font-weight: 700;">
                Customer-Specific Action Plan
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {THEME_VARS['text']}; margin: 3px 0 8px;">
                Recommended: {rec['primary_strategy']}
            </div>
            <div style="font-size: 0.82rem; color: {THEME_VARS['text']}; margin-bottom: 8px;">
                <b>Predicted Response Probability:</b> {pred_resp['percentage']}% ({pred_resp['label']})
            </div>
            <div style="font-size: 0.8rem; color: {THEME_VARS['text_muted']};">
                <b>Suggested Offer:</b> {rec['campaign_offer']}<br/>
                <b>Preferred Channel:</b> {rec['recommended_channel']}<br/>
                <b>Messaging Tone:</b> {rec['messaging_tone']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 6: DATA UPLOAD & ACTIVE RE-TRAINING
# ==============================================================================
with tabs[5]:
    st.markdown("### Upload & Validate Customer Dataset")
    st.markdown(
        "Upload a new customer marketing campaign CSV. The pipeline automatically detects delimiters "
        "(comma, semicolon, tab), validates schema integrity, imputes missing values, and re-trains models directly in memory."
    )

    up_col1, up_col2 = st.columns([7, 3])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file (e.g. customers.csv, marketing_campaign.csv):",
            type=["csv", "txt"]
        )
    with up_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Reset to Default Benchmark Dataset", use_container_width=True):
            st.session_state.active_df = get_dataset()
            st.session_state.active_models = load_models()
            st.session_state.dataset_name = "Default Benchmark Dataset (customers.csv)"
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Reset back to default benchmark dataset.")
            st.rerun()

    if uploaded_file is not None:
        try:
            raw_uploaded = load_raw_data(uploaded_file)
            st.success(f"File '{uploaded_file.name}' read successfully! Detected {len(raw_uploaded):,} rows and {len(raw_uploaded.columns)} columns.")

            # Schema Validation
            is_valid, missing_cols = validate_schema(raw_uploaded)
            if not is_valid:
                st.error(f"Schema Validation Failed! The following required columns are missing: {missing_cols}")
            else:
                st.success("All required schema columns are verified.")

                # Cleaning preview
                cleaned_up, clean_rep = clean_data(raw_uploaded)

                up_c1, up_c2, up_c3, up_c4 = st.columns(4)
                with up_c1:
                    st.metric("Initial Rows", f"{clean_rep['initial_rows']:,}")
                with up_c2:
                    st.metric("Valid Cleaned Rows", f"{clean_rep['final_rows']:,}")
                with up_c3:
                    st.metric("Missing Income Imputed", clean_rep["missing_income_imputed"])
                with up_c4:
                    st.metric("Outliers Handled", clean_rep["removed_rows"])

                st.markdown("##### Cleaned Dataset Preview (First 5 Rows):")
                st.dataframe(cleaned_up.head(5), use_container_width=True)

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # Action button to trigger instant in-memory pipeline execution
                if st.button("Apply This Dataset & Re-Train All Models Across Dashboard", type="primary", use_container_width=True):
                    prog_bar = st.progress(0, text="Initiating in-memory retraining pipeline...")

                    # Step 1: Feature Engineering
                    prog_bar.progress(25, text="Step 1/4: Engineering behavioral features and purchasing ratios...")
                    feat_df = engineer_features(cleaned_up)

                    # Step 2: K-Means Clustering & 2D PCA
                    prog_bar.progress(55, text="Step 2/4: Fitting K-Means clustering & 2D PCA projection...")
                    kmeans, scaler, pca, segmented_df, profiles = fit_clustering_pipeline(feat_df, n_clusters=4)

                    # Step 3: Supervised Response Models
                    prog_bar.progress(80, text="Step 3/4: Training supervised models (Logistic Regression, Random Forest, XGBoost)...")
                    pred_results = train_and_evaluate_models(segmented_df, include_cluster=True)

                    # Step 4: Persist to disk and update session state
                    prog_bar.progress(95, text="Step 4/4: Serializing updated models and activating session state...")
                    cleaned_up.to_csv(get_path("data/customers.csv"), index=False)
                    segmented_df.to_csv(get_path("models/processed_customers.csv"), index=False)
                    joblib.dump(scaler, get_path("models/scaler.pkl"))
                    joblib.dump(kmeans, get_path("models/clustering_model.pkl"))
                    joblib.dump(pca, get_path("models/pca_model.pkl"))
                    joblib.dump(pred_results["models"][pred_results["best_model_name"]]["model"], get_path("models/response_model.pkl"))
                    joblib.dump(pred_results["preprocessor"], get_path("models/preprocessor.pkl"))

                    with open(get_path("models/cluster_profiles.json"), "w", encoding="utf-8") as f:
                        json.dump(profiles, f, indent=2)

                    serializable = {
                        "best_model_name": pred_results["best_model_name"],
                        "comparison_table": pred_results["comparison_table"].to_dict(orient="records"),
                        "models": {
                            name: {
                                "accuracy": d["accuracy"],
                                "precision": d["precision"],
                                "recall": d["recall"],
                                "f1_score": d["f1_score"],
                                "roc_auc": d["roc_auc"],
                                "confusion_matrix": d["confusion_matrix"],
                                "roc_curve": d["roc_curve"]
                            }
                            for name, d in pred_results["models"].items()
                        },
                        "feature_names": pred_results["feature_names"],
                        "test_metadata": pred_results["test_metadata"],
                        "cleaning_report": clean_rep
                    }
                    with open(get_path("models/model_comparison.json"), "w", encoding="utf-8") as f:
                        json.dump(serializable, f, indent=2)

                    # Activate across current Streamlit session
                    st.session_state.active_df = segmented_df
                    st.session_state.active_models = {
                        "scaler": scaler,
                        "kmeans": kmeans,
                        "pca": pca,
                        "response_model": pred_results["models"][pred_results["best_model_name"]]["model"],
                        "preprocessor": pred_results["preprocessor"],
                        "profiles": profiles,
                        "comparison": serializable,
                        "k_eval": models.get("k_eval")
                    }
                    st.session_state.dataset_name = f"Uploaded: {uploaded_file.name} ({len(segmented_df):,} customers)"
                    st.cache_data.clear()
                    st.cache_resource.clear()

                    prog_bar.progress(100, text="Complete! Updated models and segments are active.")
                    st.success(f"Success! The uploaded dataset '{uploaded_file.name}' is now active across all dashboard tabs.")
                    st.rerun()

        except Exception as ex:
            st.error(f"Error reading or processing file: {ex}")


# ==============================================================================
# TAB 7: METHODOLOGY & MODEL AUDIT
# ==============================================================================
with tabs[6]:
    st.markdown("### Machine Learning Methodology & Data Leakage Prevention")

    comp_dict = models.get("comparison", {})

    st.markdown("""
    #### Data Leakage Prevention Strategy
    In supervised marketing response modeling, data leakage occurs when models learn from features containing
    future information or identifiers that are unavailable before the campaign launch.
    
    Our architecture guarantees strict zero-leakage compliance through four safeguards:
    1. **Strict Chronological Separation**:
       - `Response` represents acceptance in the **current / 6th campaign** (target event).
       - Features `AcceptedCmp1` through `AcceptedCmp5` represent pilot campaigns conducted **strictly prior** to the target campaign. They are aggregated into a pre-campaign engagement feature (`Previous_Campaigns_Accepted`).
    2. **Exclusion of Identifier & Artifact Columns**:
       - `Id` is strictly isolated for UI lookup and is **never** passed into feature matrices.
       - Constant operational columns `Z_CostContact` (always ₹3) and `Z_Revenue` (always ₹11) are automatically dropped.
    3. **Featurization Ordering**:
       - Train-test splitting (80/20 stratified) is executed **before** fitting any `StandardScaler` or `OneHotEncoder`.
       - All test evaluation is performed on completely unseen customer test folds.
    4. **Unsupervised Segments as Inputs**:
       - K-Means segmentation operates strictly without `Response`. Customer segment assignment is passed into the classifier as an unbiased behavioral feature.
    """)

    st.markdown("<hr style='border: none; border-top: 1px solid " + THEME_VARS["border"] + "; margin: 20px 0;'/>", unsafe_allow_html=True)

    if comp_dict and "comparison_table" in comp_dict:
        st.markdown("#### Supervised Model Benchmark Comparison")
        st.markdown("Evaluated on stratified unseen test holdout (20% split):")

        comp_df = pd.DataFrame(comp_dict["comparison_table"])
        st.dataframe(
            comp_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#1e3a8a"),
            use_container_width=True
        )

        st.markdown(f"**Recommended Production Model:** `{comp_dict.get('best_model_name', 'Random Forest')}` "
                    f"(Selected for optimal balance of ROC-AUC and F1-Score on imbalanced responses).")

        # ROC Curves & Confusion Matrix
        eval_c1, eval_c2 = st.columns([1, 1])

        with eval_c1:
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Receiver Operating Characteristic (ROC) Curves</div>
                <div class="chart-subtitle">True Positive Rate vs False Positive Rate across classification thresholds</div>
            """, unsafe_allow_html=True)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#71717a"), name="Random (AUC = 0.50)"))

            for m_name, m_data in comp_dict.get("models", {}).items():
                if "roc_curve" in m_data:
                    rc = m_data["roc_curve"]
                    fig_roc.add_trace(go.Scatter(
                        x=rc["fpr"], y=rc["tpr"], mode="lines",
                        name=f"{m_name} (AUC = {m_data['roc_auc']:.3f})"
                    ))
            fig_roc.update_layout(PLOT_LAYOUT, height=340, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with eval_c2:
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Confusion Matrix (Recommended Model: """ + str(comp_dict.get("best_model_name")) + """)</div>
                <div class="chart-subtitle">Classification breakdown on unseen holdout test set</div>
            """, unsafe_allow_html=True)
            best_m_data = comp_dict["models"].get(comp_dict["best_model_name"], {})
            if "confusion_matrix" in best_m_data:
                cm = best_m_data["confusion_matrix"]
                z = cm
                x = ["Predicted: 0 (No)", "Predicted: 1 (Yes)"]
                y = ["Actual: 0 (No)", "Actual: 1 (Yes)"]
                fig_cm = px.imshow(
                    z, x=x, y=y, text_auto=True,
                    color_continuous_scale="Blues" if not IS_DARK else "teal",
                    labels=dict(x="Predicted", y="Actual", color="Customers")
                )
                fig_cm.update_layout(PLOT_LAYOUT, height=340, coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        if "cleaning_report" in comp_dict:
            cl = comp_dict["cleaning_report"]
            st.markdown("#### Data Quality & Preprocessing Audit Summary")
            st.markdown(f"""
            - **Initial Raw Records:** {cl.get('initial_rows', 0):,}
            - **Final Cleaned Records:** {cl.get('final_rows', 0):,}
            - **Missing Income Imputations:** {cl.get('missing_income_imputed', 0)} (imputed using group median by education)
            - **Invalid Birth Years Removed (< 1920):** {cl.get('invalid_birth_years_removed', 0)}
            - **Extreme Outlier Incomes Handled (> ₹200,000):** {cl.get('extreme_income_outliers_removed', 0)}
            - **Constant Columns Stripped:** `{cl.get('dropped_constant_columns', [])}`
            """)
