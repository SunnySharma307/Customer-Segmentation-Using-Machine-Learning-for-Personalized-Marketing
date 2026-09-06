# Customer Segmentation & Marketing Response Prediction

A production-grade machine learning application that transforms raw customer marketing data into actionable behavioral segments and predicts marketing campaign responsiveness to deliver personalized marketing strategies.

Built with **Python**, **Scikit-Learn**, **XGBoost**, **Plotly**, and **Streamlit**.

---

##  Key Features

1. **Unsupervised Customer Segmentation (Model 1)**
   - **K-Means Clustering** with **StandardScaler** normalization.
   - **Elbow Method & Silhouette Analysis** across $K \in [2, 10]$ to rigorously select optimal clusters.
   - **2D PCA Projection** ($PC1$ & $PC2$) with interactive Plotly scatter maps and customer dossiers.
   - **Dynamic Cluster Profiling**: Programmatically names and profiles customer clusters (e.g. *High-Value Elite*, *Deal & Discount Hunters*, *Digital Convenience*, *Occasional Shoppers*) based on actual centroid characteristics rather than static numeric IDs.

2. **Supervised Campaign Response Prediction (Model 2)**
   - Binary classification targeting whether a customer will accept a marketing offer (`Response`).
   - Models trained & compared: **Logistic Regression**, **Random Forest Classifier**, and **XGBoost**.
   - Metric suite: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and ROC curves.
   - Class imbalance handling via balanced class weighting and threshold tuning.

3. **Strict Data Leakage Prevention**
   - Chronological isolation: Distinguishes pre-campaign pilot events (`AcceptedCmp1–5`) from the target campaign (`Response`).
   - Featurization ordering: Train-test splitting (80/20 stratified) executed *strictly before* fitting any scalers or encoders.
   - Elimination of identifiers (`Id`) and constant operational columns (`Z_CostContact`, `Z_Revenue`).

4. **Dual-Model Integration & Personalized Recommendation Engine**
   - Connects both models:
     $$\text{Customer} \longrightarrow \text{K-Means Segment} \longrightarrow \text{Response Classifier} \longrightarrow \text{Propensity Probability} \longrightarrow \text{Personalized Action Playbook}$$
   - Generates tailored offers, communication channels, messaging tone, and ROI tiers.

5. **Interactive Streamlit Web Dashboard**
   - Modern zinc design system with dark/light mode toggle.
   - 7 intuitive modules:
     -  **Executive Dashboard**: High-level KPIs, segment donuts, and conversion benchmarks.
     -  **Customer Segmentation**: Optimal K evaluation, 2D PCA visualizer, and segment persona cards.
     -  **Segment Deep-Dive**: Cross-segment product spending breakdown, channel preferences, and income boxplots.
     -  **Campaign Prediction**: Interactive simulator form to test what-if scenarios and generate marketing recommendations.
     -  **Customer 360° Profile**: Search customer dossier by ID with product basket breakdown.
     -  **Data Upload & Validation**: CSV drag-and-drop with delimiter sniffing and schema verification.
     -  **Methodology & Leakage Audit**: Full model comparison table, ROC curves, confusion matrix, and audit logs.

---

## ️ Project Architecture

```
customer-segmentation/
│
├── data/
│   └── customers.csv              # Customer marketing campaign dataset
│
├── models/
│   ├── clustering_model.pkl       # Fitted K-Means model
│   ├── scaler.pkl                 # StandardScaler for clustering features
│   ├── pca_model.pkl              # 2D PCA projection model
│   ├── response_model.pkl         # Trained supervised classifier
│   ├── preprocessor.pkl           # ColumnTransformer for classification
│   ├── cluster_profiles.json      # Dynamic segment profiles and descriptions
│   ├── k_evaluation.json          # Inertia and Silhouette metric scores
│   ├── model_comparison.json      # Model performance metrics & ROC data
│   └── processed_customers.csv    # Enriched dataset with segment labels & PCA
│
├── notebooks/
│   └── analysis.ipynb             # Full end-to-end data story and EDA
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Robust loading, delimiter sniffing, outlier cleaning
│   ├── feature_engineering.py     # Aggregations, tenure, age, and channel ratios
│   ├── clustering.py              # K-Means pipeline, silhouette scores, 2D PCA
│   ├── prediction.py              # Leakage-free classification & model benchmark
│   └── recommendations.py         # Personalized recommendation engine
│
├── app/
│   └── streamlit_app.py           # Multi-tab modern Streamlit web application
│
├── tests/
│   └── test_pipeline.py           # Automated unit and integration tests
│
├── train.py                       # CLI script to train models and serialize artifacts
├── requirements.txt               # Pinned package dependencies
└── README.md                      # Project documentation
```

---

##  Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train Models
Execute the complete training pipeline to clean data, optimize clusters, and train supervised classifiers:
```bash
python train.py
```

### 4. Launch Web Application
Start the Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 5. Run Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

##  Dataset Dictionary & Feature Definitions

| Feature | Type | Description |
|---|---|---|
| `Id` | Identifier | Unique customer ID (isolated from modeling) |
| `Year_Birth` | Demographic | Birth year; transformed to `Customer_Age` |
| `Education` | Categorical | Education level (`Graduation`, `PhD`, `Master`, `Basic`, `2n Cycle`) |
| `Marital_Status` | Categorical | Relationship status (`Married`, `Together`, `Single`, `Divorced`, `Widow`) |
| `Income` | Continuous | Yearly household income (missing values imputed by education median) |
| `Kidhome` / `Teenhome` | Count | Number of small children / teenagers in household |
| `Dt_Customer` | Date | Date of customer registration; transformed to `Customer_Tenure` |
| `Recency` | Integer | Number of days since last purchase |
| `MntWines` ... `MntGoldProds` | Currency | Spending across 6 distinct categories (summed into `Total_Spending`) |
| `NumWebPurchases` ... | Count | Orders completed across channels (summed into `Total_Purchases`) |
| `AcceptedCmp1` ... `5` | Binary | Historical acceptance in pilot campaigns |
| `Response` | Binary (Target) | 1 if customer accepted offer in target campaign, 0 otherwise |

---

## ️ Leakage Prevention Protocol
- **Target Event**: `Response` is the outcome of the latest campaign.
- **Featurization Separation**: Feature scaling and one-hot encoding are fit **only** on the training fold during cross-validation and holdout evaluation.
- **Operational Non-Predictors**: Columns with constant variance (`Z_CostContact`, `Z_Revenue`) and customer IDs are excluded prior to model ingestion.

---

## ️ License
Distributed under the Apache-2.0 License.
