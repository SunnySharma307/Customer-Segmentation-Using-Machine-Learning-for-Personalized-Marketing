"""
prediction.py
Supervised binary classification pipeline for campaign response prediction.
Includes leakage prevention, strict train-test separation, model comparison, and metric evaluation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

NUMERIC_FEATURES = [
    "Customer_Age",
    "Income",
    "Total_Children",
    "Customer_Tenure",
    "Recency",
    "Total_Spending",
    "Total_Purchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "Avg_Spending_Per_Purchase",
    "Web_Purchase_Ratio",
    "Store_Purchase_Ratio",
    "Catalog_Purchase_Ratio",
    "Deal_Purchase_Ratio",
    "Previous_Campaigns_Accepted",
]

CATEGORICAL_FEATURES = [
    "Education",
    "Marital_Status",
]


def prepare_modeling_data(
    df: pd.DataFrame,
    include_cluster: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepares features X and target y, enforcing zero data leakage.
    Excludes identifiers, constant columns, and future event features.
    """
    if "Response" not in df.columns:
        raise ValueError("Target column 'Response' not found in dataframe.")

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    if include_cluster and "Cluster" in df.columns:
        cat_cols = cat_cols + ["Cluster"]

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()
    y = df["Response"].astype(int).copy()

    return X, y, num_cols, cat_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer that scales numeric features
    and one-hot encodes categorical features with unknown handling.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor


def train_and_evaluate_models(
    df: pd.DataFrame,
    include_cluster: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes supervised classification workflow:
      1. Prepares leakage-free features and target
      2. Stratified train-test split
      3. Fits preprocessor ONLY on training set
      4. Trains Logistic Regression, Random Forest, and XGBoost/GradientBoosting
      5. Evaluates full metric suite on unseen test set
    """
    X, y, num_cols, cat_cols = prepare_modeling_data(df, include_cluster=include_cluster)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Class weighting to address ~15% positive class imbalance
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(1, pos_count)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced",
            random_state=random_state, n_jobs=1
        )
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=random_state
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            random_state=random_state
        )

    results = {}
    comparison_rows = []

    for name, model in models.items():
        # Fit on transformed train
        model.fit(X_train_trans, y_train)

        # Predictions on test
        y_pred = model.predict(X_test_trans)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_trans)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, thresh = roc_curve(y_test, y_prob)

        results[name] = {
            "model": model,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm,
            "roc_curve": {
                "fpr": [float(x) for x in fpr],
                "tpr": [float(x) for x in tpr],
            }
        }

        comparison_rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
        })

    # Determine best model by ROC-AUC and F1
    best_model_name = max(results.keys(), key=lambda k: results[k]["roc_auc"] * 0.6 + results[k]["f1_score"] * 0.4)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="ROC-AUC", ascending=False)

    return {
        "models": results,
        "preprocessor": preprocessor,
        "comparison_table": comparison_df,
        "best_model_name": best_model_name,
        "feature_names": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "all": num_cols + cat_cols
        },
        "test_metadata": {
            "test_size": test_size,
            "test_samples": len(y_test),
            "positive_cases": int(y_test.sum()),
            "negative_cases": int((y_test == 0).sum())
        }
    }


def predict_customer_response(
    customer_dict: Dict[str, Any],
    model: Any,
    preprocessor: ColumnTransformer,
    feature_cols: List[str],
    threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Predicts campaign response probability and classification for a single customer.
    """
    row = pd.DataFrame([customer_dict])
    # Ensure all required columns exist with defaults
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    X = row[feature_cols]
    X_trans = preprocessor.transform(X)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_trans)[0, 1])
    else:
        prob = float(model.predict(X_trans)[0])

    is_responsive = bool(prob >= threshold)

    return {
        "probability": round(prob, 4),
        "percentage": round(prob * 100, 1),
        "prediction": 1 if is_responsive else 0,
        "label": "Likely to Respond" if is_responsive else "Unlikely to Respond",
        "threshold_used": threshold
    }
