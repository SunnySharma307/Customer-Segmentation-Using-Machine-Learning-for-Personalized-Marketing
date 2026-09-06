"""
train.py
End-to-end model training script:
1. Loads and cleans customer dataset
2. Engineers behavioral and demographic features
3. Computes optimal K metrics (Elbow & Silhouette)
4. Fits K-Means clustering & 2D PCA, dynamically profiling segments
5. Trains supervised classification models (Logistic Regression, Random Forest, XGBoost)
6. Serializes trained artifacts and metrics into models/ directory
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import load_raw_data, clean_data
from src.feature_engineering import engineer_features
from src.clustering import (
    evaluate_optimal_k, fit_clustering_pipeline, CLUSTERING_FEATURES
)
from src.prediction import train_and_evaluate_models


def main():
    print("=" * 60)
    print("Customer Segmentation & Campaign Response Pipeline")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    data_path = "data/customers.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place customers.csv in data/.")

    # 1. Load and Clean Data
    print("\n[Step 1/5] Loading and cleaning dataset...")
    raw_df = load_raw_data(data_path)
    clean_df, clean_report = clean_data(raw_df)
    print(f"  Initial records: {clean_report['initial_rows']}")
    print(f"  Cleaned records: {clean_report['final_rows']}")
    print(f"  Missing Income imputed: {clean_report['missing_income_imputed']}")
    print(f"  Outliers removed: {clean_report['removed_rows']}")

    # 2. Feature Engineering
    print("\n[Step 2/5] Engineering behavioral and interaction features...")
    engineered_df = engineer_features(clean_df)
    print(f"  Engineered dataset shape: {engineered_df.shape}")

    # 3. Clustering Evaluation & Training
    print("\n[Step 3/5] Evaluating optimal K (Elbow & Silhouette)...")
    k_eval = evaluate_optimal_k(engineered_df, k_range=(2, 8))
    print(f"  Tested K: {k_eval['k_values']}")
    print(f"  Silhouette Scores: {[round(s, 4) for s in k_eval['silhouette_scores']]}")
    print(f"  Best K by Silhouette: {k_eval['recommended_k_silhouette']}")

    with open("models/k_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(k_eval, f, indent=2)

    # Use K=4 (ideal balance of high silhouette and practical business granularity)
    n_clusters = 4
    print(f"\n[Step 4/5] Fitting K-Means (K={n_clusters}) & 2D PCA...")
    kmeans, scaler, pca, segmented_df, cluster_profiles = fit_clustering_pipeline(
        engineered_df, n_clusters=n_clusters
    )

    print("  Cluster Profiles & Dynamic Business Names:")
    for c_id, prof in cluster_profiles.items():
        print(f"    Cluster {c_id} -> '{prof['name']}': {prof['count']} customers ({prof['pct']}%), "
              f"Avg Spend: ${prof['spending']}, Avg Income: ${prof['income']}, Resp Rate: {prof['response_rate']}%")

    # Save clustering artifacts
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(kmeans, "models/clustering_model.pkl")
    joblib.dump(pca, "models/pca_model.pkl")

    with open("models/cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(cluster_profiles, f, indent=2)

    # 4. Supervised Campaign Response Prediction
    print("\n[Step 5/5] Training supervised classification models...")
    pred_results = train_and_evaluate_models(segmented_df, include_cluster=True)

    print("\nModel Comparison on Unseen Test Fold:")
    print(pred_results["comparison_table"].to_string(index=False))
    print(f"\nRecommended Production Model: {pred_results['best_model_name']}")

    best_model = pred_results["models"][pred_results["best_model_name"]]["model"]
    preprocessor = pred_results["preprocessor"]

    # Save prediction artifacts
    joblib.dump(best_model, "models/response_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # Serialize evaluation metrics (excluding python model objects)
    serializable_results = {
        "best_model_name": pred_results["best_model_name"],
        "comparison_table": pred_results["comparison_table"].to_dict(orient="records"),
        "models": {
            name: {
                "accuracy": data["accuracy"],
                "precision": data["precision"],
                "recall": data["recall"],
                "f1_score": data["f1_score"],
                "roc_auc": data["roc_auc"],
                "confusion_matrix": data["confusion_matrix"],
                "roc_curve": data["roc_curve"]
            }
            for name, data in pred_results["models"].items()
        },
        "feature_names": pred_results["feature_names"],
        "test_metadata": pred_results["test_metadata"],
        "cleaning_report": clean_report
    }

    with open("models/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    # Save enriched processed dataset for fast dashboard loading
    segmented_df.to_csv("models/processed_customers.csv", index=False)
    print("\nAll model artifacts and processed datasets saved successfully into 'models/'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
