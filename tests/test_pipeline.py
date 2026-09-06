"""
test_pipeline.py
Automated unit and integration tests for customer segmentation & prediction pipelines.
"""

import unittest
import pandas as pd
import numpy as np

from src.preprocessing import clean_data, validate_schema
from src.feature_engineering import engineer_features
from src.clustering import (
    fit_clustering_pipeline, predict_cluster, CLUSTERING_FEATURES
)
from src.prediction import (
    train_and_evaluate_models, predict_customer_response
)
from src.recommendations import generate_recommendation


class TestCustomerPipeline(unittest.TestCase):

    def setUp(self):
        # Create a synthetic dataset mimicking the schema
        np.random.seed(42)
        n = 120
        self.raw_data = pd.DataFrame({
            "Id": range(1001, 1001 + n),
            "Year_Birth": np.random.choice([1955, 1965, 1975, 1985, 1990, 1899], size=n, p=[0.2, 0.25, 0.25, 0.2, 0.08, 0.02]),
            "Education": np.random.choice(["Graduation", "PhD", "Master", "Basic", "2n Cycle"], size=n),
            "Marital_Status": np.random.choice(["Married", "Single", "Together", "Divorced", "YOLO"], size=n),
            "Income": np.random.choice([np.nan, 30000.0, 50000.0, 75000.0, 120000.0, 666666.0], size=n, p=[0.05, 0.25, 0.35, 0.25, 0.08, 0.02]),
            "Kidhome": np.random.choice([0, 1, 2], size=n),
            "Teenhome": np.random.choice([0, 1], size=n),
            "Dt_Customer": pd.date_range("2012-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "Recency": np.random.randint(0, 100, size=n),
            "MntWines": np.random.randint(5, 500, size=n),
            "MntFruits": np.random.randint(0, 100, size=n),
            "MntMeatProducts": np.random.randint(5, 400, size=n),
            "MntFishProducts": np.random.randint(0, 100, size=n),
            "MntSweetProducts": np.random.randint(0, 100, size=n),
            "MntGoldProds": np.random.randint(5, 150, size=n),
            "NumDealsPurchases": np.random.randint(0, 10, size=n),
            "NumWebPurchases": np.random.randint(1, 15, size=n),
            "NumCatalogPurchases": np.random.randint(0, 10, size=n),
            "NumStorePurchases": np.random.randint(1, 15, size=n),
            "NumWebVisitsMonth": np.random.randint(1, 12, size=n),
            "AcceptedCmp1": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
            "AcceptedCmp2": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
            "AcceptedCmp3": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
            "AcceptedCmp4": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
            "AcceptedCmp5": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
            "Complain": np.zeros(n, dtype=int),
            "Z_CostContact": [3] * n,
            "Z_Revenue": [11] * n,
            "Response": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        })

    def test_schema_validation(self):
        valid, missing = validate_schema(self.raw_data)
        self.assertTrue(valid)
        self.assertEqual(len(missing), 0)

    def test_data_cleaning(self):
        cleaned, report = clean_data(self.raw_data, income_cap=200000.0, min_birth_year=1920)
        # Income should have zero nulls
        self.assertEqual(cleaned["Income"].isnull().sum(), 0)
        # Unrealistic year (< 1920) should be removed
        self.assertTrue((cleaned["Year_Birth"] >= 1920).all())
        # Extreme income (> 200000) should be removed
        self.assertTrue((cleaned["Income"] <= 200000).all())
        # Constant columns removed
        self.assertNotIn("Z_CostContact", cleaned.columns)
        self.assertNotIn("Z_Revenue", cleaned.columns)
        # Categorical normalized
        self.assertNotIn("YOLO", cleaned["Marital_Status"].unique())

    def test_feature_engineering(self):
        cleaned, _ = clean_data(self.raw_data)
        engineered = engineer_features(cleaned)

        self.assertIn("Total_Spending", engineered.columns)
        self.assertIn("Total_Purchases", engineered.columns)
        self.assertIn("Customer_Age", engineered.columns)
        self.assertIn("Customer_Tenure", engineered.columns)
        self.assertIn("Web_Purchase_Ratio", engineered.columns)

        # Spending sum correctness
        expected_spend = (
            cleaned["MntWines"] + cleaned["MntFruits"] + cleaned["MntMeatProducts"] +
            cleaned["MntFishProducts"] + cleaned["MntSweetProducts"] + cleaned["MntGoldProds"]
        )
        pd.testing.assert_series_equal(
            engineered["Total_Spending"], expected_spend, check_names=False, check_dtype=False
        )

    def test_clustering_pipeline(self):
        cleaned, _ = clean_data(self.raw_data)
        engineered = engineer_features(cleaned)
        kmeans, scaler, pca, segmented, profiles = fit_clustering_pipeline(engineered, n_clusters=3)

        self.assertIn("Cluster", segmented.columns)
        self.assertIn("Cluster_Name", segmented.columns)
        self.assertIn("PCA1", segmented.columns)
        self.assertEqual(len(profiles), 3)

        # Test single prediction
        sample = engineered.iloc[0].to_dict()
        pred = predict_cluster(sample, kmeans, scaler, pca, profiles)
        self.assertIn("cluster_id", pred)
        self.assertIn("cluster_name", pred)
        self.assertIn("pca1", pred)

    def test_supervised_prediction_and_recommendation(self):
        cleaned, _ = clean_data(self.raw_data)
        engineered = engineer_features(cleaned)
        kmeans, scaler, pca, segmented, profiles = fit_clustering_pipeline(engineered, n_clusters=3)

        pred_results = train_and_evaluate_models(segmented, test_size=0.25)
        self.assertIn("Logistic Regression", pred_results["models"])
        self.assertIn("comparison_table", pred_results)

        best_name = pred_results["best_model_name"]
        best_model = pred_results["models"][best_name]["model"]
        preprocessor = pred_results["preprocessor"]

        sample = segmented.iloc[0].to_dict()
        resp = predict_customer_response(
            sample, best_model, preprocessor,
            pred_results["feature_names"]["all"]
        )
        self.assertIn("probability", resp)
        self.assertIn("label", resp)

        rec = generate_recommendation(
            cluster_name=sample["Cluster_Name"],
            response_probability=resp["probability"],
            is_responsive=resp["prediction"] == 1,
            customer_data=sample
        )
        self.assertIn("primary_strategy", rec)
        self.assertIn("campaign_offer", rec)
        self.assertIn("action_items", rec)


if __name__ == "__main__":
    unittest.main()
