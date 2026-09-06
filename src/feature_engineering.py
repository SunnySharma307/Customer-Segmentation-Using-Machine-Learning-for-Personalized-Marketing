"""
feature_engineering.py
Domain-driven feature engineering for customer segmentation and response prediction.
"""

from typing import Optional
import pandas as pd
import numpy as np


SPENDING_COLUMNS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]

PURCHASE_COUNT_COLUMNS = [
    "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumDealsPurchases"
]

CAMPAIGN_HISTORY_COLUMNS = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
    "AcceptedCmp4", "AcceptedCmp5"
]


def engineer_features(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Computes all required behavioral, demographic, and interaction features.
    
    Features created:
      - Total_Spending: sum of all product categories
      - Total_Purchases: sum of all channel purchases
      - Total_Children: Kidhome + Teenhome
      - Has_Children: binary indicator
      - Customer_Age: age relative to dataset cohort reference year
      - Customer_Tenure: active days from registration (Dt_Customer)
      - Channel purchase ratios: Web, Store, Catalog, Deal shares
      - Average spending per purchase
      - Spending category shares (Wine, Meat, Gold)
      - Previous campaign participation (pre-target engagement)
    """
    feat_df = df.copy()

    # 1. Total Spending
    available_spend_cols = [c for c in SPENDING_COLUMNS if c in feat_df.columns]
    feat_df["Total_Spending"] = feat_df[available_spend_cols].sum(axis=1)

    # 2. Total Purchases
    available_purchase_cols = [c for c in PURCHASE_COUNT_COLUMNS if c in feat_df.columns]
    feat_df["Total_Purchases"] = feat_df[available_purchase_cols].sum(axis=1)

    # 3. Total Children & Family composition
    kid = feat_df["Kidhome"] if "Kidhome" in feat_df.columns else 0
    teen = feat_df["Teenhome"] if "Teenhome" in feat_df.columns else 0
    feat_df["Total_Children"] = kid + teen
    feat_df["Has_Children"] = (feat_df["Total_Children"] > 0).astype(int)

    # 4. Customer Age
    if "Dt_Customer" in feat_df.columns and pd.api.types.is_datetime64_any_dtype(feat_df["Dt_Customer"]):
        if reference_date is None:
            reference_date = feat_df["Dt_Customer"].max() + pd.Timedelta(days=1)
        ref_year = reference_date.year
    else:
        ref_year = 2015
        if reference_date is None:
            reference_date = pd.Timestamp("2015-01-01")

    if "Year_Birth" in feat_df.columns:
        feat_df["Customer_Age"] = ref_year - feat_df["Year_Birth"]
    else:
        feat_df["Customer_Age"] = 45  # fallback

    # 5. Customer Tenure (days since registration)
    if "Dt_Customer" in feat_df.columns:
        feat_df["Dt_Customer"] = pd.to_datetime(feat_df["Dt_Customer"], errors="coerce")
        feat_df["Customer_Tenure"] = (reference_date - feat_df["Dt_Customer"]).dt.days
        # If any missing tenure, fill with median
        if feat_df["Customer_Tenure"].isnull().any():
            feat_df["Customer_Tenure"] = feat_df["Customer_Tenure"].fillna(feat_df["Customer_Tenure"].median())
    else:
        feat_df["Customer_Tenure"] = 365

    # 6. Channel Ratios (with zero-division protection)
    safe_purchases = feat_df["Total_Purchases"].replace(0, np.nan)
    feat_df["Web_Purchase_Ratio"] = (feat_df["NumWebPurchases"] / safe_purchases).fillna(0.0).clip(0, 1)
    feat_df["Store_Purchase_Ratio"] = (feat_df["NumStorePurchases"] / safe_purchases).fillna(0.0).clip(0, 1)
    feat_df["Catalog_Purchase_Ratio"] = (feat_df["NumCatalogPurchases"] / safe_purchases).fillna(0.0).clip(0, 1)
    feat_df["Deal_Purchase_Ratio"] = (feat_df["NumDealsPurchases"] / safe_purchases).fillna(0.0).clip(0, 1)

    # 7. Average Spending Per Purchase
    feat_df["Avg_Spending_Per_Purchase"] = (feat_df["Total_Spending"] / safe_purchases).fillna(0.0)

    # 8. Spending Category Shares
    safe_spending = feat_df["Total_Spending"].replace(0, np.nan)
    if "MntWines" in feat_df.columns:
        feat_df["Wine_Share"] = (feat_df["MntWines"] / safe_spending).fillna(0.0).clip(0, 1)
    if "MntMeatProducts" in feat_df.columns:
        feat_df["Meat_Share"] = (feat_df["MntMeatProducts"] / safe_spending).fillna(0.0).clip(0, 1)
    if "MntGoldProds" in feat_df.columns:
        feat_df["Gold_Share"] = (feat_df["MntGoldProds"] / safe_spending).fillna(0.0).clip(0, 1)

    # 9. Historical Campaign Participation (pre-target events)
    hist_cmp_cols = [c for c in CAMPAIGN_HISTORY_COLUMNS if c in feat_df.columns]
    if hist_cmp_cols:
        feat_df["Previous_Campaigns_Accepted"] = feat_df[hist_cmp_cols].sum(axis=1)
        feat_df["Has_Accepted_Previous_Cmp"] = (feat_df["Previous_Campaigns_Accepted"] > 0).astype(int)
    else:
        feat_df["Previous_Campaigns_Accepted"] = 0
        feat_df["Has_Accepted_Previous_Cmp"] = 0

    return feat_df
