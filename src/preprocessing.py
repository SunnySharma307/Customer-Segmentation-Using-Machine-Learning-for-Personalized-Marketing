"""
preprocessing.py
Data loading, schema validation, outlier handling, and data cleaning routines.
"""

from typing import Tuple, Dict, Any, Optional
import io
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Dt_Customer", "Recency", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth"
]

CONSTANT_COLUMNS = ["Z_CostContact", "Z_Revenue"]


def detect_separator(file_or_path, sample_bytes: int = 4096) -> str:
    """Detect delimiter among comma, semicolon, tab."""
    if isinstance(file_or_path, str):
        with open(file_or_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(sample_bytes)
    elif hasattr(file_or_path, "read"):
        sample = file_or_path.read(sample_bytes)
        if isinstance(sample, bytes):
            sample = sample.decode("utf-8", errors="ignore")
        file_or_path.seek(0)
    else:
        sample = str(file_or_path)[:sample_bytes]

    counts = {
        ",": sample.count(","),
        ";": sample.count(";"),
        "\t": sample.count("\t"),
    }
    best_sep = max(counts, key=counts.get)
    return best_sep if counts[best_sep] > 0 else ","


def load_raw_data(file_or_path) -> pd.DataFrame:
    """
    Load customer dataset from path, uploaded file buffer, or string.
    Automatically handles comma, semicolon, and tab delimiters.
    """
    sep = detect_separator(file_or_path)
    if isinstance(file_or_path, str):
        df = pd.read_csv(file_or_path, sep=sep)
    else:
        df = pd.read_csv(file_or_path, sep=sep)

    # Standardize column casing for common variations (e.g. ID -> Id)
    col_map = {col: col.strip() for col in df.columns}
    df = df.rename(columns=col_map)
    if "ID" in df.columns and "Id" not in df.columns:
        df = df.rename(columns={"ID": "Id"})

    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validates that essential columns required for segmentation and prediction exist.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    is_valid = len(missing) == 0
    return is_valid, missing


def clean_data(
    df: pd.DataFrame,
    income_cap: float = 200000.0,
    min_birth_year: int = 1920
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs comprehensive data cleaning:
      1. Validates schema
      2. Deduplicates records
      3. Imputes missing Income using group median by Education (with fallback to global median)
      4. Filters unrealistic Year_Birth (< min_birth_year)
      5. Caps or removes extreme Income outliers (> income_cap)
      6. Converts Dt_Customer to datetime
      7. Normalizes categorical values (e.g. rare Marital_Status)
      8. Drops constant columns (Z_CostContact, Z_Revenue)
      9. Tracks all statistics for audit report
    """
    clean_df = df.copy()
    initial_rows = len(clean_df)
    
    # 1. Deduplication
    duplicates_count = clean_df.duplicated().sum()
    if duplicates_count > 0:
        clean_df = clean_df.drop_duplicates()

    # 2. Check & record missing Income
    missing_income_count = int(clean_df["Income"].isnull().sum())
    if missing_income_count > 0:
        # Median imputation grouped by Education
        clean_df["Income"] = clean_df.groupby("Education")["Income"].transform(
            lambda s: s.fillna(s.median())
        )
        # Fallback to global median if still null
        if clean_df["Income"].isnull().any():
            clean_df["Income"] = clean_df["Income"].fillna(clean_df["Income"].median())

    # 3. Filter unrealistic Year_Birth
    invalid_birth_mask = clean_df["Year_Birth"] < min_birth_year
    invalid_birth_count = int(invalid_birth_mask.sum())
    clean_df = clean_df[~invalid_birth_mask].copy()

    # 4. Filter or cap extreme Income outliers
    extreme_income_mask = clean_df["Income"] > income_cap
    extreme_income_count = int(extreme_income_mask.sum())
    clean_df = clean_df[~extreme_income_mask].copy()

    # 5. Convert Dt_Customer to datetime
    if "Dt_Customer" in clean_df.columns:
        clean_df["Dt_Customer"] = pd.to_datetime(clean_df["Dt_Customer"], errors="coerce")
        # In case of any unparseable dates, fill with median date
        if clean_df["Dt_Customer"].isnull().any():
            clean_df["Dt_Customer"] = clean_df["Dt_Customer"].fillna(clean_df["Dt_Customer"].dropna().median())

    # 6. Normalize Marital_Status
    if "Marital_Status" in clean_df.columns:
        clean_df["Marital_Status"] = clean_df["Marital_Status"].astype(str).str.strip()
        # Consolidate rare/nonsensical categories into Single or Partner
        marital_map = {
            "Married": "Married",
            "Together": "Together",
            "Single": "Single",
            "Divorced": "Divorced",
            "Widow": "Widow",
            "Alone": "Single",
            "Absurd": "Single",
            "YOLO": "Single"
        }
        clean_df["Marital_Status"] = clean_df["Marital_Status"].map(lambda x: marital_map.get(x, "Other"))

    # 7. Normalize Education
    if "Education" in clean_df.columns:
        clean_df["Education"] = clean_df["Education"].astype(str).str.strip()

    # 8. Remove constant columns
    dropped_cols = []
    for col in CONSTANT_COLUMNS:
        if col in clean_df.columns:
            clean_df = clean_df.drop(columns=[col])
            dropped_cols.append(col)

    # Reset index
    clean_df = clean_df.reset_index(drop=True)

    report = {
        "initial_rows": initial_rows,
        "final_rows": len(clean_df),
        "removed_rows": initial_rows - len(clean_df),
        "duplicates_removed": int(duplicates_count),
        "missing_income_imputed": missing_income_count,
        "invalid_birth_years_removed": invalid_birth_count,
        "extreme_income_outliers_removed": extreme_income_count,
        "dropped_constant_columns": dropped_cols,
    }

    return clean_df, report
