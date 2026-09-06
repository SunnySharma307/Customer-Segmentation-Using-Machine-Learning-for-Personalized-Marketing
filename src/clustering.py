"""
clustering.py
K-Means customer segmentation, optimal K evaluation, PCA projection, and dynamic profiling.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CLUSTERING_FEATURES = [
    "Income",
    "Recency",
    "Total_Spending",
    "Total_Purchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "Total_Children",
    "Customer_Tenure"
]


def evaluate_optimal_k(
    df: pd.DataFrame,
    features: List[str] = CLUSTERING_FEATURES,
    k_range: Tuple[int, int] = (2, 10),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates multiple K values using Inertia (Elbow) and Silhouette Score.
    Returns inertias, silhouette scores, and optimal recommendation.
    """
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_values = list(range(k_range[0], k_range[1] + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        score = float(silhouette_score(X_scaled, labels))
        silhouettes.append(score)

    best_k_idx = int(np.argmax(silhouettes))
    recommended_k = k_values[best_k_idx]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouettes,
        "recommended_k_silhouette": recommended_k,
        "max_silhouette_score": silhouettes[best_k_idx]
    }


def fit_clustering_pipeline(
    df: pd.DataFrame,
    n_clusters: int = 4,
    features: List[str] = CLUSTERING_FEATURES,
    random_state: int = 42
) -> Tuple[KMeans, StandardScaler, PCA, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Fits StandardScaler, KMeans, and 2D PCA on the dataset.
    Generates dynamic business names and profiles for each cluster.
    """
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Fit 2D PCA for visual representation
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)

    # Add cluster results and coordinates to output dataframe
    result_df = df.copy()
    result_df["Cluster"] = cluster_labels
    result_df["PCA1"] = pca_coords[:, 0]
    result_df["PCA2"] = pca_coords[:, 1]

    # Calculate cluster profiles and dynamic business names
    profiles = generate_cluster_profiles(result_df, n_clusters, features)
    result_df["Cluster_Name"] = result_df["Cluster"].map(lambda c: profiles[c]["name"])

    return kmeans, scaler, pca, result_df, profiles


def generate_cluster_profiles(
    df: pd.DataFrame,
    n_clusters: int,
    features: List[str] = CLUSTERING_FEATURES
) -> Dict[int, Dict[str, Any]]:
    """
    Analyzes cluster statistics and assigns descriptive business names dynamically
    based on actual centroid characteristics rather than static numeric IDs.
    """
    stats = {}
    total_customers = len(df)

    for c in range(n_clusters):
        c_df = df[df["Cluster"] == c]
        count = len(c_df)
        pct = (count / total_customers) * 100

        # Mean statistics
        income = float(c_df["Income"].mean())
        spending = float(c_df["Total_Spending"].mean())
        purchases = float(c_df["Total_Purchases"].mean())
        recency = float(c_df["Recency"].mean())
        web_purchases = float(c_df["NumWebPurchases"].mean())
        catalog_purchases = float(c_df["NumCatalogPurchases"].mean())
        store_purchases = float(c_df["NumStorePurchases"].mean())
        deal_purchases = float(c_df["NumDealsPurchases"].mean())
        web_visits = float(c_df["NumWebVisitsMonth"].mean())
        children = float(c_df["Total_Children"].mean())
        tenure = float(c_df["Customer_Tenure"].mean()) if "Customer_Tenure" in c_df.columns else 0.0

        # Response rate if target exists
        resp_rate = float(c_df["Response"].mean() * 100) if "Response" in c_df.columns else 0.0

        deal_ratio = deal_purchases / (purchases if purchases > 0 else 1.0)
        web_ratio = web_purchases / (purchases if purchases > 0 else 1.0)
        catalog_ratio = catalog_purchases / (purchases if purchases > 0 else 1.0)

        stats[c] = {
            "cluster_id": c,
            "count": count,
            "pct": round(pct, 1),
            "income": round(income, 2),
            "spending": round(spending, 2),
            "purchases": round(purchases, 2),
            "recency": round(recency, 1),
            "web_purchases": round(web_purchases, 1),
            "catalog_purchases": round(catalog_purchases, 1),
            "store_purchases": round(store_purchases, 1),
            "deal_purchases": round(deal_purchases, 1),
            "deal_ratio": round(deal_ratio, 3),
            "web_ratio": round(web_ratio, 3),
            "catalog_ratio": round(catalog_ratio, 3),
            "web_visits": round(web_visits, 1),
            "children": round(children, 2),
            "tenure": round(tenure, 0),
            "response_rate": round(resp_rate, 2),
        }

    # Dynamic naming algorithm:
    # 1. Highest spending + highest income -> "High-Value Elite Shoppers"
    # 2. Highest deal ratio or deal purchases -> "Budget & Deal Seekers"
    # 3. Highest web ratio or web visits -> "Digital Convenience Shoppers"
    # 4. Remaining / lowest spending / inactive -> "Occasional / Low-Engagement Shoppers"
    assigned_names = {}
    assigned_descriptions = {}
    unassigned = set(range(n_clusters))

    # Identify High-Value
    high_value_c = max(unassigned, key=lambda c: stats[c]["spending"] * 0.7 + stats[c]["income"] * 0.01)
    assigned_names[high_value_c] = "High-Value Elite"
    assigned_descriptions[high_value_c] = (
        "Affluent customers with highest total spending, significant catalog/store purchases, "
        "and low deal sensitivity. Ideal for premium tier and VIP programs."
    )
    unassigned.remove(high_value_c)

    # Identify Deal Seekers if unassigned exists
    if unassigned:
        deal_c = max(unassigned, key=lambda c: stats[c]["deal_purchases"] + stats[c]["deal_ratio"] * 5)
        assigned_names[deal_c] = "Deal & Discount Hunters"
        assigned_descriptions[deal_c] = (
            "Value-driven customers with highest discount purchases and moderate spending. "
            "Highly responsive to promotional offers, flash sales, and bundle discounts."
        )
        unassigned.remove(deal_c)

    # Identify Digital-First if unassigned exists
    if unassigned:
        web_c = max(unassigned, key=lambda c: stats[c]["web_ratio"] * 2 + stats[c]["web_purchases"])
        assigned_names[web_c] = "Digital Convenience Shoppers"
        assigned_descriptions[web_c] = (
            "Tech-savvy customers with strong preference for online shopping and regular web visits. "
            "Best targeted through email drip campaigns and web personalization."
        )
        unassigned.remove(web_c)

    # Any remaining clusters
    for rem_c in list(unassigned):
        if stats[rem_c]["spending"] < np.median([stats[x]["spending"] for x in range(n_clusters)]):
            assigned_names[rem_c] = "Occasional / Low-Engagement"
            assigned_descriptions[rem_c] = (
                "Lower purchasing frequency and smaller basket sizes. Require re-activation campaigns "
                "or low-friction onboarding to boost loyalty."
            )
        else:
            assigned_names[rem_c] = f"Mainstream Shoppers (Segment {rem_c})"
            assigned_descriptions[rem_c] = (
                "Balanced demographic with steady store and web activity across multiple categories."
            )

    for c in range(n_clusters):
        stats[c]["name"] = assigned_names.get(c, f"Cluster {c}")
        stats[c]["description"] = assigned_descriptions.get(c, "Standard customer cohort.")

    return stats


def predict_cluster(
    customer_dict: Dict[str, Any],
    kmeans: KMeans,
    scaler: StandardScaler,
    pca: Optional[PCA] = None,
    profiles: Optional[Dict[int, Dict[str, Any]]] = None,
    features: List[str] = CLUSTERING_FEATURES
) -> Dict[str, Any]:
    """
    Predicts the cluster for a single customer vector.
    """
    row = pd.DataFrame([customer_dict])
    X_scaled = scaler.transform(row[features])
    cluster_id = int(kmeans.predict(X_scaled)[0])

    cluster_name = profiles[cluster_id]["name"] if profiles and cluster_id in profiles else f"Cluster {cluster_id}"
    description = profiles[cluster_id]["description"] if profiles and cluster_id in profiles else ""

    res = {
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "description": description,
    }

    if pca is not None:
        pca_coords = pca.transform(X_scaled)
        res["pca1"] = float(pca_coords[0, 0])
        res["pca2"] = float(pca_coords[0, 1])

    return res
