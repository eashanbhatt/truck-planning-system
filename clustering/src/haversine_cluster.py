"""
clustering/src/haversine_cluster.py
-------------------------------------
Builds geographic shipment clusters using a haversine pairwise distance
matrix fed directly into AgglomerativeClustering.

Why haversine + precomputed matrix?
------------------------------------
KMeans uses Euclidean distance on lat/lon coordinates, which distorts
real-world geography (1° longitude ≠ 1° latitude in miles).  By computing
a true great-circle distance matrix first, the clustering reflects actual
driving/shipping proximity between destinations.

Algorithm
---------
1.  Compute n×n haversine distance matrix D[i,j] in miles.
2.  Feed D into AgglomerativeClustering(metric='precomputed',
    linkage='complete').
    - 'complete' linkage = max pairwise distance within a cluster.
    - This guarantees every pair of destinations inside a cluster is
      within `distance_threshold_miles` of each other.
3.  Attach cluster labels back to the DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Haversine distance matrix (vectorized)
# ---------------------------------------------------------------------------

def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Vectorized pairwise haversine distance matrix.

    Parameters
    ----------
    lats : 1-D array of latitudes  (degrees)
    lons : 1-D array of longitudes (degrees)

    Returns
    -------
    np.ndarray of shape (n, n) — distances in miles
    """
    R    = 3_959.0
    lats = np.radians(lats)
    lons = np.radians(lons)

    # Broadcast pairwise differences
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
    )
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return dist


# ---------------------------------------------------------------------------
# Cluster builder
# ---------------------------------------------------------------------------

def build_clusters(
    df: pd.DataFrame,
    distance_threshold_miles: float = 200.0,
) -> pd.DataFrame:
    """
    Group shipments into geographic clusters via haversine distance matrix.

    Shipments whose destinations are within `distance_threshold_miles` of
    every other shipment in the same cluster are placed together.

    Parameters
    ----------
    df                      : DataFrame from data_loader.load()
    distance_threshold_miles: max intra-cluster destination distance in miles

    Returns
    -------
    df with two new columns:
        cluster_id   : integer cluster label (0-based)
        cluster_size : number of shipments in this cluster
    """
    lats = df["dest_lat"].values
    lons = df["dest_lon"].values

    # Step 1 — haversine distance matrix
    dist_matrix = haversine_matrix(lats, lons)

    # Step 2 — agglomerative clustering on precomputed distances
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=distance_threshold_miles,
    )
    labels = model.fit_predict(dist_matrix)

    df = df.copy()
    df["cluster_id"] = labels

    # Step 3 — attach cluster size and centroid info
    cluster_sizes = df.groupby("cluster_id")["shipment_id"].transform("count")
    df["cluster_size"] = cluster_sizes

    return df


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-cluster summary: centroid, total weight, destinations, etc.
    Useful for inspecting what the clustering produced before bin-packing.
    """
    grp = (
        df.groupby("cluster_id")
        .agg(
            n_shipments   = ("shipment_id",   "count"),
            total_weight  = ("weight_lbs",    "sum"),
            n_destinations= ("dest_city",     "nunique"),
            must_tl_count = ("mode_flag",     lambda x: (x == "MUST_TL").sum()),
            centroid_lat  = ("dest_lat",      "mean"),
            centroid_lon  = ("dest_lon",      "mean"),
            cities        = ("dest_city",     lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values("total_weight", ascending=False)
        .reset_index(drop=True)
    )

    # Max intra-cluster haversine distance (radius of the cluster)
    def _max_dist(sub):
        lats = sub["dest_lat"].values
        lons = sub["dest_lon"].values
        if len(sub) < 2:
            return 0.0
        m = haversine_matrix(lats, lons)
        return round(float(m.max()), 1)

    grp["max_intra_dist_mi"] = grp["cluster_id"].apply(
        lambda cid: _max_dist(df[df["cluster_id"] == cid])
    )
    return grp
