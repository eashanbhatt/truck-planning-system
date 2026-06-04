"""
clustering.py
-------------
Two-stage clustering pipeline for shipment consolidation:

  Stage 1 — KMeans on destination lat/lon
            Partitions the country into geographic delivery zones (default k=8).

  Stage 2 — Agglomerative (hierarchical) clustering within each zone
            Groups nearby shipments that share a similar pickup window,
            producing candidate consolidated loads.

Usage
-----
    from src.clustering import LoadConsolidator

    lc = LoadConsolidator(n_zones=8, max_cluster_miles=150)
    clustered_df = lc.fit(shipments_df)
    summary = lc.cluster_summary()
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TL_WEIGHT_THRESHOLD = 15_000   # lbs — above this, TL is typically cheaper than LTL
TL_WEIGHT_CAPACITY  = 44_000   # lbs — legal max weight for a standard 53-ft trailer
TL_VOLUME_CAPACITY  = 2_400    # cuft — 53 ft × 8.5 ft × 9 ft (approx)


# ---------------------------------------------------------------------------
# LoadConsolidator
# ---------------------------------------------------------------------------

class LoadConsolidator:
    """
    Clusters shipments into candidate consolidated loads.

    Parameters
    ----------
    n_zones : int
        Number of geographic destination zones (KMeans k).
    max_cluster_miles : float
        Maximum radius (miles) for intra-zone Agglomerative clusters.
        Converted to degrees for the distance threshold (~1° ≈ 69 miles).
    min_cluster_size : int
        Minimum shipments required to form a cluster (singleton → stays LTL).
    seed : int
        Random state for KMeans reproducibility.
    """

    def __init__(
        self,
        n_zones: int = 8,
        max_cluster_miles: float = 150,
        min_cluster_size: int = 2,
        seed: int = 42,
    ):
        self.n_zones = n_zones
        self.max_cluster_miles = max_cluster_miles
        self.distance_threshold = max_cluster_miles / 69.0   # degrees
        self.min_cluster_size = min_cluster_size
        self.seed = seed

        self._kmeans: Optional[KMeans] = None
        self._clustered_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, shipments: pd.DataFrame) -> pd.DataFrame:
        """
        Run both clustering stages and return an annotated DataFrame.

        New columns added
        -----------------
        zone_id        : KMeans zone (0 … n_zones-1)
        cluster_id     : global unique cluster label  (e.g. "Z2_C04")
        cluster_weight : total weight of all shipments in this cluster
        cluster_volume : total volume of all shipments in this cluster
        is_tl_eligible : bool — cluster weight exceeds TL threshold
        """
        df = shipments.copy()

        # Stage 1: geographic zone assignment
        df = self._assign_zones(df)

        # Stage 2: hierarchical clustering within each zone
        df = self._cluster_within_zones(df)

        # Annotate with cluster-level aggregates
        df = self._annotate_clusters(df)

        self._clustered_df = df
        return df

    def cluster_summary(self) -> pd.DataFrame:
        """
        Return a per-cluster summary DataFrame.
        Raises RuntimeError if fit() has not been called.
        """
        if self._clustered_df is None:
            raise RuntimeError("Call fit() before cluster_summary().")

        df = self._clustered_df
        grp = df.groupby("cluster_id").agg(
            n_shipments    = ("shipment_id",       "count"),
            total_weight   = ("weight_lbs",        "sum"),
            total_volume   = ("volume_cuft",       "sum"),
            avg_distance   = ("distance_miles",    "mean"),
            baseline_cost  = ("ltl_cost_baseline", "sum"),
            region         = ("region",            lambda x: x.mode()[0]),
            pickup_date    = ("pickup_date",       "min"),
        ).reset_index()

        grp["is_tl_eligible"] = grp["total_weight"] >= TL_WEIGHT_THRESHOLD
        grp["utilisation_pct"] = (grp["total_weight"] / TL_WEIGHT_CAPACITY * 100).round(1)
        return grp.sort_values("total_weight", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """KMeans on destination lat/lon → zone_id column."""
        coords = df[["dest_lat", "dest_lon"]].values
        km = KMeans(n_clusters=self.n_zones, random_state=self.seed, n_init=10)
        df["zone_id"] = km.fit_predict(coords)
        self._kmeans = km
        return df

    def _cluster_within_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Within each KMeans zone, run Agglomerative clustering.
        Features: dest_lat, dest_lon, pickup_date (ordinal), weight.
        """
        all_parts = []
        global_cluster_counter = 0

        for zone_id, zone_df in df.groupby("zone_id"):
            if len(zone_df) == 1:
                # Singleton zone — no consolidation possible
                zone_df = zone_df.copy()
                zone_df["_local_cluster"] = 0
                zone_df["cluster_id"] = f"Z{zone_id}_C{global_cluster_counter:03d}"
                global_cluster_counter += 1
                all_parts.append(zone_df)
                continue

            # Build feature matrix
            scaler = StandardScaler()
            pickup_ordinal = pd.to_datetime(zone_df["pickup_date"]).map(
                lambda d: d.toordinal()
            ).values.reshape(-1, 1)

            coords = zone_df[["dest_lat", "dest_lon"]].values
            weight = zone_df[["weight_lbs"]].values

            # Scale features: coordinates dominate, pickup date secondary
            features = np.hstack([
                scaler.fit_transform(coords) * 1.0,
                scaler.fit_transform(pickup_ordinal) * 0.3,
                scaler.fit_transform(weight) * 0.2,
            ])

            agg = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                linkage="ward",
            )
            local_labels = agg.fit_predict(features)

            zone_df = zone_df.copy()
            zone_df["_local_cluster"] = local_labels

            # Map local labels → global cluster IDs
            for local_id in np.unique(local_labels):
                mask = zone_df["_local_cluster"] == local_id
                sub  = zone_df[mask]

                if len(sub) < self.min_cluster_size:
                    # Too small to consolidate — keep as individual LTL
                    for idx in sub.index:
                        zone_df.loc[idx, "cluster_id"] = (
                            f"Z{zone_id}_C{global_cluster_counter:03d}"
                        )
                        global_cluster_counter += 1
                else:
                    cid = f"Z{zone_id}_C{global_cluster_counter:03d}"
                    zone_df.loc[mask, "cluster_id"] = cid
                    global_cluster_counter += 1

            all_parts.append(zone_df)

        result = pd.concat(all_parts).sort_index()
        result.drop(columns=["_local_cluster"], errors="ignore", inplace=True)
        return result

    def _annotate_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cluster-level weight / volume / eligibility back to each row."""
        agg = df.groupby("cluster_id").agg(
            cluster_weight=("weight_lbs",  "sum"),
            cluster_volume=("volume_cuft", "sum"),
        )
        df = df.join(agg, on="cluster_id")
        df["is_tl_eligible"] = df["cluster_weight"] >= TL_WEIGHT_THRESHOLD
        return df
