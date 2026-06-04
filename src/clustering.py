"""
clustering.py  —  Module 1 core logic
--------------------------------------
Two-stage clustering pipeline to build a load plan:

  Stage 1 — KMeans on destination lat/lon
            Groups the country into k geographic zones.

  Stage 2 — Agglomerative (hierarchical) clustering within each zone
            Further groups nearby shipments into candidate consolidated loads.

  Stage 3 — Greedy bin-packing
            Packs each cluster into TL trucks (max 44,000 lbs, max MAX_STOPS
            distinct destinations per truck).  Leftover OPTIONAL shipments
            (weight ≤ 15,000 lbs) remain as individual LTL moves.

Output columns added to the DataFrame
--------------------------------------
    zone_id        : KMeans zone (0 … k-1)
    cluster_id     : stage-2 cluster label  e.g. "Z2_C04"
    truck_id       : assigned truck  e.g. "TRK-001"  or  "LTL"
    assigned_mode  : "TL" or "LTL"
    truck_weight   : total weight on the assigned truck
    n_stops        : number of distinct destinations on the truck
    tl_cost        : TL truck cost (0 for LTL rows)
    ltl_cost_ind   : individual LTL cost for this shipment
    plan_cost      : actual cost charged (tl_cost / n_on_truck  for TL, ltl_cost_ind for LTL)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

from src.cost_utils import ltl_cost, tl_cost, route_distance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TL_MAX_LBS   = 44_000
LTL_MAX_LBS  = 15_000
MAX_STOPS    = 3        # max distinct delivery destinations per TL truck


# ---------------------------------------------------------------------------
# LoadConsolidator
# ---------------------------------------------------------------------------

class LoadConsolidator:
    """
    Two-stage clustering + greedy bin-packing load planner.

    Parameters
    ----------
    n_zones           : KMeans k (geographic zones)
    max_cluster_miles : intra-zone Agglomerative distance threshold (miles)
    seed              : random state for reproducibility
    """

    def __init__(
        self,
        n_zones: int = 8,
        max_cluster_miles: float = 200,
        seed: int = 42,
    ):
        self.n_zones = n_zones
        self.max_cluster_miles = max_cluster_miles
        self.seed = seed
        self._plan: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_plan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full clustering pipeline and return an annotated DataFrame.

        Parameters
        ----------
        df : output of data_loader.load_shipments()

        Returns
        -------
        pd.DataFrame with zone_id, cluster_id, truck_id, assigned_mode,
        truck_weight, n_stops, tl_cost, ltl_cost_ind, plan_cost columns added.
        """
        df = df.copy()

        # Pre-compute individual LTL cost for every shipment (used in comparison)
        df["ltl_cost_ind"] = df.apply(
            lambda r: ltl_cost(r["weight_lbs"], r["distance_miles"], r["freight_class"]),
            axis=1,
        )

        # Stage 1: zone assignment
        df = self._assign_zones(df)

        # Stage 2: hierarchical clustering within each zone
        df = self._cluster_within_zones(df)

        # Stage 3: greedy bin-packing into TL trucks
        df = self._pack_trucks(df)

        self._plan = df
        return df

    def plan_summary(self) -> pd.DataFrame:
        """Per-truck summary. Raises if build_plan() not called yet."""
        if self._plan is None:
            raise RuntimeError("Call build_plan() first.")

        df = self._plan
        tl_rows  = df[df["assigned_mode"] == "TL"]
        ltl_rows = df[df["assigned_mode"] == "LTL"]

        truck_summary = (
            tl_rows.groupby("truck_id")
            .agg(
                n_shipments   = ("shipment_id",   "count"),
                total_weight  = ("weight_lbs",    "sum"),
                n_stops       = ("dest_city",     "nunique"),
                truck_cost    = ("tl_cost",       "first"),
                ltl_baseline  = ("ltl_cost_ind",  "sum"),
            )
            .reset_index()
        )
        truck_summary["savings"] = truck_summary["ltl_baseline"] - truck_summary["truck_cost"]
        truck_summary["util_pct"] = (
            truck_summary["total_weight"] / TL_MAX_LBS * 100
        ).round(1)

        return truck_summary

    # ------------------------------------------------------------------
    # Stage 1 — KMeans zones
    # ------------------------------------------------------------------

    def _assign_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[["dest_lat", "dest_lon"]].values
        k = min(self.n_zones, len(df))
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        df["zone_id"] = km.fit_predict(coords)
        return df

    # ------------------------------------------------------------------
    # Stage 2 — Agglomerative within each zone
    # ------------------------------------------------------------------

    def _cluster_within_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        dist_thresh = self.max_cluster_miles / 69.0   # approx degrees
        parts = []
        counter = 0

        for zone_id, zone_df in df.groupby("zone_id"):
            zone_df = zone_df.copy()

            if len(zone_df) == 1:
                zone_df["cluster_id"] = f"Z{zone_id}_C{counter:03d}"
                counter += 1
                parts.append(zone_df)
                continue

            # Feature matrix: lat/lon (primary) + weight (secondary)
            scaler  = StandardScaler()
            coords  = scaler.fit_transform(zone_df[["dest_lat", "dest_lon"]].values)
            weights = StandardScaler().fit_transform(zone_df[["weight_lbs"]].values)
            features = np.hstack([coords * 1.0, weights * 0.15])

            agg    = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_thresh,
                linkage="ward",
            )
            labels = agg.fit_predict(features)
            zone_df["_local"] = labels

            for local_id in np.unique(labels):
                mask = zone_df["_local"] == local_id
                zone_df.loc[mask, "cluster_id"] = f"Z{zone_id}_C{counter:03d}"
                counter += 1

            zone_df.drop(columns=["_local"], inplace=True)
            parts.append(zone_df)

        return pd.concat(parts).sort_index()

    # ------------------------------------------------------------------
    # Stage 3 — Greedy bin-packing into TL trucks
    # ------------------------------------------------------------------

    def _pack_trucks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each cluster, greedily assign shipments to TL trucks.

        Rules
        -----
        - MUST_TL shipments (weight > LTL_MAX_LBS) are packed first.
        - OPTIONAL shipments are added to open trucks if they fit.
        - A truck closes when: weight > TL_MAX_LBS OR distinct destinations > MAX_STOPS.
        - Remaining OPTIONAL shipments ship as individual LTL.
        """
        wh_lat = df["warehouse_lat"].iloc[0]
        wh_lon = df["warehouse_lon"].iloc[0]

        df = df.copy()
        df["truck_id"]      = "LTL"
        df["assigned_mode"] = "LTL"
        df["tl_cost"]       = 0.0
        df["truck_weight"]  = 0.0
        df["n_stops"]       = 0

        truck_counter = 1
        all_assignments: Dict[str, dict] = {}   # truck_id → {weight, stops, indices}

        for cluster_id, cluster_df in df.groupby("cluster_id"):
            # Sort: MUST_TL first (heaviest first), then OPTIONAL
            must_tl  = cluster_df[cluster_df["mode"] == "MUST_TL"].sort_values(
                "weight_lbs", ascending=False
            )
            optional = cluster_df[cluster_df["mode"] == "OPTIONAL"].sort_values(
                "weight_lbs", ascending=False
            )
            ordered  = pd.concat([must_tl, optional])

            current_truck  = None   # truck_id string
            current_weight = 0.0
            current_stops:  set = set()
            current_indices: List[int] = []

            def _close_truck():
                nonlocal current_truck, current_weight, current_stops, current_indices
                if current_indices:
                    stops_list = [
                        (df.at[i, "dest_lat"], df.at[i, "dest_lon"])
                        for i in set(
                            df.loc[current_indices, ["dest_lat", "dest_lon"]]
                            .drop_duplicates()
                            .apply(tuple, axis=1)
                        )
                    ]
                    route_mi = route_distance(wh_lat, wh_lon, stops_list)
                    truck_cost = tl_cost(route_mi)
                    all_assignments[current_truck] = {
                        "weight": current_weight,
                        "stops":  len(current_stops),
                        "cost":   truck_cost,
                        "indices": list(current_indices),
                    }
                current_truck  = None
                current_weight = 0.0
                current_stops  = set()
                current_indices = []

            for idx, row in ordered.iterrows():
                dest_key = (row["dest_city"], row["dest_state"])
                new_weight = current_weight + row["weight_lbs"]
                new_stops  = current_stops | {dest_key}

                fits = (new_weight <= TL_MAX_LBS) and (len(new_stops) <= MAX_STOPS)

                if current_truck is None:
                    # Open a new truck regardless (MUST_TL might need its own)
                    current_truck  = f"TRK-{truck_counter:03d}"
                    truck_counter += 1
                    current_weight = row["weight_lbs"]
                    current_stops  = {dest_key}
                    current_indices = [idx]

                    # If this single shipment already exceeds TL cap → flag warning
                    if row["weight_lbs"] > TL_MAX_LBS:
                        print(f"  ⚠  {row['shipment_id']} ({row['weight_lbs']:,.0f} lbs) "
                              f"exceeds TL capacity — shipping oversize.")

                elif fits:
                    current_weight += row["weight_lbs"]
                    current_stops.add(dest_key)
                    current_indices.append(idx)

                else:
                    _close_truck()
                    # Start fresh truck for this shipment
                    current_truck  = f"TRK-{truck_counter:03d}"
                    truck_counter += 1
                    current_weight = row["weight_lbs"]
                    current_stops  = {dest_key}
                    current_indices = [idx]

            _close_truck()   # close final open truck

        # Apply assignments back to df
        for truck_id, info in all_assignments.items():
            idxs = info["indices"]
            # Only assign to TL if at least one MUST_TL or total weight makes TL worthwhile
            total_w  = info["weight"]
            cost_tl  = info["cost"]
            cost_ltl = df.loc[idxs, "ltl_cost_ind"].sum()

            if cost_tl < cost_ltl or any(df.loc[idxs, "mode"] == "MUST_TL"):
                df.loc[idxs, "truck_id"]      = truck_id
                df.loc[idxs, "assigned_mode"] = "TL"
                df.loc[idxs, "tl_cost"]       = cost_tl
                df.loc[idxs, "truck_weight"]  = total_w
                df.loc[idxs, "n_stops"]       = info["stops"]
            # else: leave as LTL

        # Compute final plan_cost per row
        df["plan_cost"] = df.apply(
            lambda r: (
                r["tl_cost"] / max(1, (df["truck_id"] == r["truck_id"]).sum())
                if r["assigned_mode"] == "TL"
                else r["ltl_cost_ind"]
            ),
            axis=1,
        ).round(2)

        return df
