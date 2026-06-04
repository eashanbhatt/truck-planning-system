"""
clustering/src/bin_packer.py
------------------------------
Greedy bin-packing: takes the geographic clusters from haversine_cluster.py
and packs them into TL trucks.

Rules
-----
- MUST_TL shipments (weight > 15,000 lbs) are always placed on a TL truck.
- OPTIONAL shipments (≤ 15,000 lbs) are added to TL trucks if they fit AND
  if doing so is cheaper than shipping them individually as LTL.
- A truck is "full" when:
    total weight  > TL_MAX_LBS  (44,000 lbs), OR
    distinct destinations > MAX_STOPS (3 stops)
- Each cluster is packed independently — no cross-cluster truck sharing.

Cost model (used only for TL-vs-LTL decision)
----------------------------------------------
TL  : $150 base  +  $3.25 / mile  (nearest-neighbour route from warehouse)
LTL : (weight / 100) × CWT_rate(distance, freight_class)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

TL_MAX_LBS   = 44_000
LTL_MAX_LBS  = 15_000
MAX_STOPS    = 3
TL_BASE      = 150.0
TL_PER_MILE  = 3.25

LTL_CWT_TABLE = [
    (0,    300,  22.0),
    (300,  700,  32.0),
    (700,  1200, 44.0),
    (1200, 2000, 56.0),
    (2000, 9999, 68.0),
]

FC_MULT = {
    50: 0.80, 65: 0.88, 70: 0.93, 77.5: 0.97,
    85: 1.00, 92.5: 1.04, 100: 1.10, 110: 1.18,
    125: 1.28, 150: 1.40,
}


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 3_959.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def _route_miles(wh_lat, wh_lon, stops: List[Tuple]) -> float:
    """Greedy nearest-neighbour route distance (warehouse → each stop)."""
    if not stops:
        return 0.0
    remaining = list(stops)
    cur = (wh_lat, wh_lon)
    total = 0.0
    while remaining:
        dists = [_haversine(cur[0], cur[1], s[0], s[1]) for s in remaining]
        idx = int(np.argmin(dists))
        total += dists[idx]
        cur = remaining.pop(idx)
    return round(total, 1)


def _tl_cost(route_miles: float) -> float:
    return round(TL_BASE + TL_PER_MILE * route_miles, 2)


def _ltl_cost(weight_lbs: float, dist_miles: float, freight_class: float) -> float:
    base_cwt = LTL_CWT_TABLE[-1][2]
    for lo, hi, rate in LTL_CWT_TABLE:
        if lo <= dist_miles < hi:
            base_cwt = rate
            break
    mult = FC_MULT.get(float(freight_class), 1.0)
    return round((weight_lbs / 100.0) * base_cwt * mult, 2)


# ---------------------------------------------------------------------------
# Bin-packer
# ---------------------------------------------------------------------------

def pack(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pack each haversine cluster into TL trucks.

    Parameters
    ----------
    df : DataFrame from haversine_cluster.build_clusters(),
         must include warehouse_lat / warehouse_lon.

    Returns
    -------
    df with new columns:
        ltl_cost_each  : cost if this shipment ships individually as LTL
        truck_id       : "TRK-001" … or "LTL"
        assigned_mode  : "TL" or "LTL"
        tl_cost        : cost of the TL truck assigned (0 for LTL rows)
        truck_weight   : total weight on assigned truck
        truck_n_stops  : distinct destinations on assigned truck
        plan_cost      : effective cost allocated to this shipment
    """
    df = df.copy()
    wh_lat = df["warehouse_lat"].iloc[0]
    wh_lon = df["warehouse_lon"].iloc[0]

    # Pre-compute individual LTL cost for every shipment
    df["ltl_cost_each"] = df.apply(
        lambda r: _ltl_cost(r["weight_lbs"], r["wh_to_dest_mi"], r["freight_class"]),
        axis=1,
    )

    # Initialise output columns
    df["truck_id"]      = "LTL"
    df["assigned_mode"] = "LTL"
    df["tl_cost"]       = 0.0
    df["truck_weight"]  = 0.0
    df["truck_n_stops"] = 0
    df["plan_cost"]     = df["ltl_cost_each"]

    truck_num = 1

    for cluster_id, cluster_df in df.groupby("cluster_id"):
        # Sort: MUST_TL first (heaviest first), then OPTIONAL (heaviest first)
        must_tl  = cluster_df[cluster_df["mode_flag"] == "MUST_TL"].sort_values(
            "weight_lbs", ascending=False
        )
        optional = cluster_df[cluster_df["mode_flag"] == "OPTIONAL"].sort_values(
            "weight_lbs", ascending=False
        )
        ordered = pd.concat([must_tl, optional])

        # Greedy bin-packing state
        truck_weight:  float = 0.0
        truck_stops:   set   = set()
        truck_indices: list  = []

        def _flush_truck():
            nonlocal truck_num, truck_weight, truck_stops, truck_indices
            if not truck_indices:
                return

            stops_coords = list(
                df.loc[truck_indices, ["dest_lat", "dest_lon"]]
                .drop_duplicates()
                .apply(tuple, axis=1)
            )
            route_mi   = _route_miles(wh_lat, wh_lon, stops_coords)
            cost_tl    = _tl_cost(route_mi)
            cost_ltl   = df.loc[truck_indices, "ltl_cost_each"].sum()

            # Only assign TL if cheaper than all-LTL or contains a MUST_TL
            has_must_tl = any(df.loc[i, "mode_flag"] == "MUST_TL" for i in truck_indices)
            if cost_tl < cost_ltl or has_must_tl:
                tid = f"TRK-{truck_num:03d}"
                truck_num += 1
                for i in truck_indices:
                    df.at[i, "truck_id"]      = tid
                    df.at[i, "assigned_mode"] = "TL"
                    df.at[i, "tl_cost"]       = cost_tl
                    df.at[i, "truck_weight"]  = truck_weight
                    df.at[i, "truck_n_stops"] = len(truck_stops)
            # else: leave as LTL (defaults already set)

            truck_weight  = 0.0
            truck_stops   = set()
            truck_indices = []

        for idx, row in ordered.iterrows():
            dest_key   = (row["dest_city"], row["dest_state"])
            new_weight = truck_weight + row["weight_lbs"]
            new_stops  = truck_stops | {dest_key}

            fits = new_weight <= TL_MAX_LBS and len(new_stops) <= MAX_STOPS

            if not truck_indices:
                # Open new truck for this shipment
                truck_weight  = row["weight_lbs"]
                truck_stops   = {dest_key}
                truck_indices = [idx]
            elif fits:
                truck_weight  = new_weight
                truck_stops   = new_stops
                truck_indices.append(idx)
            else:
                _flush_truck()
                truck_weight  = row["weight_lbs"]
                truck_stops   = {dest_key}
                truck_indices = [idx]

        _flush_truck()   # close the last open truck

    # Compute plan_cost (TL cost split evenly among shipments on each truck)
    truck_counts = df[df["assigned_mode"] == "TL"].groupby("truck_id").size()
    df["plan_cost"] = df.apply(
        lambda r: (
            round(r["tl_cost"] / max(1, truck_counts.get(r["truck_id"], 1)), 2)
            if r["assigned_mode"] == "TL"
            else r["ltl_cost_each"]
        ),
        axis=1,
    )

    return df
