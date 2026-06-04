"""
cost_utils.py
-------------
Freight cost calculations and routing helpers shared by both modules.

Cost model
----------
TL  : $150 base + $3.25 per mile  (flat rate, one-way route from warehouse)
LTL : (weight_lbs / 100) × CWT_rate
      CWT_rate = f(distance, freight_class)  — simplified tariff schedule

Route distance (TL with multiple stops)
----------------------------------------
Uses a greedy nearest-neighbour heuristic:
    warehouse → nearest unvisited stop → next nearest → … → last stop
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Rate constants
# ---------------------------------------------------------------------------

TL_BASE_COST      = 150.0   # $ fixed per truck (fuel surcharge + accessorial)
TL_RATE_PER_MILE  = 3.25    # $/mile

# LTL CWT (per-hundred-weight) rate table  ─  simplified by distance band
# Actual carriers use NMFC class × distance matrices; this is representative.
LTL_CWT_BANDS = [
    (0,    300,  22.0),   # (min_miles, max_miles, base_cwt_rate)
    (300,  700,  32.0),
    (700,  1200, 44.0),
    (1200, 2000, 56.0),
    (2000, 9999, 68.0),
]

# Freight class multiplier on top of base CWT rate
FC_MULTIPLIER = {
    50:  0.80, 65:  0.88, 70:  0.93, 77.5: 0.97,
    85:  1.00, 92.5:1.04, 100: 1.10, 110:  1.18,
    125: 1.28, 150: 1.40, 175: 1.55, 200:  1.70,
    250: 1.90, 300: 2.10, 400: 2.50, 500:  3.00,
}


# ---------------------------------------------------------------------------
# Haversine (also available in data_loader, kept here for independence)
# ---------------------------------------------------------------------------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3_959.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# LTL cost
# ---------------------------------------------------------------------------

def ltl_cost(weight_lbs: float, distance_miles: float, freight_class: float) -> float:
    """
    Calculate LTL freight cost for a single shipment.

    Parameters
    ----------
    weight_lbs    : shipment weight
    distance_miles: warehouse-to-destination distance
    freight_class : NMFC freight class

    Returns
    -------
    float : cost in USD
    """
    # Base CWT rate from distance band
    base_cwt = LTL_CWT_BANDS[-1][2]  # default: longest band
    for lo, hi, rate in LTL_CWT_BANDS:
        if lo <= distance_miles < hi:
            base_cwt = rate
            break

    # Apply freight class multiplier (default to 1.0 if class not in table)
    fc_mult = FC_MULTIPLIER.get(float(freight_class), 1.0)
    cwt_rate = base_cwt * fc_mult

    return round((weight_lbs / 100.0) * cwt_rate, 2)


# ---------------------------------------------------------------------------
# TL cost + routing
# ---------------------------------------------------------------------------

def route_distance(
    warehouse_lat: float,
    warehouse_lon: float,
    stops: List[Tuple[float, float]],
) -> float:
    """
    Greedy nearest-neighbour route: warehouse → stops (return not included).

    Parameters
    ----------
    warehouse_lat, warehouse_lon : origin coords
    stops : list of (lat, lon) tuples for each delivery stop

    Returns
    -------
    float : total one-way route distance in miles
    """
    if not stops:
        return 0.0

    visited  = []
    remaining = list(stops)
    current   = (warehouse_lat, warehouse_lon)
    total     = 0.0

    while remaining:
        dists = [haversine_miles(current[0], current[1], s[0], s[1]) for s in remaining]
        idx   = int(np.argmin(dists))
        total += dists[idx]
        current = remaining.pop(idx)
        visited.append(current)

    return round(total, 1)


def tl_cost(route_miles: float) -> float:
    """TL flat-rate cost for a given route distance."""
    return round(TL_BASE_COST + TL_RATE_PER_MILE * route_miles, 2)


# ---------------------------------------------------------------------------
# Baseline cost (all-LTL)
# ---------------------------------------------------------------------------

def baseline_ltl_cost(df: pd.DataFrame) -> float:
    """Total cost if every shipment shipped individually as LTL."""
    return df.apply(
        lambda r: ltl_cost(r["weight_lbs"], r["distance_miles"], r["freight_class"]),
        axis=1,
    ).sum()


# ---------------------------------------------------------------------------
# Savings summary CSV writer
# ---------------------------------------------------------------------------

def save_savings_summary(plan: pd.DataFrame, output_path: str, module: str) -> None:
    """
    Write a tiered savings summary CSV with three sections:
      OVERALL  — single row with full-plan totals
      REGION   — one row per US region
      TRUCK    — one row per TL truck
      LTL      — one row for all LTL shipments combined

    Parameters
    ----------
    plan        : annotated DataFrame from build_plan() (either module)
    output_path : file path for the output CSV
    module      : label string, e.g. "Clustering" or "MILP Optimizer"
    """
    rows = []

    def _row(section, label, sub_df):
        n         = len(sub_df)
        weight    = sub_df["weight_lbs"].sum()
        baseline  = sub_df["ltl_cost_ind"].sum()
        optimized = sub_df["plan_cost"].sum()
        savings   = baseline - optimized
        pct       = round(savings / baseline * 100, 1) if baseline else 0.0
        n_tl      = (sub_df["assigned_mode"] == "TL").sum()
        n_ltl     = (sub_df["assigned_mode"] == "LTL").sum()
        return {
            "section":            section,
            "label":              label,
            "module":             module,
            "shipments":          n,
            "tl_shipments":       n_tl,
            "ltl_shipments":      n_ltl,
            "total_weight_lbs":   round(weight, 0),
            "baseline_ltl_cost":  round(baseline, 2),
            "optimized_cost":     round(optimized, 2),
            "savings":            round(savings, 2),
            "savings_pct":        pct,
        }

    # OVERALL
    rows.append(_row("OVERALL", "All Shipments", plan))

    # Per REGION
    for region, grp in plan.groupby("region"):
        rows.append(_row("REGION", region, grp))

    # Per TL TRUCK
    tl_plan = plan[plan["assigned_mode"] == "TL"]
    for truck_id, grp in tl_plan.groupby("truck_id"):
        rows.append(_row("TRUCK", truck_id, grp))

    # LTL aggregate
    ltl_plan = plan[plan["assigned_mode"] == "LTL"]
    if len(ltl_plan):
        rows.append(_row("LTL", "All LTL Shipments", ltl_plan))

    summary_df = pd.DataFrame(rows)

    # Format cost columns as currency strings for readability
    for col in ["baseline_ltl_cost", "optimized_cost", "savings"]:
        summary_df[col] = summary_df[col].map("${:,.2f}".format)
    summary_df["savings_pct"] = summary_df["savings_pct"].map("{}%".format)

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"  Savings summary saved → {output_path}")
