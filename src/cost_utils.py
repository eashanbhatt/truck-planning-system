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
