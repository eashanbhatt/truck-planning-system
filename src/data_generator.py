"""
data_generator.py
-----------------
Generates synthetic shipment data modelled after a Midwest-based print/publishing
distribution operation (similar to LSC Communications).

Each row represents one shipment with:
  - Origin: one of three Midwest distribution centres
  - Destination: one of 25 US cities across 6 regions
  - Weight (lbs), Volume (cuft), Freight class
  - Pickup date and delivery due date
  - Priority flag
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import random

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

DISTRIBUTION_CENTERS = [
    {"city": "Chicago, IL",      "lat": 41.8781, "lon": -87.6298},
    {"city": "Indianapolis, IN", "lat": 39.7684, "lon": -86.1581},
    {"city": "Columbus, OH",     "lat": 39.9612, "lon": -82.9988},
]

DESTINATION_CITIES = [
    # Northeast
    {"city": "New York, NY",      "lat": 40.7128, "lon": -74.0060, "region": "Northeast"},
    {"city": "Boston, MA",        "lat": 42.3601, "lon": -71.0589, "region": "Northeast"},
    {"city": "Philadelphia, PA",  "lat": 39.9526, "lon": -75.1652, "region": "Northeast"},
    {"city": "Baltimore, MD",     "lat": 39.2904, "lon": -76.6122, "region": "Northeast"},
    {"city": "Washington, DC",    "lat": 38.9072, "lon": -77.0369, "region": "Northeast"},
    # Southeast
    {"city": "Atlanta, GA",       "lat": 33.7490, "lon": -84.3880, "region": "Southeast"},
    {"city": "Charlotte, NC",     "lat": 35.2271, "lon": -80.8431, "region": "Southeast"},
    {"city": "Miami, FL",         "lat": 25.7617, "lon": -80.1918, "region": "Southeast"},
    {"city": "Tampa, FL",         "lat": 27.9506, "lon": -82.4572, "region": "Southeast"},
    {"city": "Nashville, TN",     "lat": 36.1627, "lon": -86.7816, "region": "Southeast"},
    # South Central
    {"city": "Dallas, TX",        "lat": 32.7767, "lon": -96.7970, "region": "South Central"},
    {"city": "Houston, TX",       "lat": 29.7604, "lon": -95.3698, "region": "South Central"},
    {"city": "Austin, TX",        "lat": 30.2672, "lon": -97.7431, "region": "South Central"},
    # Mountain
    {"city": "Denver, CO",        "lat": 39.7392, "lon": -104.9903, "region": "Mountain"},
    {"city": "Phoenix, AZ",       "lat": 33.4484, "lon": -112.0740, "region": "Mountain"},
    {"city": "Las Vegas, NV",     "lat": 36.1699, "lon": -115.1398, "region": "Mountain"},
    # Pacific
    {"city": "Los Angeles, CA",   "lat": 34.0522, "lon": -118.2437, "region": "Pacific"},
    {"city": "San Francisco, CA", "lat": 37.7749, "lon": -122.4194, "region": "Pacific"},
    {"city": "Seattle, WA",       "lat": 47.6062, "lon": -122.3321, "region": "Pacific"},
    {"city": "Portland, OR",      "lat": 45.5051, "lon": -122.6750, "region": "Pacific"},
    # Midwest
    {"city": "Minneapolis, MN",   "lat": 44.9778, "lon": -93.2650, "region": "Midwest"},
    {"city": "Detroit, MI",       "lat": 42.3314, "lon": -83.0458, "region": "Midwest"},
    {"city": "Cleveland, OH",     "lat": 41.4993, "lon": -81.6944, "region": "Midwest"},
    {"city": "St. Louis, MO",     "lat": 38.6270, "lon": -90.1994, "region": "Midwest"},
    {"city": "Kansas City, MO",   "lat": 39.0997, "lon": -94.5786, "region": "Midwest"},
]

# NMFC freight classes — lower class = denser/heavier freight (cheaper to ship)
FREIGHT_CLASSES = [50, 65, 70, 77.5, 85, 92.5, 100, 110, 125, 150]


# ---------------------------------------------------------------------------
# Haversine helper (no external dependency)
# ---------------------------------------------------------------------------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in miles."""
    R = 3_959.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_shipments(n_shipments: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate n_shipments synthetic freight records.

    Weight distribution
    -------------------
    ~30 % TL-eligible (10 000 – 44 000 lbs) — large pallet loads
    ~70 % LTL-sized   (100 – 10 000 lbs)    — partial loads

    Returns
    -------
    pd.DataFrame with columns:
        shipment_id, origin_city, origin_lat, origin_lon,
        dest_city, dest_lat, dest_lon, weight_lbs, volume_cuft,
        freight_class, distance_miles, pickup_date, delivery_due,
        priority, region, ltl_cost_baseline
    """
    np.random.seed(seed)
    random.seed(seed)

    base_date = date(2023, 10, 2)   # Monday
    records = []

    for i in range(n_shipments):
        origin = random.choice(DISTRIBUTION_CENTERS)
        dest   = random.choice(DESTINATION_CITIES)

        # Weight
        if np.random.random() < 0.30:
            weight = round(np.random.uniform(10_000, 44_000), 1)
        else:
            weight = round(np.random.uniform(100, 10_000), 1)

        # Freight class + derived volume
        fc = random.choice(FREIGHT_CLASSES[:8])
        # Density (lbs/cuft) is inversely related to freight class
        density = max(1.0, 50.0 / (fc / 50.0))
        volume  = round(min(weight / density, 2_400), 1)

        # Dates
        pickup_day  = base_date + timedelta(days=int(np.random.randint(0, 5)))
        dist_miles  = haversine_miles(origin["lat"], origin["lon"],
                                      dest["lat"],   dest["lon"])
        transit     = max(1, int(dist_miles / 500))
        delivery_due = pickup_day + timedelta(days=transit + int(np.random.randint(0, 2)))

        # Baseline LTL cost (what the old 3PL tool would charge)
        # Rate = f(distance, weight, freight class) — simplified tariff
        ltl_rate_per_cwt = 20 + (dist_miles / 100) * 1.5 + (fc - 50) * 0.05
        ltl_cost = round((weight / 100) * ltl_rate_per_cwt, 2)

        records.append({
            "shipment_id":       f"SHP{i+1:04d}",
            "origin_city":       origin["city"],
            "origin_lat":        round(origin["lat"] + np.random.normal(0, 0.05), 4),
            "origin_lon":        round(origin["lon"] + np.random.normal(0, 0.05), 4),
            "dest_city":         dest["city"],
            "dest_lat":          round(dest["lat"]   + np.random.normal(0, 0.10), 4),
            "dest_lon":          round(dest["lon"]   + np.random.normal(0, 0.10), 4),
            "weight_lbs":        weight,
            "volume_cuft":       volume,
            "freight_class":     fc,
            "distance_miles":    round(dist_miles, 1),
            "pickup_date":       pickup_day,
            "delivery_due":      delivery_due,
            "priority":          "expedited" if np.random.random() < 0.15 else "standard",
            "region":            dest["region"],
            "ltl_cost_baseline": ltl_cost,
        })

    return pd.DataFrame(records)
