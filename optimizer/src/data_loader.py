"""
optimizer/src/data_loader.py
------------------------------
Standalone CSV loader for the linear optimizer solution.
No imports from the clustering module.

Expected CSV columns
---------------------
    shipment_id, warehouse, dest_city, dest_state,
    dest_lat, dest_lon, weight_lbs, freight_class,
    pickup_date, delivery_due
"""

import numpy as np
import pandas as pd

WAREHOUSES = {
    "CHICAGO_IL":      {"lat": 41.8781, "lon": -87.6298, "city": "Chicago, IL"},
    "INDIANAPOLIS_IN": {"lat": 39.7684, "lon": -86.1581, "city": "Indianapolis, IN"},
    "COLUMBUS_OH":     {"lat": 39.9612, "lon": -82.9988, "city": "Columbus, OH"},
    "DALLAS_TX":       {"lat": 32.7767, "lon": -96.7970, "city": "Dallas, TX"},
    "LOS_ANGELES_CA":  {"lat": 34.0522, "lon": -118.2437, "city": "Los Angeles, CA"},
    "ATLANTA_GA":      {"lat": 33.7490, "lon": -84.3880, "city": "Atlanta, GA"},
}

REGION_MAP = {
    "ME": "Northeast", "NH": "Northeast", "VT": "Northeast", "MA": "Northeast",
    "RI": "Northeast", "CT": "Northeast", "NY": "Northeast", "NJ": "Northeast",
    "PA": "Northeast", "DE": "Northeast", "MD": "Northeast", "DC": "Northeast",
    "VA": "Southeast", "WV": "Southeast", "NC": "Southeast", "SC": "Southeast",
    "GA": "Southeast", "FL": "Southeast", "AL": "Southeast", "MS": "Southeast",
    "TN": "Southeast", "KY": "Southeast", "AR": "Southeast",
    "TX": "South Central", "OK": "South Central", "LA": "South Central",
    "NM": "South Central",
    "OH": "Midwest", "IN": "Midwest", "IL": "Midwest", "MI": "Midwest",
    "WI": "Midwest", "MN": "Midwest", "IA": "Midwest", "MO": "Midwest",
    "ND": "Midwest", "SD": "Midwest", "NE": "Midwest", "KS": "Midwest",
    "MT": "Mountain", "WY": "Mountain", "CO": "Mountain", "UT": "Mountain",
    "ID": "Mountain", "AZ": "Mountain", "NV": "Mountain",
    "WA": "Pacific",  "OR": "Pacific",  "CA": "Pacific",
}

LTL_MAX_LBS = 15_000
TL_MAX_LBS  = 44_000

REQUIRED = {
    "shipment_id", "warehouse", "dest_city", "dest_state",
    "dest_lat", "dest_lon", "weight_lbs", "freight_class",
    "pickup_date", "delivery_due",
}


def load(csv_path: str) -> pd.DataFrame:
    """
    Read CSV, validate, and attach derived columns for the optimizer.

    Derived columns
    ---------------
        warehouse_lat, warehouse_lon  : origin coordinates
        wh_to_dest_mi                 : haversine distance to destination
        mode_flag                     : "MUST_TL" or "OPTIONAL"
        region                        : US region (for MILP partitioning)
    """
    df = pd.read_csv(csv_path, parse_dates=["pickup_date", "delivery_due"])

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    warehouses = df["warehouse"].unique()
    if len(warehouses) > 1:
        raise ValueError(
            f"CSV has {len(warehouses)} warehouses: {list(warehouses)}. "
            "Filter to one warehouse before running."
        )
    wh_code = warehouses[0]
    if wh_code not in WAREHOUSES:
        raise ValueError(f"Unknown warehouse '{wh_code}'.")

    wh = WAREHOUSES[wh_code]
    df["warehouse_lat"] = wh["lat"]
    df["warehouse_lon"] = wh["lon"]
    df["wh_to_dest_mi"] = df.apply(
        lambda r: _haversine(wh["lat"], wh["lon"], r["dest_lat"], r["dest_lon"]),
        axis=1,
    ).round(1)
    df["mode_flag"] = df["weight_lbs"].apply(
        lambda w: "MUST_TL" if w > LTL_MAX_LBS else "OPTIONAL"
    )
    df["region"] = df["dest_state"].map(REGION_MAP).fillna("Other")
    return df.reset_index(drop=True)


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 3_959.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def print_summary(df: pd.DataFrame) -> None:
    wh  = df["warehouse"].iloc[0]
    loc = WAREHOUSES.get(wh, {}).get("city", "")
    print(f"  Warehouse    : {wh}  ({loc})")
    print(f"  Shipments    : {len(df)}")
    print(f"  Total weight : {df['weight_lbs'].sum():>12,.0f} lbs")
    print(f"  MUST_TL      : {(df['mode_flag']=='MUST_TL').sum():>4d}  (weight > {LTL_MAX_LBS:,} lbs)")
    print(f"  OPTIONAL     : {(df['mode_flag']=='OPTIONAL').sum():>4d}  (LTL-eligible)")
    print(f"  Regions      : {', '.join(sorted(df['region'].unique()))}")
