"""
data_loader.py
--------------
Loads and validates the shipments CSV.

Expected columns
----------------
    shipment_id   : unique ID
    warehouse     : origin warehouse code (must exist in WAREHOUSE_LOCATIONS)
    dest_city     : destination city name
    dest_state    : two-letter state / territory code
    dest_lat      : destination latitude
    dest_lon      : destination longitude
    weight_lbs    : shipment weight in pounds
    freight_class : NMFC freight class (50, 65, 70, … 500)
    pickup_date   : YYYY-MM-DD
    delivery_due  : YYYY-MM-DD

Derived columns added by load_shipments()
------------------------------------------
    mode          : "MUST_TL" (weight > LTL_MAX_LBS) or "OPTIONAL"
    region        : US geographic region (Northeast, Southeast, …)
    warehouse_lat : latitude of origin warehouse
    warehouse_lon : longitude of origin warehouse
    distance_miles: great-circle distance from warehouse to destination
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Known warehouse locations
# ---------------------------------------------------------------------------

WAREHOUSE_LOCATIONS: dict = {
    "CHICAGO_IL":      {"lat": 41.8781, "lon": -87.6298, "city": "Chicago, IL"},
    "INDIANAPOLIS_IN": {"lat": 39.7684, "lon": -86.1581, "city": "Indianapolis, IN"},
    "COLUMBUS_OH":     {"lat": 39.9612, "lon": -82.9988, "city": "Columbus, OH"},
    "DALLAS_TX":       {"lat": 32.7767, "lon": -96.7970, "city": "Dallas, TX"},
    "LOS_ANGELES_CA":  {"lat": 34.0522, "lon": -118.2437, "city": "Los Angeles, CA"},
    "ATLANTA_GA":      {"lat": 33.7490, "lon": -84.3880, "city": "Atlanta, GA"},
}

# ---------------------------------------------------------------------------
# US state → region mapping
# ---------------------------------------------------------------------------

REGION_MAP: dict = {
    # Northeast
    "ME": "Northeast", "NH": "Northeast", "VT": "Northeast", "MA": "Northeast",
    "RI": "Northeast", "CT": "Northeast", "NY": "Northeast", "NJ": "Northeast",
    "PA": "Northeast", "DE": "Northeast", "MD": "Northeast", "DC": "Northeast",
    # Southeast
    "VA": "Southeast", "WV": "Southeast", "NC": "Southeast", "SC": "Southeast",
    "GA": "Southeast", "FL": "Southeast", "AL": "Southeast", "MS": "Southeast",
    "TN": "Southeast", "KY": "Southeast", "AR": "Southeast",
    # South Central
    "TX": "South Central", "OK": "South Central", "LA": "South Central",
    "NM": "South Central",
    # Midwest
    "OH": "Midwest", "IN": "Midwest", "IL": "Midwest", "MI": "Midwest",
    "WI": "Midwest", "MN": "Midwest", "IA": "Midwest", "MO": "Midwest",
    "ND": "Midwest", "SD": "Midwest", "NE": "Midwest", "KS": "Midwest",
    # Mountain
    "MT": "Mountain", "WY": "Mountain", "CO": "Mountain", "UT": "Mountain",
    "ID": "Mountain", "AZ": "Mountain", "NV": "Mountain",
    # Pacific
    "WA": "Pacific", "OR": "Pacific", "CA": "Pacific", "AK": "Pacific",
    "HI": "Pacific",
}

# ---------------------------------------------------------------------------
# Capacity constants
# ---------------------------------------------------------------------------

TL_MAX_LBS  = 44_000   # hard weight limit per TL truck
LTL_MAX_LBS = 15_000   # max weight for an individual LTL shipment
                        # shipments above this MUST go TL

REQUIRED_COLUMNS = {
    "shipment_id", "warehouse", "dest_city", "dest_state",
    "dest_lat", "dest_lon", "weight_lbs", "freight_class",
    "pickup_date", "delivery_due",
}


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    R = 3_959.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_shipments(csv_path: str) -> pd.DataFrame:
    """
    Read and validate the shipments CSV.  Adds derived columns.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame with all original columns plus:
        warehouse_lat, warehouse_lon, distance_miles, mode, region

    Raises
    ------
    ValueError   if required columns are missing or warehouse is unknown.
    """
    df = pd.read_csv(csv_path, parse_dates=["pickup_date", "delivery_due"])

    # Validate columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Validate single warehouse
    warehouses = df["warehouse"].unique()
    if len(warehouses) > 1:
        raise ValueError(
            f"CSV contains multiple warehouses: {list(warehouses)}. "
            "Filter to a single warehouse before running."
        )

    wh_code = warehouses[0]
    if wh_code not in WAREHOUSE_LOCATIONS:
        raise ValueError(
            f"Unknown warehouse '{wh_code}'. "
            f"Add it to WAREHOUSE_LOCATIONS in data_loader.py."
        )

    wh = WAREHOUSE_LOCATIONS[wh_code]
    df["warehouse_lat"] = wh["lat"]
    df["warehouse_lon"] = wh["lon"]

    # Distance from warehouse to destination
    df["distance_miles"] = df.apply(
        lambda r: haversine_miles(
            wh["lat"], wh["lon"], r["dest_lat"], r["dest_lon"]
        ),
        axis=1,
    ).round(1)

    # TL/LTL mode flag
    df["mode"] = df["weight_lbs"].apply(
        lambda w: "MUST_TL" if w > LTL_MAX_LBS else "OPTIONAL"
    )

    # Region
    df["region"] = df["dest_state"].map(REGION_MAP).fillna("Other")

    return df.reset_index(drop=True)


def summary(df: pd.DataFrame) -> None:
    """Print a quick data summary to stdout."""
    wh = df["warehouse"].iloc[0]
    print(f"\n  Warehouse   : {wh}  ({WAREHOUSE_LOCATIONS.get(wh, {}).get('city', '')})")
    print(f"  Shipments   : {len(df)}")
    print(f"  Total weight: {df['weight_lbs'].sum():>10,.0f} lbs")
    print(f"  MUST_TL     : {(df['mode']=='MUST_TL').sum():>4}  (weight > {LTL_MAX_LBS:,} lbs)")
    print(f"  OPTIONAL    : {(df['mode']=='OPTIONAL').sum():>4}  (LTL-eligible)")
    print(f"  Regions     : {', '.join(sorted(df['region'].unique()))}")
    print(f"  Date range  : {df['pickup_date'].min().date()} → {df['pickup_date'].max().date()}")
