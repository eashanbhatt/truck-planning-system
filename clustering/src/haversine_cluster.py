"""
clustering/src/haversine_cluster.py
-------------------------------------
Builds geographic shipment clusters using a **composite distance matrix**
that combines haversine distance with bearing difference.

Why bearing matters for truck routing
--------------------------------------
Two destinations might be geographically close (low haversine) but in
completely opposite directions from the warehouse — putting them on the
same truck creates a back-and-forth route that wastes miles and time.

By penalising large bearing differences, clusters naturally form "shipping
lanes": groups of destinations that leave the warehouse in roughly the same
compass direction, producing efficient multi-stop TL routes.

Composite distance formula
--------------------------
For each pair of destinations i, j:

    bearing_i   = forward azimuth from warehouse → dest_i  (0–360°)
    bearing_j   = forward azimuth from warehouse → dest_j  (0–360°)
    bearing_diff = circular min(|b_i − b_j|, 360 − |b_i − b_j|)   [0–180°]

    composite[i,j] = haversine_weight × haversine[i,j]
                   + bearing_weight   × (bearing_diff[i,j] / 180) × max_haversine

Both components are expressed in the same unit (miles-equivalent) so the
`distance_threshold` parameter in `build_clusters()` remains in miles and
is directly interpretable.

Example
-------
Chicago → New York  : haversine ≈  790 mi, bearing ≈  84° (East)
Chicago → Boston    : haversine ≈  980 mi, bearing ≈  72° (East)
Chicago → Miami     : haversine ≈ 1380 mi, bearing ≈ 151° (Southeast)
Chicago → Los Angeles: haversine ≈ 2020 mi, bearing ≈ 247° (West)

NY–Boston pair  : haversine ≈ 215 mi, bearing diff ≈  12° → low composite → same cluster ✓
NY–LA pair      : haversine ≈ 2450 mi, bearing diff ≈ 163° → high composite → different clusters ✓
NY–Miami pair   : haversine ≈ 1280 mi, bearing diff ≈  67° → medium composite

Algorithm
---------
1. Compute bearing from warehouse to every destination.
2. Compute n×n haversine distance matrix between all destination pairs.
3. Compute n×n bearing difference matrix (pairwise, circular).
4. Build composite matrix (miles-equivalent).
5. Feed composite matrix into AgglomerativeClustering(metric='precomputed',
   linkage='complete').
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Bearing helpers
# ---------------------------------------------------------------------------

def bearing_to(wh_lat: float, wh_lon: float,
               dest_lat: float, dest_lon: float) -> float:
    """
    Forward azimuth (bearing) from warehouse to a single destination.

    Parameters
    ----------
    wh_lat, wh_lon     : warehouse coordinates (degrees)
    dest_lat, dest_lon : destination coordinates (degrees)

    Returns
    -------
    float : bearing in degrees [0, 360)
            0° = North, 90° = East, 180° = South, 270° = West
    """
    lat1 = np.radians(wh_lat)
    lat2 = np.radians(dest_lat)
    dlon = np.radians(dest_lon - wh_lon)

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    b = np.degrees(np.arctan2(x, y))
    return float((b + 360) % 360)


def bearing_vector(wh_lat: float, wh_lon: float,
                   dest_lats: np.ndarray, dest_lons: np.ndarray) -> np.ndarray:
    """
    Vectorized forward azimuth from warehouse to an array of destinations.

    Parameters
    ----------
    wh_lat, wh_lon : warehouse coordinates
    dest_lats      : array of destination latitudes  (n,)
    dest_lons      : array of destination longitudes (n,)

    Returns
    -------
    np.ndarray of shape (n,) — bearing in degrees [0, 360)
    """
    lat1  = np.radians(wh_lat)
    lat2  = np.radians(dest_lats)
    dlon  = np.radians(dest_lons - wh_lon)

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    b = np.degrees(np.arctan2(x, y))
    return (b + 360) % 360


def bearing_diff_matrix(bearings: np.ndarray) -> np.ndarray:
    """
    Pairwise circular bearing difference matrix.

    Parameters
    ----------
    bearings : 1-D array of bearings in degrees [0, 360)

    Returns
    -------
    np.ndarray of shape (n, n)
        Values in [0, 180] — the minimum angular separation between
        any two bearings, accounting for the circular 0°/360° wrap.
    """
    # Broadcast pairwise differences
    diff = np.abs(bearings[:, None] - bearings[None, :]) % 360
    # Circular minimum: max angular separation is 180°
    return np.minimum(diff, 360 - diff)


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

    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Composite distance matrix
# ---------------------------------------------------------------------------

def composite_matrix(
    dest_lats: np.ndarray,
    dest_lons: np.ndarray,
    wh_lat: float,
    wh_lon: float,
    haversine_weight: float = 0.6,
    bearing_weight: float = 0.4,
) -> tuple:
    """
    Build the composite distance matrix combining haversine + bearing.

    Composite formula (miles-equivalent)
    -------------------------------------
        composite[i,j] = haversine_weight × haversine[i,j]
                       + bearing_weight   × (bearing_diff[i,j] / 180) × max_haversine

    Expressing bearing penalty in miles makes the composite threshold
    directly interpretable: a threshold of 200 miles means two destinations
    are grouped only if their combined haversine distance and directional
    divergence is equivalent to ≤ 200 miles.

    Parameters
    ----------
    dest_lats, dest_lons : destination coordinate arrays
    wh_lat, wh_lon       : warehouse coordinates
    haversine_weight     : weight for haversine component (default 0.6)
    bearing_weight       : weight for bearing component   (default 0.4)

    Returns
    -------
    composite   : np.ndarray (n, n) — composite distance in miles-equivalent
    hav_mat     : np.ndarray (n, n) — raw haversine distances in miles
    bear_mat    : np.ndarray (n, n) — raw bearing differences in degrees
    bearings    : np.ndarray (n,)   — bearing from warehouse to each dest
    """
    # Haversine distance matrix (miles)
    hav_mat = haversine_matrix(dest_lats, dest_lons)

    # Bearing from warehouse to each destination
    bearings = bearing_vector(wh_lat, wh_lon, dest_lats, dest_lons)

    # Pairwise bearing difference matrix (degrees [0, 180])
    bear_mat = bearing_diff_matrix(bearings)

    # Bearing penalty in miles-equivalent
    max_hav = hav_mat.max() if hav_mat.max() > 0 else 1.0
    bear_miles = (bear_mat / 180.0) * max_hav

    # Composite
    composite = haversine_weight * hav_mat + bearing_weight * bear_miles

    return composite, hav_mat, bear_mat, bearings


# ---------------------------------------------------------------------------
# Cluster builder
# ---------------------------------------------------------------------------

def build_clusters(
    df: pd.DataFrame,
    wh_lat: float,
    wh_lon: float,
    distance_threshold_miles: float = 200.0,
    haversine_weight: float = 0.6,
    bearing_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Group shipments into geographic clusters using the composite distance matrix.

    Two destinations are placed in the same cluster only if their composite
    distance (haversine + bearing penalty) is ≤ `distance_threshold_miles`.

    Using complete linkage guarantees that every pair of destinations within
    a cluster satisfies this constraint.

    Parameters
    ----------
    df                      : DataFrame from data_loader.load()
    wh_lat, wh_lon          : warehouse coordinates (for bearing calculation)
    distance_threshold_miles: composite distance threshold in miles-equivalent
    haversine_weight        : weight for haversine component (default 0.6)
    bearing_weight          : weight for bearing component   (default 0.4)

    Returns
    -------
    df with new columns:
        bearing_deg     : bearing from warehouse to this destination (0–360°)
        cluster_id      : integer cluster label (0-based)
        cluster_size    : number of shipments in this cluster
    """
    lats = df["dest_lat"].values
    lons = df["dest_lon"].values

    # Build composite distance matrix
    comp, hav_mat, bear_mat, bearings = composite_matrix(
        lats, lons, wh_lat, wh_lon,
        haversine_weight=haversine_weight,
        bearing_weight=bearing_weight,
    )

    # Attach bearing to each shipment row
    df = df.copy()
    df["bearing_deg"] = np.round(bearings, 1)

    # Agglomerative clustering on precomputed composite distances
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=distance_threshold_miles,
    )
    labels = model.fit_predict(comp)

    df["cluster_id"]   = labels
    df["cluster_size"] = df.groupby("cluster_id")["shipment_id"].transform("count")

    return df


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cluster summary including haversine spread and bearing spread.
    Helps explain why each cluster was formed.
    """
    lats = df["dest_lat"].values
    lons = df["dest_lon"].values
    hav = haversine_matrix(lats, lons)

    def _max_hav(cid):
        idxs = df.index[df["cluster_id"] == cid].tolist()
        if len(idxs) < 2:
            return 0.0
        sub = hav[np.ix_(idxs, idxs)]
        return round(float(sub.max()), 1)

    grp = (
        df.groupby("cluster_id")
        .agg(
            n_shipments    = ("shipment_id",   "count"),
            total_weight   = ("weight_lbs",    "sum"),
            n_destinations = ("dest_city",     "nunique"),
            must_tl_count  = ("mode_flag",     lambda x: (x == "MUST_TL").sum()),
            min_bearing    = ("bearing_deg",   "min"),
            max_bearing    = ("bearing_deg",   "max"),
            mean_bearing   = ("bearing_deg",   "mean"),
            cities         = ("dest_city",     lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values("total_weight", ascending=False)
        .reset_index(drop=True)
    )

    grp["max_intra_dist_mi"] = grp["cluster_id"].apply(_max_hav)
    grp["bearing_spread_deg"] = (grp["max_bearing"] - grp["min_bearing"]).round(1)
    grp["mean_bearing"] = grp["mean_bearing"].round(1)

    # Human-readable compass direction from mean bearing
    grp["direction"] = grp["mean_bearing"].apply(_compass)

    return grp


def _compass(bearing: float) -> str:
    """Convert a bearing in degrees to a compass label (N, NE, E, …)."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx  = int((bearing + 22.5) / 45) % 8
    return dirs[idx]
