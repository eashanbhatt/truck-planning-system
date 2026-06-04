"""
optimizer/src/milp.py
----------------------
Mixed-Integer Linear Program (MILP) for truck load optimisation.
Solver: PuLP + CBC (open-source, no licence required).

This module is completely independent of the clustering solution.
It makes no geographic proximity assumptions — it minimises total
freight cost mathematically.

Approach
--------
Shipments are partitioned by US region for tractability.
One MILP sub-problem is solved per region.

MILP Formulation (per region)
------------------------------
Sets
  S  — shipments in this region
  T  — potential trucks {0 … N_T-1}
  D  — distinct destination keys in S

Decision variables
  x[s, t] ∈ {0, 1}   1 if shipment s is loaded on truck t
  y[t]    ∈ {0, 1}   1 if truck t is activated
  z[d, t] ∈ {0, 1}   1 if destination d is served by truck t

Objective
  min  Σ_t  TL_cost(t) · y[t]
     + Σ_s  LTL_cost(s) · (1 − Σ_t x[s,t])

Constraints
  (1)  Σ_t x[s,t] ≤ 1                         ∀s   one truck per shipment
  (2)  Σ_s w[s]·x[s,t] ≤ TL_MAX · y[t]        ∀t   weight capacity
  (3)  x[s,t] ≤ z[dest(s), t]                  ∀s,t link shipment → destination
  (4)  Σ_d z[d,t] ≤ MAX_STOPS · y[t]           ∀t   stop limit
  (5)  x[s,t] ≤ y[t]                            ∀s,t link shipment → truck
  (6)  Σ_t x[s,t] = 1   ∀s ∈ MUST_TL           forced onto a truck

Cost model
----------
  TL  : $150 base + $3.25/mile  (nearest-neighbour route from warehouse)
  LTL : (weight / 100) × CWT_rate × freight_class_multiplier
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary,
    lpSum, value, PULP_CBC_CMD, LpStatus,
)

TL_MAX_LBS  = 44_000
LTL_MAX_LBS = 15_000
MAX_STOPS   = 3
TL_BASE     = 150.0
TL_PER_MILE = 3.25
SOLVER_SEC  = 60

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
# Cost helpers (self-contained — no imports from clustering)
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 3_959.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def _route_miles(wh_lat, wh_lon, stops: List) -> float:
    """Greedy nearest-neighbour route distance."""
    if not stops:
        return 0.0
    remaining, cur, total = list(stops), (wh_lat, wh_lon), 0.0
    while remaining:
        dists = [_haversine(cur[0], cur[1], s[0], s[1]) for s in remaining]
        idx   = int(np.argmin(dists))
        total += dists[idx]
        cur    = remaining.pop(idx)
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


def ltl_cost_series(df: pd.DataFrame) -> pd.Series:
    """Compute individual LTL cost for every row in df."""
    return df.apply(
        lambda r: _ltl_cost(r["weight_lbs"], r["wh_to_dest_mi"], r["freight_class"]),
        axis=1,
    )


# ---------------------------------------------------------------------------
# MILP solver
# ---------------------------------------------------------------------------

class LinearOptimizer:
    """
    Solves the TL/LTL assignment MILP for each US region independently.

    Parameters
    ----------
    solver_verbose : print CBC solver output (default False)
    """

    def __init__(self, solver_verbose: bool = False):
        self.solver_verbose = solver_verbose
        self._plan: Optional[pd.DataFrame] = None

    def build_plan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the MILP for every region and return the annotated DataFrame.

        New columns
        -----------
            ltl_cost_each   : cost if shipped individually as LTL
            truck_id        : "TRK-001" … or "LTL"
            assigned_mode   : "TL" or "LTL"
            tl_cost         : cost of the assigned TL truck (0 for LTL)
            plan_cost       : effective cost charged to this shipment
            solver_status   : "Optimal", "Feasible", or "Fallback"
        """
        df = df.copy()
        df["ltl_cost_each"]  = ltl_cost_series(df)
        df["truck_id"]       = "LTL"
        df["assigned_mode"]  = "LTL"
        df["tl_cost"]        = 0.0
        df["solver_status"]  = "LTL"

        truck_num = 1
        wh_lat    = df["warehouse_lat"].iloc[0]
        wh_lon    = df["warehouse_lon"].iloc[0]

        for region, region_df in df.groupby("region"):
            assignments, truck_num = self._solve_region(
                region, region_df, wh_lat, wh_lon, truck_num
            )
            for truck_id, info in assignments.items():
                idxs = info["indices"]
                df.loc[idxs, "truck_id"]      = truck_id
                df.loc[idxs, "assigned_mode"] = "TL"
                df.loc[idxs, "tl_cost"]       = info["cost"]
                df.loc[idxs, "solver_status"] = info["status"]

        # plan_cost: TL cost split evenly; LTL cost for individual shipments
        truck_sizes = df[df["assigned_mode"] == "TL"].groupby("truck_id").size()
        df["plan_cost"] = df.apply(
            lambda r: (
                round(r["tl_cost"] / max(1, truck_sizes.get(r["truck_id"], 1)), 2)
                if r["assigned_mode"] == "TL"
                else r["ltl_cost_each"]
            ),
            axis=1,
        )
        self._plan = df
        return df

    # ------------------------------------------------------------------
    # Per-region MILP
    # ------------------------------------------------------------------

    def _solve_region(
        self,
        region: str,
        region_df: pd.DataFrame,
        wh_lat: float,
        wh_lon: float,
        truck_num: int,
    ):
        ship_ids = region_df["shipment_id"].tolist()
        if not ship_ids:
            return {}, truck_num

        total_w  = region_df["weight_lbs"].sum()
        n_trucks = max(1, int(np.ceil(total_w / TL_MAX_LBS)) + 1)

        dest_keys   = region_df.apply(
            lambda r: f"{r['dest_city']}|{r['dest_state']}", axis=1
        ).tolist()
        unique_dests = list(set(dest_keys))
        dest_of      = dict(zip(ship_ids, dest_keys))

        ltl_costs = dict(zip(ship_ids, region_df["ltl_cost_each"].values))
        weights   = dict(zip(ship_ids, region_df["weight_lbs"].values))
        must_tl   = set(region_df.loc[region_df["mode_flag"] == "MUST_TL", "shipment_id"])

        # TL cost proxy: avg distance × rate (actual cost computed post-solve)
        avg_dist      = region_df["wh_to_dest_mi"].mean()
        tl_cost_proxy = _tl_cost(avg_dist * 1.3)   # 1.3× multi-stop allowance

        # ------ Build LP ------
        prob = LpProblem(f"MILP_{region.replace(' ', '_')}", LpMinimize)
        T    = list(range(n_trucks))

        x = {(s, t): LpVariable(f"x_{s}_{t}", cat=LpBinary)
             for s in ship_ids for t in T}
        y = {t: LpVariable(f"y_{t}", cat=LpBinary) for t in T}
        z = {(d, t): LpVariable(f"z_{d}_{t}", cat=LpBinary)
             for d in unique_dests for t in T}

        # Objective
        prob += (
            lpSum(tl_cost_proxy * y[t] for t in T)
            + lpSum(
                ltl_costs[s] * (1 - lpSum(x[s, t] for t in T))
                for s in ship_ids
            )
        )

        # Constraints
        for s in ship_ids:
            prob += lpSum(x[s, t] for t in T) <= 1, f"assign_{s}"
        for t in T:
            prob += (
                lpSum(weights[s] * x[s, t] for s in ship_ids)
                <= TL_MAX_LBS * y[t],
                f"weight_{t}",
            )
            prob += (
                lpSum(z[d, t] for d in unique_dests)
                <= MAX_STOPS * y[t],
                f"stops_{t}",
            )
            for s in ship_ids:
                prob += x[s, t] <= y[t],              f"link_sy_{s}_{t}"
                prob += x[s, t] <= z[dest_of[s], t],  f"link_sd_{s}_{t}"
        for s in must_tl:
            prob += lpSum(x[s, t] for t in T) == 1, f"force_{s}"
        for t in range(1, n_trucks):
            prob += y[t] <= y[t - 1], f"sym_{t}"   # symmetry breaking

        solver = PULP_CBC_CMD(msg=self.solver_verbose, timeLimit=SOLVER_SEC)
        prob.solve(solver)
        status = LpStatus[prob.status]

        if status not in ("Optimal", "Feasible"):
            return self._fallback(region_df, wh_lat, wh_lon, truck_num)

        # ------ Extract ------
        assignments: Dict = {}
        for t in T:
            if value(y[t]) is None or value(y[t]) < 0.5:
                continue
            on_truck = [
                s for s in ship_ids
                if value(x[s, t]) is not None and value(x[s, t]) > 0.5
            ]
            if not on_truck:
                continue
            sub   = region_df[region_df["shipment_id"].isin(on_truck)]
            stops = list(
                sub[["dest_lat", "dest_lon"]].drop_duplicates().apply(tuple, axis=1)
            )
            route_mi    = _route_miles(wh_lat, wh_lon, stops)
            actual_cost = _tl_cost(route_mi)
            tid         = f"TRK-{truck_num:03d}"
            truck_num  += 1
            assignments[tid] = {
                "indices": sub.index.tolist(),
                "cost":    actual_cost,
                "status":  status,
            }

        return assignments, truck_num

    def _fallback(self, region_df, wh_lat, wh_lon, truck_num):
        """Fallback: each MUST_TL on its own truck; OPTIONAL stays LTL."""
        assignments = {}
        for _, row in region_df[region_df["mode_flag"] == "MUST_TL"].iterrows():
            dist = _route_miles(wh_lat, wh_lon, [(row["dest_lat"], row["dest_lon"])])
            tid  = f"TRK-{truck_num:03d}"
            truck_num += 1
            assignments[tid] = {
                "indices": [row.name],
                "cost":    _tl_cost(dist),
                "status":  "Fallback",
            }
        return assignments, truck_num
