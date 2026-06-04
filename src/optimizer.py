"""
optimizer.py  —  Module 2 core logic
--------------------------------------
Mixed-Integer Linear Program (MILP) using PuLP + open-source CBC solver.

Approach
--------
Shipments are first grouped by US region (Northeast, Southeast, etc.)
for tractability.  Within each region the MILP decides:

  • Which shipments to consolidate onto TL trucks
  • Which shipments remain as individual LTL moves

to minimise total freight cost.

MILP Formulation (per region)
------------------------------
Sets
  S  — shipments in this region
  T  — potential trucks  {1 … N_T}  (upper bound pre-computed)
  D  — distinct destination cities in S

Variables
  x[s, t] ∈ {0,1}   1 if shipment s is on truck t
  y[t]    ∈ {0,1}   1 if truck t is activated
  z[d, t] ∈ {0,1}   1 if destination d is served by truck t

Objective
  min  Σ_t  TL_cost(t) · y[t]
     + Σ_s  LTL_cost(s) · (1 − Σ_t x[s,t])

Constraints
  (1)  Σ_t x[s,t] ≤ 1                          ∀ s     assignment
  (2)  Σ_s weight[s]·x[s,t] ≤ TL_MAX · y[t]   ∀ t     weight cap
  (3)  x[s,t] ≤ z[dest(s), t]                  ∀ s,t   link ship→dest
  (4)  Σ_d z[d,t] ≤ MAX_STOPS · y[t]           ∀ t     stop limit
  (5)  x[s,t] ≤ y[t]                            ∀ s,t   link ship→truck
  (6)  MUST_TL shipments: Σ_t x[s,t] = 1        ∀ s∈MUST_TL   (forced onto a truck)

Solver: CBC (bundled with PuLP — no licence required)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary,
    lpSum, value, PULP_CBC_CMD, LpStatus,
)

from src.cost_utils import ltl_cost, tl_cost, route_distance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TL_MAX_LBS   = 44_000
LTL_MAX_LBS  = 15_000
MAX_STOPS    = 3
SOLVER_TIME  = 60     # seconds per region sub-problem


# ---------------------------------------------------------------------------
# LoadOptimizer
# ---------------------------------------------------------------------------

class LoadOptimizer:
    """
    MILP-based load planner.  Solves one sub-problem per geographic region.

    Parameters
    ----------
    solver_verbose : print CBC solver output (default False)
    """

    def __init__(self, solver_verbose: bool = False):
        self.solver_verbose = solver_verbose
        self._plan: Optional[pd.DataFrame] = None
        self._baseline_cost: float = 0.0
        self._optimized_cost: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_plan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the MILP for every region and return an annotated DataFrame.

        New columns added
        -----------------
            truck_id        : "TRK-001" … or "LTL"
            assigned_mode   : "TL" or "LTL"
            tl_cost         : cost of the assigned TL truck (0 for LTL rows)
            ltl_cost_ind    : individual LTL cost for this shipment
            plan_cost       : effective cost charged to this shipment
            solver_status   : "Optimal" / "Feasible" / "LTL-fallback"
        """
        df = df.copy()

        # Pre-compute individual LTL cost
        df["ltl_cost_ind"] = df.apply(
            lambda r: ltl_cost(r["weight_lbs"], r["distance_miles"], r["freight_class"]),
            axis=1,
        )

        # Initialise output columns
        df["truck_id"]      = "LTL"
        df["assigned_mode"] = "LTL"
        df["tl_cost"]       = 0.0
        df["solver_status"] = "LTL-fallback"

        truck_counter = 1
        wh_lat = df["warehouse_lat"].iloc[0]
        wh_lon = df["warehouse_lon"].iloc[0]

        for region, region_df in df.groupby("region"):
            assignments, counter_out = self._solve_region(
                region, region_df, wh_lat, wh_lon, truck_counter
            )
            truck_counter = counter_out

            for truck_id, info in assignments.items():
                idxs = info["indices"]
                df.loc[idxs, "truck_id"]      = truck_id
                df.loc[idxs, "assigned_mode"] = "TL"
                df.loc[idxs, "tl_cost"]       = info["cost"]
                df.loc[idxs, "solver_status"] = info["status"]

        # Final plan_cost per row
        truck_sizes = df[df["assigned_mode"] == "TL"].groupby("truck_id").size()
        df["plan_cost"] = df.apply(
            lambda r: (
                r["tl_cost"] / max(1, truck_sizes.get(r["truck_id"], 1))
                if r["assigned_mode"] == "TL"
                else r["ltl_cost_ind"]
            ),
            axis=1,
        ).round(2)

        self._plan           = df
        self._baseline_cost  = df["ltl_cost_ind"].sum()
        self._optimized_cost = df["plan_cost"].sum()
        return df

    def plan_summary(self) -> pd.DataFrame:
        """Per-truck summary table."""
        if self._plan is None:
            raise RuntimeError("Call build_plan() first.")

        df      = self._plan
        tl_rows = df[df["assigned_mode"] == "TL"]

        summary = (
            tl_rows.groupby("truck_id")
            .agg(
                n_shipments  = ("shipment_id",   "count"),
                total_weight = ("weight_lbs",    "sum"),
                n_stops      = ("dest_city",     "nunique"),
                truck_cost   = ("tl_cost",       "first"),
                ltl_baseline = ("ltl_cost_ind",  "sum"),
                region       = ("region",        "first"),
                status       = ("solver_status", "first"),
            )
            .reset_index()
        )
        summary["savings"]  = summary["ltl_baseline"] - summary["truck_cost"]
        summary["util_pct"] = (summary["total_weight"] / TL_MAX_LBS * 100).round(1)
        return summary.sort_values("savings", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Per-region MILP
    # ------------------------------------------------------------------

    def _solve_region(
        self,
        region: str,
        region_df: pd.DataFrame,
        wh_lat: float,
        wh_lon: float,
        truck_counter: int,
    ) -> tuple[Dict, int]:
        """
        Solve the MILP for one region.  Returns assignment dict + updated counter.
        """
        ship_ids  = region_df["shipment_id"].tolist()
        indices   = region_df.index.tolist()
        n         = len(ship_ids)

        if n == 0:
            return {}, truck_counter

        # Upper bound on trucks: total weight / TL cap + 1 buffer
        total_w  = region_df["weight_lbs"].sum()
        n_trucks = max(1, int(np.ceil(total_w / TL_MAX_LBS)) + 1)

        # Distinct destinations
        dest_keys = region_df[["dest_city", "dest_state"]].apply(
            lambda r: f"{r['dest_city']}_{r['dest_state']}", axis=1
        ).tolist()
        unique_dests = list(set(dest_keys))

        # Shipment → destination index
        dest_idx = {s: dest_keys[i] for i, s in enumerate(ship_ids)}

        # Pre-compute costs
        ltl_costs = dict(zip(ship_ids, region_df["ltl_cost_ind"].values))
        must_tl   = set(
            region_df.loc[region_df["mode"] == "MUST_TL", "shipment_id"].tolist()
        )

        # Average TL cost proxy: use centroid distance of region
        avg_lat  = region_df["dest_lat"].mean()
        avg_lon  = region_df["dest_lon"].mean()
        avg_dist = route_distance(wh_lat, wh_lon, [(avg_lat, avg_lon)])
        tl_cost_proxy = tl_cost(avg_dist * 1.3)  # 1.3× to account for multi-stop detour

        # ------ Build MILP ------
        prob = LpProblem(f"LoadOpt_{region.replace(' ', '_')}", LpMinimize)
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

        weights = dict(zip(ship_ids, region_df["weight_lbs"].values))

        for t in T:
            # Weight capacity
            prob += (
                lpSum(weights[s] * x[s, t] for s in ship_ids)
                <= TL_MAX_LBS * y[t],
                f"weight_{t}",
            )
            # Stop limit
            prob += (
                lpSum(z[d, t] for d in unique_dests)
                <= MAX_STOPS * y[t],
                f"stops_{t}",
            )

        for s in ship_ids:
            # Each shipment on at most 1 truck
            prob += lpSum(x[s, t] for t in T) <= 1, f"assign_{s}"

            for t in T:
                # Link shipment → truck
                prob += x[s, t] <= y[t], f"link_{s}_{t}"
                # Link shipment → destination-on-truck
                prob += x[s, t] <= z[dest_idx[s], t], f"dest_link_{s}_{t}"

        # Force MUST_TL shipments onto a truck
        for s in must_tl:
            prob += lpSum(x[s, t] for t in T) == 1, f"must_tl_{s}"

        # Symmetry breaking
        for t in range(1, n_trucks):
            prob += y[t] <= y[t - 1], f"sym_{t}"

        solver = PULP_CBC_CMD(msg=self.solver_verbose, timeLimit=SOLVER_TIME)
        prob.solve(solver)
        status = LpStatus[prob.status]

        if status not in ("Optimal", "Feasible"):
            # Fall back: MUST_TL shipments get their own truck each
            return self._fallback(region_df, wh_lat, wh_lon, truck_counter)

        # ------ Extract solution ------
        assignments: Dict[str, dict] = {}

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
                sub[["dest_lat", "dest_lon"]].drop_duplicates()
                .apply(tuple, axis=1)
            )
            route_mi   = route_distance(wh_lat, wh_lon, stops)
            actual_cost = tl_cost(route_mi)

            truck_id = f"TRK-{truck_counter:03d}"
            truck_counter += 1

            assignments[truck_id] = {
                "indices": sub.index.tolist(),
                "cost":    actual_cost,
                "status":  status,
            }

        return assignments, truck_counter

    def _fallback(
        self,
        region_df: pd.DataFrame,
        wh_lat: float,
        wh_lon: float,
        truck_counter: int,
    ) -> tuple[Dict, int]:
        """
        Fallback when MILP fails: put each MUST_TL shipment on its own truck.
        OPTIONAL shipments remain LTL.
        """
        assignments = {}
        must_tl = region_df[region_df["mode"] == "MUST_TL"]

        for _, row in must_tl.iterrows():
            dist     = route_distance(wh_lat, wh_lon, [(row["dest_lat"], row["dest_lon"])])
            cost     = tl_cost(dist)
            truck_id = f"TRK-{truck_counter:03d}"
            truck_counter += 1
            assignments[truck_id] = {
                "indices": [row.name],
                "cost":    cost,
                "status":  "LTL-fallback",
            }

        return assignments, truck_counter
