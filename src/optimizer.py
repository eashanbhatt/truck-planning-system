"""
optimizer.py
------------
Mixed-Integer Linear Program (MILP) for truck load optimisation.

For each geographic cluster produced by LoadConsolidator, the optimizer
decides how many TL trucks to dispatch and which shipments ride on each,
minimising total freight cost subject to weight and volume capacity.

Formulation
-----------
Sets
  S = {shipments in cluster}
  T = {1 … N_trucks}  (upper bound on trucks needed)

Variables
  x[s, t] ∈ {0, 1}   1 if shipment s is loaded on truck t
  y[t]    ∈ {0, 1}   1 if truck t is activated

Objective
  min  Σ_t  TL_cost(t) · y[t]
     + Σ_s  LTL_cost(s) · (1 − Σ_t x[s,t])

Constraints
  Σ_t x[s,t] ≤ 1                          ∀ s   (each shipment on ≤1 truck)
  Σ_s weight[s]·x[s,t] ≤ CAP_W · y[t]    ∀ t   (weight capacity)
  Σ_s volume[s]·x[s,t] ≤ CAP_V · y[t]    ∀ t   (volume capacity)
  x[s,t] ≤ y[t]                           ∀ s,t (link)

Usage
-----
    from src.optimizer import LoadOptimizer
    from src.clustering import LoadConsolidator

    lc = LoadConsolidator()
    clustered = lc.fit(shipments)
    summary   = lc.cluster_summary()

    opt = LoadOptimizer()
    plan = opt.optimize(clustered, summary)
    opt.print_summary()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary,
    lpSum, value, PULP_CBC_CMD, LpStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TL_WEIGHT_CAPACITY = 44_000   # lbs
TL_VOLUME_CAPACITY = 2_400    # cuft
TL_RATE_PER_MILE   = 3.25     # $/mile (flat rate regardless of weight)
TL_BASE_COST       = 150.0    # $ fixed accessorial / fuel surcharge per truck


@dataclass
class LoadPlanEntry:
    """One row in the final load plan."""
    cluster_id:    str
    truck_id:      str
    shipments:     List[str]      # shipment_ids
    total_weight:  float
    total_volume:  float
    mode:          str            # "TL" or "LTL"
    tl_cost:       float
    ltl_cost:      float
    optimized_cost: float
    savings:       float


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _ltl_cost(weight_lbs: float, distance_miles: float, freight_class: float) -> float:
    """
    Simplified LTL tariff (per-CWT rate × hundredweight).
    Rate increases with distance and freight class.
    """
    rate_per_cwt = 20.0 + (distance_miles / 100) * 1.5 + (freight_class - 50) * 0.05
    return round((weight_lbs / 100) * rate_per_cwt, 2)


def _tl_cost(distance_miles: float) -> float:
    """Flat TL rate: base + per-mile charge."""
    return round(TL_BASE_COST + TL_RATE_PER_MILE * distance_miles, 2)


# ---------------------------------------------------------------------------
# LoadOptimizer
# ---------------------------------------------------------------------------

class LoadOptimizer:
    """
    Solves the TL/LTL assignment MILP for each cluster independently,
    then aggregates results into a full load plan.
    """

    def __init__(
        self,
        tl_weight_cap: float = TL_WEIGHT_CAPACITY,
        tl_volume_cap: float = TL_VOLUME_CAPACITY,
        tl_rate_per_mile: float = TL_RATE_PER_MILE,
        solver_msg: bool = False,
    ):
        self.tl_weight_cap   = tl_weight_cap
        self.tl_volume_cap   = tl_volume_cap
        self.tl_rate_per_mile = tl_rate_per_mile
        self.solver_msg      = solver_msg

        self._load_plan: List[LoadPlanEntry] = []
        self._total_baseline: float = 0.0
        self._total_optimized: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        clustered_df: pd.DataFrame,
        cluster_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Optimise every cluster and build the load plan.

        Returns
        -------
        pd.DataFrame
            One row per truck / LTL shipment with cost details.
        """
        self._load_plan = []

        for _, cluster_row in cluster_summary.iterrows():
            cid = cluster_row["cluster_id"]
            shipments = clustered_df[clustered_df["cluster_id"] == cid].copy()

            if len(shipments) == 0:
                continue

            entries = self._optimize_cluster(cid, shipments, cluster_row)
            self._load_plan.extend(entries)

        plan_df = self._to_dataframe()
        self._total_baseline  = clustered_df["ltl_cost_baseline"].sum()
        self._total_optimized = plan_df["optimized_cost"].sum()
        return plan_df

    def print_summary(self) -> None:
        """Print a formatted cost comparison to stdout."""
        if not self._load_plan:
            print("No plan generated — call optimize() first.")
            return

        plan_df = self._to_dataframe()
        tl_loads   = plan_df[plan_df["mode"] == "TL"]
        ltl_loads  = plan_df[plan_df["mode"] == "LTL"]

        savings    = self._total_baseline - self._total_optimized
        pct_saving = savings / self._total_baseline * 100 if self._total_baseline else 0

        sep = "─" * 54
        print(f"\n{'═'*54}")
        print(f"  LOAD OPTIMISER — RESULTS SUMMARY")
        print(f"{'═'*54}")
        print(f"\n{'COST BREAKDOWN':}")
        print(sep)
        print(f"  {'Metric':<32} {'Value':>18}")
        print(sep)
        print(f"  {'Baseline (all LTL)':<32} {'${:>14,.0f}'.format(self._total_baseline)}")
        print(f"  {'Optimised total cost':<32} {'${:>14,.0f}'.format(self._total_optimized)}")
        print(f"  {'Savings':<32} {'${:>14,.0f}'.format(savings)}")
        print(f"  {'Savings %':<32} {pct_saving:>17.1f}%")
        print(sep)

        print(f"\n{'LOAD PLAN SUMMARY':}")
        print(sep)
        print(f"  {'TL loads dispatched':<32} {len(tl_loads):>18,}")
        print(f"  {'LTL shipments (individual)':<32} {len(ltl_loads):>18,}")
        print(f"  {'Avg TL utilisation':<32} "
              f"{(tl_loads['total_weight'].mean() / self.tl_weight_cap * 100):>17.1f}%"
              if len(tl_loads) else "  {'Avg TL utilisation':<32} {'N/A':>18}")
        print(f"  {'Total TL cost':<32} {'${:>14,.0f}'.format(tl_loads['optimized_cost'].sum())}")
        print(f"  {'Total LTL cost':<32} {'${:>14,.0f}'.format(ltl_loads['optimized_cost'].sum())}")
        print(f"{'═'*54}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _optimize_cluster(
        self,
        cluster_id: str,
        shipments: pd.DataFrame,
        cluster_meta: pd.Series,
    ) -> List[LoadPlanEntry]:
        """
        Solve the MILP for a single cluster.
        Falls back to all-LTL if the cluster is too small or the solver fails.
        """
        n = len(shipments)
        avg_dist = cluster_meta["avg_distance"]

        # Upper bound on trucks needed (ceiling of total weight / capacity)
        total_weight = shipments["weight_lbs"].sum()
        n_trucks = max(1, int(np.ceil(total_weight / self.tl_weight_cap)) + 1)

        # Precompute per-shipment costs
        shipments = shipments.copy()
        shipments["ltl_cost"] = shipments.apply(
            lambda r: _ltl_cost(r["weight_lbs"], r["distance_miles"], r["freight_class"]),
            axis=1,
        )
        tl_cost_per_truck = _tl_cost(avg_dist)
        baseline_cost = shipments["ltl_cost"].sum()

        # Skip MIP if cluster is very small — pure LTL is obviously optimal
        if total_weight < 2_000 or n == 1:
            return self._all_ltl(cluster_id, shipments, baseline_cost)

        # ----- Build MILP -----
        prob = LpProblem(f"LoadOpt_{cluster_id}", LpMinimize)

        ship_ids  = shipments["shipment_id"].tolist()
        truck_ids = list(range(n_trucks))

        x = {(s, t): LpVariable(f"x_{s}_{t}", cat=LpBinary)
             for s in ship_ids for t in truck_ids}
        y = {t: LpVariable(f"y_{t}", cat=LpBinary) for t in truck_ids}

        # Objective
        prob += (
            lpSum(tl_cost_per_truck * y[t] for t in truck_ids)
            + lpSum(
                shipments.loc[shipments["shipment_id"] == s, "ltl_cost"].values[0]
                * (1 - lpSum(x[s, t] for t in truck_ids))
                for s in ship_ids
            )
        )

        # Constraints
        weights = dict(zip(shipments["shipment_id"], shipments["weight_lbs"]))
        volumes = dict(zip(shipments["shipment_id"], shipments["volume_cuft"]))

        for s in ship_ids:
            prob += lpSum(x[s, t] for t in truck_ids) <= 1, f"assign_{s}"

        for t in truck_ids:
            prob += (
                lpSum(weights[s] * x[s, t] for s in ship_ids)
                <= self.tl_weight_cap * y[t],
                f"weight_cap_{t}",
            )
            prob += (
                lpSum(volumes[s] * x[s, t] for s in ship_ids)
                <= self.tl_volume_cap * y[t],
                f"vol_cap_{t}",
            )
            for s in ship_ids:
                prob += x[s, t] <= y[t], f"link_{s}_{t}"

        # Symmetry breaking — prefer lower-indexed trucks
        for t in range(1, n_trucks):
            prob += y[t] <= y[t - 1], f"sym_{t}"

        solver = PULP_CBC_CMD(msg=self.solver_msg, timeLimit=30)
        prob.solve(solver)

        if LpStatus[prob.status] != "Optimal":
            # Fall back to all-LTL
            return self._all_ltl(cluster_id, shipments, baseline_cost)

        # ----- Extract solution -----
        entries: List[LoadPlanEntry] = []
        assigned: set = set()

        for t in truck_ids:
            if value(y[t]) is None or value(y[t]) < 0.5:
                continue
            on_truck = [
                s for s in ship_ids
                if value(x[s, t]) is not None and value(x[s, t]) > 0.5
            ]
            if not on_truck:
                continue

            sub = shipments[shipments["shipment_id"].isin(on_truck)]
            tw  = sub["weight_lbs"].sum()
            tv  = sub["volume_cuft"].sum()
            ltl = sub["ltl_cost"].sum()

            entries.append(LoadPlanEntry(
                cluster_id=cluster_id,
                truck_id=f"{cluster_id}_TL{t+1}",
                shipments=on_truck,
                total_weight=tw,
                total_volume=tv,
                mode="TL",
                tl_cost=tl_cost_per_truck,
                ltl_cost=ltl,
                optimized_cost=tl_cost_per_truck,
                savings=ltl - tl_cost_per_truck,
            ))
            assigned.update(on_truck)

        # Remaining shipments ship LTL individually
        unassigned = [s for s in ship_ids if s not in assigned]
        for s in unassigned:
            row = shipments[shipments["shipment_id"] == s].iloc[0]
            entries.append(LoadPlanEntry(
                cluster_id=cluster_id,
                truck_id=f"{cluster_id}_LTL_{s}",
                shipments=[s],
                total_weight=row["weight_lbs"],
                total_volume=row["volume_cuft"],
                mode="LTL",
                tl_cost=0.0,
                ltl_cost=row["ltl_cost"],
                optimized_cost=row["ltl_cost"],
                savings=0.0,
            ))

        return entries

    def _all_ltl(
        self,
        cluster_id: str,
        shipments: pd.DataFrame,
        baseline_cost: float,
    ) -> List[LoadPlanEntry]:
        """Return individual LTL entries for every shipment in the cluster."""
        entries = []
        for _, row in shipments.iterrows():
            entries.append(LoadPlanEntry(
                cluster_id=cluster_id,
                truck_id=f"{cluster_id}_LTL_{row['shipment_id']}",
                shipments=[row["shipment_id"]],
                total_weight=row["weight_lbs"],
                total_volume=row["volume_cuft"],
                mode="LTL",
                tl_cost=0.0,
                ltl_cost=row["ltl_cost"],
                optimized_cost=row["ltl_cost"],
                savings=0.0,
            ))
        return entries

    def _to_dataframe(self) -> pd.DataFrame:
        if not self._load_plan:
            return pd.DataFrame()
        return pd.DataFrame([vars(e) for e in self._load_plan])
