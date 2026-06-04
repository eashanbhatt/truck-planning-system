"""
clustering/src/reporter.py
---------------------------
Console output and CSV writers for the clustering solution.

Outputs
-------
1. <output_dir>/clustering_plan.csv     — full row-per-shipment load plan
2. <output_dir>/clustering_savings.csv  — tiered savings summary
                                          (OVERALL / CLUSTER / TRUCK / LTL)
"""

import os
import pandas as pd
from tabulate import tabulate

TL_MAX_LBS = 44_000


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_results(plan: pd.DataFrame, baseline: float, elapsed: float) -> None:
    optimized = plan["plan_cost"].sum()
    savings   = baseline - optimized
    pct       = savings / baseline * 100 if baseline else 0.0

    tl_df    = plan[plan["assigned_mode"] == "TL"]
    ltl_df   = plan[plan["assigned_mode"] == "LTL"]
    n_trucks = tl_df["truck_id"].nunique()

    sep = "─" * 54
    print(f"\n{'═'*54}")
    print("  SOLUTION 1 — HAVERSINE CLUSTERING  (results)")
    print(f"{'═'*54}")
    print(f"\n  {'Metric':<36} {'Value':>15}")
    print(sep)
    print(f"  {'Baseline cost (all LTL)':<36} {'${:>13,.0f}'.format(baseline)}")
    print(f"  {'Optimised cost':<36} {'${:>13,.0f}'.format(optimized)}")
    print(f"  {'Total savings':<36} {'${:>13,.0f}'.format(savings)}")
    print(f"  {'Savings %':<36} {pct:>14.1f}%")
    print(sep)
    print(f"  {'TL trucks dispatched':<36} {n_trucks:>15,}")
    print(f"  {'Shipments consolidated onto TL':<36} {len(tl_df):>15,}")
    print(f"  {'Shipments remaining as LTL':<36} {len(ltl_df):>15,}")
    if n_trucks:
        avg_util = (
            tl_df.groupby("truck_id")["weight_lbs"].sum().mean()
            / TL_MAX_LBS * 100
        )
        print(f"  {'Avg TL weight utilisation':<36} {avg_util:>14.1f}%")
    print(f"  {'Distance metric':<36} {'Haversine (great-circle)':>15}")
    print(f"  {'Clustering algorithm':<36} {'AgglomerativeClustering':>15}")
    print(f"  {'Linkage':<36} {'complete':>15}")
    print(f"  {'Plan generated in':<36} {elapsed:>14.2f}s")
    print(f"{'═'*54}\n")

    # Truck details table
    truck_tbl = _truck_table(plan, tl_df)
    if not truck_tbl.empty:
        print("  TL Truck Details:")
        print(tabulate(truck_tbl, headers="keys", tablefmt="rounded_outline",
                       showindex=False))
        print()

    # LTL table
    if not ltl_df.empty:
        ltl_tbl = ltl_df[
            ["shipment_id", "dest_city", "dest_state",
             "weight_lbs", "freight_class", "ltl_cost_each"]
        ].copy()
        ltl_tbl["weight_lbs"]    = ltl_tbl["weight_lbs"].map("{:,.0f}".format)
        ltl_tbl["ltl_cost_each"] = ltl_tbl["ltl_cost_each"].map("${:,.0f}".format)
        ltl_tbl.columns = ["Shipment", "Dest City", "ST",
                            "Weight (lbs)", "FC", "LTL Cost"]
        print("  LTL Shipments (individual):")
        print(tabulate(ltl_tbl, headers="keys", tablefmt="rounded_outline",
                       showindex=False))
        print()


def _truck_table(plan: pd.DataFrame, tl_df: pd.DataFrame) -> pd.DataFrame:
    if tl_df.empty:
        return pd.DataFrame()

    summary = (
        tl_df.groupby("truck_id")
        .agg(
            n_shipments  = ("shipment_id",   "count"),
            total_weight = ("weight_lbs",    "sum"),
            n_stops      = ("dest_city",     "nunique"),
            tl_cost      = ("tl_cost",       "first"),
            ltl_baseline = ("ltl_cost_each", "sum"),
            cluster      = ("cluster_id",    "first"),
        )
        .reset_index()
    )
    summary["savings"]  = summary["ltl_baseline"] - summary["tl_cost"]
    summary["util_pct"] = (summary["total_weight"] / TL_MAX_LBS * 100).round(1)

    out = summary[["truck_id", "cluster", "n_shipments", "total_weight",
                   "n_stops", "tl_cost", "ltl_baseline", "savings", "util_pct"]].copy()
    out["total_weight"] = out["total_weight"].map("{:,.0f}".format)
    out["tl_cost"]      = out["tl_cost"].map("${:,.0f}".format)
    out["ltl_baseline"] = out["ltl_baseline"].map("${:,.0f}".format)
    out["savings"]      = out["savings"].map("${:,.0f}".format)
    out["util_pct"]     = out["util_pct"].map("{:.1f}%".format)
    out.columns = ["Truck", "Cluster", "Shpmnts", "Weight (lbs)",
                   "Stops", "TL Cost", "LTL Baseline", "Savings", "Util%"]
    return out


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

PLAN_COLS = [
    "shipment_id", "warehouse", "dest_city", "dest_state",
    "weight_lbs", "freight_class", "pickup_date", "delivery_due",
    "mode_flag", "cluster_id", "cluster_size",
    "truck_id", "assigned_mode", "truck_weight", "truck_n_stops",
    "wh_to_dest_mi", "tl_cost", "ltl_cost_each", "plan_cost",
]


def save_plan(plan: pd.DataFrame, path: str) -> None:
    """Save the full row-per-shipment load plan CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [c for c in PLAN_COLS if c in plan.columns]
    plan[cols].to_csv(path, index=False)
    print(f"  Load plan saved    → {path}")


def save_savings(plan: pd.DataFrame, path: str) -> None:
    """
    Save a tiered savings summary CSV with sections:
      OVERALL  — total plan
      CLUSTER  — per haversine cluster
      TRUCK    — per TL truck
      LTL      — all LTL shipments aggregated
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    rows = []

    def _add(section, label, sub):
        n        = len(sub)
        weight   = sub["weight_lbs"].sum()
        baseline = sub["ltl_cost_each"].sum()
        optimized = sub["plan_cost"].sum()
        savings  = baseline - optimized
        pct      = round(savings / baseline * 100, 1) if baseline else 0.0
        rows.append({
            "section":           section,
            "label":             label,
            "n_shipments":       n,
            "tl_shipments":      (sub["assigned_mode"] == "TL").sum(),
            "ltl_shipments":     (sub["assigned_mode"] == "LTL").sum(),
            "total_weight_lbs":  round(weight, 0),
            "baseline_ltl_cost": f"${baseline:,.2f}",
            "optimized_cost":    f"${optimized:,.2f}",
            "savings":           f"${savings:,.2f}",
            "savings_pct":       f"{pct}%",
        })

    # OVERALL
    _add("OVERALL", "All Shipments", plan)

    # Per CLUSTER
    for cid, grp in plan.groupby("cluster_id"):
        cities = ", ".join(sorted(grp["dest_city"].unique()))
        _add("CLUSTER", f"Cluster {cid} ({cities})", grp)

    # Per TRUCK
    tl_df = plan[plan["assigned_mode"] == "TL"]
    for tid, grp in tl_df.groupby("truck_id"):
        _add("TRUCK", tid, grp)

    # LTL aggregate
    ltl_df = plan[plan["assigned_mode"] == "LTL"]
    if len(ltl_df):
        _add("LTL", "All LTL Shipments", ltl_df)

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Savings summary saved → {path}")
