"""
compare.py
----------
Side-by-side savings comparison: Clustering vs Linear Optimizer.

Reads the output CSVs produced by both solutions and prints a
formatted comparison table, then writes comparison.csv.

Run
---
    # Run both solutions first (if outputs don't exist yet):
    python clustering/run.py
    python optimizer/run.py

    # Then compare:
    python compare.py

    # Or point to custom output files:
    python compare.py \\
        --clustering-savings clustering/output/clustering_savings.csv \\
        --clustering-plan    clustering/output/clustering_plan.csv    \\
        --optimizer-savings  optimizer/output/milp_savings.csv        \\
        --optimizer-plan     optimizer/output/milp_plan.csv           \\
        --output             output/comparison.csv
"""

import argparse
import os
import sys
import pandas as pd
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CLUS_SAV  = os.path.join(BASE, "clustering", "output", "clustering_savings.csv")
DEFAULT_CLUS_PLAN = os.path.join(BASE, "clustering", "output", "clustering_plan.csv")
DEFAULT_OPT_SAV   = os.path.join(BASE, "optimizer",  "output", "milp_savings.csv")
DEFAULT_OPT_PLAN  = os.path.join(BASE, "optimizer",  "output", "milp_plan.csv")
DEFAULT_OUT       = os.path.join(BASE, "output",     "comparison.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dollar(val) -> float:
    """Convert '$1,234.56' → 1234.56."""
    return float(str(val).replace("$", "").replace(",", ""))


def _parse_pct(val) -> float:
    """Convert '62.4%' → 62.4."""
    return float(str(val).replace("%", ""))


def _load_savings(path: str, label: str) -> dict:
    """Extract key metrics from a savings CSV."""
    if not os.path.exists(path):
        sys.exit(f"  ERROR: {label} savings file not found: {path}\n"
                 f"  Run the solution first.")
    df = pd.read_csv(path)

    overall = df[df["section"] == "OVERALL"].iloc[0]
    trucks  = df[df["section"] == "TRUCK"]
    ltl_row = df[df["section"] == "LTL"]

    baseline  = _parse_dollar(overall["baseline_ltl_cost"])
    optimized = _parse_dollar(overall["optimized_cost"])
    savings   = _parse_dollar(overall["savings"])
    pct       = _parse_pct(overall["savings_pct"])
    n_tl_ships = int(overall["tl_shipments"])
    n_ltl_ships = int(overall["ltl_shipments"])
    n_trucks   = len(trucks)

    ltl_cost = _parse_dollar(ltl_row["optimized_cost"].values[0]) if len(ltl_row) else 0.0
    tl_cost  = optimized - ltl_cost

    # Per-truck savings from trucks section
    truck_savings = trucks["savings"].apply(_parse_dollar) if len(trucks) else pd.Series([0.0])

    return {
        "baseline":      baseline,
        "optimized":     optimized,
        "savings":       savings,
        "savings_pct":   pct,
        "n_tl_ships":    n_tl_ships,
        "n_ltl_ships":   n_ltl_ships,
        "n_trucks":      n_trucks,
        "tl_cost":       tl_cost,
        "ltl_cost":      ltl_cost,
        "best_truck_savings": truck_savings.max() if len(truck_savings) else 0.0,
        "avg_truck_savings":  truck_savings.mean() if len(truck_savings) else 0.0,
    }


def _load_plan_metrics(path: str, label: str) -> dict:
    """Compute truck-level metrics from a plan CSV."""
    if not os.path.exists(path):
        sys.exit(f"  ERROR: {label} plan file not found: {path}")
    df = pd.read_csv(path)
    tl = df[df["assigned_mode"] == "TL"]

    if tl.empty:
        return {"avg_util": 0.0, "avg_stops": 0.0, "avg_tl_weight": 0.0,
                "max_util": 0.0, "pct_full_trucks": 0.0}

    per_truck = tl.groupby("truck_id").agg(
        weight=("weight_lbs", "sum"),
        stops=("dest_city",   "nunique"),
    )
    per_truck["util_pct"] = per_truck["weight"] / 44_000 * 100

    return {
        "avg_util":        round(per_truck["util_pct"].mean(), 1),
        "max_util":        round(per_truck["util_pct"].max(), 1),
        "avg_stops":       round(per_truck["stops"].mean(), 1),
        "avg_tl_weight":   round(per_truck["weight"].mean(), 0),
        "pct_full_trucks": round((per_truck["util_pct"] >= 80).sum() / len(per_truck) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _winner(c_val, o_val, higher_is_better=True) -> tuple:
    """Return (c_label, o_label) with '★' marking the better value."""
    if higher_is_better:
        better_c = c_val > o_val
    else:
        better_c = c_val < o_val
    c_mark = " ★" if better_c else ""
    o_mark = " ★" if not better_c else ""
    return c_mark, o_mark


def print_comparison(c: dict, o: dict, cm: dict, om: dict) -> None:
    sep  = "─" * 66
    sep2 = "═" * 66

    print(f"\n{sep2}")
    print("  SAVINGS COMPARISON  —  Clustering  vs  Linear Optimizer (MILP)")
    print(sep2)
    print(f"\n  {'Metric':<38} {'Clustering':>12} {'Optimizer':>12}")
    print(sep)

    # Overall cost
    print(f"\n  COST SUMMARY")
    print(f"  {'Baseline (all-LTL)':<38} "
          f"{'${:,.0f}'.format(c['baseline']):>12} "
          f"{'${:,.0f}'.format(o['baseline']):>12}")

    c_mark, o_mark = _winner(c['savings'], o['savings'])
    print(f"  {'Optimised cost':<38} "
          f"{'${:,.0f}'.format(c['optimized']):>12} "
          f"{'${:,.0f}'.format(o['optimized']):>12}")
    print(f"  {'Total savings':<38} "
          f"{'${:,.0f}'.format(c['savings']) + c_mark:>12} "
          f"{'${:,.0f}'.format(o['savings']) + o_mark:>12}")

    c_mark, o_mark = _winner(c['savings_pct'], o['savings_pct'])
    print(f"  {'Savings %':<38} "
          f"{str(c['savings_pct']) + '%' + c_mark:>12} "
          f"{str(o['savings_pct']) + '%' + o_mark:>12}")

    delta = o['savings'] - c['savings']
    print(f"\n  {'Optimizer advantage':<38} "
          f"{'—':>12} "
          f"{'${:,.0f}'.format(delta):>12}")

    # Mode split
    print(f"\n  {'─'*64}")
    print(f"\n  LOAD PLAN SPLIT")
    print(f"  {'TL trucks dispatched':<38} {c['n_trucks']:>12,} {o['n_trucks']:>12,}")

    c_mark, o_mark = _winner(c['n_tl_ships'], o['n_tl_ships'])
    print(f"  {'Shipments on TL':<38} "
          f"{str(c['n_tl_ships']) + c_mark:>12} "
          f"{str(o['n_tl_ships']) + o_mark:>12}")

    c_mark, o_mark = _winner(c['n_ltl_ships'], o['n_ltl_ships'], higher_is_better=False)
    print(f"  {'Shipments staying LTL':<38} "
          f"{str(c['n_ltl_ships']) + c_mark:>12} "
          f"{str(o['n_ltl_ships']) + o_mark:>12}")

    print(f"  {'TL freight cost':<38} "
          f"{'${:,.0f}'.format(c['tl_cost']):>12} "
          f"{'${:,.0f}'.format(o['tl_cost']):>12}")
    print(f"  {'LTL freight cost':<38} "
          f"{'${:,.0f}'.format(c['ltl_cost']):>12} "
          f"{'${:,.0f}'.format(o['ltl_cost']):>12}")

    # Truck efficiency
    print(f"\n  {'─'*64}")
    print(f"\n  TRUCK EFFICIENCY")

    c_mark, o_mark = _winner(cm['avg_util'], om['avg_util'])
    print(f"  {'Avg TL weight utilisation':<38} "
          f"{str(cm['avg_util']) + '%' + c_mark:>12} "
          f"{str(om['avg_util']) + '%' + o_mark:>12}")

    c_mark, o_mark = _winner(cm['max_util'], om['max_util'])
    print(f"  {'Max TL weight utilisation':<38} "
          f"{str(cm['max_util']) + '%' + c_mark:>12} "
          f"{str(om['max_util']) + '%' + o_mark:>12}")

    c_mark, o_mark = _winner(cm['pct_full_trucks'], om['pct_full_trucks'])
    print(f"  {'Trucks ≥ 80% full':<38} "
          f"{str(cm['pct_full_trucks']) + '%' + c_mark:>12} "
          f"{str(om['pct_full_trucks']) + '%' + o_mark:>12}")

    print(f"  {'Avg stops per TL truck':<38} "
          f"{cm['avg_stops']:>12.1f} {om['avg_stops']:>12.1f}")

    print(f"  {'Avg weight per TL truck (lbs)':<38} "
          f"{cm['avg_tl_weight']:>12,.0f} {om['avg_tl_weight']:>12,.0f}")

    # Per-truck savings
    c_mark, o_mark = _winner(c['best_truck_savings'], o['best_truck_savings'])
    print(f"  {'Best single-truck savings':<38} "
          f"{'${:,.0f}'.format(c['best_truck_savings']) + c_mark:>12} "
          f"{'${:,.0f}'.format(o['best_truck_savings']) + o_mark:>12}")

    print(f"  {'Avg per-truck savings':<38} "
          f"{'${:,.0f}'.format(c['avg_truck_savings']):>12} "
          f"{'${:,.0f}'.format(o['avg_truck_savings']):>12}")

    # Method summary
    print(f"\n  {'─'*64}")
    print(f"\n  METHOD")
    print(f"  {'Grouping logic':<38} {'Haversine+Bearing':>12} {'MILP/region':>12}")
    print(f"  {'Optimality':<38} {'Heuristic':>12} {'Exact':>12}")
    print(f"  {'Speed':<38} {'<0.1s':>12} {'~0.3s':>12}")
    print(f"\n  ★ = better value for that metric")
    print(f"{sep2}\n")


def print_region_comparison(
    c_savings: pd.DataFrame,
    o_savings: pd.DataFrame,
) -> None:
    """
    Compare savings by shared geographic grouping.
    Clustering uses CLUSTER labels; optimizer uses REGION labels.
    We show both independently side by side.
    """
    print(f"  CLUSTERING — by geographic cluster (haversine + bearing):")
    c_rows = c_savings[c_savings["section"] == "CLUSTER"][
        ["label", "n_shipments", "tl_shipments", "ltl_shipments",
         "baseline_ltl_cost", "optimized_cost", "savings", "savings_pct"]
    ].copy()
    c_rows.columns = ["Cluster", "Shpmnts", "TL", "LTL",
                      "Baseline", "Optimised", "Savings", "Savings%"]
    print(tabulate(c_rows, headers="keys", tablefmt="rounded_outline", showindex=False))

    print(f"\n  OPTIMIZER — by US region (MILP):")
    o_rows = o_savings[o_savings["section"] == "REGION"][
        ["label", "n_shipments", "tl_shipments", "ltl_shipments",
         "baseline_ltl_cost", "optimized_cost", "savings", "savings_pct"]
    ].copy()
    o_rows.columns = ["Region", "Shpmnts", "TL", "LTL",
                      "Baseline", "Optimised", "Savings", "Savings%"]
    print(tabulate(o_rows, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def save_comparison(c: dict, o: dict, cm: dict, om: dict, path: str) -> None:
    """Write a flat comparison CSV with one row per metric."""
    rows = [
        # Section, Metric, Clustering, Optimizer, Better
        ("Cost",       "Baseline LTL Cost ($)",       f"${c['baseline']:,.2f}",       f"${o['baseline']:,.2f}",       "—"),
        ("Cost",       "Optimised Cost ($)",           f"${c['optimized']:,.2f}",       f"${o['optimized']:,.2f}",       "Optimizer"),
        ("Cost",       "Total Savings ($)",            f"${c['savings']:,.2f}",         f"${o['savings']:,.2f}",         "Optimizer"),
        ("Cost",       "Savings %",                    f"{c['savings_pct']}%",          f"{o['savings_pct']}%",          "Optimizer"),
        ("Cost",       "Optimizer Advantage ($)",      "—",                             f"${o['savings']-c['savings']:,.2f}", "—"),
        ("Load Split", "TL Trucks Dispatched",         str(c['n_trucks']),              str(o['n_trucks']),              "Optimizer"),
        ("Load Split", "Shipments on TL",              str(c['n_tl_ships']),            str(o['n_tl_ships']),            "Clustering" if c['n_tl_ships'] > o['n_tl_ships'] else "Optimizer"),
        ("Load Split", "Shipments Staying LTL",        str(c['n_ltl_ships']),           str(o['n_ltl_ships']),           "Clustering" if c['n_ltl_ships'] < o['n_ltl_ships'] else "Optimizer"),
        ("Load Split", "TL Freight Cost ($)",          f"${c['tl_cost']:,.2f}",         f"${o['tl_cost']:,.2f}",         "—"),
        ("Load Split", "LTL Freight Cost ($)",         f"${c['ltl_cost']:,.2f}",        f"${o['ltl_cost']:,.2f}",        "—"),
        ("Efficiency", "Avg TL Utilisation (%)",       f"{cm['avg_util']}%",            f"{om['avg_util']}%",            "Optimizer" if om['avg_util'] > cm['avg_util'] else "Clustering"),
        ("Efficiency", "Max TL Utilisation (%)",       f"{cm['max_util']}%",            f"{om['max_util']}%",            "Optimizer" if om['max_util'] > cm['max_util'] else "Clustering"),
        ("Efficiency", "Trucks >= 80% Full (%)",       f"{cm['pct_full_trucks']}%",     f"{om['pct_full_trucks']}%",     "Optimizer" if om['pct_full_trucks'] > cm['pct_full_trucks'] else "Clustering"),
        ("Efficiency", "Avg Stops per TL Truck",       f"{cm['avg_stops']}",            f"{om['avg_stops']}",            "—"),
        ("Efficiency", "Avg Weight per Truck (lbs)",   f"{cm['avg_tl_weight']:,.0f}",   f"{om['avg_tl_weight']:,.0f}",   "—"),
        ("Efficiency", "Best Single-Truck Savings ($)",f"${c['best_truck_savings']:,.2f}", f"${o['best_truck_savings']:,.2f}", "Optimizer" if o['best_truck_savings'] > c['best_truck_savings'] else "Clustering"),
        ("Efficiency", "Avg Per-Truck Savings ($)",    f"${c['avg_truck_savings']:,.2f}",  f"${o['avg_truck_savings']:,.2f}",  "Optimizer" if o['avg_truck_savings'] > c['avg_truck_savings'] else "Clustering"),
        ("Method",     "Grouping Logic",               "Haversine + Bearing",           "US Region Partition",           "—"),
        ("Method",     "Algorithm",                    "AgglomerativeClustering",        "MILP (PuLP + CBC)",             "—"),
        ("Method",     "Optimality",                   "Heuristic",                     "Exact (per region)",            "—"),
        ("Method",     "Speed",                        "<0.1s",                         "~0.3s",                         "—"),
    ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = pd.DataFrame(rows, columns=["section", "metric", "clustering", "optimizer", "better"])
    out.to_csv(path, index=False)
    print(f"  Comparison saved   → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare clustering vs optimizer savings")
    p.add_argument("--clustering-savings", default=DEFAULT_CLUS_SAV)
    p.add_argument("--clustering-plan",    default=DEFAULT_CLUS_PLAN)
    p.add_argument("--optimizer-savings",  default=DEFAULT_OPT_SAV)
    p.add_argument("--optimizer-plan",     default=DEFAULT_OPT_PLAN)
    p.add_argument("--output",             default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*66}")
    print("  LOADING RESULTS …")
    print(f"{'═'*66}")
    print(f"  Clustering  : {args.clustering_savings}")
    print(f"  Optimizer   : {args.optimizer_savings}")

    # Load savings summaries
    c_sav_df = pd.read_csv(args.clustering_savings)
    o_sav_df = pd.read_csv(args.optimizer_savings)

    c = _load_savings(args.clustering_savings, "Clustering")
    o = _load_savings(args.optimizer_savings,  "Optimizer")
    cm = _load_plan_metrics(args.clustering_plan, "Clustering")
    om = _load_plan_metrics(args.optimizer_plan,  "Optimizer")

    # Print comparison
    print_comparison(c, o, cm, om)
    print_region_comparison(c_sav_df, o_sav_df)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_comparison(c, o, cm, om, args.output)
    print()


if __name__ == "__main__":
    main()
