"""
run_optimizer.py  —  Module 2
================================
Load plan built using a Mixed-Integer Linear Program (MILP).
Solver: PuLP with open-source CBC backend — no licence required.

The MILP minimises total freight cost (TL $/mile + LTL $/cwt)
subject to weight capacity and max-stops constraints per truck.

Usage
-----
    python run_optimizer.py --input data/sample_shipments.csv
    python run_optimizer.py --input data/sample_shipments.csv --output output/milp_plan.csv --verbose
"""

import argparse
import os
import sys
import time

import pandas as pd
from tabulate import tabulate

from src.data_loader import load_shipments, summary as data_summary
from src.optimizer   import LoadOptimizer
from src.cost_utils  import baseline_ltl_cost, save_savings_summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Module 2 — MILP-based truck load planner (PuLP / CBC)"
    )
    p.add_argument("--input",   default="data/sample_shipments.csv",
                   help="Path to input CSV  (default: data/sample_shipments.csv)")
    p.add_argument("--output",  default="output/milp_plan.csv",
                   help="Path for load plan CSV  (default: output/milp_plan.csv)")
    p.add_argument("--savings", default="output/milp_savings.csv",
                   help="Path for savings summary CSV  (default: output/milp_savings.csv)")
    p.add_argument("--verbose", action="store_true",
                   help="Print CBC solver logs for each region")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def print_cost_summary(df: pd.DataFrame, baseline: float, elapsed: float) -> None:
    optimized = df["plan_cost"].sum()
    savings   = baseline - optimized
    pct       = savings / baseline * 100 if baseline else 0

    tl_rows  = df[df["assigned_mode"] == "TL"]
    ltl_rows = df[df["assigned_mode"] == "LTL"]
    n_trucks = tl_rows["truck_id"].nunique()

    sep = "─" * 52
    print(f"\n{'═'*52}")
    print("  MODULE 2 — MILP OPTIMISER  (results)")
    print(f"{'═'*52}")
    print(f"\n  {'Metric':<34} {'Value':>14}")
    print(sep)
    print(f"  {'Baseline cost (all LTL)':<34} {'${:>12,.0f}'.format(baseline)}")
    print(f"  {'Optimised cost (MILP)':<34} {'${:>12,.0f}'.format(optimized)}")
    print(f"  {'Total savings':<34} {'${:>12,.0f}'.format(savings)}")
    print(f"  {'Savings %':<34} {pct:>13.1f}%")
    print(sep)
    print(f"  {'TL trucks dispatched':<34} {n_trucks:>14,}")
    print(f"  {'Shipments on TL':<34} {len(tl_rows):>14,}")
    print(f"  {'Shipments staying LTL':<34} {len(ltl_rows):>14,}")
    if n_trucks:
        avg_util = tl_rows.groupby("truck_id")["weight_lbs"].sum().mean() / 44_000 * 100
        print(f"  {'Avg TL weight utilisation':<34} {avg_util:>13.1f}%")
    print(f"  {'Solver':<34} {'PuLP + CBC (open-source)':>14}")
    print(f"  {'Plan generated in':<34} {elapsed:>12.2f}s")
    print(f"{'═'*52}\n")


def print_truck_table(summary_df: pd.DataFrame) -> None:
    display = summary_df[
        ["truck_id", "region", "n_shipments", "total_weight",
         "n_stops", "truck_cost", "ltl_baseline", "savings", "util_pct", "status"]
    ].copy()
    display["total_weight"] = display["total_weight"].map("{:,.0f}".format)
    display["truck_cost"]   = display["truck_cost"].map("${:,.0f}".format)
    display["ltl_baseline"] = display["ltl_baseline"].map("${:,.0f}".format)
    display["savings"]      = display["savings"].map("${:,.0f}".format)
    display["util_pct"]     = display["util_pct"].map("{:.1f}%".format)
    display.columns = ["Truck", "Region", "Shpmnts", "Weight (lbs)",
                        "Stops", "TL Cost", "LTL Baseline", "Savings",
                        "Util%", "Status"]
    print("  TL Truck Details (MILP solution):")
    print(tabulate(display, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()


def print_region_breakdown(df: pd.DataFrame, baseline: float) -> None:
    region_grp = df.groupby("region").agg(
        n_ships   = ("shipment_id",   "count"),
        tl_ships  = ("assigned_mode", lambda x: (x == "TL").sum()),
        ltl_ships = ("assigned_mode", lambda x: (x == "LTL").sum()),
        opt_cost  = ("plan_cost",     "sum"),
        base_cost = ("ltl_cost_ind",  "sum"),
    ).reset_index()
    region_grp["savings"] = region_grp["base_cost"] - region_grp["opt_cost"]
    region_grp["pct"]     = (region_grp["savings"] / region_grp["base_cost"] * 100).round(1)

    display = region_grp.copy()
    for col in ["opt_cost", "base_cost", "savings"]:
        display[col] = display[col].map("${:,.0f}".format)
    display["pct"] = display["pct"].map("{:.1f}%".format)
    display.columns = ["Region", "Total", "→TL", "→LTL",
                        "Optimised", "Baseline", "Savings", "Savings%"]
    print("  Cost Breakdown by Region:")
    print(tabulate(display, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()


def print_ltl_table(df: pd.DataFrame) -> None:
    ltl = df[df["assigned_mode"] == "LTL"][
        ["shipment_id", "dest_city", "dest_state", "region",
         "weight_lbs", "freight_class", "ltl_cost_ind"]
    ].copy()
    ltl["weight_lbs"]   = ltl["weight_lbs"].map("{:,.0f}".format)
    ltl["ltl_cost_ind"] = ltl["ltl_cost_ind"].map("${:,.0f}".format)
    ltl.columns = ["Shipment", "Dest City", "ST", "Region",
                    "Weight (lbs)", "FC", "LTL Cost"]
    print("  LTL Shipments (individual):")
    print(tabulate(ltl, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"\n{'═'*52}")
    print("  TRUCK LOAD PLANNING  —  MODULE 2: MILP OPTIMISER")
    print(f"{'═'*52}")

    # 1. Load
    print(f"\n[1/3] Loading shipments from '{args.input}' …")
    try:
        df = load_shipments(args.input)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"  ERROR: {e}")
    data_summary(df)

    baseline = baseline_ltl_cost(df)
    print(f"\n  Baseline cost (all LTL): ${baseline:,.0f}")

    # 2. Optimise
    n_regions = df["region"].nunique()
    print(f"\n[2/3] Solving MILP for {n_regions} region(s) using PuLP + CBC …")
    t0  = time.time()
    opt = LoadOptimizer(solver_verbose=args.verbose)
    plan = opt.build_plan(df)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    # 3. Output
    print(f"\n[3/3] Building output …")
    truck_summary = opt.plan_summary()
    print_cost_summary(plan, baseline, elapsed)
    print_truck_table(truck_summary)
    print_region_breakdown(plan, baseline)
    print_ltl_table(plan)

    # 4. Save CSV
    output_cols = [
        "shipment_id", "warehouse", "dest_city", "dest_state",
        "weight_lbs", "freight_class", "pickup_date", "delivery_due",
        "mode", "region", "distance_miles",
        "truck_id", "assigned_mode",
        "tl_cost", "ltl_cost_ind", "plan_cost", "solver_status",
    ]
    plan[output_cols].to_csv(args.output, index=False)
    print(f"  Load plan saved    → {args.output}")
    save_savings_summary(plan, args.savings, module="MILP Optimizer")
    print()


if __name__ == "__main__":
    main()
