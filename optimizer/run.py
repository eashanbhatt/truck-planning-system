"""
optimizer/run.py  —  Solution 2: Linear Optimizer (MILP)
==========================================================
Builds a truck load plan using Mixed-Integer Linear Programming.
Solver: PuLP + CBC (open-source, no licence required).

This solution is completely independent of the clustering approach.
It makes no assumptions about geographic proximity — instead it
minimises total freight cost subject to weight and stop constraints.

Pipeline
--------
1. Load shipments CSV (one warehouse)
2. Compute individual LTL cost for every shipment (baseline)
3. Partition shipments by US region (for tractability)
4. Solve one MILP per region:
   - Variables: x[shipment, truck], y[truck], z[destination, truck]
   - Objective: min Σ TL_cost·y[t] + Σ LTL_cost·(1-Σx[s,t])
   - Constraints: weight ≤ 44k lbs/truck, stops ≤ 3/truck,
                  MUST_TL forced onto truck
5. Output load plan CSV + savings summary CSV

Run
---
    python optimizer/run.py
    python optimizer/run.py --input data/sample_shipments.csv
    python optimizer/run.py --verbose   # show CBC solver logs
    python optimizer/run.py --help

Outputs (written to optimizer/output/)
---------------------------------------
    milp_plan.csv     — row per shipment with truck assignment
    milp_savings.csv  — OVERALL / REGION / TRUCK / LTL savings breakdown
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load, print_summary
from src.milp        import LinearOptimizer, ltl_cost_series
from src.reporter    import print_results, save_plan, save_savings


def parse_args():
    p = argparse.ArgumentParser(
        description="Solution 2 — Linear optimizer (MILP) truck load planner"
    )
    p.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "sample_shipments.csv"),
        help="Path to shipments CSV",
    )
    p.add_argument(
        "--plan-out",
        default=os.path.join(os.path.dirname(__file__), "output", "milp_plan.csv"),
        help="Output path for load plan CSV",
    )
    p.add_argument(
        "--savings-out",
        default=os.path.join(os.path.dirname(__file__), "output", "milp_savings.csv"),
        help="Output path for savings summary CSV",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print CBC solver logs for each region sub-problem",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*54}")
    print("  SOLUTION 2 — LINEAR OPTIMIZER (MILP)")
    print("  PuLP + CBC  |  Open-source  |  No licence required")
    print(f"{'═'*54}")

    # ── 1. Load ───────────────────────────────────────────────────────────
    print(f"\n[1/3]  Loading  '{args.input}' …")
    try:
        df = load(args.input)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"  ERROR: {e}")
    print_summary(df)

    baseline = ltl_cost_series(df).sum()
    print(f"\n  Baseline cost (all LTL): ${baseline:,.0f}")
    print(f"  Regions to solve: {', '.join(sorted(df['region'].unique()))}")

    # ── 2. Optimise ───────────────────────────────────────────────────────
    n_regions = df["region"].nunique()
    print(f"\n[2/3]  Solving MILP for {n_regions} region(s) using PuLP + CBC …")
    t0  = time.time()
    opt = LinearOptimizer(solver_verbose=args.verbose)
    plan = opt.build_plan(df)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    # ── 3. Report + Save ──────────────────────────────────────────────────
    print(f"\n[3/3]  Saving outputs …")
    print_results(plan, baseline, elapsed)
    os.makedirs(os.path.dirname(args.plan_out), exist_ok=True)
    save_plan(plan, args.plan_out)
    save_savings(plan, args.savings_out)
    print()


if __name__ == "__main__":
    main()
