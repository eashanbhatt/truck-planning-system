"""
clustering/run.py  —  Solution 1: Haversine Clustering
========================================================
Builds a truck load plan using geographic proximity clustering.

Pipeline
--------
1. Load shipments CSV (one warehouse)
2. Compute pairwise haversine distance matrix between all destinations
3. AgglomerativeClustering (metric='precomputed', linkage='complete')
   → groups destinations within `--threshold` miles into one cluster
4. Greedy bin-packing within each cluster
   → packs shipments into TL trucks (max 44k lbs, max 3 stops)
   → leaves OPTIONAL shipments as LTL if TL is not cost-effective
5. Output load plan CSV + savings summary CSV

Run
---
    python clustering/run.py
    python clustering/run.py --input data/sample_shipments.csv --threshold 250
    python clustering/run.py --help

Outputs (written to clustering/output/)
----------------------------------------
    clustering_plan.csv     — row per shipment with truck assignment
    clustering_savings.csv  — OVERALL / CLUSTER / TRUCK / LTL savings breakdown
"""

import argparse
import os
import sys
import time

# Add project root to path so src imports work when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader       import load, print_summary
from src.haversine_cluster import build_clusters, cluster_summary
from src.bin_packer        import pack
from src.reporter          import print_results, save_plan, save_savings


def parse_args():
    p = argparse.ArgumentParser(
        description="Solution 1 — Haversine clustering truck load planner"
    )
    p.add_argument(
        "--input", default=os.path.join(os.path.dirname(__file__), "..", "data", "sample_shipments.csv"),
        help="Path to shipments CSV",
    )
    p.add_argument(
        "--threshold", type=float, default=200.0,
        help="Max intra-cluster haversine distance in miles (default: 200)",
    )
    p.add_argument(
        "--plan-out", default=os.path.join(os.path.dirname(__file__), "output", "clustering_plan.csv"),
        help="Output path for load plan CSV",
    )
    p.add_argument(
        "--savings-out", default=os.path.join(os.path.dirname(__file__), "output", "clustering_savings.csv"),
        help="Output path for savings summary CSV",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*54}")
    print("  SOLUTION 1 — HAVERSINE CLUSTERING")
    print("  AgglomerativeClustering (precomputed haversine)")
    print(f"{'═'*54}")

    # ── 1. Load ───────────────────────────────────────────────────────────
    print(f"\n[1/4]  Loading  '{args.input}' …")
    try:
        df = load(args.input)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"  ERROR: {e}")
    print_summary(df)

    # Baseline: every shipment ships individually as LTL
    from src.bin_packer import _ltl_cost
    baseline = sum(
        _ltl_cost(r["weight_lbs"], r["wh_to_dest_mi"], r["freight_class"])
        for _, r in df.iterrows()
    )
    print(f"\n  Baseline cost (all LTL): ${baseline:,.0f}")

    # ── 2. Cluster ────────────────────────────────────────────────────────
    print(f"\n[2/4]  Building haversine distance matrix + clustering "
          f"(threshold = {args.threshold:.0f} mi) …")
    t0 = time.time()
    clustered = build_clusters(df, distance_threshold_miles=args.threshold)
    n_clusters = clustered["cluster_id"].nunique()
    print(f"  {n_clusters} clusters formed from {len(df)} shipments")

    cs = cluster_summary(clustered)
    print(f"\n  Cluster summary (top 8 by weight):")
    from tabulate import tabulate
    display = cs.head(8)[
        ["cluster_id", "n_shipments", "total_weight", "n_destinations",
         "must_tl_count", "max_intra_dist_mi", "cities"]
    ].copy()
    display["total_weight"] = display["total_weight"].map("{:,.0f}".format)
    display.columns = ["Cluster", "Shpmnts", "Weight (lbs)", "Dests",
                       "MUST_TL", "Max Dist (mi)", "Cities"]
    print(tabulate(display, headers="keys", tablefmt="rounded_outline", showindex=False))

    # ── 3. Pack ───────────────────────────────────────────────────────────
    print(f"\n[3/4]  Bin-packing clusters into TL trucks …")
    plan    = pack(clustered)
    elapsed = time.time() - t0

    # ── 4. Report + Save ──────────────────────────────────────────────────
    print(f"\n[4/4]  Saving outputs …")
    print_results(plan, baseline, elapsed)
    save_plan(plan, args.plan_out)
    save_savings(plan, args.savings_out)
    print()


if __name__ == "__main__":
    main()
