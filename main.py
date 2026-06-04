"""
main.py
-------
End-to-end demo of the Truck Load Planning System.

Pipeline
--------
1. Generate synthetic shipments (mimicking LSC Communications' ops)
2. Stage 1 clustering  — KMeans zones on destination lat/lon
3. Stage 2 clustering  — Agglomerative within each zone
4. MILP optimisation   — assign shipments to TL trucks, minimise cost
5. Print results table
6. Render visualisations

Run
---
    python main.py
    python main.py --shipments 300 --zones 10 --seed 7
"""

import argparse
import time
from tabulate import tabulate

from src.data_generator import generate_shipments
from src.clustering      import LoadConsolidator
from src.optimizer       import LoadOptimizer
from src.visualizer      import plot_clusters, plot_savings, plot_load_plan, plot_dendrogram


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Truck Load Planning System demo")
    p.add_argument("--shipments", type=int, default=200,
                   help="Number of shipments to generate (default: 200)")
    p.add_argument("--zones",     type=int, default=8,
                   help="KMeans zones / regions (default: 8)")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--no-plots",  action="store_true",
                   help="Skip matplotlib visualisations")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    banner = "═" * 56
    print(f"\n{banner}")
    print("  TRUCK LOAD PLANNING SYSTEM")
    print("  KMeans + Hierarchical Clustering · PuLP MILP Optimizer")
    print(f"{banner}\n")

    # ── 1. Data ──────────────────────────────────────────────────────────
    print(f"[1/4] Generating {args.shipments} synthetic shipments …")
    t0 = time.time()
    shipments = generate_shipments(n_shipments=args.shipments, seed=args.seed)
    print(f"      Done in {time.time()-t0:.2f}s  |  "
          f"{shipments['region'].nunique()} regions  |  "
          f"Total weight: {shipments['weight_lbs'].sum():,.0f} lbs\n")

    # ── 2 & 3. Clustering ─────────────────────────────────────────────────
    print(f"[2/4] Clustering shipments (KMeans k={args.zones} → Agglomerative) …")
    t0 = time.time()
    lc = LoadConsolidator(n_zones=args.zones, max_cluster_miles=150, seed=args.seed)
    clustered = lc.fit(shipments)
    summary   = lc.cluster_summary()
    n_clusters    = len(summary)
    n_tl_eligible = summary["is_tl_eligible"].sum()
    print(f"      Done in {time.time()-t0:.2f}s  |  "
          f"{n_clusters} clusters formed  |  "
          f"{n_tl_eligible} TL-eligible (≥15k lbs)\n")

    # Print cluster summary table (top 10)
    top = summary.head(10)[
        ["cluster_id", "n_shipments", "total_weight",
         "total_volume", "utilisation_pct", "is_tl_eligible", "region"]
    ].copy()
    top["total_weight"] = top["total_weight"].map("{:,.0f}".format)
    top["total_volume"] = top["total_volume"].map("{:,.0f}".format)
    print("  Top 10 clusters by weight:")
    print(tabulate(top, headers="keys", tablefmt="rounded_outline",
                   showindex=False))
    print()

    # ── 4. Optimisation ───────────────────────────────────────────────────
    print("[3/4] Running MILP optimisation (PuLP / CBC) …")
    t0 = time.time()
    opt  = LoadOptimizer(solver_msg=False)
    plan = opt.optimize(clustered, summary)
    elapsed = time.time() - t0
    print(f"      Solved in {elapsed:.1f}s\n")

    opt.print_summary()

    # ── 5. Detailed plan sample ───────────────────────────────────────────
    tl_sample = (
        plan[plan["mode"] == "TL"]
        .sort_values("savings", ascending=False)
        .head(8)[["truck_id", "total_weight", "total_volume",
                  "tl_cost", "ltl_cost", "savings"]]
        .copy()
    )
    for col in ["total_weight", "total_volume"]:
        tl_sample[col] = tl_sample[col].map("{:,.0f}".format)
    for col in ["tl_cost", "ltl_cost", "savings"]:
        tl_sample[col] = tl_sample[col].map("${:,.0f}".format)

    print("  Top TL loads by savings:")
    print(tabulate(tl_sample, headers="keys", tablefmt="rounded_outline",
                   showindex=False))
    print()

    # ── 6. Visualisations ─────────────────────────────────────────────────
    if not args.no_plots:
        print("[4/4] Rendering visualisations …")
        plot_clusters(clustered)
        plot_savings(clustered, plan)
        plot_load_plan(plan)
        plot_dendrogram(clustered, zone_id=0)
    else:
        print("[4/4] Plots skipped (--no-plots).")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
