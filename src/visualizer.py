"""
visualizer.py
-------------
Plotting utilities for the truck planning system.

    plot_clusters(df)       - scatter map of shipments coloured by cluster zone
    plot_savings(plan_df)   - bar chart comparing TL vs LTL cost by region
    plot_load_plan(plan_df) - weight utilisation per TL truck
    plot_dendrogram(df)     - hierarchical clustering dendrogram for a single zone
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Consistent colour palette
PALETTE = sns.color_palette("tab10", 12)
ACCENT  = "#1a56db"
ORANGE  = "#d94f00"


# ---------------------------------------------------------------------------
# 1. Cluster scatter map
# ---------------------------------------------------------------------------

def plot_clusters(df: pd.DataFrame, save_path: str = None) -> None:
    """
    US scatter map of shipment destinations coloured by KMeans zone.

    Parameters
    ----------
    df : clustered DataFrame (output of LoadConsolidator.fit())
    save_path : if provided, save the figure to this path
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f8faff")
    ax.set_facecolor("#f0f5ff")

    zones = df["zone_id"].unique()
    colours = {z: PALETTE[i % len(PALETTE)] for i, z in enumerate(sorted(zones))}

    for zone, grp in df.groupby("zone_id"):
        ax.scatter(
            grp["dest_lon"], grp["dest_lat"],
            c=[colours[zone]], s=grp["weight_lbs"] / 400 + 20,
            alpha=0.75, edgecolors="white", linewidths=0.4,
            label=f"Zone {zone}",
        )

    # Mark distribution centres
    origins = df[["origin_city", "origin_lat", "origin_lon"]].drop_duplicates()
    ax.scatter(
        origins["origin_lon"], origins["origin_lat"],
        marker="*", s=300, c=ORANGE, zorder=5, label="Distribution Centre",
    )

    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 52)
    ax.set_xlabel("Longitude", fontsize=10, color="#475569")
    ax.set_ylabel("Latitude",  fontsize=10, color="#475569")
    ax.set_title(
        "Shipment Destinations — KMeans Zone Clustering\n"
        "(bubble size ∝ shipment weight)",
        fontsize=13, fontweight="bold", color="#0f172a", pad=14,
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Savings bar chart
# ---------------------------------------------------------------------------

def plot_savings(clustered_df: pd.DataFrame, plan_df: pd.DataFrame,
                 save_path: str = None) -> None:
    """
    Side-by-side bar chart: baseline LTL cost vs optimised cost by region.
    """
    # Join plan back to region info
    exploded = plan_df.explode("shipments").rename(columns={"shipments": "shipment_id"})
    merged = exploded.merge(
        clustered_df[["shipment_id", "region", "ltl_cost_baseline"]],
        on="shipment_id", how="left",
    )

    region_baseline = merged.groupby("region")["ltl_cost_baseline"].sum()
    region_opt      = merged.groupby("region")["optimized_cost"].sum()

    regions = sorted(region_baseline.index)
    x = np.arange(len(regions))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#f8faff")
    ax.set_facecolor("#f8faff")

    bars1 = ax.bar(x - w/2, [region_baseline[r] for r in regions],
                   width=w, color=ORANGE, alpha=0.75, label="Baseline (all LTL)",
                   edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w/2, [region_opt[r] for r in regions],
                   width=w, color=ACCENT, alpha=0.85, label="Optimised (TL + LTL)",
                   edgecolor="white", linewidth=0.8)

    # Savings annotations
    for b1, b2 in zip(bars1, bars2):
        saving = b1.get_height() - b2.get_height()
        pct    = saving / b1.get_height() * 100 if b1.get_height() else 0
        ax.annotate(
            f"−{pct:.0f}%",
            xy=(b2.get_x() + b2.get_width() / 2, b2.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=8, color="#16a34a", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=10)
    ax.set_ylabel("Total Freight Cost ($)", fontsize=10, color="#475569")
    ax.set_title("Freight Cost: Baseline vs Optimised by Region",
                 fontsize=13, fontweight="bold", color="#0f172a", pad=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. TL utilisation chart
# ---------------------------------------------------------------------------

def plot_load_plan(plan_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Horizontal bar chart showing weight utilisation for every TL truck.
    """
    tl = plan_df[plan_df["mode"] == "TL"].copy()
    if tl.empty:
        print("No TL loads to display.")
        return

    tl = tl.sort_values("total_weight", ascending=True).head(20)  # top 20 for readability
    tl["util_pct"] = tl["total_weight"] / TL_WEIGHT_CAPACITY * 100

    fig, ax = plt.subplots(figsize=(10, max(4, len(tl) * 0.45)))
    fig.patch.set_facecolor("#f8faff")
    ax.set_facecolor("#f8faff")

    colours = [ACCENT if u >= 70 else "#93c5fd" for u in tl["util_pct"]]
    bars = ax.barh(tl["truck_id"], tl["util_pct"], color=colours,
                   edgecolor="white", linewidth=0.6)

    ax.axvline(70, color=ORANGE, linestyle="--", linewidth=1.2,
               label="70% utilisation target")
    ax.axvline(100, color="#64748b", linestyle=":", linewidth=1.0)

    for bar, w in zip(bars, tl["total_weight"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{w:,.0f} lbs", va="center", fontsize=8, color="#475569")

    ax.set_xlabel("Weight Utilisation (%)", fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_title("TL Truck Weight Utilisation",
                 fontsize=13, fontweight="bold", color="#0f172a", pad=14)
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Dendrogram for a single zone
# ---------------------------------------------------------------------------

def plot_dendrogram(df: pd.DataFrame, zone_id: int = 0,
                    save_path: str = None) -> None:
    """
    Scipy dendrogram for Agglomerative clustering within a single zone.
    Useful for explaining the hierarchical consolidation logic.
    """
    zone_df = df[df["zone_id"] == zone_id].copy()
    if len(zone_df) < 3:
        print(f"Zone {zone_id} has fewer than 3 shipments — dendrogram not meaningful.")
        return

    features = zone_df[["dest_lat", "dest_lon"]].values
    Z = linkage(features, method="ward")

    fig, ax = plt.subplots(figsize=(max(8, len(zone_df) * 0.5), 5))
    fig.patch.set_facecolor("#f8faff")
    ax.set_facecolor("#f8faff")

    dendrogram(
        Z,
        labels=zone_df["dest_city"].str.split(",").str[0].tolist(),
        leaf_rotation=45, leaf_font_size=8,
        color_threshold=0.6 * max(Z[:, 2]),
        ax=ax,
    )

    ax.set_title(
        f"Hierarchical Clustering Dendrogram — Zone {zone_id}",
        fontsize=12, fontweight="bold", color="#0f172a",
    )
    ax.set_ylabel("Ward Linkage Distance")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Keep constant accessible from module level
TL_WEIGHT_CAPACITY = 44_000
