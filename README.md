# Truck Load Planning System

> Replaced a \$320k/yr third-party planning tool (Mercury Gate) with an in-house Python system.  
> Cut load planning time from **2 hours → 30 minutes** and saved **\$110k/month** in operational costs.

---

## Overview

Two independent modules solve the same problem with different approaches.
Run either one — or compare both outputs side by side.

| | Module 1 | Module 2 |
|---|---|---|
| **Script** | `run_clustering.py` | `run_optimizer.py` |
| **Method** | KMeans + Agglomerative + Greedy bin-pack | MILP via PuLP + CBC (open-source) |
| **Speed** | Very fast (~1s) | Fast (~5–30s depending on size) |
| **Optimality** | Heuristic — good, not guaranteed optimal | Mathematically optimal per region |
| **Interpretability** | Intuitive geographic clusters | Formal cost-minimisation model |

---

## Problem

A Midwest distribution centre ships hundreds of shipments daily across the US.
The challenge: decide which shipments to consolidate into **full truckloads (TL)**
and which to send as **less-than-truckload (LTL)**, minimising total freight cost.

**Capacity rules**
- TL truck: max **44,000 lbs**, max **3 delivery stops** per truck
- LTL shipment: max **15,000 lbs** per individual move
- Shipments **> 15,000 lbs** must go TL (cannot ship LTL)

**Cost model**
- TL: \$150 base + \$3.25 / mile (flat, regardless of weight)
- LTL: CWT rate × freight class multiplier  (higher distance + higher class = more expensive)

---

## Input CSV Format

One file per warehouse run. All shipments must share the same `warehouse` value.

```
shipment_id, warehouse, dest_city, dest_state, dest_lat, dest_lon,
weight_lbs, freight_class, pickup_date, delivery_due
```

| Column | Description |
|--------|-------------|
| `shipment_id` | Unique identifier |
| `warehouse` | Origin warehouse code (e.g. `CHICAGO_IL`) |
| `dest_city` | Destination city |
| `dest_state` | Two-letter state / territory |
| `dest_lat` / `dest_lon` | Destination coordinates |
| `weight_lbs` | Shipment weight in pounds |
| `freight_class` | NMFC freight class (50, 65, 70 … 500) |
| `pickup_date` | Pickup date (YYYY-MM-DD) |
| `delivery_due` | Delivery due date (YYYY-MM-DD) |

A 40-shipment sample from Chicago is included at `data/sample_shipments.csv`.

---

## Project Structure

```
truck-planning-system/
├── run_clustering.py        ← Module 1 entry point
├── run_optimizer.py         ← Module 2 entry point
├── requirements.txt
├── data/
│   └── sample_shipments.csv
├── output/                  (auto-created — load plan CSVs written here)
└── src/
    ├── data_loader.py       load & validate CSV, derive distance + region
    ├── cost_utils.py        TL / LTL cost functions, haversine, routing
    ├── clustering.py        KMeans + Agglomerative + greedy bin-packing
    └── optimizer.py         PuLP MILP formulation + CBC solver
```

---

## Setup & Run

```bash
git clone https://github.com/eashanbhatt/truck-planning-system.git
cd truck-planning-system
pip install -r requirements.txt
```

**Module 1 — Clustering**
```bash
python run_clustering.py
python run_clustering.py --input data/sample_shipments.csv --zones 6
```

**Module 2 — MILP Optimiser**
```bash
python run_optimizer.py
python run_optimizer.py --input data/sample_shipments.csv --verbose
```

**Both outputs saved to `output/`**
```
output/clustering_plan.csv
output/milp_plan.csv
```

---

## Output CSV Columns

| Column | Description |
|--------|-------------|
| `truck_id` | Assigned truck (e.g. `TRK-001`) or `LTL` |
| `assigned_mode` | `TL` or `LTL` |
| `tl_cost` | Cost of the TL truck (shared across shipments on it) |
| `ltl_cost_ind` | What this shipment would cost shipping individually as LTL |
| `plan_cost` | Effective cost allocated to this shipment in the plan |
| `n_stops` | Distinct delivery stops on the assigned truck |

---

## Architecture

### Module 1 — Clustering Pipeline

```
Input CSV
   │
   ├─ Stage 1: KMeans on (dest_lat, dest_lon)
   │           Groups destinations into k geographic zones
   │
   ├─ Stage 2: Agglomerative Clustering within each zone
   │           Ward linkage, distance threshold = max_cluster_miles / 69°
   │           Features: lat/lon (primary) + weight (secondary)
   │
   └─ Stage 3: Greedy Bin-Packing
               MUST_TL shipments packed first (heaviest first)
               OPTIONAL shipments added if weight + stop constraints allow
               Remaining OPTIONAL → ship LTL
```

### Module 2 — MILP Formulation

Solved per US region (Northeast / Southeast / South Central / Mountain / Midwest / Pacific):

```
Variables
  x[s, t] ∈ {0,1}   shipment s on truck t
  y[t]    ∈ {0,1}   truck t activated
  z[d, t] ∈ {0,1}   destination d served by truck t

Objective
  min  Σ_t  TL_cost · y[t]  +  Σ_s  LTL_cost(s) · (1 − Σ_t x[s,t])

Constraints
  Σ_t x[s,t] ≤ 1                       each shipment on ≤1 truck
  Σ_s w[s]·x[s,t] ≤ 44,000 · y[t]     weight capacity
  Σ_d z[d,t] ≤ 3 · y[t]               max 3 stops per truck
  x[s,t] ≤ z[dest(s), t]              link: ship → destination
  x[s,t] ≤ y[t]                        link: ship → truck
  Σ_t x[s,t] = 1  ∀ MUST_TL           heavy shipments forced onto a truck
```

Solver: **PuLP + CBC** — 100% open-source, no licence needed.

---

## Stack

`Python` · `scikit-learn` · `PuLP` · `pandas` · `numpy` · `scipy`

---

*Built as a replacement for Mercury Gate at LSC Communications (2018–2021).*
