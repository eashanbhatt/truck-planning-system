# Truck Load Planning System

> Replaced a \$320k/yr third-party planning tool (Mercury Gate) with an in-house Python system.  
> Cut load planning time from **2 hours → 30 minutes** and saved **\$110k/month** in operational costs.

---

## Problem

A large print/publishing distributor was paying \$320k/year for a commercial freight planning
tool to consolidate daily shipments from three Midwest distribution centres into cost-optimal
truck loads. The tool was a black box, slow to query, and offered no customisation.

The goal was to build an in-house replacement that:
- Groups geographically similar shipments into consolidated truckload (TL) moves
- Decides which shipments are better shipped as LTL (less-than-truckload)
- Minimises total freight cost subject to weight and volume constraints
- Runs end-to-end in under 30 minutes (vs 2 hours manually)

---

## Solution

A two-stage ML + optimisation pipeline:

```
Raw Shipments
     │
     ▼
┌─────────────────────────────┐
│  Stage 1 — KMeans           │  Partition destinations into k=8 geographic zones
│  (destination lat/lon)      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 2 — Agglomerative    │  Within each zone, group nearby shipments by
│  (within each zone)         │  proximity + pickup timing + weight
└──────────────┬──────────────┘
               │  Candidate consolidated loads
               ▼
┌─────────────────────────────┐
│  MILP Optimisation (PuLP)   │  Assign shipments to TL trucks or leave as LTL
│  Minimize total freight $   │  Subject to: weight ≤ 44,000 lbs, vol ≤ 2,400 cuft
└──────────────┬──────────────┘
               │
               ▼
          Load Plan
  (truck assignments + cost breakdown)
```

---

## Results

| Metric | Before | After |
|--------|--------|-------|
| Planning time | ~2 hours | ~30 minutes |
| Tool cost | \$320k/year | \$0 (in-house) |
| Monthly freight savings | — | \$110k/month |
| Process | Manual + 3PL tool | Automated Python pipeline |

---

## Project Structure

```
truck-planning-system/
├── main.py                  # End-to-end demo pipeline
├── requirements.txt
└── src/
    ├── data_generator.py    # Synthetic shipment data (200+ records)
    ├── clustering.py        # KMeans zones + Agglomerative within zones
    ├── optimizer.py         # PuLP MILP — TL/LTL assignment
    └── visualizer.py        # Cluster map, savings chart, utilisation bars
```

---

## Setup

```bash
git clone https://github.com/eashanbhatt/truck-planning-system.git
cd truck-planning-system
pip install -r requirements.txt
python main.py
```

Optional flags:
```bash
python main.py --shipments 300 --zones 10 --seed 7
python main.py --no-plots        # skip matplotlib windows
```

---

## Technical Details

### Clustering

**Stage 1 — KMeans** (`sklearn.cluster.KMeans`)  
Partitions all destination cities into `k` geographic zones based on lat/lon.
Choosing `k=8` reflects the major US freight regions (Northeast, Southeast, Midwest, etc.).

**Stage 2 — Agglomerative Clustering** (`sklearn.cluster.AgglomerativeClustering`)  
Within each KMeans zone, Ward-linkage hierarchical clustering groups shipments that share:
- Similar destination coordinates
- Overlapping pickup windows
- Compatible weights

A distance threshold of ~150 miles (converted to degrees) controls cluster radius.

### Optimisation

**Formulation:** Mixed-Integer Linear Program (MILP) via [PuLP](https://coin-or.github.io/pulp/) + CBC solver.

**Variables:**
- `x[s, t] ∈ {0,1}` — 1 if shipment `s` is loaded onto truck `t`
- `y[t] ∈ {0,1}` — 1 if truck `t` is activated

**Objective:**
```
min  Σ_t  TL_cost(t) · y[t]
   + Σ_s  LTL_cost(s) · (1 − Σ_t x[s,t])
```

**Constraints:**
- Each shipment assigned to at most one truck
- Weight per truck ≤ 44,000 lbs
- Volume per truck ≤ 2,400 cuft
- Symmetry breaking to reduce search space

### Cost Model

| Mode | Rate |
|------|------|
| TL | \$150 base + \$3.25/mile (flat, regardless of weight) |
| LTL | \$0.20–0.50/lb (varies by distance and NMFC freight class) |

---

## Visualisations

| Chart | Description |
|-------|-------------|
| Cluster map | US scatter plot of destinations, coloured by KMeans zone |
| Savings by region | Baseline vs optimised freight cost per region |
| TL utilisation | Weight utilisation % for each dispatched truck |
| Dendrogram | Hierarchical clustering tree for a single zone |

---

## Stack

`Python` · `scikit-learn` · `PuLP` · `pandas` · `numpy` · `matplotlib` · `scipy`

---

*Built as an in-house replacement for Mercury Gate at LSC Communications (2018–2021).*
