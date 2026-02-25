# Data Directory Consolidation Plan

## Problem

The project has **two parallel data directories** totaling ~2GB, with significant duplication:

| Directory | Size | Purpose | Referenced by Code? |
|-----------|------|---------|-------------------|
| `data/` | 1.0GB | Canonical data dir (input, processed, profiles, shapefiles) | **Yes** — `data_transform.py`, `data_service.py`, `config.py`, tests |
| `site_scoring/data/` | 1.0GB | Legacy/staging area with renamed files + stale copies | **No** — no Python code imports from here |

### Key Findings

1. **927MB duplicate**: `site_scores_revenue_and_diagnostics.csv` exists identically in both directories (confirmed via MD5)
2. **Different naming conventions**: `data/input/` has distance-computed files (`site_walmart_distances.csv`), while `site_scoring/data/input/` has raw geodata sources (`walmart_geodata.csv`) — these are **complementary**, not duplicates
3. **34MB of dead weight**: `site_scoring/data/stale data/` has old file copies with zero code references
4. **Python package removed**: `site_scoring/data/__init__.py` and `registry.py` are already deleted (per git status)

### File Inventory

**`data/input/` (canonical, referenced by code)**:
- `site_scores_revenue_and_diagnostics.csv` (927MB) — main revenue data
- `nearest_site_distances.csv` (6.5MB) — computed distances
- `site_interstate_distances.csv` (3.8MB) — computed distances
- `site_kroger_distances.csv` (2.7MB) — computed distances
- `site_mcdonalds_distances.csv` (2.8MB) — computed distances
- `site_walmart_distances.csv` (2.7MB) — computed distances
- `site_target_distances.csv` (2.7MB) — computed distances

**`site_scoring/data/input/` (unreferenced by code)**:
- `site_scores_revenue_and_diagnostics.csv` (927MB) — **DUPLICATE, delete**
- `sites_base_data_set.csv` (17MB) — raw site metadata → **move to `data/input/`**
- `mcdonalds_geodata.csv` (7.5MB) — raw geodata → **move to `data/input/`**
- `walmart_geodata.csv` (6.7MB) — raw geodata → **move to `data/input/`**
- `nearest_site_geodata.csv` (6.5MB) — raw geodata → **move to `data/input/`**
- `salesforce_site_revenue.csv` (5.1MB) — revenue source → **move to `data/input/`**
- `interstate_geodata.csv` (3.8MB) — raw geodata → **move to `data/input/`**
- `target_geo_data.csv` (1.6MB) — raw geodata → **move to `data/input/`**

**`site_scoring/data/input/platinum/`**:
- `sites_base_data_set_2.csv` (47KB) — variant data → **move to `data/input/platinum/`**

**`site_scoring/data/stale data/` (34MB, zero code references)**:
- 9 old CSV files → **DELETE entirely**

---

## Implementation Steps

### Step 1: Delete stale data (zero risk)
```
rm -rf site_scoring/data/stale data/
```
**Verification**: No Python code references this directory (confirmed via grep)
**Recovery**: Files are versioned — they were likely copied from other locations

### Step 2: Move unique files from `site_scoring/data/input/` → `data/input/`
```
mv site_scoring/data/input/sites_base_data_set.csv    data/input/
mv site_scoring/data/input/mcdonalds_geodata.csv       data/input/
mv site_scoring/data/input/walmart_geodata.csv         data/input/
mv site_scoring/data/input/nearest_site_geodata.csv    data/input/
mv site_scoring/data/input/salesforce_site_revenue.csv data/input/
mv site_scoring/data/input/interstate_geodata.csv      data/input/
mv site_scoring/data/input/target_geo_data.csv         data/input/
```
**Verification**: No code currently references these files by path; they're source/geodata files used for offline computation
**Recovery**: `git checkout` can restore from previous locations

### Step 3: Move platinum subfolder
```
mkdir -p data/input/platinum/
mv site_scoring/data/input/platinum/sites_base_data_set_2.csv data/input/platinum/
```

### Step 4: Delete the 927MB duplicate
```
rm site_scoring/data/input/site_scores_revenue_and_diagnostics.csv
```
**Verification**: MD5 hashes match — confirmed identical
**Recovery**: The canonical copy remains at `data/input/`

### Step 5: Remove empty `site_scoring/data/` directory tree
```
rm -rf site_scoring/data/
```
The Python package files (`__init__.py`, `registry.py`) are already deleted per git status. Only `__pycache__` and `.DS_Store` files remain.

### Step 6: Update `.gitignore` if needed
Ensure `data/input/` CSVs are properly gitignored (they're large binary-like files).

---

## What Does NOT Change

- **`data/input/`** — all existing files stay in place (this is the canonical dir)
- **`data/processed/`** — parquet files and exports untouched
- **`data/profiles/`** — profiling CSVs untouched
- **`data/shapefiles/`** — shapefile data untouched
- **All Python code** — no code changes needed because no code references `site_scoring/data/input/`
- **`site_scoring/outputs/`** — experiment artifacts untouched

## Result

| Metric | Before | After |
|--------|--------|-------|
| Total disk usage | ~2.0GB across 2 dirs | ~1.0GB in 1 dir |
| Data directories | 2 (`data/`, `site_scoring/data/`) | 1 (`data/`) |
| Duplicate files | 927MB duplicated | 0 |
| Stale/dead files | 34MB | 0 |
| Code changes needed | — | **None** |

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Notebook references `site_scoring/data/` paths | Low | Grep found no references; check notebooks manually |
| Future scripts need old file paths | Low | Files are moved, not deleted; paths are discoverable in `data/input/` |
| `platinum1_1/` package references old paths | Low | That package has its own paths module; check before executing |

## Rollback

All moves can be reversed with `git checkout` for tracked files. For untracked files, the moves are symmetric — just reverse the `mv` commands.
