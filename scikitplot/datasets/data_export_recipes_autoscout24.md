# Generic Dataset Export Recipes (AutoScout24 examples)

This guide is **dataset-agnostic** (works for any CSV/Parquet) and uses **AutoScout24** as a concrete example.
It focuses on **reproducible, representative ML subsets** for EDA, modeling, and MLOps pipelines.

---

## Files produced by `data_export.py`

For each exported subset, you typically get:

- Dataset file: `subset_<strategy>_<n>.parquet` (or `.csv`)
- Optional manifest: `subset_<strategy>_<n>.manifest.json` (enabled by `--write-manifest`)
- Optional profile: `subset_<strategy>_<n>.profile.json` (enabled by `--write-profile`)

You may also get a preparation profile (depending on options) such as `prepared.profile.json`.

---

## Quick principles (user + dev)

### Reproducibility
- **hash strategy**: deterministic (same dataset → same subset) using stable hashing of `--id-col`.
- **random strategy**: deterministic if you pin `--seed`.
- **stratified strategy**: strict proportional allocation across strata groups, then deterministic within group (hash or random).

### Representativeness (generic)
To make a subset representative for “generic ML tasks”, ensure your subset includes:
1) **Target/label** (e.g., `price` for regression; `seller_is_dealer` for classification),
2) **Entity id** (stable selection, dedup, deterministic splits),
3) **Major segmentation fields** (e.g., country/type/fuel/used),
4) **A usage/condition proxy** (e.g., mileage, registration_date).

### Safety (avoid accidental data loss)
Never do a global `.dropna()` over all columns. Instead, use:
- `--required-cols colA colB ...`
to drop rows missing only the columns your task truly needs.

---

## AutoScout24 sizing guidance

Your dataset has **~118k rows**, so:
- **10%** ≈ **11,838** rows (using `--percentages 10 --rounding round`)
- If you need an *exact* **12,000**, use `--sizes 12000`.

Recommended ladder for stable iteration:
- 0.1% (smoke tests), 1% (fast EDA), 10% (~12k), 25%, 50%, 100%

---

# Export recipes

> All examples assume you are in a shell where `python` resolves to the environment that has `pandas` installed.

---
## Quickstart

```bash
# from scikitplot import datasets
!python -m scikitplot.datasets._data_export \
  --input autoscout24_dataset_20251108.csv \
  --output-dir data/subsets \
  --percentages 0.1 \
  --rounding round \
  --strategy hash \
  --id-col id \
  --dedup \
  --required-cols id price make model registration_date mileage_km_raw vehicle_type fuel_category is_used country_code \
  --keep-cols id price make model model_version registration_date mileage_km_raw vehicle_type body_type fuel_category primary_fuel \
      transmission power_kw power_hp nr_seats nr_doors country_code zip city latitude longitude is_used seller_is_dealer offer_type \
      description equipment_comfort equipment_entertainment  equipment_extra equipment_safety\
  --format csv \
  --write-manifest \
  --write-profile
```

## 1) Best default: representative 10% (hash + dedup + core columns)

**When to use**
- EDA + baseline regression/classification
- General “dev subset” for iterative pipelines
- Works well for most generic tasks

```bash
python data_export.py \
  --input autoscout24_dataset_20251108.csv \
  --output-dir data/subsets \
  --percentages 10 \
  --rounding round \
  --strategy hash \
  --id-col id \
  --dedup \
  --required-cols id price make model registration_date mileage_km_raw vehicle_type fuel_category is_used country_code \
  --keep-cols id price make model model_version registration_date mileage_km_raw vehicle_type body_type fuel_category primary_fuel transmission power_kw power_hp nr_seats nr_doors country_code zip city latitude longitude is_used seller_is_dealer offer_type description \
  --format parquet \
  --write-manifest \
  --write-profile
```

**Why these parameters help**
- `--dedup` avoids repeated entities (stable evaluation/debug).
- `--required-cols ...` ensures rows are useful for most tasks.
- `--keep-cols ...` avoids sparse/noisy columns and speeds iteration.

---

## 2) Exact size: deterministic 12,000 rows (hash)

If you want exactly **12,000** rows (instead of ~11,838):

```bash
python data_export.py \
  --input autoscout24_dataset_20251108.csv \
  --output-dir data/subsets \
  --sizes 12000 \
  --strategy hash \
  --id-col id \
  --dedup \
  --required-cols id price make model mileage_km_raw fuel_category vehicle_type country_code \
  --keep-cols id price make model model_version registration_date mileage_km_raw vehicle_type body_type fuel_category transmission power_kw power_hp nr_seats nr_doors country_code city latitude longitude is_used seller_is_dealer description \
  --format parquet \
  --write-manifest \
  --write-profile
```

---

## 3) Most representative: stratified 10% (balanced segments)

**When to use**
- You care about stable segment distributions (country/type/fuel/used/dealer)
- Classification tasks and monitoring tasks

```bash
python data_export.py \
  --input autoscout24_dataset_20251108.csv \
  --output-dir data/subsets \
  --percentages 10 \
  --rounding round \
  --strategy stratified \
  --strata-cols country_code vehicle_type fuel_category is_used seller_is_dealer \
  --within random \
  --seed 42 \
  --required-cols price make model mileage_km_raw fuel_category vehicle_type country_code \
  --keep-cols id price make model model_version registration_date mileage_km_raw vehicle_type body_type fuel_category transmission power_kw power_hp nr_seats nr_doors country_code city latitude longitude is_used seller_is_dealer description \
  --format parquet \
  --write-manifest \
  --write-profile
```

**Dev note**
- Adding too many `--strata-cols` can produce many tiny groups. Keep strata to the “big segmentation columns”.

---

## 4) NLP / RAG subset (require description)

Use this when you plan to train text models or do retrieval on descriptions.

```bash
python data_export.py \
  --input autoscout24_dataset_20251108.csv \
  --output-dir data/subsets \
  --percentages 10 \
  --rounding round \
  --strategy hash \
  --id-col id \
  --dedup \
  --required-cols id description make model price \
  --keep-cols id description make model model_version price registration_date mileage_km_raw vehicle_type fuel_category country_code city \
  --format parquet \
  --write-manifest \
  --write-profile
```

---

## 5) Learning-curve ladder (multiple sizes in one run)

Useful for quick “does the model stabilize?” checks.

```bash
python data_export.py \
  --input autoscout24_dataset_20251108.csv \
  --output-dir data/subsets \
  --percentages 0.1 1 5 10 25 50 100 \
  --rounding round \
  --strategy hash \
  --id-col id \
  --dedup \
  --required-cols id price make model mileage_km_raw fuel_category vehicle_type country_code \
  --keep-cols id price make model registration_date mileage_km_raw vehicle_type fuel_category transmission power_kw power_hp country_code city latitude longitude is_used seller_is_dealer description \
  --format parquet \
  --write-manifest \
  --write-profile
```

---

## 6) Huge CSV: streaming hash (memory-bounded, multi-size)

Use streaming when the input is too large to fit comfortably in memory.
**Note:** streaming mode supports **absolute `--sizes` only** (not percentages/fractions).

```bash
python data_export.py \
  --input autoscout24_dataset_20251108.csv \
  --output-dir data/subsets \
  --sizes 1000 5000 12000 \
  --strategy hash \
  --id-col id \
  --stream-hash-csv \
  --chunksize 50000 \
  --format parquet \
  --write-manifest \
  --write-profile
```

---

# How to interpret the manifest + profile (MS file)

## Manifest (`*.manifest.json`)
This file is your “audit trail” for how the subset was created.

**Typical fields**
- `created_utc`
- `source` (path/URL + input format)
- `prepare` (query, required_cols, keep/drop cols, dedup)
- `export` (size, strategy, strategy parameters)
- `rows` (prepared row counts; in streaming mode includes seen/eligible counts)
- `columns`
- `output_file`

**Quick view**
```bash
cat data/subsets/subset_hash_11838.manifest.json
```

## Profile (`*.profile.json`)
Lightweight stats to speed up EDA and debugging:
- shape (rows/cols)
- dtypes
- null counts
- small cardinality summaries (if enabled by the script settings)

**Quick view**
```bash
cat data/subsets/subset_hash_11838.profile.json
```

---

# Task examples on the exported ~12k subset

Assume you exported either:
- `subset_hash_11838.parquet` (10% of ~118k), or
- `subset_hash_12000.parquet` (exact)

Below are **end-to-end examples** for EDA, visualization, regression, classification, ANN, NLP, and a simple RAG-style retrieval demo.

> If you have `autoscout24_tasks.py` in the same directory, you can run these directly.

---

## A) EDA + visualization (fast, practical)
```bash
python autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet eda
```

Produces:
- missing-value summary
- top categories (make, fuel_category)
- basic plots (histograms/scatter)

---

## B) Regression: price prediction (tabular baseline)
```bash
python autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet regression_price
```

Typical pipeline:
- clean numeric + categorical
- one-hot encode categoricals
- baseline regressor and metrics

---

## C) Classification: dealer vs private seller
```bash
python autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet classification_dealer
```

Label:
- `seller_is_dealer` (boolean)

---

## D) ANN (tabular neural net baseline)
```bash
python autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet ann_price
```

Notes:
- ANN is optional for tabular; tree/linear often wins, but ANN is useful for experimentation.

---

## E) NLP: text → classification baseline
```bash
python autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet nlp_dealer
```

Uses:
- `description` text
- simple vectorization + classifier baseline

---

## F) RAG-style retrieval demo (lightweight, local)
```bash
python autoscout24_tasks.py \
  --data data/subsets/subset_hash_11838.parquet rag_demo \
  --query "low mileage electric hatchback" \
  --top-k 5
```

This is a simple “retrieve top matches” approach. For production RAG, you would:
- embed `description` + structured fields
- store in vector DB / ANN index
- retrieve + rerank
- generate answers with an LLM

---

# Troubleshooting (common strict errors)

## “Missing columns”
- Your `--keep-cols` / `--required-cols` references columns not present in the input.
Fix by running a quick schema check:
```python
import pandas as pd
df = pd.read_csv("autoscout24_dataset_20251108.csv", nrows=5)
print(df.columns.tolist())
```

## “percentage too small”
- If prepared rows are small and you ask for e.g. 0.01% with rounding mode,
the computed size may become 0.
Fix:
- increase percentage
- or use `--sizes` explicitly
- or change `--rounding ceil` if you want minimum 1 row

## “fractions/percentages not supported with --stream-hash-csv”
Streaming is single-pass and requires an absolute target size; use `--sizes`.

---

## Recommended “starter bundle” for AutoScout24

If you want one command that works for most ML tasks and yields a ~12k dev subset:

- Use recipe **#1** (10% hash with core required/keep columns).
- Then run EDA + a baseline model (regression or classification) from the task examples.

---

**End.**
