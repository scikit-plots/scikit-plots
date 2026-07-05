# scikit-plots `learn/data_preparation_and_analysis` — Regeneration & Maintenance Guide

**One document to rebuild, extend, or recreate this course — no chat history required.**

| | |
|---|---|
| **Folder** | `docs/source/learn/data_preparation_and_analysis/` |
| **Renders at** | `https://scikit-plots.github.io/dev/learn/data_preparation_and_analysis/index.html` |
| **Scope** | An **ordered 56-lesson course** on the applied predictive-modelling workflow (**8 stages**: foundations → associations → market basket → segmentation → regression → classification → trees → evaluation) + one staged hub |
| **Generator** | `build_data_preparation.py` (Python 3, standard library only) |
| **Determinism** | Same inputs → **byte-identical** output, every run (ordered, no RNG) |
| **Status** | 56 / 56 lessons complete, all cross-linked, all links + underlines valid |
| **Seal sha256** | `41cd74771e818433290f349c70f22068db4603d66e10a499ed1cb7b709186302` (concatenated `01-…56-*.rst` + `index.rst`, sorted) |

> **Golden rules** (violate none):
> 1. **Idempotent** — rebuilding never changes output unless an input changed.
> 2. **Verified-only** — every fact and formula is grounded in a real source; nothing invented.
> 3. **Re-express, never copy** — the source corpus is *framing only*; all prose is original.
> 4. **Fail-fast** — a title/key mismatch aborts the build; there are no silent gaps.

This folder is the **fourth** built on the shared pattern, after the terminology glossary
(`learn/terminology/`), the time-series course (`learn/time_series/`) and the deep-learning course
(`learn/deep_learning/`). The engine is a direct clone of those generators, specialised for **this**
curriculum — the largest so far, and the one most tightly coupled to scikit-plots itself (its final
stage *is* the library's evaluation-chart domain). See Section 8 to recreate the inputs from nothing.

---

## 0. TL;DR — rebuild in one command

```bash
cd docs/source/learn/data_preparation_and_analysis
python build_data_preparation.py   # clears old NN-*.rst, rewrites index.rst + all 56 lessons
```

Then validate (Section 4). Expected evidence lines:

```
Wrote index.rst + 56 lesson pages (56 with full content, 0 stub) across 8 stages.
broken :doc: 0 | broken :ref: 0 | underline errors 0
```

No arguments, no environment, no network. Pure Python standard library.

---

## 1. What lives in this folder

| File | Role | Edited by hand? |
|---|---|---|
| `build_data_preparation.py` | The deterministic generator: STAGES registry, 56 GLOSS one-liners, all emission logic, fail-fast guards | **Yes** (registry/GLOSS only) |
| `dpa_content.py` | `CONTENT[title] → RST body` and `MINDMAP[title] → related titles` for all 56 lessons | **Yes** (this is where lessons live) |
| `dpa_inventory.tsv` | **Frozen** 56-row inventory: `title ⇥ url ⇥ stage`, in curriculum order | Only to append a new lesson |
| `index.rst` | **GENERATED** staged hub: banner, intro, warning/note, 8 stage sections of cards, hidden ordered toctree | **Never** |
| `01-…56-*.rst` | **GENERATED** lesson pages (2-digit id = curriculum order) | **Never** |
| `DATA_PREPARATION_AND_ANALYSIS.md` | This guide | Yes |

Anything not listed (e.g. `__pycache__/`) is disposable and must never be synced or zipped.

---

## 2. Architecture — how the pieces fit

```
dpa_inventory.tsv ──┐            (56 frozen rows: title, url, stage)
                    ├─► build_data_preparation.py ─► index.rst + 01-…56-*.rst
dpa_content.py ─────┘            (CONTENT + MINDMAP, keys = EXACT titles)
        ▲
        └── GLOSS + STAGES live inside the generator (one-liners + stage registry)
```

* The **inventory** is the single source of truth for *what exists and in which order*.
  Lesson number = row position (1-based) → filename prefix `01`–`56`.
* The **generator** joins inventory ⇄ GLOSS ⇄ STAGES ⇄ CONTENT/MINDMAP, aborting on any mismatch.
* Every page gets: `.. _dpa-<slug>:` anchor, overlined title, stage/level badge,
  **◀ Prev / Next ▶** navigation (auto from order), the body (or a stub), a `.. seealso::`
  block of MINDMAP lateral links, the source URL, and controlled `.. tags::`.
* The **hub** (`index.rst`) carries the page anchor `_data-preparation-and-analysis-index:`,
  three reader levels, the **held-out-test-data warning** (the course's standing rule), one
  `grid`/`grid-item-card` section per stage, and a hidden ordered toctree.

### The 8 stages (in order)

| # | key | emoji | lessons | theme |
|---|---|---|---|---|
| 1 | `foundations` | 📋 | 01–06 | why analyse, CRISP-DM, big data, profiling, IEEE 754 |
| 2 | `associations` | 🔗 | 07–16 | Chicago taxi data, correlation, χ²/Cramér's V, tests, η² |
| 3 | `market_basket` | 🛒 | 17–22 | association rules, Apriori, mlxtend, cross-selling |
| 4 | `segmentation` | 🧩 | 23–30 | sampling, LCG, train/test split, clustering, RFM |
| 5 | `regression` | 📈 | 31–37 | least squares, multiple regression, selection, Shapley |
| 6 | `classification` | 🎯 | 38–43 | logistic regression, MLE, separation, deviance, retention case |
| 7 | `trees` | 🌳 | 44–49 | CART, piecewise models, interactions, cluster profiling |
| 8 | `evaluation` | 📊 | 50–56 | confusion matrix, thresholds, residuals, **ROC/AUC, lift** |

Stage 8 is scikit-plots' home turf: those lessons name the library's charts (confusion matrix,
ROC, cumulative-gain/lift, residual plots) directly.

---

## 3. Regenerating the pages (the build)

```bash
python build_data_preparation.py
```

What one run does, in order:

1. Loads and counts the inventory (**must be exactly 56 rows**; else abort, exit 1).
2. Cross-checks inventory ⇄ GLOSS both ways; checks every stage key against STAGES.
3. Imports `dpa_content.py`; verifies every CONTENT key, MINDMAP key **and MINDMAP
   neighbour** is an exact inventory title.
4. Deletes only `[0-9][0-9]-*.rst` (never the toolkit, never `index.rst` until rewrite).
5. Emits the 56 pages, then the hub. Deterministic: sorted iteration, no RNG, no timestamps.

Nothing is written if any check fails — a broken input can never half-destroy the output.

---

## 4. Validating (run after every change — mandatory)

Run from the **parent** `learn/` directory. This is the exact validator used throughout:

```python
import re, glob, os
files = sorted(glob.glob("data_preparation_and_analysis/*.rst"))
defined, docnames = set(), set()
for f in files:
    t = open(f, encoding="utf-8").read()
    for m in re.finditer(r'^\.\.\s+_([A-Za-z0-9_-]+):\s*$', t, re.M):
        defined.add(m.group(1))
    dn = f[:-4]; docnames.add(dn); docnames.add(os.path.basename(dn))
defined.add("terminology-index")          # external anchor this course links to
bd = br = ue = 0
for f in files:
    t = open(f, encoding="utf-8").read()
    for m in re.finditer(r':doc:`(?:[^`<]*<)?([^`>]+)>?`', t):
        tg = m.group(1).strip()
        if tg not in docnames and os.path.basename(tg) not in docnames:
            bd += 1; print("BADDOC", f, tg)
    for m in re.finditer(r':ref:`(?:[^`<]*<)?([A-Za-z0-9_-]+)>?`', t):
        if m.group(1) not in defined:
            br += 1; print("BADREF", f, m.group(1))
    L = t.split("\n")
    for i in range(len(L) - 1):
        tt, uu = L[i], L[i + 1]
        if (uu.strip() and re.fullmatch(r'[=\-^"~]{3,}', uu.strip())
                and len(set(uu.strip())) == 1
                and not re.fullmatch(r'[=\-^"~]{3,}', tt.strip())
                and tt.strip() and not tt.startswith("   ")      # ← code-block aware
                and len(uu.strip()) < len(tt.rstrip())):
            ue += 1; print("UNDER", f, repr(tt))
print(f"broken :doc: {bd} | broken :ref: {br} | underline errors {ue}")
```

Must print `broken :doc: 0 | broken :ref: 0 | underline errors 0`. The
`not tt.startswith("   ")` guard is essential: lesson bodies contain indented
`.. code-block:: python` content that would otherwise false-positive as bad underlines.

**Idempotency proof** (run after validation; sh/dash-safe, no bashisms):

```sh
B=$(cat $(ls [0-9][0-9]-*.rst | sort) index.rst | sha256sum | cut -d' ' -f1)
python3 build_data_preparation.py > /dev/null 2>&1
A=$(cat $(ls [0-9][0-9]-*.rst | sort) index.rst | sha256sum | cut -d' ' -f1)
[ "$B" = "$A" ] && echo "OK idempotent ($A)" || echo "DRIFT"
```

With all 56 lessons present, `$A` must equal the **seal sha** in the header table.

---

## 5. Updating & extending (add or edit lessons)

### Edit an existing lesson

Edit its raw-string body in `dpa_content.py` → rebuild → validate → idempotency. Never touch
the generated `.rst`.

### Add a new lesson

1. Append one row to `dpa_inventory.tsv` (`title ⇥ url ⇥ stage`; stage must exist in STAGES).
2. Add one GLOSS one-liner in the generator, and bump `N_LESSONS`.
3. Add `CONTENT[title]` + `MINDMAP[title]` in `dpa_content.py`.
4. Rebuild; the new page, its prev/next links, the stage card and the toctree all appear
   automatically. (Note: inserting mid-sequence renumbers every later page — by design;
   ids are curriculum positions, not permanent handles.)

### The exact-title rule (critical)

Every CONTENT/MINDMAP key and every MINDMAP neighbour must match its inventory title
**byte-for-byte**. This course has the trickiest titles of the four hubs:

| Character | Escape | Appears in |
|---|---|---|
| en dash `–` | `\u2013` | `Taxi Trips – 2022 dataset…`, `Binary Classification Models – Conceptual…`, `AUC–ROC Curve: …` |
| e-acute `é` | `\u00e9` | `Harald Cramér` |
| eta² `η²` | `\u03b7\u00b2` | `Eta Squared (η²): Effect Size in ANOVA` |
| underscore | — | `association_rules: Generating…` (literal snake_case) |

In Python builders **always write these as `\uXXXX` escapes** inside key strings (so equality
is unambiguous); inside CONTENT prose write the characters literally. When appending content
via shell, use a single-quoted heredoc (`cat >> dpa_content.py << 'PYEOF'`) so nothing is
interpolated.

### Batch cadence

Content was written ~3 lessons per batch (19 batches), each batch: distinctive web search →
authoritative grounding → **re-express in original words** → append CONTENT+MINDMAP →
rebuild → validate → idempotency → sync → re-zip, with evidence reported every time. Keep
that rhythm for any future additions.

---

## 6. Page content contract (what a good lesson contains)

* **3–5 short titled sections** (H2, `-----` underlines), each a focused idea; ~250–450 words total.
* **Formulas in `.. math::`**, verified against an authoritative source before writing
  (e.g. Pearson r, Spearman ρ, Kendall τ, Cramér's V, η², Shapley φᵢ, the logistic/logit pair,
  deviance χ², studentized residuals, lift).
* **Python-first**: `pandas` / `scikit-learn` / `statsmodels` / `mlxtend` calls in
  `.. code-block:: python` where the lesson is operational; the **Chicago Taxi Trips (2022)**
  dataset is the running example thread.
* **scikit-plots tie-ins** wherever a chart exists for the concept (confusion matrix, ROC,
  gain/lift, residual plots) — Stage 8 especially.
* **Narrative continuity**: each lesson may reference its neighbours ("the next lesson…"),
  since the order is frozen; MINDMAP gives 4 lateral links per lesson.
* **Honest caveats** (association ≠ causation; significance ≠ importance; greedy selection
  inflates significance; report on held-out data only).

---

## 7. RST / formatting conventions (what keeps the build clean)

* Section underline length **≥** header length — Python `len()` counts é/η/²/–/× as 1 column;
  when in doubt pad the underline a few chars longer.
* `.. code-block:: python` bodies indented **3 spaces**; the validator skips 3-space-indented
  lines when checking underlines.
* CONTENT bodies are **raw strings** `r"""…"""` — LaTeX backslashes stay literal.
* Avoid bare `$` in prose (wrap in ``code``); bare `%` is fine.
* Emit Unicode characters **literally in prose**, escapes **in keys** (Section 5).
* Cards on the hub use `^^^` separators — the validator's underline check ignores them.

---

## 8. Regenerating from scratch (if you start with nothing but this guide)

The corpus is the **Data Preparation and Analysis** category (56 posts) of
`https://insightful-data-lab.com/` (author "Ju Yeon Eum"), harvested across
`…/category/data-preparation-and-analysis/` pages 1–6. Post URL pattern:
`https://insightful-data-lab.com/{date}/{slug}/` with `date = 2026/01/14` for stages 1–4 and
`2026/01/16` for stages 5–8; the exact slugs are frozen in `dpa_inventory.tsv`
(three notable ones: eta-squared uses the percent-encoded slug
`eta-squared-%ce%b7%c2%b2-effect-size-in-anova`; lesson 46 keeps the legacy slug
`interactions-in-cart-decision-trees`; Cross-Selling is `cross-selling-2`).

### Curriculum map (the 56 lessons, in order)

**📋 Stage 1 — Foundations (01–06):** Why Do We Analyze Data? · The Process of Data Analysis ·
CRISP-DM for Data Science · Big Data: Definition, Characteristics, Evolution, and Business
Impact · The First Step in Knowing Your Data · IEEE 754 Floating-Point Standard

**🔗 Stage 2 — Associations & Correlation (07–16):** Discovering Associations Through Data:
From Everyday Patterns to Chicago Taxi Trips (September 2022) · Taxi Trips – 2022 dataset from
the City of Chicago open data portal · Objective Selection of the Bin Width for a Time
Histogram · Measuring Associations in Data · Measuring Associations Between Two Continuous
Variables · Correlation Coefficients in Python (Pearson, Spearman, Kendall) · Karl Pearson ·
Harald Cramér · What Are Statistical Tests? · Eta Squared (η²): Effect Size in ANOVA

**🛒 Stage 3 — Market Basket & Association Rules (17–22):** Understanding Market Baskets and
Ideal Customers · What Can Association Rules Tell Us? · How Association Rules Are Discovered:
Concepts, Scale, Measures, and the Apriori Approach · Apriori: Frequent Itemsets via the
Apriori Algorithm · association_rules: Generating Association Rules from Frequent Itemsets
(mlxtend) · Cross-Selling

**🧩 Stage 4 — Sampling, Partitioning & Segmentation (23–30):** Stratified Random Sampling ·
Linear Congruential Random Number Generator (LCG) · Partitioning Observations to Train
Objective Models · Putting Similar Observations into Clusters · Clustering · Recency,
Frequency, and Monetary Value (RFM) · RFM Analysis · Creating Segments of Observations for
Business Reasons (RFM)

**📈 Stage 5 — Regression (31–37):** Least Squares Regression · Multiple Linear Regression ·
Feature Importance in Linear Regression · Forward Selection: Definition and Core Idea ·
Forward Selection and Model Interpretation in Linear Regression · Understanding Forward and
Backward Stepwise Regression · How Shapley Values Work

**🎯 Stage 6 — Classification & Logistic Regression (38–43):** Logistic Regression: Modeling
Binary Outcomes via Odds and Log-Odds · Maximum Likelihood (MLE): Fitting a Distribution to
Observed Data · Assessing Model Fit in Logistic Regression · Complete and Quasi-Complete
Separation in Logistic Regression · Forward Selection with Nested Models and Deviance Tests ·
Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student
Retention

**🌳 Stage 7 — Decision Trees (44–49):** Motivation of Decision Trees: An Incremental Model of
Decision-Making · The CART Algorithm · Decision Trees as Piecewise Models and Their Predictive
Structure · How CART Decision Trees Model Interactions · Cluster Profiling Using Decision
Trees · Using Decision Trees to Explain Clustering Results

**📊 Stage 8 — Model Evaluation (50–56):** Assessing the Quality of Prediction Models · Binary
Classification Models – Conceptual Framework and Evaluation Metrics · Nominal Classification
Models: Model State and Evaluation Metrics · Binary Classification Model Evaluation and
Threshold Optimization · Identifying Outliers Using Residuals and Studentized Residuals ·
AUC–ROC Curve: Evaluating Classification Model Performance · Lift Analysis for Direct Mail
Campaigns: Concept, Process, and Business Value

Recreate `dpa_inventory.tsv` from this map + the URL pattern, rebuild the generator per
Sections 2–3, then write content per Sections 5–6, grounding every lesson before writing.

---

## 9. Deterministic sync + zip (only if publishing to a separate output dir)

```sh
rm -rf __pycache__
OUT=/path/to/outputs/learn/data_preparation_and_analysis
find "$OUT" -type f -exec rm -f {} \; 2>/dev/null; rm -rf "$OUT"; mkdir -p "$OUT"
cp ./*.rst ./*.py ./*.tsv ./*.md "$OUT/"
# (do NOT copy __pycache__; exclude *.pyc from any zip)
```

The defensive file-by-file delete guards against transient I/O errors on network mounts
(observed once during the build; this pattern recovered it cleanly).

---

## 10. Fail-fast guard & invariants (the safety net)

The build **aborts with exit 1 and writes nothing** if any of these hold:

* inventory row count ≠ `N_LESSONS` (56);
* any inventory title lacks a GLOSS entry, or any GLOSS key is not in the inventory;
* any inventory stage is not a STAGES key;
* any CONTENT key, MINDMAP key, or **MINDMAP neighbour** is not an exact inventory title.

Both fail-fast paths were smoke-tested (bad CONTENT key → exit 1; bad MINDMAP neighbour →
exit 1; restore → clean rebuild). Standing invariants: 57 `.rst` files (56 lessons + hub);
every `:doc:`/`:ref:` resolves; every underline ≥ its header; `len(CONTENT) == len(MINDMAP)
== 56`; rebuild is byte-identical (seal sha in the header table); the hub's held-out-test
warning is never removed.

---

## 11. Suggestions & future work

* **Sibling hubs next**: Bayesian Data Analysis (144 posts) and Data Analytics (216 posts,
  8 sub-sections — likely a multi-course hub). The pre-existing `bayesian_data_analysis/`
  hand-written hub already cross-links terminology theme anchors; salvage its framing the way
  this course salvaged its own predecessor hub.
* **Executable figures**: Stage 8 lessons could gain `plot::` directives rendering live
  scikit-plots ROC/lift/confusion charts once the docs build budget allows.
* **Terminology deep-links**: individual metric mentions (precision, recall, AUC…) could link
  to their glossary term pages, not just the hub anchor.
* **Hub scaling**: if this hub ever feels heavy, the terminology **v2 filterable browser**
  pattern (collapsed dropdowns + live JS filter + A–Z) is a drop-in upgrade.

---

## 12. Quick rule card (pin this)

1. Inventory is frozen; ids = curriculum order.
2. Keys match titles **byte-for-byte** (`\u2013`, `\u00e9`, `\u03b7\u00b2` — Section 5).
3. Ground first, then write **original** prose; formulas verified before use.
4. Rebuild → validate (0/0/0) → idempotency (seal sha) → sync → zip. Every time.
5. Never edit generated `.rst`; never ship `__pycache__`.
6. Report performance on **held-out test data** — the course's own rule applies to its authors.
