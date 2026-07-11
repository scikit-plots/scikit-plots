# Bayesian Data Analysis — course guide

This folder builds the **Bayesian Data Analysis** hub of the scikit-plots `learn/`
documentation: a 144-lesson course following Gelman, Carlin, Stern, Dunson, Vehtari
& Rubin, *Bayesian Data Analysis* (3rd ed.), rewritten in original words and
cross-linked into a browsable Sphinx site. It is the fifth subfolder, and it follows
the same self-contained generator pattern as `terminology/`, `time_series/`,
`deep_learning/`, and `data_preparation_and_analysis/`.

The course renders at
`https://scikit-plots.github.io/dev/learn/bayesian_data_analysis/`.

---

## What is in this folder

| File | Role |
| --- | --- |
| `BAYESIAN_DATA_ANALYSIS.md` | This guide. |
| `build_bayesian.py` | The generator. Reads the inventory + content, writes `index.rst` and the 144 lesson pages. Deterministic and idempotent. |
| `bda_content.py` | The corpus: `CONTENT` (full lesson bodies) and `MINDMAP` (4 cross-links per lesson), keyed by exact lesson title. |
| `bda_inventory.tsv` | **Frozen.** 144 rows (`title⇥url⇥stage`) in curriculum order. The single source of truth for titles, ordering, and stage assignment. |
| `index.rst` | Generated. The v2 filterable browser (see below). |
| `NNN-*.rst` | Generated. One page per lesson, 3-digit zero-padded id (`001`–`144`). |

Everything the generator needs lives in this folder. Nothing outside it is read or
written except the Sphinx build itself.

---

## Curriculum: 5 parts, 16 stages, 144 lessons

The course is organised into 5 **parts**, each spanning several **stages**; every
lesson belongs to exactly one stage. Part-level reading levels: Part 1 is beginner,
Parts 2–3 intermediate, Parts 4–5 advanced.

| Part | Stage (`key`) | 🔣 | Lessons | Title |
| --- | --- | --- | --- | --- |
| **I · Fundamentals** | `bayes_idea` | 🎲 | 10 | The Bayesian Idea |
| | `single_param` | 📍 | 9 | Single-Parameter Models & Priors |
| | `multiparam` | 🧮 | 8 | Multiparameter Models |
| | `asymptotics` | 📏 | 5 | Asymptotics & Frequentist Ties |
| | `hierarchical` | 🏛️ | 7 | Hierarchical Models |
| **II · Checking & Deciding** | `checking` | 🔍 | 10 | Model Checking & Comparison |
| | `collection_decision` | 🗳️ | 12 | Data Collection & Decisions |
| **III · Computation** | `simulation` | 🧰 | 7 | Simulation Basics |
| | `mcmc` | ⛓️ | 12 | MCMC: Gibbs, Metropolis & HMC |
| | `approximation` | 🎛️ | 10 | Modal & Variational Approximation |
| **IV · Regression** | `regression` | 📈 | 8 | Regression Foundations |
| | `hier_regression` | 🏗️ | 7 | Hierarchical Regression |
| | `glm` | 🔗 | 7 | Generalized Linear Models |
| | `robust_missing` | 🛡️ | 11 | Robustness & Missing Data |
| **V · Nonlinear & Nonparametric** | `gp_basis` | 🌊 | 9 | Basis Functions & Gaussian Processes |
| | `nonparametric` | ♾️ | 12 | Mixtures & Nonparametric Bayes |

Total: **144** lessons. The stage registry (`STAGES` in `build_bayesian.py`) holds
`(emoji, title, part, blurb)` per key; `STAGE_ORDER` fixes their sequence; `PARTS`
maps part numbers to names.

---

## How the generator works

`build_bayesian.py` is a pure function of its three inputs. One run:

1. `HERE = Path(__file__).resolve().parent`; `sys.path.insert(0, str(HERE))` so the
   content module imports regardless of the working directory.
2. Load `bda_inventory.tsv` (144 rows) and `bda_content.py` (`CONTENT`, `MINDMAP`).
3. **Fail-fast validation** (aborts with exit 1, writing nothing, on any of):
   - inventory row count ≠ `N_LESSONS` (144);
   - any inventory title without a `CONTENT` entry, or any `CONTENT`/`MINDMAP` key
     not an **exact** inventory title;
   - any `MINDMAP` neighbour not an exact inventory title;
   - any stage key not in `STAGES`;
   - duplicate titles.
4. Clear only files matching `[0-9]+-*.rst` (never touches hand files).
5. Emit `index.rst` (v2 browser) and one `NNN-*.rst` per lesson.

Determinism: everything is sorted, no RNG, no timestamps — so two runs are
**byte-identical**. This is verified every build via sha256 and proven at seal time
by a clean-room rebuild.

### Each lesson page

Anchor → auto H1 (overline + underline, `title_bar = max(72, len+2)`) → stage/level
badge → ◀ Prev / Next ▶ nav → body (`CONTENT`) → `.. seealso::` block (the 4
`MINDMAP` links) → source link → `.. tags::`. Every full lesson is: intuition →
key equation in `.. math::` → a real `scipy.stats` / PyMC / ArviZ snippet → a
worked-number line → an honest caveat.

### Anchors and the `bda-` prefix

`ANCHOR_PREFIX = "bda"`. Each page carries a stable anchor `_bda-NNN:`; the index
carries the page anchor `_bayesian-data-analysis-index:` and stable per-stage
anchors `_bda-stage-<key>:`.

---

## The v2 filterable browser (`index.rst`)

Identical in mechanism to the other four sealed hubs:

- a dependency-free JS **filter box** (`id="term-filter"`) matching lesson title OR
  gloss keyword, auto-opening matching dropdowns, showing a live match count, and
  degrading gracefully to a plain list when JS is off;
- a collapsed `.. dropdown::` per stage (class `details.sd-dropdown`) with a blurb
  and lines `NN · Title — gloss`, the stages grouped under the **5 Part headers**;
- an A–Z master dropdown (class `term-az`);
- the salvaged hub framing: the 🎲 banner, three reading levels, the PyMC/ArviZ
  note, and the scikit-plots map (GMM AIC/BIC + residual-distribution charts) with
  Gelman / PyMC / ArviZ sources.

---

## Legacy anchor compatibility (`COMPAT_ANCHORS`)

The pre-existing hand-written hub used 15 `bayes-*` anchors. To avoid breaking any
inbound links, every lesson page emits these as compatibility anchors (after its own
page anchor) resolving to the index top:

```
bayes-discovery   bayes-foundations  bayes-computation  bayes-advanced
bayes-skplt-map   bayes-sources      bayes-theorem      bayes-pieces
bayes-credible    bayes-conjugacy    bayes-mcmc         bayes-ppc
bayes-hierarchical  bayes-mixtures   bayes-dp
```

---

## Landmines (exact titles — copy verbatim, never retype)

The inventory preserves several source-title oddities. Their `CONTENT`/`MINDMAP`
keys **must** use the exact `\uXXXX` escapes so the keys match the frozen titles;
retyping with ASCII look-alikes will trip the fail-fast guard.

### Ligatures

| Lesson | Char | Escape | Title |
| --- | --- | --- | --- |
| 057 | ﬀ | `\ufb00` | Bayesian decision theory in di**ﬀ**erent contexts |
| 073 | ﬀ | `\ufb00` | E**ﬀ**ective number of simulation draws |
| 075 | ﬃ | `\ufb03` | E**ﬃ**cient Gibbs samplers |
| 076 | ﬃ | `\ufb03` | E**ﬃ**cient Metropolis jumping rules |
| 099 | ﬃ | `\ufb03` | Regression coe**ﬃ**cients exchangeable in batches |
| 104 | ﬃ | `\ufb03` | Analysis of variance and the batching of coe**ﬃ**cients |
| 127 | ﬃ | `\ufb03` | Basis selection and shrinkage of coe**ﬃ**cients |

Census: **5 ﬃ + 2 ﬀ** in page titles. (Verify: `python3 -c` counting `\ufb03` /
`\ufb00` in each page's H1 line.)

### Other special characters

| Lesson | Chars | Title |
| --- | --- | --- |
| 004 | θ `\u03b8`, en-dash `\u2013` | Discrete Bayesian Examples – Genetics and Spell Checking (with θ) |
| 006 | em-dash `\u2014` | Example — Probabilities from Football Point Spreads |
| 007 | em-dash `\u2014` | Example — Calibration for Record Linkage |

### Preserved source typo

| Lesson | Title (typo kept verbatim) |
| --- | --- |
| 110 | State-level **opinons** from national polls |

Do **not** "correct" it to *opinions*; the key must match the frozen inventory.

### Disambiguated duplicate

Two lessons share a near-identical title, kept distinct by the `(continued)` suffix,
with distinct filenames, distinct H1s, and cross-links to each other:

| Lesson | Title | File |
| --- | --- | --- |
| 074 | Example: hierarchical normal model | `074-example-hierarchical-normal-model.rst` |
| 086 | Example: hierarchical normal model (continued) | `086-example-hierarchical-normal-model-continued.rst` |

074 is the Gibbs treatment; 086 is the modal/approximation treatment.

---

## RST & validation conventions

Shared with the other four subfolders:

- **Section underlines** must be ≥ header length. Unicode (é, ², χ, η, ﬃ, ﬀ,
  en-/em-dash) counts as **one column** via Python `len()`; the generator sizes the
  H1 bar accordingly.
- **`.. code-block:: python`** bodies are indented 3 spaces. The validator **skips**
  any line starting with 3 spaces when checking underlines, so code content never
  false-positives.
- **Nested `.. math::`** under a bullet: marker at indent 2, body at indent 5
  (marker + 3). Balanced-brace and indentation checks confirm these.
- Avoid a bare `$` in prose (it can trip MathJax); a bare `%` is fine.
- Some code blocks use `.. code-block:: stan` (lesson 080); this is intentional.

Per-batch scans that must stay clean: every `:doc:`/`:ref:` target resolves; every
underline ≥ header; every `.. math::` has balanced braces (top-level and nested);
every code block imports what it uses (no undefined `np.`/`pm.`/`az.`/`stats.`); no
undefined ALLCAPS globals; the ligature census is 5 ﬃ + 2 ﬀ.

### Validator report

`build → validate → idempotency` prints, each run:

```
validate: doc 0 | ref 0 | underline 0  ||  content 144 mindmap 144  ||  idem OK <sha12>
```

The defined-anchor set includes the five hub page anchors: `terminology-index`,
`time-series-index`, `deep-learning-index`, `data-preparation-and-analysis-index`,
`bayesian-data-analysis-index`.

---

## Rebuilding

From this folder:

```bash
python3 build_bayesian.py
```

Writes `index.rst` + 144 pages. To rebuild, validate, check idempotency, and sync +
rezip the whole `learn/` tree in one call, use the repository helper:

```bash
python3 /home/claude/_buildcheck.py bayesian_data_analysis --sync
```

(The helper lives outside `learn/` and is never shipped.)

### Environment notes

- `bash` here is `/bin/sh` (dash): no `${var:0:N}` slicing, and `$'\uXXXX'` escapes
  do **not** expand — use Python to inspect Unicode/ligatures.
- The output mount can throw transient I/O errors mid-sync; the helper deletes
  file-by-file, `sync`s, then removes and recreates the directory to recover.
- The working tree resets between sessions; the source of truth persists under
  `/mnt/user-data/outputs/learn/bayesian_data_analysis/`. Restore by copying it back.

---

## Provenance

Content is an original rewrite of the topics in *Bayesian Data Analysis* (3rd ed.),
grounded per batch against authoritative sources (the primary papers behind each
worked example — Gelman & King on incumbency and election forecasting, Gelman–Fagan–
Kiss on police stops, Gelman–Bois–Jiang on toxicokinetics, Chung et al. on
boundary-avoiding priors, Vehtari et al. on R̂/ESS, Teh et al. on the HDP, and so on).
Each lesson is written in original words to respect copyright, cross-linked into the
mind-map, and mapped where relevant to scikit-plots' diagnostic plots.
