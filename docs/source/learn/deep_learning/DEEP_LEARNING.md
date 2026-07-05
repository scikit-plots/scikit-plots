# scikit-plots `learn/deep_learning` — Regeneration & Maintenance Guide

**One document to rebuild, extend, or recreate this course — no chat history required.**

| | |
|---|---|
| **Folder** | `docs/source/learn/deep_learning/` |
| **Renders at** | `https://scikit-plots.github.io/dev/learn/deep_learning/index.html` |
| **Scope** | An **ordered 17-lesson course** on neural-network foundations (deeplearning.ai C1) + one staged hub |
| **Generator** | `build_deep_learning.py` (Python 3, standard library only) |
| **Determinism** | Same inputs → **byte-identical** output, every run (ordered, no RNG) |
| **Status** | 17 / 17 lessons complete, all cross-linked, all links + underlines valid |

> **Golden rules** (violate none):
> 1. **Idempotent** — rebuilding never changes output unless an input changed.
> 2. **Verified-only** — every fact and formula is grounded in a real source; nothing invented.
> 3. **Re-express, never copy** — the source corpus is *framing only*; all prose is original.
> 4. **Fail-fast** — a title/key mismatch aborts the build; there are no silent gaps.

This folder is the **third** built on the shared pattern, after the terminology glossary
(`learn/terminology/`) and the time-series course (`learn/time_series/`). The engine here is a direct
clone of the time-series generator, specialised for **this** curriculum — see Section 8 if you ever
need to recreate the inputs from nothing.

---

## 0. TL;DR — rebuild in one command

```bash
cd docs/source/learn/deep_learning
python build_deep_learning.py    # clears old NN-*.rst, rewrites index.rst + all 17 lessons
```

Then validate (Section 4). Expected evidence line:

```
broken :doc: 0 | broken :ref: 0 | underline errors 0
```

No arguments, no environment, no network. Pure Python standard library
(`csv`, `re`, `sys`, `unicodedata`, `pathlib`).

---

## 1. What lives in this folder

```
deep_learning/
├── DEEP_LEARNING.md        ← this guide (the only doc you need to read)
├── build_deep_learning.py  ← the deterministic generator (edit for stages / glosses / layout)
├── dl_content.py           ← CONTENT + MINDMAP: the per-lesson bodies and lateral links (grows)
├── dl_inventory.tsv        ← 17 frozen `title <TAB> url <TAB> stage` rows, in CURRICULUM order
├── index.rst               ← GENERATED: the staged learning-path hub (3 view-levels, 4 stages)
├── 01-<slug>.rst           ← GENERATED: one lesson page per post (NN = curriculum order) …
│   …                          each with ◀ Previous / Next ▶ navigation
└── 17-<slug>.rst           ← GENERATED: … through lesson 17
```

**Source-of-truth files you edit** (then regenerate): `dl_inventory.tsv`, `build_deep_learning.py`,
`dl_content.py`. **Generated files you never hand-edit**: `index.rst`, every `NN-*.rst`.

---

## 2. Architecture — how the pieces fit

```
dl_inventory.tsv ─┐
 (title/url/stage) │
                   ├─►  build_deep_learning.py  ─►  index.rst  (staged hub + hidden toctree)
STAGES + GLOSS ────┤        (deterministic)     └►  01-*.rst … 17-*.rst  (one page/lesson,
 (inside script)   │                                with prev/next + See-also)
                   │
dl_content.py ─────┘
 (CONTENT bodies +
  MINDMAP links)
```

Three inputs, joined **by exact title string**:

1. **`dl_inventory.tsv`** — the frozen syllabus. Row *i* (1-based, after the header) fixes lesson
   *i*'s permanent id `NN` **and its reading order**. Columns: `title`, `url` (source, for
   traceability), `stage`. **17 rows.** Order is the **canonical Ng course sequence**, not the
   source's publish order (Section 8). Append only; never reorder (it renumbers pages).
2. **`STAGES` + `GLOSS`** (dicts inside `build_deep_learning.py`) — `STAGES[key] = (emoji, title,
   level, blurb)` defines the four course stages; `GLOSS[title]` is the one-line summary shown on
   each hub card.
3. **`CONTENT` + `MINDMAP`** (dicts inside `dl_content.py`) — `CONTENT[title]` is the full RST body;
   `MINDMAP[title]` lists **lateral** "See also" links (exact titles). Prev/next links are **not**
   stored — the generator derives them from curriculum order.

The generator emits, per lesson: an auto H1, a stage/level badge, **◀ Previous / Next ▶**
navigation, the body (or a working stub), a "Related lessons" See-also, and a source link. The hub
`index.rst` groups all lessons into **4 stages** and adds a hidden ordered toctree for sidebar nav.

### The 4 stages (in order)

| # | Stage key | Title | Level |
|---|---|---|---|
| 1 | `intro` | Introduction to Deep Learning | beginner |
| 2 | `neuron` | Logistic Regression as a Neuron | beginner |
| 3 | `calculus` | Derivatives & the Computation Graph | intermediate |
| 4 | `training` | Backprop & Vectorization | intermediate |

`STAGE_ORDER` fixes the sequence; lessons appear in inventory order within each stage.

---

## 3. Regenerating the pages (the build)

```bash
cd deep_learning
python build_deep_learning.py
```

What it does, in order:

1. Loads `dl_inventory.tsv` → the ordered list of 17 `(title, url, stage)` rows.
2. **Fail-fast checks** (Section 10) — inventory ↔ GLOSS, stages ↔ STAGES, CONTENT/MINDMAP keys ↔
   inventory titles.
3. Deletes existing `NN-*.rst` pages (glob `[0-9][0-9]-*.rst`) so the set never drifts, then writes
   all 17 fresh. **Only** that glob is removed — toolkit files (`.py`, `.tsv`, `.md`) and `index.rst`
   are never touched by the clear step.
4. Overwrites `index.rst` with the regenerated staged hub.

**Idempotency guarantee (verified):** the build is fully ordered and uses no randomness, so running
it twice with unchanged inputs produces byte-for-byte identical output. `__file__`-relative paths
(`HERE = Path(__file__).resolve().parent`, `OUT = HERE/"index.rst"`, `PAGES_DIR = HERE`) plus
`sys.path.insert(0, str(HERE))` mean it runs from any working directory and imports `dl_content`
cleanly.

---

## 4. Validating (run after every change — mandatory)

The build must always leave three counters at zero. Paste this validator and run it from the
folder's parent. **Note the code-block guard:** lesson bodies contain ``.. code-block:: python``
snippets, so the underline check **skips indented lines** (those starting with three spaces) to avoid
mistaking code for a section underline.

```python
import re, glob, os
files = sorted(glob.glob("deep_learning/*.rst"))
defined, docnames = set(), set()
for f in files:
    t = open(f, encoding="utf-8").read()
    for m in re.finditer(r'^\.\.\s+_([A-Za-z0-9_-]+):\s*$', t, re.M): defined.add(m.group(1))
    for m in re.finditer(r'^\s*:name:\s+([A-Za-z0-9_-]+)\s*$', t, re.M): defined.add(m.group(1))
    dn = f[:-4]; docnames.add(dn); docnames.add(os.path.basename(dn))
defined.add("terminology-index")            # cross-folder anchor referenced by the hub
bd = br = ue = 0
for f in files:
    t = open(f, encoding="utf-8").read()
    for m in re.finditer(r':doc:`(?:[^`<]*<)?([^`>]+)>?`', t):
        tg = m.group(1).strip()
        if tg not in docnames and os.path.basename(tg) not in docnames: bd += 1
    for m in re.finditer(r':ref:`(?:[^`<]*<)?([A-Za-z0-9_-]+)>?`', t):
        if m.group(1) not in defined: br += 1
    L = t.split("\n")
    for i in range(len(L) - 1):
        head, under = L[i], L[i + 1]
        if under.strip() and re.fullmatch(r'[=\-^"~]{3,}', under.strip()) \
           and len(set(under.strip())) == 1 \
           and not re.fullmatch(r'[=\-^"~]{3,}', head.strip()) \
           and head.strip() and not head.startswith("   ") \
           and len(under.strip()) < len(head.rstrip()):
            ue += 1; print("  UNDERLINE SHORT:", f, repr(head))
print(f"broken :doc: {bd} | broken :ref: {br} | underline errors {ue}")
```

**Report this exact evidence line every time you change content:**
`broken :doc: 0 | broken :ref: 0 | underline errors 0`.

**Title verification must be done in Python, never with a bash read-loop** — Unicode titles
(the en-dash, parentheses) survive Python string comparison but are mangled by shell word-splitting.
Completeness check:

```python
import dl_content as dc
inv = [l.split("\t")[0].strip() for l in open("dl_inventory.tsv", encoding="utf-8")]
missing = [inv[i] for i in range(1, 18) if inv[i] not in dc.CONTENT]   # row 0 is the header
print("lessons missing full content:", len(missing), missing)
print("MINDMAP entries:", len(dc.MINDMAP))
```

---

## 5. Updating & extending (add or edit lessons)

### Edit an existing lesson
1. Edit `CONTENT["<exact title>"]` (and/or `MINDMAP["<exact title>"]`) in `dl_content.py`.
2. `python build_deep_learning.py`  →  3. validate.

### Add a new lesson
1. **Append** a row to `dl_inventory.tsv`: `title<TAB>url<TAB>stage` — append only; the new row
   becomes the next `NN` and slots into its stage.
2. Add a `GLOSS["<title>"]` line in `build_deep_learning.py` (and a new `STAGES` entry if it starts a
   new stage; extend `STAGE_ORDER`).
3. Add `CONTENT["<title>"] = r"""…"""` and `MINDMAP["<title>"] = [<lateral titles>]` in
   `dl_content.py` (append with a **single-quoted heredoc** `cat >> dl_content.py << 'PYEOF'` so
   UTF-8 and LaTeX backslashes survive verbatim).
4. `python build_deep_learning.py`  →  5. validate.

### The exact-title rule (critical)
Every `CONTENT` / `MINDMAP` / `GLOSS` key must equal an inventory title **character-for-character**.
The fail-fast guard rejects mismatches. **One title in this course carries an en-dash** (`–`,
U+2013): *"Logistic Regression – Loss Function and Cost Function"*. Several carry **parentheticals**
(e.g. *"Binary Classification and Logistic Regression (Neural Network Basics)"*, *"Logistic
Regression (Binary Classification Model)"*). Reproduce them exactly, and write Unicode literally in
the raw strings (a literal `–`, not `\u2013`).

### Batch cadence
Work in batches of **~3 lessons** in curriculum order. Each finished page carries: a plain-language
idea, the **formula / derivation** where quantitative (sigmoid, cross-entropy, the `a − y` gradient,
vectorised updates), a **worked example** or short ``.. code-block:: python`` snippet, **pitfalls**,
connections to **`numpy` / scikit-plots / PyTorch / Keras**, and a curated **MINDMAP** of ~4 lateral
lessons. Validate after every batch.

---

## 6. Page content contract (what a good lesson contains)

Each `CONTENT[title]` is a raw RST string. The generator supplies the H1, the stage/level badge,
prev/next navigation and the See-also block; the body provides the substance. A strong lesson:

- opens with a **plain-English idea**, re-expressed (not paraphrased from the source);
- states the **formula** in a `.. math::` block when quantitative (omit for narrative lessons like
  the Hinton interview);
- shows a short **``numpy`` snippet** where it clarifies (the vectorised forward/backward pass);
- explains **how to read / apply it**, **when it holds**, and **pitfalls**;
- draws **`numpy` / scikit-plots (`visualkeras`) / PyTorch / Keras** connections where natural;
- ends implicitly with the auto **"Related lessons"** See-also from `MINDMAP[title]`.

Keep every lesson **self-contained** — a reader arriving by search should not need another page —
while the **prev/next** links preserve the intended reading path.

---

## 7. RST / formatting conventions (what keeps the build clean)

- **Section underlines must be ≥ the header length.** Sphinx errors if an underline is *shorter*;
  equal or longer is fine. **Pad each underline a few dashes longer** for safety. Only *section*
  underlines are hand-authored inside `CONTENT`; the page H1 is generated for you.
- **Code blocks** use ``.. code-block:: python`` with the body indented three spaces. The validator's
  underline check skips indented lines so code never trips it (Section 4).
- **Unicode counts as one column.** En/em-dashes, Greek (`α β σ`), `∂`, `≈`, `×` each occupy one
  column — Python's `len()` matches Sphinx, so measure headings with `len()`.
- **Bodies are raw strings** (`r"""…"""`) so LaTeX backslashes stay literal (`\frac`, `\sigma`,
  `\partial`, `\mathbf`). Write Unicode characters **directly** (a literal `–`, not `\u2013`).
- **Math** uses `.. math::` (display) or `:math:`…`` (inline), LaTeX syntax. Inline code (numpy
  calls) uses double back-ticks: ``np.dot(w.T, X)``.
- **Dollar signs:** avoid a bare `$` in prose (it can trigger math parsing); none are needed here.
  A bare `%` renders fine.
- Available Sphinx extensions include `sphinx_design` (`grid`, `grid-item-card`, `dropdown`,
  `badge`), `sphinx_tags`, `sphinx_copybutton`, and math directives. Do not use extensions that are
  not enabled.

---

## 8. Regenerating from scratch (if you start with nothing but this guide)

If `dl_inventory.tsv` / `dl_content.py` were ever lost, this is the methodology that produced them.

**Source corpus (context and framing only — never copied):**
`https://insightful-data-lab.com/category/deep-learning/` (17 posts, author *Ju Yeon Eum*). All 17
sit under the sub-module **"1. Neural Networks and Deep Learning"**; the site lists four further
sub-modules that are currently **empty** (0 posts) — see Section 11. The corpus tracks **Andrew Ng's
deeplearning.ai "Neural Networks and Deep Learning" course (Course 1), Weeks 1–2**, ending at
vectorised logistic regression.

**Freshness anchor (last verified):** all 17 posts are dated **2025-04-07** (the whole category
shares that single date). If newer posts appear — especially under the four empty sub-modules — the
inventory may need new rows and stages.

**The process:**
1. **Harvest all 17 titles deterministically** by paginating the category
   (`/category/deep-learning/page/N/`, ~10 posts/page → 2 pages). Record `title <TAB> url`
   (slug pattern `https://insightful-data-lab.com/2025/04/07/<slug>/`).
2. **Order by the canonical Ng sequence, not the publish order.** Because every post shares one
   date, the site's timestamp order is only a rough proxy; impose the course's own path:
   *Introduction to Deep Learning* → *Logistic Regression as a Neuron* → *Derivatives & the
   Computation Graph* → *Backprop & Vectorization*. Assign each lesson a `stage`.
3. **Two non-standard posts are handled deliberately:** *Geoffrey Hinton Interview* becomes a short
   **conceptual/historical** page (backprop's origins, the 2006 revival, capsules — not a
   transcript); *Why Deep Learning is Taking Off* becomes the **data + compute + algorithms**
   scaling-drivers page.
4. **Reframe as Python / numpy-first.** The source teaches the math with Python; this reference uses
   the **`numpy`** stack for the from-scratch derivations and notes **PyTorch / Keras** equivalents,
   plus scikit-plots' `visualkeras` for architecture views. The mathematics is identical.
5. **Ground each lesson** against authoritative references (the course notes, Stanford CS230 / CS229
   material, textbooks) before writing — never rely on memory for a formula or condition.
6. **Verify exact titles + lateral neighbours in Python** (never a bash read-loop).
7. **Rewrite in original words**, append `CONTENT` + `MINDMAP`, add `GLOSS` + `stage`.
8. **Regenerate → validate** (`broken :doc: 0 | broken :ref: 0 | underline errors 0`).

### Curriculum map (the 17 lessons, in order)

| # | Lesson | Stage |
|---|---|---|
| 01 | What is a Neural Network? | `intro` |
| 02 | Supervised Learning and Neural Networks | `intro` |
| 03 | Why Deep Learning is Taking Off | `intro` |
| 04 | Geoffrey Hinton Interview | `intro` |
| 05 | Binary Classification and Logistic Regression (Neural Network Basics) | `neuron` |
| 06 | Logistic Regression (Binary Classification Model) | `neuron` |
| 07 | Logistic Regression – Loss Function and Cost Function | `neuron` |
| 08 | Gradient Descent in Logistic Regression | `calculus` |
| 09 | Derivatives | `calculus` |
| 10 | More Derivative Examples | `calculus` |
| 11 | Computation Graph | `calculus` |
| 12 | Derivatives with a Computation Graph | `calculus` |
| 13 | Logistic Regression Gradient Descent | `training` |
| 14 | Gradient Descent on m Training Examples | `training` |
| 15 | Vectorization in Logistic Regression | `training` |
| 16 | More Vectorization Examples | `training` |
| 17 | Vectorizing Logistic Regression | `training` |

---

## 9. Deterministic sync + zip (only if publishing to a separate output dir)

Mirror the built folder to a publish directory, then zip. A transient copy error occasionally
appears; it recovers with no data loss because the source folder is never mutated:

```bash
OUT=/path/to/output/learn/deep_learning
sync
rm -rf "$OUT" && mkdir -p "$OUT" && cp ./*.rst ./*.py ./*.tsv ./*.md "$OUT/"
# (do NOT copy __pycache__; exclude *.pyc from any zip)
```

The version-controlled source tree is always intact; only the copy is retried.

---

## 10. Fail-fast guard & invariants (the safety net)

`build_deep_learning.py` aborts (non-zero exit, nothing written) if any hold:

- the inventory does not have exactly **17** lesson rows;
- an inventory title has **no** `GLOSS` entry, or a `GLOSS` entry is **not** an inventory title;
- an inventory `stage` is **not** in `STAGES`;
- a `CONTENT` key is **not** an exact inventory title;
- a `MINDMAP` key or any neighbour it references is **not** an exact inventory title.

Verified during the scaffold (bad content key → exit 1, nothing written; bad mind-map neighbour →
exit 1). **Invariants after a successful build:**

- `dl_inventory.tsv` has exactly **17** lesson rows (plus a header).
- Lesson ids `01`–`17` map 1-to-1 to curriculum order and never change.
- Output is **byte-identical** run-to-run for unchanged inputs.
- Every `:doc:` and `:ref:` resolves; every section underline is ≥ its heading.
- `len(CONTENT) == 17` and `len(MINDMAP) == 17` when the course is complete.

---

## 11. Suggestions & future work

- **The four empty sub-modules are the natural continuation.** The source's Deep Learning category
  lists five sub-modules; only the first is populated. As the others gain posts, add them as new
  stages (or sibling course folders) using the same generator + inventory + content + fail-fast +
  validate loop:

  | deeplearning.ai course | Source sub-module | Posts (last checked) |
  |---|---|---|
  | C1 · Neural Networks and Deep Learning **(this folder)** | 1. Neural Networks and Deep Learning | **17 ✓ complete** |
  | C2 · Improving Deep Neural Networks | 2. Improving Deep Neural Networks | 0 |
  | C3 · Structuring Machine Learning Projects | 3. Structuring Machine Learning Projects | 0 |
  | C4 · Convolutional Neural Networks | 4. Convolutional Neural Networks | 0 |
  | C5 · Sequence Models | 5. Sequence Models | 0 |

- **Keep the Sphinx build warning-free.** The toolkit files (`.py`, `.tsv`, this `.md`) live beside
  the `.rst` lessons. Sphinx ignores `.py`/`.tsv`, but a `.md` in a source tree can emit a
  "document isn't included in any toctree" warning. Add to `exclude_patterns` in `conf.py`:
  ```python
  exclude_patterns += [
      "learn/deep_learning/DEEP_LEARNING.md",
      "learn/deep_learning/*.py",
      "learn/deep_learning/*.tsv",
  ]
  ```
  (or move the toolkit into an underscore-prefixed `deep_learning/_toolkit/`, which Sphinx skips by
  default, adjusting `HERE` so pages still land in `deep_learning/`).
- **Wire the hub into navigation.** Ensure the parent `learn/index.rst` toctree references
  `deep_learning/index.rst` (it does today, as a `grid-item-card`). Note the generated `index.rst`
  **replaces** the earlier hand-written hub — its framing (neuron-to-networks arc, the three reader
  levels, the `visualkeras`/PyTorch/Keras connections, the terminology cross-link) was salvaged into
  the generator.
- **Add a CI gate.** Run `python build_deep_learning.py` then the Section 4 validator on every PR;
  fail on idempotency drift or any non-zero counter.
- **Continuing the wider corpus.** The same loop transfers to the remaining source categories:

  | Source category | Posts | Folder |
  |---|---|---|
  | Terminology | 431 ✓ complete | `learn/terminology/` |
  | Introduction to Time Series | 18 ✓ complete | `learn/time_series/` |
  | **Deep Learning (this folder)** | **17 ✓ complete** | `learn/deep_learning/` |
  | Bayesian Data Analysis | 144 | `learn/bayesian_data_analysis/` |
  | Data Analytics (8 sub-sections) | 216 | — |
  | Data Preparation and Analysis | 56 | `learn/data_preparation_and_analysis/` |

---

## 12. Quick rule card (pin this)

- **Rebuild:** `cd deep_learning && python build_deep_learning.py`. Pure stdlib, no args, no network.
- **Idempotent:** unchanged inputs → identical bytes. Never hand-edit `index.rst` or `NN-*.rst`.
- **Edit content** in `dl_content.py`; **add lessons** by appending to `dl_inventory.tsv`
  (+ `GLOSS` + `CONTENT` + `MINDMAP`, + a `STAGES` entry if new), then rebuild.
- **Keys must match inventory titles exactly** (the en-dash in lesson 07, parentheticals). Fail-fast
  enforces this.
- **Underlines ≥ heading length** (pad longer); Unicode counts as one column; write Unicode literally
  in raw strings; indent ``code-block`` bodies three spaces; avoid bare `$`.
- **Validate every change:** `broken :doc: 0 | broken :ref: 0 | underline errors 0`. Verify titles in
  **Python**, never a bash read-loop.
- **Order is the canonical Ng sequence**, not the source's publish order. **Prev/next** is auto;
  **MINDMAP** is lateral.
- **Rewrite, never copy** the source; **verified sources only** — invent nothing.
- **~3 lessons per batch**, each deep (idea, formula, worked example / numpy snippet, pitfalls,
  links).
