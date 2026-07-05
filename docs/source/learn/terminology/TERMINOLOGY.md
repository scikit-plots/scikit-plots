# scikit-plots `learn/terminology` — Regeneration & Maintenance Guide

**One document to rebuild, extend, or recreate this entire folder — no chat history required.**

| | |
|---|---|
| **Folder** | `docs/source/learn/terminology/` |
| **Renders at** | `https://scikit-plots.github.io/dev/learn/terminology/index.html` |
| **Scope** | 431 ML / data-science terminology pages + one grid index |
| **Generator** | `build_terminology.py` (Python 3, standard library only) |
| **Determinism** | Same inputs → **byte-identical** output, every run (sorted, no RNG) |
| **Status** | 431 / 431 pages complete, all cross-linked, all links + underlines valid |

> **Golden rules** (violate none):
> 1. **Idempotent** — rebuilding never changes output unless an input changed.
> 2. **Verified-only** — every fact and citation is grounded in a real source; nothing invented.
> 3. **Re-express, never copy** — the source corpus is *framing only*; all prose is rewritten in original words (copyright).
> 4. **Fail-fast** — a title/key mismatch aborts the build; there are no silent gaps.

---

## 0. TL;DR — rebuild in one command

```bash
cd docs/source/learn/terminology
python build_terminology.py      # clears old NNN-*.rst, rewrites index.rst + all 431 pages
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
terminology/
├── TERMINOLOGY.md          ← this guide (the only doc you need to read)
├── build_terminology.py    ← the deterministic generator (edit for themes/glosses/layout)
├── term_content.py         ← CONTENT + MINDMAP: the per-term bodies and cross-links (grows)
├── term_inventory.tsv      ← 431 frozen `title <TAB> url` rows (the master list, stable order)
├── index.rst               ← GENERATED: the semantic grid hub (26 theme cards, 3 levels)
├── 001-<slug>.rst          ← GENERATED: one self-contained page per term …
│   …                          (NNN = inventory row order, a stable permanent id)
└── 431-<slug>.rst          ← GENERATED: … through term 431
```

**Source-of-truth files you edit** (then regenerate): `term_inventory.tsv`,
`build_terminology.py`, `term_content.py`.
**Generated files you never hand-edit**: `index.rst`, every `NNN-*.rst`
(they are overwritten on each build).

---

## 2. Architecture — how the pieces fit

```
term_inventory.tsv ─┐
 (431 titles+urls)  │
                    ├─►  build_terminology.py  ─►  index.rst  (grid hub)
ENRICH  ────────────┤        (deterministic)   └►  001-*.rst … 431-*.rst  (one page/term)
 (theme + gloss,    │
  inside the script)│
                    │
term_content.py ────┘
 (CONTENT bodies +
  MINDMAP links)
```

Three inputs, joined **by exact title string**:

1. **`term_inventory.tsv`** — the frozen master list. Row *i* (1-based) fixes term *i*'s
   permanent numeric id `NNN`. Column 1 is the title, column 2 the source URL. **431 rows.**
   Do not reorder rows (it would renumber every page). Append only.
2. **`ENRICH`** (a dict inside `build_terminology.py`) — maps every title to
   `(theme_key, one-line rewritten gloss)`. Used for the grid cards and the per-page summary line.
3. **`CONTENT` and `MINDMAP`** (dicts inside `term_content.py`) —
   `CONTENT[title]` is the full RST body of the page; `MINDMAP[title]` is the list of related
   exact titles that become the page's "Related terms" cross-links.

The generator emits, for each term, a page with an auto-generated H1 (from the title), the
theme/level badges, the body from `CONTENT`, and mind-map links from `MINDMAP`; plus a single
`index.rst` grid hub grouping all terms into **26 themes across 3 levels**.

### The 26 themes (grouped by level, in registry order)

| Level | Theme key → display name |
|---|---|
| **foundations** | `probstats` Probability & Statistics Foundations · `inference` Statistical Inference & Power · `concepts` AI & ML Concepts |
| **applied** | `imbalance` Imbalanced Learning & Resampling · `metrics` Classification & Averaging Metrics · `evaluation` Model Evaluation & Uncertainty · `abtest` A/B Testing & Experimentation · `growth` Business & Growth Analytics · `validation` Validation & Cross-Validation · `training` Model Training & Optimization · `platforms` ML Platforms & Tools · `ops` Operations & Supply Chain · `features` Data Preparation & Features |
| **advanced** | `fairness` Fairness & Calibration · `bayes` Bayesian Inference · `bandits` Sequential Methods & Bandits · `signal` Signal Processing & Time Series · `mlops` MLOps, Serving & Monitoring · `drift` Distribution Shift & Drift · `repr` Representations & Embeddings · `ranking` Ranking & Interleaving · `causal` Causal Inference & Uplift · `risk` Risk & Probabilistic Forecasting · `recsys` Recommender Systems · `calibration` Probability Calibration · `xai` Explainability & Governance |

Level order is `["foundations", "applied", "advanced"]`; within a level, themes appear in
registry order. Each theme is defined in the `THEMES` dict as `(emoji, display, level, blurb)`.

---

## 3. Regenerating the pages (the build)

```bash
cd terminology
python build_terminology.py
```

What it does, in order:

1. Loads `term_inventory.tsv` → the ordered list of 431 `(title, url)` pairs.
2. **Fail-fast check** — aborts if any inventory title lacks an `ENRICH` entry, or any `ENRICH`
   entry is not an inventory title (Section 10).
3. Imports `CONTENT` and `MINDMAP` from `term_content.py` and **fail-fast checks** that every
   key is an exact inventory title.
4. Deletes existing `NNN-*.rst` pages (glob `[0-9][0-9][0-9]-*.rst`) so the set never drifts,
   then writes all 431 fresh. **Only** files matching that glob are removed — the toolkit files
   (`.py`, `.tsv`, `.md`) and `index.rst` are never touched by the clear step.
5. Overwrites `index.rst` with the regenerated grid hub.

**Idempotency guarantee (verified):** because the build is fully sorted and uses no randomness,
running it twice with unchanged inputs produces a byte-for-byte identical tree. A SHA-256 of the
concatenated pages is stable across runs and across relocating the folder.

`__file__`-relative paths (`HERE = Path(__file__).resolve().parent`) mean the script works from
any working directory; it also inserts `HERE` on `sys.path` so `term_content` imports cleanly.

---

## 4. Validating (run after every change — mandatory)

The build must always leave three counters at zero: broken `:doc:` links, broken `:ref:` links,
and section-underline errors. Paste this validator and run it from the folder's parent:

```python
import re, glob, os
files = sorted(glob.glob("terminology/**/*.rst", recursive=True))
defined, docnames = set(), set()
for f in files:
    t = open(f, encoding="utf-8").read()
    for m in re.finditer(r'^\.\.\s+_([A-Za-z0-9_-]+):\s*$', t, re.M): defined.add(m.group(1))
    for m in re.finditer(r'^\s*:name:\s+([A-Za-z0-9_-]+)\s*$', t, re.M): defined.add(m.group(1))
    dn = f[:-4]; docnames.add(dn); docnames.add(os.path.basename(dn))
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
        if under.strip() == "^^^": continue                     # card separators are not underlines
        if head.strip() and re.fullmatch(r'[=\-^"~]{3,}', under.strip()) \
           and len(set(under.strip())) == 1 \
           and not re.fullmatch(r'[=\-^"~]{3,}', head.strip()) \
           and len(under.strip()) < len(head.rstrip()):
            ue += 1; print("  UNDERLINE SHORT:", f, repr(head))
print(f"broken :doc: {bd} | broken :ref: {br} | underline errors {ue}")
```

**Report this exact evidence line every time you change content:**
`broken :doc: 0 | broken :ref: 0 | underline errors 0`.

**Title verification must be done in Python, never with a bash read-loop** — Unicode titles
(accents, superscripts, dashes, a trailing period) survive Python string comparison but are
mangled by shell word-splitting. Example check that every page's title matches the inventory:

```python
import term_content as tc
inv = [l.split("\t")[0].strip() for l in open("term_inventory.tsv", encoding="utf-8")]
missing = [inv[i] for i in range(1, 432) if inv[i] not in tc.CONTENT]   # row 0 is the header
print("terms missing full content:", len(missing), missing[:10])
print("MINDMAP entries:", len(tc.MINDMAP))
```

---

## 5. Updating & extending (add or edit terms)

### Edit an existing term
1. Edit `CONTENT["<exact title>"]` (and/or `MINDMAP["<exact title>"]`) in `term_content.py`.
2. `python build_terminology.py`  →  4. validate.

### Add a new term
1. **Append** a row to `term_inventory.tsv`: `title<TAB>url` (append only — never reorder;
   the new row becomes the next `NNN`).
2. Add an `ENRICH` entry in `build_terminology.py`: `"<title>": ("<theme_key>", "<one-line gloss>")`.
3. Add `CONTENT["<title>"] = r"""…"""` and `MINDMAP["<title>"] = [<related exact titles>]` in
   `term_content.py`. `term_content.py` is appended with a **single-quoted heredoc**
   (`cat >> term_content.py << 'PYEOF'`) so UTF-8 and LaTeX backslashes are preserved verbatim.
4. `python build_terminology.py`  →  5. validate.

### The exact-title rule (critical)
Every `CONTENT`/`MINDMAP`/`ENRICH` key **must equal an inventory title character-for-character**.
The fail-fast guard rejects any mismatch. Titles in this corpus include tricky codepoints — all
must be reproduced exactly:

| Character | Codepoint | Example title |
|---|---|---|
| `é` | U+00E9 | *Naïve Bayes*, accented terms |
| `²` | U+00B2 | `R² (R-squared)` |
| `χ` | U+03C7 | `Chi-square (χ²) Test` |
| en-dash `–` | U+2013 | ranges inside titles |
| em-dash `—` | U+2014 | titles using em-dashes |
| `&`, `@`, `$` | — | e.g. `Inference Cost (Inference $)` |
| straight `'` | U+0027 | apostrophes (not the curly U+2019) |
| trailing `.` | — | `Bayesian Inference.` (yes, the period is part of the title) |

### Batch cadence
Work in batches of **~3 terms**. Each finished page carries: a plain-language definition,
the **formula / key relationship** where the term is quantitative, a **worked intuition**,
**edge cases**, **common pitfalls**, **when-to-use (and when not)** guidance, **ML connections**,
and a curated **MINDMAP** of ~6 related terms. Validate after every batch.

---

## 6. Page content contract (what a good page contains)

Each `CONTENT[title]` is a raw RST string. The generator supplies the H1 (page title) and the
theme/level scaffolding; the body provides the substance. A strong page:

- Opens with a **plain-English "what it is"**, rewritten from understanding (not the source text).
- States the **formula** in a `.. math::` block when the term is quantitative (omit for
  non-mathematical terms).
- Explains **how to read / interpret it**, **when to use it and when not**, **edge cases**, and
  **pitfalls**.
- Draws **ML / scikit-plots / scikit-learn connections** where they exist.
- Ends implicitly with the auto-generated **"Related terms"** block, sourced from `MINDMAP[title]`
  (list neighbours by their exact titles; the build turns them into working cross-links).

Keep every page **self-contained** — a reader arriving by search should not need any other page.

---

## 7. RST / formatting conventions (what keeps the build clean)

- **Section underlines must be ≥ the header length.** Sphinx errors if an underline is *shorter*
  than its title; equal or longer is fine. **Pad each underline a few dashes longer than the
  heading** for safety. Only *section* underlines are hand-authored inside `CONTENT`; the page's
  H1 is generated for you.
- **Unicode counts as one column.** `é`, `²`, `χ`, en/em-dashes, `μ`, `α`, `β`, `≈`, `×`, and
  accented letters each occupy a single column — Python's `len()` matches Sphinx's counting, so
  measure headings with `len()`.
- **Card `^^^` separators are not underlines.** The validator skips any line that is exactly
  `^^^`; don't treat those as section underlines.
- **Dollar signs:** a bare `$` in prose can trigger math parsing — wrap amounts in ``inline
  code`` or spell out "dollars". A bare `%` renders fine. A single unpaired `$` inside a *title*
  renders literally and is safe.
- **Bodies are raw strings** (`r"""…"""`) so LaTeX backslashes stay literal (`\frac`, `\sqrt`).
- **Math** uses `.. math::` (display) or `:math:`…`` (inline), LaTeX syntax.
- Available Sphinx extensions in this project include `sphinx_design` (`grid`, `card`,
  `dropdown`, `tab-set`, `badge`), `sphinx_tags`, `sphinx_togglebutton`, `sphinx_copybutton`,
  and math directives. Do not use extensions that are not enabled.

---

## 8. Regenerating from scratch (if you start with nothing but this guide)

If `term_inventory.tsv` / `term_content.py` were ever lost, this is the original end-to-end
methodology that produced them. It is deterministic and repeatable.

**Source corpus (context and framing only — never copied):**
`https://insightful-data-lab.com/category/00terminology/` (431 posts, author *Ju Yeon Eum*).
Post URLs follow `https://insightful-data-lab.com/YYYY/MM/DD/<slug>/`.

**Freshness anchors (last verified):**
- Site-wide last post: **2026-01-17** — `…/2026/01/17/exponential-smoothing-models/`.
- Terminology-category last post: **2025-08-30** — `…/2025/08/30/subsampling/`.

If newer posts exist past these dates, the inventory may need new rows; otherwise the 431-row
list is complete.

**The process, per term:**
1. **Harvest all 431 titles deterministically** by paginating the category
   (`/category/00terminology/page/N/`, ~10 posts/page). The RSS feed
   (`/category/00terminology/feed/`) and the WordPress REST API
   (`/wp-json/wp/v2/posts?per_page=100&categories=<id>`) are spot-check aids, **not** substitutes
   for full pagination. Produce a flat `title <TAB> url` list → `term_inventory.tsv`.
2. **Ground each term.** Run a distinctive-anchor search on the title. For common terms, standard
   references suffice; for SEO-buried or ambiguous terms, find an **authoritative** source
   (official docs, papers, primary references) before writing. Never rely on memory for specifics
   — search and confirm.
3. **Verify the exact title, codepoints, and neighbours in Python** (never a bash read-loop) so
   special characters match the inventory exactly and every `MINDMAP` neighbour is a real title.
4. **Rewrite in original words.** Express each definition from understanding; do not paraphrase
   the source line-by-line. This is a hard copyright rule.
5. **Append** `CONTENT[title]` and `MINDMAP[title]` to `term_content.py` (single-quoted heredoc),
   assign the term a `theme_key` + gloss in `ENRICH`.
6. **Regenerate → validate** (`broken :doc: 0 | broken :ref: 0 | underline errors 0`).

---

## 9. Deterministic sync + zip (only if publishing to a separate output dir)

When mirroring the built folder to a publish/output directory, a transient filesystem error can
occasionally appear on the copy step; it always recovers with no data loss because the source
folder is never mutated. Use this self-recovering sync, then re-zip:

```bash
OUT=/path/to/output/learn                       # adjust to your publish dir
sync
if rm -f "$OUT/terminology/"*.rst 2>/dev/null && cp terminology/*.rst "$OUT/terminology/" 2>/dev/null; then
  echo "  sync OK"
else
  echo "  transient I/O — rebuilding target dir"
  mv "$OUT/terminology" "/tmp/.term_corrupt_$(date +%s)" 2>/dev/null
  mkdir -p "$OUT/terminology" && cp terminology/*.rst "$OUT/terminology/"
  rm -rf /tmp/.term_corrupt_* 2>/dev/null
fi
cp terminology/build_terminology.py terminology/term_content.py \
   terminology/term_inventory.tsv terminology/TERMINOLOGY.md "$OUT/terminology/"
```

The source tree under version control is always intact; only the copy is retried.

---

## 10. Fail-fast guard & invariants (the safety net)

`build_terminology.py` aborts (non-zero exit, nothing written) when any of these hold:

- an inventory title has **no** `ENRICH` entry, or an `ENRICH` entry is **not** an inventory title;
- a `CONTENT` key is **not** an exact inventory title;
- a `MINDMAP` key or any neighbour it references is **not** an exact inventory title.

This is what guarantees there are no silent gaps and no dangling cross-links. **Invariants that
must always hold after a successful build:**

- `term_inventory.tsv` has exactly **431** term rows (plus a header row).
- Page ids `001`–`431` map 1-to-1 to inventory order and never change.
- Output is **byte-identical** run-to-run for unchanged inputs.
- Every `:doc:` and `:ref:` resolves; every section underline is ≥ its heading.
- `len(CONTENT) == 431` and `len(MINDMAP) == 431` when the glossary is complete.

---

## 11. Suggestions & future work

- **Keep the Sphinx build warning-free.** The toolkit files (`.py`, `.tsv`, and this `.md`) now
  live beside the `.rst` pages. Sphinx ignores `.py`/`.tsv`, but a `.md` in a source tree can emit
  a "document isn't included in any toctree" warning. Either add them to `exclude_patterns` in
  `conf.py`:
  ```python
  exclude_patterns += [
      "learn/terminology/TERMINOLOGY.md",
      "learn/terminology/*.py",
      "learn/terminology/*.tsv",
  ]
  ```
  or (alternative) move the toolkit into an underscore-prefixed subfolder
  (`terminology/_toolkit/`), which Sphinx skips by default — then set `HERE = Path(__file__)…parent.parent`
  in the script so it still writes pages into `terminology/`.
- **Wire the hub into navigation.** Ensure the parent `learn/index.rst` toctree references
  `terminology/index.rst` (a `grid-item-card` mirroring the sibling `glossary`/`resources` cards),
  or the page renders but is unreachable from the sidebar.
- **Add a CI gate.** Run `python build_terminology.py` then the Section 4 validator on every PR;
  fail the build if the tree changes unexpectedly (idempotency drift) or any counter is non-zero.
- **If `term_content.py` grows unwieldy** (currently ~0.75 MB), consider splitting `CONTENT` into
  per-theme modules imported into one namespace, keeping the exact-title contract intact.
- **Applying this pattern to the sibling subfolders.** The same generator + inventory + content +
  fail-fast + validate loop transfers directly to the other source categories, which are next:

  | Source category | Posts |
  |---|---|
  | Bayesian Data Analysis | 144 |
  | Data Analytics (Foundations 27 · Data-Driven Decisions 27 · Data Preparation 25 · Data cleaning & preparation 32 · Analyze Data 30 · Data visualization 27 · Data Analysis Using Python 33 · Job Search 15) | 216 |
  | Data Preparation and Analysis | 56 |
  | Deep Learning (Neural Networks and Deep Learning 17; remaining sub-modules currently 0) | 17 |
  | Introduction to Time Series | 18 |
  | **Terminology (this folder)** | **431 ✓ complete** |

---

## 12. Quick rule card (pin this)

- **Rebuild:** `cd terminology && python build_terminology.py`. Pure stdlib, no args, no network.
- **Idempotent:** unchanged inputs → identical bytes. Never hand-edit `index.rst` or `NNN-*.rst`.
- **Edit content** in `term_content.py`; **add terms** by appending to `term_inventory.tsv`
  (+ `ENRICH` + `CONTENT` + `MINDMAP`), then rebuild.
- **Keys must match inventory titles exactly** (accents, `²`, `χ`, dashes, `&`, `@`, `$`,
  straight `'`, trailing `.`). Fail-fast enforces this.
- **Underlines ≥ heading length** (pad longer); Unicode counts as one column; `^^^` is not an
  underline; avoid bare `$` in prose.
- **Validate every change:** `broken :doc: 0 | broken :ref: 0 | underline errors 0`. Verify titles
  in **Python**, never a bash read-loop.
- **Rewrite, never copy** the source. **Verified sources only** — invent nothing.
- **~3 terms per batch**, each deep (definition, formula, examples, edge cases, pitfalls,
  when-to-use, ML links, mind-map).

## Hub layout (v2 — filterable browser)

`index.rst` is emitted as a **collapsed, filterable browser** instead of the original
always-expanded card grid (431 links were visible at once; the page now loads as ~30 rows):

- a **live filter box** (small dependency-free JS, generator-emitted): typing filters every
  term by name *or gloss keyword*, auto-opens matching theme dropdowns, and shows a match
  count; with JS disabled the page degrades to plain collapsible sections;
- one **`.. dropdown::` per theme** (26, grouped under the three level headers), each with
  the theme blurb and a two-column `hlist` of its terms;
- one **A–Z master dropdown** listing all terms alphabetically (regenerated + sorted
  automatically);
- **anchor contract preserved**: `_terminology-index:`, the legacy level anchors, and every
  `_term-theme-<key>:` label (externally referenced, e.g. by the Bayesian hub) still resolve.

Adding a term is unchanged (inventory row + GLOSS/ENRICH entry); counts, dropdown contents
and the A–Z list regenerate deterministically. Term pages are untouched by the redesign.
