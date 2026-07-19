# NEW_COURSE_PLAYBOOK.md — building a new `learn/` course

A reusable recipe for adding a course to the scikit-plots `learn/` hub, distilled
from the six existing courses (`terminology`, `time_series`, `deep_learning`,
`data_preparation_and_analysis`, `bayesian_data_analysis`, `data_analytics`). Hand
this file plus `scikit-plots-learn.zip` to a fresh chat and it has everything it
needs to start: the templates are the existing courses, the validator is the
in-folder `_buildcheck.py`, and the two build modes are described below.

> **Golden rules (all courses):** everything a course needs lives in its own folder;
> the generator is a **pure, deterministic, idempotent** function of its inputs
> (byte-identical on every run); validate every turn with the folder's
> `_buildcheck.py`; ship no `.pyc`/`__pycache__`; and every generated lesson page
> carries the **AI-generated-content admonition** (see §6).

---

## 1. Pick the mode first

Two modes, chosen by how the source content may be used.

| | **Mode A — Rewrite** | **Mode B — Verbatim mirror (CC BY)** |
| --- | --- | --- |
| When | Source is copyrighted / all-rights-reserved; you must express it in your own words. | Source is openly licensed (e.g. **CC BY 4.0**) and you reproduce it **verbatim** with attribution. |
| Content origin | Written fresh by the model, grounded in the source's ideas. | Fetched **as-is** from the source's reStructuredText. |
| Copyright posture | Avoids reproduction; original prose. | Explicit attribution + license notice on every page + a `LICENSE` file. |
| Existing example | The six current courses (rewrites of `insightful-data-lab.com`). | `hands-on/edtech/` (mirror of `labs.bilimedtech.com`, CC BY 4.0). See §7. |
| Cross-links | `MINDMAP` (4 neighbours) + `GLOSS`. | Preserve the source's own `toctree` structure; keep its internal links. |

Both modes share the folder layout (§2), the generator contract (§3), the AI notice
(§6), and validation (§8). Modes A and B differ only in **where the body text comes
from** and **what per-page notices are attached**.

---

## 2. Folder layout (both modes)

One self-contained folder under `learn/` (or a nested path such as
`learn/hands-on/edtech/`). Everything the generator reads or writes lives inside it.

```
<course>/
  build_<course>.py        # the generator (pure, deterministic, idempotent)
  <prefix>_content.py       # CONTENT{}, MINDMAP{}, and (optional) GLOSS{} — keyed by exact title
  <prefix>_inventory.tsv    # FROZEN: header + one row per lesson, in curriculum order
  <COURSE>.md               # the course guide (twin of DATA_ANALYTICS.md / BAYESIAN_DATA_ANALYSIS.md)
  _buildcheck.py            # drop-in, identical across every course (see §8)
  index.rst                 # generated hub/browser
  <NNN>-<slug>.rst          # generated lesson pages (flat), OR
  <n>_<section>/…           # generated per-section subfolders (nested, like data_analytics)
  LICENSE_<course>          # Mode B only: the full license text (e.g. CC BY 4.0)
```

**Flat vs nested.** Flat = one `index.rst` and all lesson pages at the top level
(five of the six courses). Nested = a top hub `index.rst` plus one `index.rst` per
section, with lesson pages in per-section subfolders and **per-section 3-digit
numbering** (`data_analytics`). Choose nested when the course has natural top-level
sections each with many lessons. The `_buildcheck.py` handles both automatically.

---

## 3. The generator contract (both modes)

`build_<course>.py` is a pure function of its inputs. Requirements:

- `HERE = Path(__file__).resolve().parent`; `sys.path.insert(0, str(HERE))` so the
  content module imports from any working directory.
- **Fail-fast validation** before writing anything (exit 1 on): inventory row count
  mismatch; any inventory title lacking a `CONTENT` entry; any
  `CONTENT`/`MINDMAP`/`GLOSS` key not an exact inventory title; any `MINDMAP`
  neighbour not an exact title; any unknown section/stage; duplicate titles.
- Clear only generated files (`[0-9]+-*.rst`) before regenerating; never touch the
  guide, `_buildcheck.py`, `LICENSE`, or the hand inputs.
- **Determinism:** everything sorted, no RNG, no timestamps → two runs are
  byte-identical. Prove it every run (the `_buildcheck.py` idempotency check).
- Emit globally-unique anchors with a course prefix (`<prefix>-<section>-NNN`,
  `<prefix>-<section>-index`, a hub anchor `<course>-index`, etc.). Keep any legacy
  anchors as compatibility anchors if replacing a hand-written page.

Each lesson page: anchor(s) → auto H1 (bar sized by `len(title)`) → section/stage
badge → ◀ Prev / Next ▶ nav (+ ↑ Section / ↑ Hub for nested) → **AI notice (§6)** →
body → related-links (`.. hint::` from `MINDMAP`) → source (`.. seealso::`).

---

## 4. Shared six-phase workflow

1. **Harvest** the source into a `<prefix>_inventory.tsv` (title, url, section,
   stage). For a Sphinx source site, follow the `toctree` (§7). **Freeze it** — it is
   the single source of truth for titles/order/structure.
2. **Build the generator on stubs** — prove the whole tree renders and validates with
   every lesson a stub, before writing any content. Register all sections/stages.
3. **Fill content in small batches** (Mode A) or **fetch pages** (Mode B). Every
   batch: read exact frozen titles via `awk`; **pre-check every cross-link target**
   against the inventory *before* writing; ground named tools/frameworks/figures with
   `web_search`; append; rebuild; run `_buildcheck.py`; confirm `doc 0 | ref 0 |
   underline 0` and `idem OK`.
4. **Write the guide** `<COURSE>.md` (twin of an existing one), documenting the
   curriculum table, generator mechanics, anchor scheme, browser, landmines,
   validation, and rebuild steps. Verify every claim against the actual build.
5. **Clean-room seal** — rebuild from scratch in a fresh directory from only the
   inputs (generator, content, inventory, guide) and prove byte-identical output.
6. **Deliver** — sync into `/mnt/user-data/outputs/learn/<course>` and rezip the
   whole `learn/` tree (`_buildcheck.py --sync` does this).

---

## 5. Mode A — Rewrite (the six existing courses)

Write each lesson **in original words** (avoid reproducing the source). Structure:
intuition → key idea → a real worked example (`.. code-block::` where technical) → an
honest **"The caveat"** section; section-closers hand off to the next section. Add
`MINDMAP[title]` (4 exact-title neighbours) and `GLOSS[title]` (one-liner). Append via
`cat >> <prefix>_content.py << 'PYEOF' … PYEOF`.

**Landmines seen in practice (guard for these):**

- **Exact frozen titles** — always read titles from the inventory with `awk`; guessed
  titles have been wrong repeatedly. Pre-check every neighbour before writing.
- **Special characters in titles** (em-dash `—`, subscript `₂`, `&`, parentheses) —
  write them **literally** into a quoted-heredoc `r"""…"""` body; slugs strip to clean
  ASCII. `len()` counts each as one column, so H1/underlines stay correct.
- **Preserved source typos** — reproduce byte-for-byte (title *and* slug); do not
  "correct" them, or the clean-room seal breaks.
- **Triple-quote collision** — if a lesson's code block shows a `"""` docstring, that
  `CONTENT` entry **must** use the `r'''…'''` delimiter, or the inner `"""` ends the
  string and the build fails with a `SyntaxError`.
- **Short heading underlines** — "Why … matters"-style headings need the dash count ≥
  the heading length; the `_buildcheck.py` underline check catches these every run.
- **Shell is `/bin/sh` (dash)** — `$'\uXXXX'` does not expand in heredocs; append any
  Unicode inventory rows via Python. (Literal Unicode inside a quoted-heredoc body is
  fine.)

See `DATA_ANALYTICS.md` §"Landmines" for the concrete table and the `r'''` rule.

---

## 6. The AI-generated-content admonition (both modes, required)

Every generated lesson page must open its body with an **`.. important::`** notice so
readers at any level (newbie → lead → senior) know the page is AI-assisted and must be
verified. It is emitted from a module-level constant in each generator, placed right
after the nav and before the body:

```python
AI_NOTICE = (
    ".. important::\n"
    "\n"
    "   **AI-generated content.** This page was written with the assistance of an\n"
    "   AI language model and is provided as a learning aid. Despite careful\n"
    "   review, it may still contain mistakes, omissions, or out-of-date\n"
    "   information. Whether you are new to the topic, a team lead, or a senior\n"
    "   practitioner, treat it as a starting point rather than an authoritative\n"
    "   reference: read it critically and independently verify anything you act on\n"
    "   (code, commands, figures, and factual claims) against official\n"
    "   documentation and primary sources before relying on it."
)
```

Emit it once per lesson with the generator's line-append function (`a(AI_NOTICE)` /
`w(AI_NOTICE)`), followed by a blank line. All six current generators already do this
(patched together). A new generator should copy the constant and the one emit call.

**Mode B note:** for a verbatim mirror you are reproducing someone else's work, so the
"AI-generated" wording would be inaccurate for the *body*. Instead attach the
**attribution/license admonition** (§7) to every mirrored page, and reserve the
AI-notice only for any pages *you* author (e.g. a landing/index you write yourself).

---

## 7. Mode B — Verbatim mirror under CC BY (the `hands-on/edtech/` build)

Target: reproduce `https://labs.bilimedtech.com` **exactly, without changes**, into
`learn/hands-on/edtech/`, under **CC BY 4.0**, with proper attribution.

### 7.1 Fetch strategy (proven)

`web_fetch` only accepts URLs already surfaced in the conversation — **constructed
`_sources/…` URLs are rejected**. So fetch link-by-link:

1. Fetch a page's rendered **`.html`** (the root `index.html` is the entry point). The
   RTD theme page exposes: real `href` links to every child page, and a **"View page
   source"** link to that page's `_sources/<path>.rst.txt`.
2. Fetch that **`_sources/<path>.rst.txt`** (now allowed, because the link appeared in
   the HTML result) to get the **exact reStructuredText**.
3. Parse its `.. toctree::` blocks for child document paths, fetch each child's
   `.html` (its link appeared in the parent's HTML), and recurse to the leaves.

The root `toctree` for this site is:

```
Labs:       cloud-computing/index · nasm/index · c/index · operating-systems/index
Workshops:  workshops/index
```

Each section `index` has its own `toctree` listing the leaf lab pages. Mirror the
**exact source path structure** under `hands-on/edtech/` (e.g.
`hands-on/edtech/c/index.rst`, `hands-on/edtech/c/<lab>.rst`).

### 7.2 Reproduce verbatim

Write each fetched `.rst.txt` to its mirrored path **unchanged** (same directives,
`toctree`s, code-blocks, internal `:doc:`/`:ref:` links). Do **not** rewrite, reorder,
or "improve" — CC BY verbatim reproduction is the whole point. The one allowed edit is
the site index: you may drop the `Search` section (Sphinx generates search itself), as
noted in the plan.

### 7.3 Attribution + license on every page

CC BY 4.0 requires attribution (author, source, license link) and a change notice.
Attach a Sphinx admonition to **every mirrored page** (an `.. admonition::` /
`.. note::` — pick one class and use it consistently). Suggested block:

```rst
.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__
   (specific page: `<full source page URL>`__), licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0)
   <https://creativecommons.org/licenses/by/4.0/deed.en>`__.
   See :doc:`LICENSE <LICENSE_edtech>` for the full license text.
```

Fill `<full source page URL>` with that page's canonical source (its `.html` URL) so
each page cites its exact origin. If you make **any** adaptation to a page, the notice
must say so explicitly (change notice) — but the plan is verbatim, so state "without
modification".

### 7.4 The LICENSE file

Fetch the full legal text once and save it as `hands-on/edtech/LICENSE_edtech` (or
`.rst`), from:

- `https://creativecommons.org/licenses/by/4.0/legalcode.txt`

Reference it from every page's notice via `:doc:`LICENSE <LICENSE_edtech>``. Add a
short header line to the file: *"All educational content in this folder is reproduced
without modification from https://labs.bilimedtech.com under the Creative Commons
Attribution 4.0 International (CC BY 4.0) license."* (Fetching that exact URL in a new
chat: the CC domain may need to be surfaced first — `web_search` for the CC BY 4.0
legal code, then fetch the result link, or fetch the `deed.en` page which links to the
legal code.)

### 7.5 Canonical URL / duplicate-content (SEO)

Verbatim reproduction risks a duplicate-content penalty. Point each mirrored page's
**canonical URL at the original source page** so search engines credit the original.
The source pages already declare `canonical: https://labs.bilimedtech.com/…`. In the
Sphinx build, set this per page — e.g. a raw-HTML `<link rel="canonical" href="<source
page URL>">` in the page head via a `.. meta::`/`html_baseurl` mechanism, or the
theme's canonical option. Document the exact hook in the `EDTECH.md` guide when built.

### 7.6 Structure recap

```
learn/hands-on/edtech/
  index.rst                    # adapted site index (Search section removed)
  LICENSE_edtech               # full CC BY 4.0 text + header
  cloud-computing/index.rst  + leaf labs
  nasm/index.rst             + leaf labs
  c/index.rst                + leaf labs
  operating-systems/index.rst+ leaf labs
  workshops/index.rst        + leaf workshops
  _buildcheck.py               # drop-in validator
  EDTECH.md                    # course guide (attribution/license/fetch-map documented)
```

The inventory for a mirror can record `title · source_url · section · leaf`, harvested
by walking the toctrees; freeze it so re-fetches and extensions are reproducible.

---

## 8. Validation with `_buildcheck.py` (both modes)

Every course carries an **identical** `_buildcheck.py`. Run it from inside the folder:

```bash
python3 _buildcheck.py            # build + validate + idempotency (+ census)
python3 _buildcheck.py --sync     # …then mirror to outputs/learn and rezip
```

It auto-discovers the generator and content module, takes all paths relative to its
own directory (so **flat and nested both work**), and prints:

```
[<course>] <generator's last line>
validate: doc 0 | ref 0 | underline 0  ||  content N mindmap N [gloss N]  ||  lessons F/T full  ||  idem OK <sha12>
```

A green run is `doc 0 | ref 0 | underline 0`, `idem OK`, `F/T` at target. A folder
with **no generator** (a hand-maintained static folder, or a verbatim mirror with no
generator step) runs in **validate-only** mode — it still checks every `:doc:`/`:ref:`
link and underline. Its cross-course `:ref:` anchor set (`DEFINED_ANCHORS`) lists every
sibling hub anchor; add the new course's hub anchor there when introducing one, so
inter-course links validate everywhere.

> Mode B option: a mirror can skip a generator entirely and just place the fetched
> `.rst` files directly (with the attribution notice injected as you write each file).
> Then `_buildcheck.py` runs validate-only over the mirrored tree. If you prefer a
> generator (e.g. to inject the attribution notice uniformly and keep re-fetches
> idempotent), follow the §3 contract with the body sourced from a `SOURCES{}` dict of
> fetched raw RST keyed by page path.

---

## 9. Extending or editing a course later

- **Add lessons:** append rows to the (unfreeze → edit → refreeze) inventory and add
  `CONTENT`/`MINDMAP`/`GLOSS` (Mode A) or fetch the new pages (Mode B); rebuild; run
  `_buildcheck.py`. Numbering shifts are handled by the generator (per-section for
  nested).
- **Edit a lesson:** change the `CONTENT` entry (Mode A) or re-fetch the source page
  (Mode B); rebuild; validate. The idempotency check confirms nothing else moved.
- **Re-mirror (Mode B):** re-run the fetch for changed pages only; the frozen inventory
  + per-page source URLs make this reproducible. If the source changed, update the
  page's change notice accordingly.
- **Introduce a new course:** copy an existing course as the template, add its hub
  anchor to every `_buildcheck.py`'s `DEFINED_ANCHORS`, and follow §4.

---

## 10. Quick-start checklist for the new chat

```
[ ] Upload scikit-plots-learn.zip + this playbook.
[ ] Confirm mode: A (rewrite) or B (verbatim CC-BY mirror).
[ ] Confirm flat vs nested and rough scale (sections / lessons).
[ ] Harvest + FREEZE the inventory (follow the toctree for a Sphinx source).
[ ] Build the generator on stubs; validate green.
[ ] Fill content (A) / fetch pages (B) in small batches; _buildcheck every turn.
[ ] Every page: AI notice (A) or attribution+license notice (B).
[ ] Mode B: fetch LICENSE_<course>; wire per-page canonical to the source.
[ ] Write the <COURSE>.md guide; verify claims against the build.
[ ] Clean-room seal (byte-identical rebuild).
[ ] _buildcheck.py --sync; deliver the zip.
```

For `hands-on/edtech/`: **Mode B**, nested under `hands-on/`, source
`https://labs.bilimedtech.com` (© 2022 BilimEdtech, CC BY 4.0), verbatim, structure per
§7.6.
