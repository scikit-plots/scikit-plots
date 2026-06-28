# Terminology Page — Migration & Review Guide

**Target file:** `docs/source/learn/terminology/index.rst`
**Renders at:** `https://scikit-plots.github.io/dev/learn/terminology/index.html`
**Source of context:** `https://insightful-data-lab.com/category/00terminology/` (431 posts)
**Status of current draft:** 1717 lines, 8 domains, ~40 terms already written (uploaded `index.rst`)

This guide is the **plan and rulebook** for finishing the terminology hub across several
sessions. It is deliberately a separate document because the source corpus (431 posts) and
the RST file are both too large to process in a single pass. Read this once, then each
follow-up session executes one **Run** from Section 7 against exact line numbers.

---

## 0. TL;DR — what to do, in order

1. **Wire the page into navigation** — it is not in the parent toctree yet (Section 8.1). One-line gap; do this first or the page never appears.
2. **Adopt the per-term contract** (Section 3.2) and the **sphinx_design templates** (Section 3.3) verbatim so every future term looks identical.
3. **Fix the tag block** to match the project's controlled vocabulary and place it at the page **bottom** (Section 3.4). The draft currently violates the documented convention.
4. **Harvest all 431 titles** with the deterministic method in Section 6, producing a flat list.
5. **Slot titles into the Master Taxonomy** (Section 4) — extend from 8 → 15 domains so every term has a home.
6. **Execute Runs 1..N** (Section 7), one domain batch per session, appending terms with verified content only.
7. **Keep `Sources` verified-only** (Section 3.5). Never invent a citation.

---

## 1. Goal & non-goals

### Goal
Build a single, self-contained **terminology reference hub** that:

- categorises **every** ML / data-science term a scikit-plots user meets, from "what is a True Positive?" to "macro-averaged AUROC under class imbalance";
- serves **newbie → intermediate → senior** readers through progressive disclosure (collapsed detail, level tabs, a discovery grid);
- is **RST/Sphinx-native** (the project builds with Sphinx + PyData theme), not Markdown;
- ends with a **verified Sources** section — real, resolvable links only;
- exposes **sphinx_tags** so the page is discoverable via the existing tag pages and search.

### Non-goals (explicitly out of scope)
- **Not** a 1:1 copy of the 431 source posts. The source is *context and framing*; every definition is **re-expressed in our own words** to avoid copyright and licence issues (Section 3.6).
- **Not** 431 separate `.rst` files. One hub page with collapsible entries, optionally split later if it grows past a comfortable build/scroll size.
- **Not** a place to fabricate terms. If a term is not in the harvested list and not standard ML canon, it does not go in.

---

## 2. Verified current state (facts, fetched — not assumed)

| Item | Verified value |
|---|---|
| Source platform | WordPress.com site, "Your Gateway to Data Mastery" (author Ju Yeon Eum / Jenny) |
| Terminology category size | **431 posts** |
| Source pagination | `/category/00terminology/page/2/` … (~10 posts/page → ~44 pages) |
| Source post URL pattern | `https://insightful-data-lab.com/YYYY/MM/DD/<slug>/` |
| Sibling source categories | Bayesian Data Analysis (144); Data Analytics (216) — Foundations 27, Data-Driven Decisions 27, Data Preparation 25, Data cleaning & preparation 32, Analyze Data 30, Data visualization 27, Data Analysis Using Python 33, Job Search 15; Data Preparation and Analysis (56); Deep Learning (17); Introduction to Time Series (18) |
| Target path | `docs/source/learn/terminology/index.rst` |
| Parent `learn/index.rst` toctree | **cheatsheet, resources, glossary** — terminology is **absent** |
| Parent layout pattern | `grid-item-card` wrapping a `toctree` per child |
| Section underline convention | `#` part (overline) · `*` chapter (overline) · `=` section · `-` subsection · `^` subsubsection · `"` paragraph |
| Tag namespaces in use | `component:`, `domain:`, `level:`, `model-type:`, `model-workflow:`, `plot-type:`, `purpose:`, `topic:`, `internal:` |
| Tag values seen | domain: {cython, mlflow, neural network, statistics}; level: {beginner, intermediate}; model-type: {classification, clustering, regression}; purpose: {reference, showcase} |
| Tag directive rule | placed at **bottom** of page; form `subcategory: tag`; one or two words; reuse existing; avoid single-use |
| Draft size / coverage | 1717 lines; 8 domains; ~40 dropdown terms; Discovery tabs + Quick Reference table + Sources |

### Page-1 source titles (verified by fetch — the only titles confirmed so far)
`Subsampling` · `Class Weighting` · `SMOTE (Synthetic Minority Over-sampling Technique)` ·
`Oversampling` · `Low-pass Filtering` · `NearMiss (Distance-based Undersampling)` ·
`Cluster-based undersampling` · `Random Undersampling` · `Signal Processing` · `Time Series`

> Everything beyond these 10 titles must come from the harvest in Section 6 before it is treated as real. Titles from prior chat summaries are **leads, not facts** until re-fetched.

---

## 3. The Rules

These are the binding rules for every edit. They are derived from the project's own
conventions (fetched from the live docs) plus the constraints you set.

### 3.1 RST / Sphinx structural rules
- **File is `.rst`**, parsed by Sphinx. No Markdown syntax in the body (MyST is enabled for `.md` files, but this page is RST — keep it RST for consistency with siblings).
- **Underline hierarchy** exactly as the project uses it (table above). The hub page uses `=` (with overline) for the page title and `-` for each Domain section. Do **not** introduce new underline characters.
- **One H1-equivalent per page**: the `=`-overlined `Terminology` title. Domains are `-` subsections.
- **Every internal anchor** uses the `terminology-<slug>` convention, e.g. `.. _terminology-roc-auroc:`. Slugs are lowercase, hyphenated, stable (used by `:ref:` links and discovery cards).
- **Only use extensions that are enabled.** Confirmed available and safe to use here: `sphinx_design` (`tab-set`, `grid`, `grid-item-card`, `dropdown`, `badge`), `sphinx_tags`, `sphinx_togglebutton`, `sphinx_copybutton`, `sphinx.ext.doctest`, `matplotlib ... plot_directive` / `mathmpl`, `IPython` directives, `sphinx_prompt`. Math via `.. math::` (LaTeX) is fine.
- **No raw HTML for structure.** The draft's centred title uses a `:raw-html:` block — acceptable for the banner only. Everything else must be real directives so it renders in all output formats and stays accessible.

### 3.2 Per-term entry contract (the canonical shape of one term)
Every term is one `.. dropdown::` (collapsed by default, except the first in a domain which
may be `:open:`). Inside, answer these in this fixed order — this mirrors NumPyDoc ordering
(definition → math → usage → API → references):

1. **What is it?** — one plain-English paragraph, rewritten, no jargon dump.
2. **Formula / Key Relationship** — exact `.. math::` statement (omit only if the term is non-mathematical).
3. **When to use it** *(and when not to)* — the decision context.
4. **scikit-plots / scikit-learn connection** — the real API call that produces/consumes the concept, in a `.. code-block:: python`.
5. **See also** — `.. seealso::` with `:ref:` cross-links to related terms (this is what creates the "nested exploration" experience).

**Rules for the contract:**
- Sections 1, 3, 4 are **mandatory**. Section 2 is mandatory for any quantitative term. Section 5 is mandatory whenever ≥1 related term exists on the page.
- Code blocks must be **runnable shape** (correct imports, correct function names). If you are unsure a function exists, verify against the scikit-plots / scikit-learn API before writing it — do not guess signatures.
- Keep each entry **self-contained**: a reader who jumps straight to it via `Ctrl+F` should not need to have read earlier entries.

### 3.3 sphinx_design component patterns (copy-paste, do not improvise)

**Domain section + first term:**
```rst
.. _terminology-<domain-slug>:

Domain N — <Domain Title>
----------------------------------------------------------------------

<One- or two-sentence statement of what this domain covers and why it
matters for model evaluation in scikit-plots.>

.. dropdown:: <Term Name>
   :color: primary
   :icon: <octicon-name>
   :name: terminology-<term-slug>
   :open:

   **What is it?**

   <definition, our own words>

   **Formula / Key Relationship**

   .. math::

      <latex>

   **When to use it**

   <guidance, including when NOT to use it>

   **scikit-plots / scikit-learn connection**

   .. code-block:: python

      <runnable api call>

   .. seealso::

      :ref:`terminology-<related-slug>` · :ref:`terminology-<related-slug-2>`
```

**Subsequent term (collapsed):** identical, minus `:open:`.

**Discovery grid card** (used only in the "Discovery at a Glance" tabs near the top):
```rst
.. grid-item-card:: <emoji> <Short Label>
   :link: terminology-<target-slug>
   :link-type: ref
   :class-card: sd-border-1

   <One-line teaser that makes someone want to click.>
```

**Colour / icon convention** (keep consistent so colour carries meaning):

| Domain group | `:color:` | Example `:icon:` |
|---|---|---|
| Core classification metrics | `primary` | `table`, `check-circle`, `goal` |
| Curves & AUROC | `info` | `graph` |
| Averaging / multiclass | `success` | `stack` |
| Imbalance / sampling | `warning` | `git-branch` |
| Statistics / Bayesian | `secondary` | `beaker` |
| Fairness / responsible AI | `danger` | `law` |
| Signal / time series | `secondary` | `broadcast`, `clock` |

### 3.4 sphinx_tags rules (corrected to match project convention)
The project's documented rule: **place `.. tags::` at the bottom of the page**, tags in the
form `subcategory: tag`, one or two words, **reusing the existing controlled vocabulary** and
avoiding tags that would link to only one page.

- **Move the tag block to the bottom** (just above or just below `Sources`). The current draft places it near the top (≈ line 40) — that contradicts the convention. (sphinx_tags renders the badge wherever the directive sits; if you want a *visual* level indicator at the top, use `sd-badge` from sphinx_design instead of the tags directive.)
- **Use only existing namespaces/values.** For this page the correct set is:
  ```rst
  .. tags::
     purpose: reference,
     domain: statistics,
     model-type: classification,
     level: beginner,
     level: intermediate,
     level: advanced
  ```
- `purpose: reference` is the canonical tag for a reference page (the project's own tagging guide page is tagged `topic: tagging, purpose: reference`).
- Do **not** invent `domain: machine-learning`, `level: expert`, etc. If a genuinely new value is needed (e.g. `domain: fairness`), it must (a) be one/two words, (b) apply to more than one page, and (c) be added deliberately, not casually.

### 3.5 Sources rules (verified-only)
- The `Sources` section lists **only links that resolve and that were actually consulted.** Every URL must be checkable at build time.
- Allowed source types: official library docs (scikit-learn, imbalanced-learn, SciPy, scikit-plots), DOI-bearing papers/textbooks, and reputable course material (e.g. Google ML Crash Course). The insightful-data-lab category link is cited once as the *context/framing* source — not as a citation for specific factual claims.
- **No fabricated DOIs, no invented author/year, no "according to X" where X was never read.** If a claim needs a citation you do not have, either find a real one or soften the claim to standard textbook knowledge without a fake reference.
- Group sources (API docs / papers / learning resources) as the draft already does — that structure is good; keep it.

### 3.6 Content-integrity rules (no hallucination, no copying)
- **Re-express, don't reproduce.** The source posts are framing; write every definition fresh. Do not paste source sentences.
- **Standard canon is fine without a per-term citation** (e.g. the precision formula). Reserve citations for non-obvious claims, named methods (SMOTE, Hand & Till AUROC), and theorems (Nyquist–Shannon).
- **If you don't know, mark it.** Pending content gets an explicit placeholder (Section 7.3), never a confident guess.
- **API calls are claims too.** A wrong function name is a hallucination. Verify `skplt.metrics.*` / `sklearn.*` names before committing.

### 3.7 UX rules ("Discovery at a glance" + nested exploration)
- **Top of page = a map, not a wall of text.** Keep the three-level `tab-set` (Foundations / Metrics / Advanced) with discovery cards. This is the senior-friendly "glance" layer.
- **Detail is collapsed by default** (`dropdown`), so a newbie is never overwhelmed; they expand what they need.
- **Cross-links create depth.** Every `seealso` turns a flat glossary into a graph the reader can wander — this is the "explore even with no prior knowledge" goal.
- **A `Quick Reference` selector table** near the end lets seniors skip straight to "which metric for which situation". Keep/extend it.

---

## 4. The Master Taxonomy (the home for all 431 terms)

The current 8 domains cover the **model-evaluation core** well but cannot hold the full 431,
which span Bayesian methods, deep learning, feature engineering, EDA, etc. Extend to **15
domains**. Domains 1–8 already exist in the draft; 9–15 are **new buckets to add**.

> Legend: ✅ in draft · ➕ new bucket · terms marked **[v]** are verified-by-fetch, others are standard-canon placeholders to confirm during harvest.

| # | Domain | Status | Seed / verified terms |
|---|---|---|---|
| 1 | Confusion Matrix & Core Classification Metrics | ✅ | Confusion Matrix, TP/TN/FP/FN, Accuracy, Precision, Recall, Specificity, F1, F-beta |
| 2 | ROC, AUROC & Threshold Curves | ✅ | ROC Curve, AUROC, PR Curve / PR-AUC, Gini |
| 3 | Averaging & Multiclass/Multilabel Metrics | ✅ | Binary/Multiclass/Multilabel, Macro/Micro/Weighted avg, Macro/Micro AUROC, OvR/OvO |
| 4 | Class Imbalance & Resampling | ✅ | Class Imbalance, Class Weighting **[v]**, Oversampling **[v]**, SMOTE **[v]**, Undersampling, Random Undersampling **[v]**, NearMiss **[v]**, Cluster-based undersampling **[v]**, Subsampling **[v]** |
| 5 | Regression Metrics & Diagnostics | ➕ | MAE, MSE, RMSE, R²/Adjusted R², residual analysis, Q–Q plot (scikit-plots has `plot_residuals_distribution`) |
| 6 | Probability, Statistics & Hypothesis Testing | ✅ (expand) | Probability & Distributions, Bootstrap CIs, Mann–Whitney U; **add**: KS statistic (scikit-plots `plot_ks_statistic`), p-value, confidence vs. credible interval |
| 7 | Bayesian Methods | ➕ | Prior, Posterior, Likelihood, Bayes' theorem, MAP, conjugacy, MCMC (source has 144 Bayesian posts — large bucket) |
| 8 | Calibration & Probability Quality | ✅ | Calibration / Reliability diagram, Brier score (scikit-plots `plot_calibration_curve`) |
| 9 | Fairness, Bias & Responsible AI | ✅ (was D6) | Demographic Parity, Equal Opportunity, Equalized Odds, Predictive Parity |
| 10 | Signal Processing & Time Series | ✅ (was D8) | Time Series **[v]**, Signal Processing **[v]**, Low-pass Filtering **[v]**, Aliasing / Nyquist |
| 11 | Feature Engineering & Data Preparation | ➕ | Scaling/standardisation, encoding, imputation, missing-data handling (source "Data cleaning & preparation" = 32 posts) |
| 12 | Model Selection, Validation & Tuning | ➕ | Train/val/test split, cross-validation, AIC/AICc/BIC (scikit-plots GMM example), learning curve, hyperparameters |
| 13 | Clustering & Unsupervised Learning | ➕ | k-means, inertia/WSS, silhouette, elbow (scikit-plots `plot_elbow`, `plot_silhouette`), PCA / explained variance |
| 14 | Deep Learning & Neural Networks | ➕ | Neuron/layer, activation, loss, gradient descent, epoch/batch, over/underfitting (source "Deep Learning" = 17 posts) |
| 15 | Data Analytics & Visualization Foundations | ➕ | EDA, distributions, central tendency/spread, plot types (source "Foundations" = 27 posts) |

**Renumbering note:** the draft labels Fairness as "Domain 6" and Calibration as "Domain 7".
The table above re-orders so related domains sit together (Calibration next to Statistics;
Fairness later). Pick one numbering and apply it consistently — renumbering is a mechanical,
low-risk edit best done in its own commit.

---

## 5. Gap analysis

- **Covered now:** ~40 terms across 8 domains ≈ **9%** of the 431 source posts (many of the remaining 391 are not yet enumerated).
- **Verified titles in hand:** 10 (page 1) — all already represented in the draft's Domain 4 / Domain 8.
- **Structural gaps:** (a) page not in parent toctree; (b) tag block misplaced and missing `purpose: reference` + `model-type: classification`; (c) 7 domains (Regression, Bayesian, Feature Eng, Model Selection, Clustering, Deep Learning, EDA) have no bucket yet.
- **Biggest content debt:** Bayesian (source has 144 posts) and Data Analytics (216) are huge source areas with **zero** coverage in the draft. These will dominate the later runs.

---

## 6. Title harvest method (deterministic, reproducible)

The user wants **all 431 titles explicitly listed and categorised**. They cannot be invented;
they must be pulled. Use this exact procedure — it is verified against the live site structure.

### 6.1 Primary method — category pagination (confirmed to work)
The category is paginated. Iterate pages **1 → ~44**:

```
https://insightful-data-lab.com/category/00terminology/
https://insightful-data-lab.com/category/00terminology/page/2/
https://insightful-data-lab.com/category/00terminology/page/3/
...
https://insightful-data-lab.com/category/00terminology/page/44/
```

On each page, each post appears as a heading link:
```
## [<Title>](https://insightful-data-lab.com/YYYY/MM/DD/<slug>/)
```
Extract the **title text** and the **slug** (the slug is the future `terminology-<slug>`
anchor). Stop when "Older posts" disappears (last page).

**Per-session budget:** fetching 44 pages in one chat is heavy. Do it in **harvest batches of
~8–10 pages per session** (≈ 80–100 titles), appending to a running `terminology_titles.tsv`:

```
title<TAB>slug<TAB>source_url<TAB>proposed_domain
```

### 6.2 Secondary method — RSS feed (spot-check / freshness)
WordPress category feeds exist at `/category/00terminology/feed/`. Useful to confirm the most
recent posts and to detect new ones later, but the feed returns a limited window — **not** a
substitute for full pagination.

### 6.3 Optional — WordPress REST API (only if the host exposes it)
WordPress sites often expose `/wp-json/wp/v2/posts?per_page=100&categories=<id>`. If reachable,
it returns titles+slugs in JSON, 100 per page — far fewer requests. Requires the numeric
category id for `00terminology`. Treat as a nice-to-have; pagination (6.1) is the guaranteed path.

### 6.4 Output of the harvest
A single flat list of **431 rows** (title, slug, url, proposed domain). This list is the
backbone of every subsequent Run and the literal answer to "list all titles, categorised".

---

## 7. Multi-run roadmap (one batch per session)

Each Run is sized to fit comfortably in one session and to touch the **draft `index.rst`** at
**known insertion points** (line numbers from the uploaded 1717-line draft; re-confirm before
editing since earlier edits shift them).

### Run 0 — Structure & hygiene (fast, do first)
- **0a** Wire page into parent toctree (Section 8.1).
- **0b** Move `.. tags::` block from ≈ line 40 to page bottom; add `purpose: reference`, `model-type: classification`.
- **0c** Decide final domain numbering (Section 4 note) and apply.
- **0d** Add empty, labelled stubs for new Domains 5, 7, 11, 12, 13, 14, 15 (placeholder per Section 7.3) so the skeleton matches the Master Taxonomy.
- *Acceptance:* page builds clean, appears in the Learn sidebar, tag badges render on the tag index.

### Run 1 — Harvest batch A + Domain 4/10 reconciliation
- Harvest pages 1–10 (Section 6.1) → titles A.
- Confirm every title A is either already in the draft or appended to Domain 4 (Imbalance) / Domain 10 (Signal & TS).
- *Acceptance:* all batch-A titles accounted for in the taxonomy TSV with a domain.

### Run 2 — Harvest batches B–D (pages 11–44)
- Complete the title harvest; finalise the 431-row TSV with `proposed_domain` for each.
- *Acceptance:* TSV row count = 431; no row missing a domain.

### Runs 3–9 — Content fill, one domain per run
Order by value and by how much verifiable canon exists:
1. **Run 3** — Domain 5 Regression metrics (small, high-value, scikit-plots has the plots).
2. **Run 4** — Domain 12 Model Selection / Validation (CV, AIC/BIC, learning curve).
3. **Run 5** — Domain 13 Clustering (elbow/silhouette/PCA map to scikit-plots APIs).
4. **Run 6** — Domain 6 Statistics expansion (KS, p-value, CI vs credible interval).
5. **Run 7** — Domain 7 Bayesian (largest; may itself span 2–3 sessions).
6. **Run 8** — Domain 14 Deep Learning.
7. **Run 9** — Domain 11 Feature Eng + Domain 15 EDA foundations.

For each content run: take the harvested titles for that domain, write each as a per-term
dropdown (Section 3.2/3.3), add `seealso` cross-links, and append any new verified sources.

### Run 10 — Final pass
- Fill Discovery grid cards for every new domain.
- Extend the Quick Reference selector table (regression + clustering rows).
- Verify all `:ref:` targets resolve (no broken cross-links).
- Final Sources audit (every link resolves).

### 7.3 Placeholder convention (for not-yet-written content)
Use a visible, greppable stub so nothing reads as finished when it isn't:
```rst
.. _terminology-<slug>:

.. dropdown:: <Term Name>
   :color: secondary
   :icon: hourglass

   .. admonition:: Draft pending
      :class: caution

      Term scheduled for Run N. Source: <source_url>.
      Do not cite until written and reviewed.
```
Grep `Draft pending` to find everything outstanding.

---

## 8. Integration checklist

### 8.1 Wire the page into the parent toctree (REQUIRED — currently missing)
In `docs/source/learn/index.rst`, add a `grid-item-card` mirroring the existing ones
(cheatsheet / resources / glossary):

```rst
    .. grid-item-card::
        :padding: 3

        **terminology**
        ^^^
        .. toctree::
            :maxdepth: 2

            Terminology <terminology/index.rst>
```
Place it where it reads best (a sensible spot is right after **glossary**, since terminology
and glossary are siblings). Without this, the page builds but is **orphaned** and unreachable
from navigation.

### 8.2 Build & verify
- Build the docs locally (the project's standard Sphinx build).
- Confirm: page renders; dropdowns expand; math renders; copy buttons appear on code blocks; the three discovery tabs switch; no `WARNING: undefined label` for any `terminology-*` ref.
- Confirm the tag badges appear and that the page is listed under `purpose: reference`, `domain: statistics`, `model-type: classification`, and each `level:` page in `_tags/`.

### 8.3 Cross-link hygiene
- Every `:link: terminology-<x>` in a discovery card must have a matching `.. _terminology-<x>:` target. The draft already defines most; new domains must define theirs **before** a card points at them.

---

## 9. Verified sources (consulted for this guide)

All links below were fetched and resolved during preparation of this guide.

**Source corpus (context/framing only)**
- Terminology category (431 posts): https://insightful-data-lab.com/category/00terminology/

**scikit-plots project conventions (the basis for the Rules)**
- Documentation Tagging Guidelines: https://scikit-plots.github.io/dev/devel/guide_document_tag.html
- Tags overview (controlled vocabulary): https://scikit-plots.github.io/dev/_tags/tagsindex.html
- Learn landing page source (`learn/index.rst`): https://raw.githubusercontent.com/scikit-plots/scikit-plots/refs/heads/main/docs/source/learn/index.rst
- Learn section (live): https://scikit-plots.github.io/dev/learn/index.html

**Library references (for per-term API claims and citations in the page itself)**
- scikit-learn `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
- imbalanced-learn API: https://imbalanced-learn.org/stable/references/index.html
- SciPy `scipy.signal`: https://docs.scipy.org/doc/scipy/reference/signal.html

> Note: papers/DOIs already cited inside the draft (`Fawcett 2006`, `Hand & Till 2001`, `Chawla et al. 2002`, `Chouldechova 2017`, `Shannon 1949`, etc.) were not re-verified for this guide. Re-confirm each DOI resolves during Run 10's Sources audit before treating them as verified.

---

## 10. Quick rule card (pin this)

- **RST only**, project underline hierarchy, `terminology-<slug>` anchors.
- **One term = one `dropdown`**, in the fixed 5-part order; first per domain `:open:`.
- **Colour carries meaning** (Section 3.3 table); icons are octicons.
- **`.. tags::` at the bottom**, controlled vocabulary, `purpose: reference` always.
- **Sources = verified links only.** No invented DOIs or function names.
- **Re-express, never copy** the source posts.
- **Unknown ⇒ `Draft pending` stub**, never a confident guess.
- **Wire into the parent toctree** or the page is invisible.
- **Harvest before you categorise**; the 431 titles come from pagination, not memory.
