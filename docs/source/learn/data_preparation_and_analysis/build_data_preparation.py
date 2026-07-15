#!/usr/bin/env python3
# ======================================================================
# build_data_preparation.py  —  deterministic generator for
#                               learn/data_preparation_and_analysis
# ----------------------------------------------------------------------
# Reads dpa_inventory.tsv (title <TAB> url <TAB> stage), joins it with the
# STAGES + GLOSS tables below and the CONTENT / MINDMAP store in
# dpa_content.py, and emits an idempotent, ORDERED LEARNING PATH:
#   * index.rst        -> staged hub (8 stages) + hidden ordered toctree
#   * NN-<slug>.rst     -> one lesson page per post (NN = curriculum order),
#                          each with prev/next navigation, a re-expressed
#                          body, lateral "See also" links, and a source.
#
# Re-runnable: same inputs -> byte-identical output (ordered, no RNG).
# Old NN-*.rst pages are cleared first so the set never drifts; only the
# NN-*.rst glob is removed (toolkit files + index.rst are never touched).
#
# Fail-fast: the build aborts (nothing written) if any inventory title
# lacks a GLOSS entry (or vice-versa), any inventory stage is unknown, or
# any CONTENT / MINDMAP key (or MINDMAP neighbour) is not an exact
# inventory title.  No silent gaps, no dangling cross-links.
#
# To edit/extend: change dpa_content.py (bodies), append a row to
# dpa_inventory.tsv (+ a GLOSS line here), then re-run.
# See DATA_PREPARATION_AND_ANALYSIS.md.
# ======================================================================
from __future__ import annotations
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # make dpa_content importable regardless of CWD
INVENTORY = HERE / "dpa_inventory.tsv"
OUT = HERE / "index.rst"               # generated in place, inside the folder
PAGES_DIR = HERE                        # NN-*.rst lesson pages are siblings of this script
N_LESSONS = 56

# ---- stage registry: key -> (emoji, title, level, blurb) -------------
STAGES: dict[str, tuple[str, str, str, str]] = {
    "foundations": ("\U0001F4CB", "Foundations", "beginner",
                    "Why we analyse data, the CRISP-DM process, big data, and how numbers are stored."),
    "associations": ("\U0001F517", "Associations & Correlation", "beginner",
                     "Measuring relationships between variables: correlation, statistical tests and effect size."),
    "market_basket": ("\U0001F6D2", "Market Basket & Association Rules", "intermediate",
                      "Mining frequent itemsets and association rules for cross-selling with Apriori."),
    "segmentation": ("\U0001F9E9", "Sampling, Partitioning & Segmentation", "intermediate",
                     "Sampling and train / test partitioning, then grouping observations by clustering and RFM."),
    "regression": ("\U0001F4C8", "Regression", "intermediate",
                   "Linear and multiple regression, feature selection, and feature importance."),
    "classification": ("\U0001F3AF", "Classification & Logistic Regression", "advanced",
                       "Modelling binary outcomes with logistic regression, maximum likelihood, and model fit."),
    "trees": ("\U0001F333", "Decision Trees", "advanced",
              "CART decision trees as piecewise models, the interactions they capture, and profiling clusters."),
    "evaluation": ("\U0001F4CA", "Model Evaluation", "advanced",
                   "Judging models: prediction quality, classification metrics, ROC / AUC, lift, and residuals."),
}
STAGE_ORDER = ["foundations", "associations", "market_basket", "segmentation",
               "regression", "classification", "trees", "evaluation"]

# ---- one-line summary per lesson (used on the index cards) ------------
GLOSS: dict[str, str] = {
    # --- Foundations ---
    "Why Do We Analyze Data?":
        "The goals of data analysis \u2014 turning raw records into decisions, predictions, and understanding.",
    "The Process of Data Analysis":
        "The end-to-end workflow from question to insight, and where each technique fits.",
    "CRISP-DM for Data Science":
        "The Cross-Industry Standard Process: six phases from business understanding to deployment.",
    "Big Data: Definition, Characteristics, Evolution, and Business Impact":
        "The volume, velocity and variety of big data, and why it reshaped analytics.",
    "The First Step in Knowing Your Data":
        "Inspecting types, ranges and missingness before any modelling begins.",
    "IEEE 754 Floating-Point Standard":
        "How computers store real numbers \u2014 precision, rounding, and the pitfalls that follow.",
    # --- Associations & Correlation ---
    "Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022)":
        "Finding relationships in data, illustrated with the Chicago taxi-trips dataset.",
    "Taxi Trips \u2013 2022 dataset from the City of Chicago open data portal":
        "The open dataset used throughout \u2014 its fields, scale, and how to load it.",
    "Objective Selection of the Bin Width for a Time Histogram":
        "Choosing histogram bin width by a principled rule rather than guesswork.",
    "Measuring Associations in Data":
        "How to quantify whether and how strongly two variables move together.",
    "Measuring Associations Between Two Continuous Variables":
        "Covariance and correlation for two numeric variables, and what they capture.",
    "Correlation Coefficients in Python (Pearson, Spearman, Kendall)":
        "Three correlation measures \u2014 linear, rank, and concordance \u2014 computed in Python.",
    "Karl Pearson":
        "The statistician behind the correlation coefficient and much of modern statistics.",
    "Harald Cram\u00e9r":
        "The mathematician whose V measures association between categorical variables.",
    "What Are Statistical Tests?":
        "The logic of hypothesis testing: null hypotheses, p-values, and significance.",
    "Eta Squared (\u03b7\u00b2): Effect Size in ANOVA":
        "An effect-size measure for how much a grouping explains a numeric variable's variance.",
    # --- Market Basket & Association Rules ---
    "Understanding Market Baskets and Ideal Customers":
        "What transaction baskets reveal about customer behaviour and product affinity.",
    "What Can Association Rules Tell Us?":
        "Reading if-then rules over co-purchased items to guide merchandising.",
    "How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach":
        "Support, confidence and lift, and how Apriori tames the combinatorial search.",
    "Apriori: Frequent Itemsets via the Apriori Algorithm":
        "Finding frequent itemsets efficiently using the downward-closure property.",
    "association_rules: Generating Association Rules from Frequent Itemsets (mlxtend)":
        "Turning frequent itemsets into ranked rules with mlxtend in Python.",
    "Cross-Selling":
        "Using discovered rules to recommend complementary products.",
    # --- Sampling, Partitioning & Segmentation ---
    "Stratified Random Sampling":
        "Sampling that preserves subgroup proportions for representative data.",
    "Linear Congruential Random Number Generator (LCG)":
        "A classic pseudo-random generator \u2014 how reproducible randomness is produced.",
    "Partitioning Observations to Train Objective Models":
        "Splitting data into train and test sets to measure honest performance.",
    "Putting Similar Observations into Clusters":
        "The idea of grouping records by similarity \u2014 unsupervised segmentation.",
    "Clustering":
        "Algorithms like k-means that partition observations into homogeneous groups.",
    "Recency, Frequency, and Monetary Value (RFM)":
        "Three behavioural dimensions that summarise a customer's value.",
    "RFM Analysis":
        "Scoring and ranking customers on recency, frequency, and monetary value.",
    "Creating Segments of Observations for Business Reasons (RFM)":
        "Turning RFM scores into actionable customer segments.",
    # --- Regression ---
    "Least Squares Regression":
        "Fitting a line by minimising the sum of squared residuals.",
    "Multiple Linear Regression":
        "Predicting an outcome from several features with a linear model.",
    "Feature Importance in Linear Regression":
        "Judging which predictors matter most from coefficients and standardisation.",
    "Forward Selection: Definition and Core Idea":
        "Building a model by adding the most useful feature one at a time.",
    "Forward Selection and Model Interpretation in Linear Regression":
        "Applying forward selection to a regression and reading the result.",
    "Understanding Forward and Backward Stepwise Regression":
        "Adding and removing features by a criterion to search model space.",
    "How Shapley Values Work":
        "Fairly attributing a prediction to each feature using cooperative game theory.",
    # --- Classification & Logistic Regression ---
    "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds":
        "Modelling a yes/no probability through the log-odds linear form.",
    "Maximum Likelihood (MLE): Fitting a Distribution to Observed Data":
        "Choosing parameters that make the observed data most probable.",
    "Assessing Model Fit in Logistic Regression":
        "Predictive-power and pseudo-R\u00b2 measures for a fitted logistic model.",
    "Complete and Quasi-Complete Separation in Logistic Regression":
        "When perfect class separation breaks maximum-likelihood estimation.",
    "Forward Selection with Nested Models and Deviance Tests":
        "Comparing nested logistic models via the deviance likelihood-ratio test.",
    "Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention":
        "A worked case study fitting and reading a student-retention model.",
    # --- Decision Trees ---
    "Motivation of Decision Trees: An Incremental Model of Decision-Making":
        "Why splitting data by questions yields an interpretable predictor.",
    "The CART Algorithm":
        "Classification and Regression Trees \u2014 recursive binary splitting by impurity.",
    "Decision Trees as Piecewise Models and Their Predictive Structure":
        "Reading a tree as a piecewise-constant model over feature regions.",
    "How CART Decision Trees Model Interactions":
        "Why trees naturally capture feature interactions through nested splits.",
    "Cluster Profiling Using Decision Trees":
        "Describing clusters by training a tree to predict cluster membership.",
    "Using Decision Trees to Explain Clustering Results":
        "Turning opaque clusters into readable rules with a surrogate tree.",
    # --- Model Evaluation ---
    "Assessing the Quality of Prediction Models":
        "The general question of how good a model is, and by which yardstick.",
    "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics":
        "The confusion matrix and the metrics derived from it for two classes.",
    "Nominal Classification Models: Model State and Evaluation Metrics":
        "Extending evaluation to multi-class, unordered outcomes.",
    "Binary Classification Model Evaluation and Threshold Optimization":
        "Choosing a decision threshold to balance precision, recall, and cost.",
    "Identifying Outliers Using Residuals and Studentized Residuals":
        "Spotting influential and anomalous points from scaled residuals.",
    "AUC\u2013ROC Curve: Evaluating Classification Model Performance":
        "The ROC curve and the area under it as threshold-free performance measures.",
    "Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value":
        "Ranking prospects by model score to target a campaign efficiently.",
}

BAR = "=" * 70
SUB = "-" * 70


def slugify(text: str) -> str:
    """Lowercase, hyphenated, ASCII slug (stable; used for filenames + anchors)."""
    t = text.replace("\u2013", "-").replace("\u2014", "-")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-").lower()
    return t


def load_inventory() -> list[tuple[str, str, str]]:
    """Return [(title, url, stage), ...] in curriculum (file) order."""
    out: list[tuple[str, str, str]] = []
    with INVENTORY.open(encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)  # skip 'title  url  stage'
        for row in rd:
            if not row or not row[0].strip():
                continue
            title, url, stage = row[0].strip(), row[1].strip(), row[2].strip()
            out.append((title, url, stage))
    return out


def title_bar(title: str) -> str:
    """An overline/underline rule guaranteed >= the title length."""
    return "=" * max(72, len(title) + 2)


def main() -> int:
    inv = load_inventory()
    titles = [t for t, _u, _s in inv]
    title_set = set(titles)

    # ---- fail-fast: inventory <-> GLOSS <-> STAGES -------------------
    if len(inv) != N_LESSONS:
        print(f"ERROR: inventory has {len(inv)} lessons, expected {N_LESSONS}", file=sys.stderr)
        return 1
    miss_gloss = [t for t in titles if t not in GLOSS]
    extra_gloss = [t for t in GLOSS if t not in title_set]
    if miss_gloss or extra_gloss:
        if miss_gloss:
            print("ERROR: inventory titles with no GLOSS entry:", file=sys.stderr)
            for t in miss_gloss:
                print("   -", repr(t), file=sys.stderr)
        if extra_gloss:
            print("ERROR: GLOSS entries not in inventory:", file=sys.stderr)
            for t in extra_gloss:
                print("   -", repr(t), file=sys.stderr)
        return 1
    bad_stage = [(t, s) for t, _u, s in inv if s not in STAGES]
    if bad_stage:
        print("ERROR: inventory stages not in STAGES:", file=sys.stderr)
        for t, s in bad_stage:
            print(f"   - {s!r} ({t!r})", file=sys.stderr)
        return 1

    # ---- fail-fast: CONTENT / MINDMAP keys are exact inventory titles -
    try:
        from dpa_content import CONTENT  # type: ignore
    except Exception:
        CONTENT = {}
    try:
        from dpa_content import MINDMAP  # type: ignore
    except Exception:
        MINDMAP = {}
    bad_content = [k for k in CONTENT if k not in title_set]
    if bad_content:
        print("ERROR: CONTENT keys not in inventory:", file=sys.stderr)
        for k in bad_content:
            print("   -", repr(k), file=sys.stderr)
        return 1
    bad_mm = [k for k in MINDMAP if k not in title_set]
    bad_nb = sorted({n for k in MINDMAP for n in MINDMAP[k] if n not in title_set})
    if bad_mm or bad_nb:
        if bad_mm:
            print("ERROR: MINDMAP keys not in inventory:", file=sys.stderr)
            for k in bad_mm:
                print("   -", repr(k), file=sys.stderr)
        if bad_nb:
            print("ERROR: MINDMAP neighbours not in inventory:", file=sys.stderr)
            for n in bad_nb:
                print("   -", repr(n), file=sys.stderr)
        return 1

    # ---- derived maps -----------------------------------------------
    idx = {t: i + 1 for i, t in enumerate(titles)}          # 1-based lesson number
    slug = {t: slugify(t) for t in titles}
    docname = {t: f"{idx[t]:02d}-{slug[t]}" for t in titles}
    anchor = {t: f"dpa-{slug[t]}" for t in titles}

    # ---- clear stale lesson pages (idempotent set) ------------------
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for old in PAGES_DIR.glob("[0-9][0-9]-*.rst"):
        old.unlink()

    # ---- emit one page per lesson -----------------------------------
    n_rich = 0
    for pos, (title, url, stage) in enumerate(inv):
        i = pos + 1
        emoji, stage_title, level, _blurb = STAGES[stage]
        sn = STAGE_ORDER.index(stage) + 1
        L: list[str] = []
        a = L.append

        a(f".. _{anchor[title]}:")
        a("")
        bar = title_bar(title)
        a(bar)
        a(title)
        a(bar)
        a("")
        a(f"**Stage {sn} \u00b7 {emoji} {stage_title}**  \u00b7  "
          f"Lesson {i:02d} of {N_LESSONS}  \u00b7  *{level}*")
        a("")

        # prev / next navigation
        nav_parts: list[str] = []
        if i > 1:
            p = inv[pos - 1][0]
            nav_parts.append(f":doc:`\u25c0 Previous \u00b7 {p} <{docname[p]}>`")
        if i < N_LESSONS:
            nx = inv[pos + 1][0]
            nav_parts.append(f":doc:`Next \u00b7 {nx} \u25b6 <{docname[nx]}>`")
        nav_parts.append(":doc:`\u2191 Section <index>`")
        if nav_parts:
            a("   \u00b7   ".join(nav_parts))
            a("")

        a("")

        # body: full content if written, else a working stub
        if title in CONTENT:
            n_rich += 1
            body = CONTENT[title].strip("\n")
            a(body)
        else:
            a(GLOSS[title])
            a("")
            a(".. admonition:: Lesson in progress")
            a("   :class: note")
            a("")
            a("   The full write-up for this lesson is being prepared. Until then, use the")
            a("   one-line summary above and follow the navigation to adjacent lessons.")
        a("")

        # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#admonitions-messages-and-warnings
        # Note      → "Be aware of this clarification or detail."          # 📝 Neutral observations, assumptions, clarifications, conventions, or exceptions.
        # See also  → "Explore these related topics and resources."        # 📚 Internal/external references, further reading, related topics, prerequisites.
        # Hint      → "This may help you understand the concept."          # 💡 Intuition, conceptual connections, mind maps, learning aids.
        # Tip       → "This may help you work more effectively."           # 💡 Best practices, shortcuts, recommendations, efficient/advice workflows.
        # Info      → "Here's additional background or context."           # ℹ️ Background, implementation notes, **sources used by this page**, supplementary factual information where the information came from.
        # Important → "Do not overlook this; it's essential."              # ⭐ Critical/Essential concepts, requirements, or limitations.

        # lateral cross-links
        if MINDMAP.get(title):
            a(".. hint::")
            a("")
            links = "  \u00b7  ".join(f":doc:`{n} <{docname[n]}>`" for n in MINDMAP[title])
            a(f"   **Related lessons:** {links}")
            a("")

        # source (context/traceability)
        a(".. seealso::")
        a("")
        a(f"   **Source article** Adapted (context, re-expressed) in our own words from: `{url} <{url}>`__ "
          f"(insightful-data-lab.com).")
        a("")

        # tags
        a(f".. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: {level}")
        a("")

        (PAGES_DIR / f"{docname[title]}.rst").write_text("\n".join(L), encoding="utf-8")

    # ---- emit the staged learning-path index ------------------------
    by_stage: dict[str, list[str]] = {k: [] for k in STAGE_ORDER}
    for t, _u, s in inv:
        by_stage[s].append(t)

    I: list[str] = []
    w = I.append
    w(":html_theme.sidebar_secondary.remove:")
    w("")
    w(".. role:: raw-html(raw)")
    w("   :format: html")
    w("")
    w(".. |br| raw:: html")
    w("")
    w("   <br/>")
    w("")
    w(".. _data-preparation-and-analysis-index:")
    w("")
    w(":raw-html:`<div style=\"text-align:center\"><strong>` \U0001F4CA Data Preparation & Analysis")
    w("|br| Building, scoring and trusting predictive models")
    w("|br| |full_version| - |today|")
    w(":raw-html:`</strong></div>`")
    w("")
    w(BAR)
    w("Data Preparation & Analysis")
    w(BAR)
    w("")
    w("This course covers the **applied predictive-modelling workflow** end to end \u2014 framing a")
    w("prediction problem, exploring and preparing data, fitting models, and, most importantly for")
    w("**scikit-plots**, *evaluating* them with the right chart for the right question. It is the")
    w("practical companion to the :ref:`Terminology reference <terminology-index>`: terminology")
    w("defines the metrics, this course shows the workflow that produces and reads them \u2014 as an")
    w("ordered, self-contained sequence of 56 lessons across 8 stages.")
    w("")
    w("Read it at any depth:")
    w("")
    w("* **newcomers** \u2014 the intuition behind analysis, associations and model evaluation;")
    w("* **practitioners** \u2014 choosing between regression, trees, ROC, lift and threshold tuning;")
    w("* **reviewers** \u2014 diagnostics (residuals, outliers, separation) before shipping a model.")
    w("")
    w(".. warning::")
    w("")
    w("   Report performance on **held-out test data**, never the data a model was fit on. The")
    w("   sampling-and-partitioning stage exists precisely so the numbers you quote are honest.")
    w("")
    w(".. note::")
    w("")
    w("   Follow the lessons in order with **Next \u25b6**, or jump in by stage below. Snippets use")
    w("   real ``scikitplot`` / ``scikit-learn`` / ``mlxtend`` calls, and the evaluation charts")
    w("   (ROC, lift, calibration, residuals) are scikit-plots' specialty. This course pairs with")
    w("   the :ref:`Terminology reference <terminology-index>` (which defines every metric used).")
    w("")
    w(SUB.replace("-", "="))   # a visual transition rule (== line, blank-separated)
    w("")

    # ---- live filter: type-to-search across every term (progressive JS) ----
    # Static, dependency-free, deterministic. Without JS the page degrades
    # gracefully to plain collapsible dropdowns.
    # ---- v2 hub: live filter + collapsed stage dropdowns + A-Z --------
    # Same pattern/classes as learn/terminology (details.sd-dropdown, .term-az)
    n_items = len(titles)
    w(".. raw:: html")
    w("")
    w('   <div style="text-align:center;margin:0.4rem 0 0.4rem">')
    w('   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"')
    w(f'          placeholder="&#128269;&nbsp; Type to filter {n_items} lessons &mdash; by title or keyword&hellip;"')
    w('          style="width:100%;max-width:100%;padding:0.55rem 1rem;font-size:1rem;')
    w('                 border:1px solid var(--pst-color-border,#ccc);border-radius:0.55rem;box-sizing:border-box;')  # 1px solid rgba(128,128,128,0.45)
    w('                 background:transparent;color:inherit"/>')
    w('   <div id="term-filter-count" style="opacity:0.65;font-size:0.85rem;')
    w('        min-height:1.2em;margin-top:0.35rem"></div>')
    w("   </div>")
    w("   <script>")
    w("   document.addEventListener('DOMContentLoaded',function(){")
    w("     var inp=document.getElementById('term-filter');if(!inp){return;}")
    w("     var dds=[].slice.call(document.querySelectorAll('details.sd-dropdown'));")
    w("     var az=document.querySelector('details.term-az');")
    w("     var items=[];")
    w("     dds.forEach(function(d){[].slice.call(d.querySelectorAll('li')).forEach(")
    w("       function(li){items.push({li:li,d:d,t:li.textContent.toLowerCase()});});});")
    w("     var cnt=document.getElementById('term-filter-count');")
    w("     inp.addEventListener('input',function(){")
    w("       var q=inp.value.trim().toLowerCase();var n=0;")
    w("       dds.forEach(function(d){d.tHits=0;});")
    w("       items.forEach(function(it){")
    w("         var hit=!q||it.t.indexOf(q)!==-1;")
    w("         it.li.style.display=hit?'':'none';")
    w("         if(hit){it.d.tHits+=1;if(az&&it.d===az){n+=1;}}});")
    w("       dds.forEach(function(d){")
    w("         if(q){d.style.display=d.tHits?'':'none';d.open=d.tHits>0;}")
    w("         else{d.style.display='';d.open=false;}});")
    w(f"       if(cnt){{cnt.textContent=(q&&az)?(n+' of {n_items} match'+(n===1?'':'s')):'';}}")
    w("     });")
    w("   });")
    w("   </script>")
    w("")

    for stage in STAGE_ORDER:
        emoji, stage_title, level, blurb = STAGES[stage]
        sn = STAGE_ORDER.index(stage) + 1
        lessons = by_stage[stage]
        # stable per-stage anchor (linkable from other hubs)
        w(f".. _dpa-stage-{stage}:")
        w("")
        w(f".. dropdown:: {emoji} Stage {sn} \u2014 {stage_title}  \u00b7  {len(lessons)} lessons")
        w("   :animate: fade-in-slide-down")
        w("")
        w(f"   *{blurb}*  \u00b7  *{level}*")
        w("")
        for t in lessons:
            w(f"   * :doc:`{idx[t]:02d} \u00b7 {t} <{docname[t]}>` \u2014 {GLOSS[t]}")
        w("")

    # ---- dictionary view: one A-Z master list (auto-sorted) ----------
    az_head = "\U0001F524 Every lesson, A\u2013Z index"
    w(az_head)
    w("-" * (len(az_head) + 2))
    w("")
    w(".. dropdown:: \U0001F520 Open the full alphabetical index")
    w("   :class-container: term-az")
    w("")
    w("   .. hlist::")
    w("      :columns: 2")
    w("")
    for t in sorted(titles, key=str.casefold):
        w(f"      * :doc:`{t} <{docname[t]}>`")
    w("")

    # hidden ordered toctree so Sphinx builds the sequence + sidebar nav
    w(".. toctree::")
    w("   :hidden:")
    w("   :maxdepth: 1")
    w("")
    for t in titles:
        w(f"   {docname[t]}")
    w("")
    w(".. tags:: purpose: reference, topic: data analysis, topic: data preparation")
    w("")

    OUT.write_text("\n".join(I), encoding="utf-8")

    print(f"Wrote index.rst + {N_LESSONS} lesson pages "
          f"({n_rich} with full content, {N_LESSONS - n_rich} stub) "
          f"across {len(STAGE_ORDER)} stages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
