#!/usr/bin/env python3
# ======================================================================
# build_bayesian.py — deterministic generator for learn/bayesian_data_analysis
# ----------------------------------------------------------------------
# 144-lesson ordered course tracking Gelman et al., *Bayesian Data
# Analysis* (3rd ed.), grouped into 16 stages under 5 Parts. Same engine
# as the other learn/ hubs; emits the shared v2 browser index (live
# filter + collapsed stage dropdowns + A-Z) and one page per lesson with
# prev/next nav, body/stub, See-also links, source, tags.
#
# Idempotent (ordered, no RNG -> byte-identical). Fail-fast: aborts with
# exit 1 (writing nothing) on any inventory/GLOSS/STAGES mismatch or any
# CONTENT/MINDMAP key or neighbour that is not an exact inventory title.
# See BAYESIAN_DATA_ANALYSIS.md.
# ======================================================================
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
INVENTORY = HERE / "bda_inventory.tsv"
OUT = HERE / "index.rst"
PAGES_DIR = HERE
N_LESSONS = 144
ANCHOR_PREFIX = "bda"

# ---- Parts (top grouping, like terminology's level headers) ----------
PARTS = {
    1: "Fundamentals", 2: "Checking & Deciding", 3: "Computation",
    4: "Regression", 5: "Nonlinear & Nonparametric"}
PART_ORDER = [1, 2, 3, 4, 5]
PART_LEVEL = {
    1: "beginner", 2: "intermediate", 3: "intermediate",
    4: "advanced", 5: "advanced"}

# ---- stage registry: key -> (emoji, title, part, blurb) --------------

STAGES: dict[str, tuple[str, str, int, str]] = {
    "bayes_idea": ("\U0001F3B2", "The Bayesian Idea", 1,
                   "The three steps, notation, probability as uncertainty, and first worked examples."),
    "single_param": ("\U0001F4CD", "Single-Parameter Models & Priors", 1,
                     "Binomial and normal models; informative, noninformative and weakly-informative priors."),
    "multiparam": ("\U0001F9EE", "Multiparameter Models", 1,
                   "Nuisance parameters, the multinomial and multivariate normal, and the bioassay example."),
    "asymptotics": ("\U0001F4CF", "Asymptotics & Frequentist Ties", 1,
                    "Normal approximation, large-sample theory, and how Bayesian and frequentist inference relate."),
    "hierarchical": ("\U0001F3DB\uFE0F", "Hierarchical Models", 1,
                     "Exchangeability, the eight-schools model, and meta-analysis through partial pooling."),
    "checking": ("\U0001F50D", "Model Checking & Comparison", 2,
                 "Posterior predictive checks, predictive accuracy, Bayes factors, and model expansion."),
    "collection_decision": ("\U0001F5F3\uFE0F", "Data Collection & Decisions", 2,
                            "Ignorability, surveys and experiments, censoring, and Bayesian decision analysis."),
    "simulation": ("\U0001F9F0", "Simulation Basics", 3,
                   "Numerical integration, rejection and importance sampling, and how many draws are needed."),
    "mcmc": ("\u26D3\uFE0F", "MCMC: Gibbs, Metropolis & HMC", 3,
             "Samplers, convergence diagnostics, Hamiltonian Monte Carlo, and Stan."),
    "approximation": ("\U0001F39B\uFE0F", "Modal & Variational Approximation", 3,
                      "Posterior modes, EM, variational inference, and expectation propagation."),
    "regression": ("\U0001F4C8", "Regression Foundations", 4,
                   "Bayesian classical regression, causal inference, and regularization."),
    "hier_regression": ("\U0001F3D7\uFE0F", "Hierarchical Regression", 4,
                        "Batched coefficients, varying intercepts and slopes, and election forecasting."),
    "glm": ("\U0001F517", "Generalized Linear Models", 4,
            "GLM likelihoods, priors for logistic regression, and overdispersed Poisson models."),
    "robust_missing": ("\U0001F6E1\uFE0F", "Robustness & Missing Data", 4,
                       "Heavy-tailed errors, overdispersion, and multiple imputation."),
    "gp_basis": ("\U0001F30A", "Basis Functions & Gaussian Processes", 5,
                 "Splines, Gaussian-process regression, and functional data analysis."),
    "nonparametric": ("\u267E\uFE0F", "Mixtures & Nonparametric Bayes", 5,
                      "Mixtures, label switching, Bayesian histograms, and Dirichlet processes."),
}
STAGE_ORDER = ["bayes_idea", "single_param", "multiparam", "asymptotics", "hierarchical",
               "checking", "collection_decision", "simulation", "mcmc", "approximation",
               "regression", "hier_regression", "glm", "robust_missing",
               "gp_basis", "nonparametric"]

# ---- legacy anchors from the hand-written hub (preserve the contract) -
# Each resolves to the top of the generated index (kept so any external
# :ref: to the old hub still builds). Documented in the guide.
COMPAT_ANCHORS = [
    "bayes-discovery", "bayes-foundations", "bayes-computation", "bayes-advanced",
    "bayes-skplt-map", "bayes-sources", "bayes-theorem", "bayes-pieces",
    "bayes-credible", "bayes-conjugacy", "bayes-mcmc", "bayes-ppc",
    "bayes-hierarchical", "bayes-mixtures", "bayes-dp",
]

# ---- one-line summary per lesson (index cards + stubs) ----------------
GLOSS: dict[str, str] = {
    # --- Part I / 1 The Bayesian Idea ---
    "The three steps of Bayesian data analysis": "Set up a full probability model, condition on data, then evaluate the fit.",
    "General Notation for Statistical Inference": "The notation for parameters, data and distributions used throughout.",
    "Bayesian Inference": "Drawing conclusions as posterior probability statements about parameters.",
    "Discrete Bayesian Examples \u2013 Genetics and Spell Checking (with \u03b8)": "Bayes' rule on discrete unknowns, from carrier genetics to spell-checking.",
    "Probability as a Measure of Uncertainty": "Interpreting probability as a quantified degree of belief.",
    "Example \u2014 Probabilities from Football Point Spreads": "Reading calibrated probabilities off betting point spreads.",
    "Example \u2014 Calibration for Record Linkage": "Assigning match probabilities when merging records across files.",
    "Some Useful Results from Probability Theory": "Key identities \u2014 conditioning, expectation, transformation \u2014 the analysis relies on.",
    "Computation and Software": "The computing tools and workflow for fitting Bayesian models.",
    "Bayesian Inference in Applied Statistics": "How the Bayesian approach plays out across real applications.",
    # --- Part I / 2 Single-Parameter Models & Priors ---
    "Estimating a Probability from Binomial Data": "The Beta\u2013Binomial model: inferring a success probability.",
    "Posterior as a Compromise Between Data and Prior Information": "How the posterior blends prior belief with observed evidence.",
    "Summarizing Posterior Inference": "Point estimates, intervals and quantiles from a posterior.",
    "Informative Prior Distributions": "Priors that encode real prior knowledge about a parameter.",
    "Normal Distribution with Known Variance": "Conjugate inference for a normal mean when the variance is fixed.",
    "Other Standard Single-Parameter Models": "Poisson, exponential and related conjugate one-parameter models.",
    "Informative Prior Distribution for Cancer Rates": "A worked informative-prior analysis of county cancer rates.",
    "Noninformative Prior Distributions": "Priors meant to let the data dominate the inference.",
    "Weakly Informative Prior Distributions": "Priors that gently regularize without imposing strong belief.",
    # --- Part I / 3 Multiparameter Models ---
    "Averaging Over Nuisance Parameters": "Marginalizing out parameters you do not care about.",
    "Normal Data with a Noninformative Prior Distribution": "Joint inference for a normal mean and variance under a flat prior.",
    "Normal Data with a Conjugate Prior Distribution": "The normal\u2013inverse-gamma conjugate analysis for mean and variance.",
    "Multinomial Model for Categorical Data": "Dirichlet\u2013multinomial inference for category probabilities.",
    "Multivariate Normal Model with Known Variance": "Inferring a mean vector under a known covariance.",
    "Multivariate Normal with Unknown Mean and Variance": "Joint inference for mean and covariance via the inverse-Wishart.",
    "Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate)": "A nonconjugate logistic dose\u2013response fit on a grid and by sampling.",
    "Summary of Elementary Modeling and Computation": "Consolidating the single- and multi-parameter modeling toolkit.",
    # --- Part I / 4 Asymptotics & Frequentist Ties ---
    "Normal Approximations to the Posterior Distribution": "Approximating a posterior by a normal around its mode.",
    "Large-Sample Theory": "How posteriors concentrate and become normal as data grow.",
    "Counterexamples to large-sample (asymptotic) Bayesian theorems": "When the normal-approximation guarantees fail.",
    "Frequency Evaluations of Bayesian Inferences": "Judging Bayesian procedures by their long-run frequency behavior.",
    "Bayesian interpretations of other statistical methods": "Seeing penalized and classical methods as Bayesian in disguise.",
    # --- Part I / 5 Hierarchical Models ---
    "Constructing a Parameterized Prior Distribution": "Building a prior whose parameters are themselves estimated.",
    "Exchangeability and hierarchical models": "Exchangeability as the justification for hierarchical structure.",
    "Bayesian analysis of conjugate hierarchical models": "Fitting hierarchical models that admit conjugate updates.",
    "Normal model with exchangeable parameters": "The normal hierarchical model with a shared population distribution.",
    "Example: parallel experiments in eight schools": "The canonical partial-pooling example across eight studies.",
    "Hierarchical modeling applied to a meta-analysis": "Pooling effect estimates across studies hierarchically.",
    "Weakly Informative Priors for Variance Parameters": "Sensible priors for group-level variance in hierarchies.",
    # --- Part II / 6 Model Checking & Comparison ---
    "The Place of Model Checking in Applied Bayesian Statistics": "Why checking fit is integral, not optional, to modeling.",
    "Do the Inferences from the Model Make Sense?": "Sanity-checking posterior conclusions against knowledge.",
    "Posterior predictive checking": "Comparing data replicated from the model to the real data.",
    "Graphical posterior predictive checks": "Visual comparisons of replicated and observed data.",
    "Model checking for the educational testing example": "Applying predictive checks to the eight-schools fit.",
    "Measures of predictive accuracy": "Log predictive density, WAIC and cross-validation scores.",
    "Model comparison based on predictive performance": "Choosing models by out-of-sample predictive ability.",
    "Model comparison using Bayes factors": "Comparing models via marginal-likelihood ratios.",
    "Continuous model expansion": "Embedding a model in a richer family to test assumptions.",
    "Implicit assumptions and model expansion: an example": "Surfacing hidden assumptions by expanding a model.",
    # --- Part II / 7 Data Collection & Decisions ---
    "Bayesian inference requires a model for data collection": "Why the sampling/collection mechanism must be modeled.",
    "Data-collection models and ignorability": "When the data-collection process can be safely ignored.",
    "Sample surveys": "Bayesian inference for survey sampling designs.",
    "Designed experiments": "Modeling randomized experiments the Bayesian way.",
    "Sensitivity and the role of randomization": "How randomization protects inference from hidden bias.",
    "Observational studies": "Inference when treatment is not randomized.",
    "Censoring and truncation": "Modeling data that are cut off or partially observed.",
    "Bayesian decision theory in di\ufb00erent contexts": "Choosing actions to maximize expected utility.",
    "Using regression predictions: survey incentives": "A decision analysis of incentive choices via regression.",
    "Multistage decision making: medical screening": "Sequential decisions under uncertainty in screening.",
    "Hierarchical decision analysis for home radon": "A hierarchical decision model for radon remediation.",
    "Personal vs. institutional decision analysis": "Whose utilities and information a decision should use.",
    # --- Part III / 8 Simulation Basics ---
    "Numerical integration": "Quadrature and simulation for intractable integrals.",
    "Distributional approximations": "Approximating posteriors with tractable distributions.",
    "Direct simulation and rejection sampling": "Drawing exact samples by proposal and acceptance.",
    "Importance sampling": "Reweighting draws from a proposal to target the posterior.",
    "How many simulation draws are needed?": "Sizing Monte Carlo samples for a target accuracy.",
    "Computing environments": "The software environments for Bayesian computation.",
    "Debugging Bayesian computing": "Diagnosing and fixing errors in model fitting.",
    # --- Part III / 9 MCMC ---
    "Gibbs sampler": "Sampling each parameter in turn from its full conditional.",
    "Metropolis and Metropolis-Hastings algorithms": "Accept\u2013reject proposals that target the posterior.",
    "Using Gibbs and Metropolis as building blocks": "Composing samplers for complex models.",
    "Inference and assessing convergence": "R-hat, effective sample size and mixing diagnostics.",
    "E\ufb00ective number of simulation draws": "Effective sample size after autocorrelation.",
    "Example: hierarchical normal model": "A worked Gibbs sampler for the normal hierarchy.",
    "E\ufb03cient Gibbs samplers": "Reparameterizations that speed up Gibbs sampling.",
    "E\ufb03cient Metropolis jumping rules": "Tuning proposal scale for efficient Metropolis moves.",
    "Further extensions to Gibbs and Metropolis": "Slice, reversible-jump and auxiliary-variable extensions.",
    "Hamiltonian Monte Carlo": "Gradient-guided proposals for efficient exploration.",
    "Hamiltonian Monte Carlo for a hierarchical model": "Applying HMC to a hierarchical example.",
    "Stan: developing a computing environment": "The Stan language and NUTS sampler for Bayesian models.",
    # --- Part III / 10 Modal & Variational Approximation ---
    "Finding posterior modes": "Optimization to locate the posterior's peak.",
    "Boundary-avoiding priors for modal summaries": "Priors that keep modal estimates off the boundary.",
    "Normal and related mixture approximations": "Approximating posteriors with normal mixtures.",
    "Finding marginal posterior modes using EM": "The EM algorithm for marginal mode-finding.",
    "Conditional and marginal posterior approximations": "Approximating conditionals and marginals in stages.",
    "Example: hierarchical normal model (continued)": "The hierarchical normal model via modal approximation.",
    "Variational inference": "Fitting the closest tractable distribution to the posterior.",
    "Expectation propagation": "Iteratively matching moments for posterior approximation.",
    "Other approximations": "Laplace, INLA and further approximate-inference methods.",
    "Unknown normalizing factors": "Inference when the likelihood's constant is intractable.",
    # --- Part IV / 11 Regression Foundations ---
    "Conditional modeling": "Modeling an outcome conditional on predictors.",
    "Bayesian analysis of classical regression": "Posterior inference for the normal linear model.",
    "Regression for causal inference: incumbency and voting": "Using regression to estimate an incumbency effect.",
    "Goals of regression analysis": "Prediction, explanation and causal aims of regression.",
    "Assembling the matrix of explanatory variables": "Coding, transforming and interacting predictors.",
    "Regularization and dimension reduction": "Shrinkage priors and reducing predictor dimension.",
    "Unequal variances and correlations": "Modeling heteroscedastic and correlated errors.",
    "Including numerical prior information": "Adding quantitative prior knowledge to regression.",
    # --- Part IV / 12 Hierarchical Regression ---
    "Regression coe\ufb03cients exchangeable in batches": "Grouping coefficients into exchangeable batches.",
    "Example: forecasting U.S. presidential elections": "A hierarchical election-forecasting regression.",
    "Interpreting a normal prior distribution as extra data": "Reading a coefficient prior as pseudo-observations.",
    "Varying intercepts and slopes": "Letting regression coefficients vary across groups.",
    "Computation: batching and transformation": "Efficient computation for hierarchical regressions.",
    "Analysis of variance and the batching of coe\ufb03cients": "ANOVA as hierarchical batching of effects.",
    "Hierarchical models for batches of variance components": "Priors over multiple variance components.",
    # --- Part IV / 13 Generalized Linear Models ---
    "Standard generalized linear model likelihoods": "Binomial, Poisson and other GLM data models.",
    "Working with generalized linear models": "Fitting, checking and interpreting Bayesian GLMs.",
    "Weakly informative priors for logistic regression": "Regularizing logistic coefficients to avoid separation.",
    "Overdispersed Poisson regression for police stops": "A hierarchical Poisson model of stop-and-frisk data.",
    "State-level opinons from national polls": "Multilevel regression and poststratification of opinion.",
    "Models for multivariate and multinomial responses": "GLMs for vector and categorical outcomes.",
    "Loglinear models for multivariate discrete data": "Loglinear structure for contingency tables.",
    # --- Part IV / 14 Robustness & Missing Data ---
    "Aspects of robustness": "Making inference resilient to model misspecification.",
    "Overdispersed versions of standard models": "Heavy-tailed and overdispersed extensions.",
    "Posterior inference and computation": "Computing posteriors for robust models.",
    "Robust inference for the eight schools": "A t-based robust reanalysis of eight schools.",
    "Robust regression using t-distributed errors": "Down-weighting outliers with t errors.",
    "Notation": "Notation for the missing-data framework.",
    "Multiple imputation": "Filling in missing values with several posterior draws.",
    "Missing data in the multivariate normal and t models": "Imputation under multivariate normal and t models.",
    "Example: multiple imputation for a series of polls": "Imputing missing responses across repeated polls.",
    "Missing values with counted data": "Handling missingness in count data.",
    "Example: an opinion poll in Slovenia": "A worked missing-data analysis of a plebiscite poll.",
    # --- Part V / 15 Basis Functions & Gaussian Processes ---
    "Example: serial dilution assay": "A nonlinear Bayesian calibration for dilution assays.",
    "Example: population toxicokinetics": "A hierarchical nonlinear pharmacokinetic model.",
    "Splines and weighted sums of basis functions": "Flexible curves built from weighted basis functions.",
    "Basis selection and shrinkage of coe\ufb03cients": "Shrinking basis coefficients to control smoothness.",
    "Non-normal models and regression surfaces": "Nonlinear and non-normal regression surfaces.",
    "Gaussian process regression": "Priors over functions for flexible regression.",
    "Example: birthdays and birthdates": "A Gaussian-process decomposition of a birth time series.",
    "Latent Gaussian process models": "GPs as latent components inside larger models.",
    "Functional data analysis": "Modeling whole curves as data objects.",
    # --- Part V / 16 Mixtures & Nonparametric Bayes ---
    "Density estimation and regression": "Estimating whole distributions, not just parameters.",
    "Setting up and interpreting mixture models": "Modeling populations as blends of components.",
    "Example: reaction times and schizophrenia": "A mixture model separating response-time subgroups.",
    "Label switching and posterior computation": "Handling component permutation in mixture posteriors.",
    "Unspecified number of mixture components": "Letting the number of components be unknown.",
    "Mixture models for classification and regression": "Mixtures as flexible supervised models.",
    "Bayesian histograms": "Nonparametric density via random histograms.",
    "Dirichlet process prior distributions": "Priors over distributions with unbounded support.",
    "Dirichlet process mixtures": "Infinite mixtures whose clusters grow with the data.",
    "Beyond density estimation": "Nonparametric priors for broader modeling tasks.",
    "Hierarchical dependence": "Dependent nonparametric priors across groups.",
    "Density regression": "Letting an entire density vary with predictors.",
}

BAR = "=" * 70


def slugify(text: str) -> str:
    t = text.replace("\u2013", "-").replace("\u2014", "-")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-").lower()


def load_inventory() -> list[tuple[str, str, str]]:
    out = []
    with INVENTORY.open(encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd)
        for row in rd:
            if row and row[0].strip():
                out.append((row[0].strip(), row[1].strip(), row[2].strip()))
    return out


def title_bar(title: str) -> str:
    return "=" * max(72, len(title) + 2)


def main() -> int:
    inv = load_inventory()
    titles = [t for t, _u, _s in inv]
    tset = set(titles)

    if len(inv) != N_LESSONS:
        print(f"ERROR: inventory has {len(inv)} lessons, expected {N_LESSONS}", file=sys.stderr); return 1
    miss = [t for t in titles if t not in GLOSS]
    extra = [t for t in GLOSS if t not in tset]
    if miss or extra:
        for t in miss:
            print("ERROR: no GLOSS for", repr(t), file=sys.stderr)
        for t in extra:
            print("ERROR: GLOSS not in inventory", repr(t), file=sys.stderr)
        return 1
    bad_stage = [(t, s) for t, _u, s in inv if s not in STAGES]
    if bad_stage:
        for t, s in bad_stage:
            print(f"ERROR: stage {s!r} ({t!r})", file=sys.stderr)
        return 1
    if len(tset) != len(titles):
        from collections import Counter
        for t, c in Counter(titles).items():
            if c > 1:
                print("ERROR: duplicate title", repr(t), file=sys.stderr)
        return 1

    try:
        from bda_content import CONTENT  # type: ignore
    except Exception:
        CONTENT = {}
    try:
        from bda_content import MINDMAP  # type: ignore
    except Exception:
        MINDMAP = {}
    bc = [k for k in CONTENT if k not in tset]
    bm = [k for k in MINDMAP if k not in tset]
    bn = sorted({n for k in MINDMAP for n in MINDMAP[k] if n not in tset})
    if bc or bm or bn:
        for k in bc:
            print("ERROR: CONTENT key not in inventory", repr(k), file=sys.stderr)
        for k in bm:
            print("ERROR: MINDMAP key not in inventory", repr(k), file=sys.stderr)
        for n in bn:
            print("ERROR: MINDMAP neighbour not in inventory", repr(n), file=sys.stderr)
        return 1

    idx = {t: i + 1 for i, t in enumerate(titles)}
    slug = {t: slugify(t) for t in titles}
    docname = {t: f"{idx[t]:03d}-{slug[t]}" for t in titles}
    anchor = {t: f"{ANCHOR_PREFIX}-{slug[t]}" for t in titles}
    by_stage: dict[str, list[str]] = {k: [] for k in STAGE_ORDER}
    for t, _u, s in inv:
        by_stage[s].append(t)

    for old in PAGES_DIR.glob("[0-9][0-9][0-9]-*.rst"):
        old.unlink()

    n_rich = 0
    for pos, (title, url, stage) in enumerate(inv):
        i = pos + 1
        emoji, st_title, part, blurb = STAGES[stage]
        sn = STAGE_ORDER.index(stage) + 1
        level = PART_LEVEL[part]
        L: list[str] = []; a = L.append
        a(f".. _{anchor[title]}:"); a("")
        b = title_bar(title); a(b); a(title); a(b); a("")
        a(f"**Part {part} \u00b7 Stage {sn} \u00b7 {emoji} {st_title}**  \u00b7  "
          f"Lesson {i:03d} of {N_LESSONS}  \u00b7  *{level}*"); a("")
        nav = []
        if i > 1:
            p = inv[pos - 1][0]; nav.append(f":doc:`\u25c0 Previous \u00b7 {p} <{docname[p]}>`")
        if i < N_LESSONS:
            nx = inv[pos + 1][0];
            nav.append(f":doc:`Next \u00b7 {nx} \u25b6 <{docname[nx]}>`")
        nav.append(":doc:`\u2191 Section <index>`")
        if nav:
            a("   \u00b7   ".join(nav)); a("")
        a("")
        if title in CONTENT:
            n_rich += 1; a(CONTENT[title].strip("\n"))
        else:
            a(GLOSS[title]); a("")
            a(".. admonition:: Lesson in progress"); a("   :class: note"); a("")
            a("   The full write-up is being prepared. Until then, use the summary above")
            a("   and the navigation to adjacent lessons.")
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
            a(".. hint::"); a("")
            a("   **Related lessons:** " + "  \u00b7  ".join(
                f":doc:`{n} <{docname[n]}>`" for n in MINDMAP[title])); a("")

        # source (context/traceability)
        a(".. seealso::")
        a("")
        a(f"   **Source article** Adapted (context, re-expressed) in our own words from: `{url} <{url}>`__ "
          f"(insightful-data-lab.com).")
        a("")

        # tags
        a(f".. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: {level}"); a("")
        (PAGES_DIR / f"{docname[title]}.rst").write_text("\n".join(L), encoding="utf-8")

    # ---- v2 browser index ------------------------------------------
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
    w(".. _bayesian-data-analysis-index:")
    w("")
    for lbl in COMPAT_ANCHORS:   # legacy anchors preserved (resolve to page top)
        w(f".. _{lbl}:")
    w("")
    w(":raw-html:`<div style=\"text-align:center\"><strong>` \U0001F3B2 Bayesian Data Analysis")
    w("|br| Reasoning about uncertainty with priors, likelihoods and posteriors")
    w("|br| |full_version| - |today|")
    w(":raw-html:`</strong></div>`"); w("")
    w(BAR); w("Bayesian Data Analysis"); w(BAR); w("")
    w("Bayesian analysis treats unknown quantities as **probability distributions** and updates them")
    w("with data: instead of a single best estimate you get a full **posterior**. This course follows")
    w("the arc of Gelman et al., *Bayesian Data Analysis* \u2014 from first principles through computation")
    w("and regression up to the **nonparametric** models (mixtures, Dirichlet processes) that close it")
    w(f"\u2014 as an ordered sequence of {len(titles)} lessons across 16 stages in 5 parts.")
    w("")
    w("Read it at any depth:")
    w("")
    w("* **newcomers** \u2014 the intuition of prior \u2192 likelihood \u2192 posterior;")
    w("* **practitioners** \u2014 how to compute, check and compare posteriors;")
    w("* **researchers** \u2014 hierarchical, GP and nonparametric (infinite-mixture) models.")
    w("")
    w(".. note::")
    w("")
    w("   **Type in the filter box** for instant lookup by title or keyword, expand a stage to browse,")
    w("   or open the A\u2013Z index at the bottom. Code snippets use real ``scipy.stats`` / ``PyMC`` /")
    w("   ``ArviZ`` / ``scikit-learn`` calls. This course pairs with the")
    w("   :ref:`Terminology reference <terminology-index>` (probability and distributions).")
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

    for part in PART_ORDER:
        ph = f"{PARTS[part]}"  # Part {part} \u2014 {PARTS[part]} Part 1 — Fundamentals
        w(ph)
        w("-" * max(72, len(ph) + 2))
        w("")
        for stage in [s for s in STAGE_ORDER if STAGES[s][2] == part]:
            emoji, st_title, _p, blurb = STAGES[stage]
            sn = STAGE_ORDER.index(stage) + 1
            lessons = by_stage[stage]
            w(f".. _{ANCHOR_PREFIX}-stage-{stage}:"); w("")
            w(f".. dropdown:: {emoji} Stage {sn} \u2014 {st_title}  \u00b7  {len(lessons)} lessons")
            w("   :animate: fade-in-slide-down"); w("")
            w(f"   *{blurb}*"); w("")
            for t in lessons:
                w(f"   * :doc:`{idx[t]:03d} \u00b7 {t} <{docname[t]}>` \u2014 {GLOSS[t]}")
            w("")

    az = "\U0001F524 Every lesson, A\u2013Z"
    w(az); w("-" * (len(az) + 2)); w("")
    w(".. dropdown:: Open the full alphabetical index")
    w("   :class-container: term-az"); w("")
    w("   .. hlist::"); w("      :columns: 2"); w("")
    for t in sorted(titles, key=str.casefold):
        w(f"      * :doc:`{t} <{docname[t]}>`")
    w("")

    # salvaged scikit-plots map + sources (from the hand-written hub)
    w("\U0001F5FA\uFE0F scikit-plots & the Bayesian stack")
    w("-" * 40); w("")
    w(".. dropdown:: Where scikit-plots and the PPL stack fit")
    w(""); w("   scikit-plots' role here is diagnostic and model-selection visual support; the heavy")
    w("   lifting is done by the probabilistic-programming stack.")
    w(""); w("   * **Gaussian Mixture Models (AIC / BIC)** \u2014 choose the number of components:")
    w("     https://scikit-plots.github.io/dev/auto_examples/stats/plot_gaussian_mixture_models.html")
    w("   * **Residuals distribution** \u2014 distributional / Q\u2013Q model checks:")
    w("     https://scikit-plots.github.io/dev/auto_examples/stats/plot_residuals_distribution_script.html")
    w("   * **PyMC** \u2014 probabilistic programming: https://www.pymc.io/")
    w("   * **ArviZ** \u2014 diagnostics and plots for Bayesian inference: https://python.arviz.org/")
    w("")
    w(".. dropdown:: Sources & standard reference")
    w(""); w("   Framing re-expressed in our own words; API calls verified against official docs.")
    w(""); w("   * Source context (144 posts): https://insightful-data-lab.com/category/bayesian-data-analysis/")
    w("   * SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html")
    w("   * scikit-learn mixtures: https://scikit-learn.org/stable/modules/mixture.html")
    w("   * Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, *Bayesian Data Analysis* (3rd ed.):")
    w("     http://www.stat.columbia.edu/~gelman/book/")
    w("   * Terminology reference: :ref:`terminology-index`")
    w("")

    w(".. toctree::"); w("   :hidden:"); w("   :maxdepth: 1"); w("")
    for t in titles:
        w(f"   {docname[t]}")
    w("")
    w(".. tags:: purpose: reference, topic: data analysis, domain: bayesian"); w("")

    OUT.write_text("\n".join(I), encoding="utf-8")
    print(f"Wrote index.rst + {N_LESSONS} lesson pages "
          f"({n_rich} full, {N_LESSONS - n_rich} stub) across {len(STAGE_ORDER)} stages / {len(PART_ORDER)} parts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
