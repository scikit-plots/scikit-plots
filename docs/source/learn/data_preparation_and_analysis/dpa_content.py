"""
CONTENT + MINDMAP for learn/data_preparation_and_analysis lessons.

Keys must be EXACT titles from dpa_inventory.tsv (see DATA_PREPARATION_AND_ANALYSIS.md).
CONTENT[title] = raw RST body; MINDMAP[title] = lateral 'See also' titles.
Populated batch by batch in curriculum order.
"""

CONTENT: dict[str, str] = {}
MINDMAP: dict[str, list[str]] = {}


# ----------------------------------------------------------------------
# Stage 1 — Foundations
# ----------------------------------------------------------------------

CONTENT["Why Do We Analyze Data?"] = r"""
From data to decisions
------------------------

Data on its own is inert — rows and numbers. **Analysis** is what turns it into something useful:
patterns you can see, questions you can answer, and, ultimately, **decisions** you can act on. Every
technique in this course exists to move data one step further along that chain, from raw records
toward a choice someone has to make.

Four kinds of question
------------------------

It helps to name **what** you are asking. Analysts group questions into four types, each harder and
more valuable than the last:

* **Descriptive** — *what happened?* (how have sales trended this year);
* **Diagnostic** — *why did it happen?* (why did churn rise last quarter);
* **Predictive** — *what is likely to happen?* (which customers will leave next);
* **Prescriptive** — *what should we do?* (which customer to call to keep them).

Choosing the right one
------------------------

The right type is set by the **business problem**, not by the fanciest tool available. A dashboard
answering "what happened" may be all a question needs; a churn-prevention campaign needs a
**predictive** model and, ideally, a **prescriptive** recommendation on top. Much of this course is
about matching the question to a method — and then, in the final stage, **checking the answer is
trustworthy** before anyone relies on it.
"""

CONTENT["The Process of Data Analysis"] = r"""
It starts with a question
---------------------------

Data analysis is a **process**, not a single act — and it always begins with a **question**, not with
the data. A clear question ("which customers are about to leave?") decides what data you need, which
method fits, and how you will know the answer is good. Skip it, and you risk an elegant analysis of
the wrong thing.

The steps
-----------

The workflow moves through recognisable stages:

* **frame** the question and its success criterion;
* **collect** the relevant data;
* **clean and prepare** it into a usable form;
* **explore** it to build intuition and spot problems;
* **model or analyse** to answer the question;
* **evaluate** whether the answer actually holds;
* **communicate or deploy** the result so it drives a decision.

This course walks these stages roughly in order — foundations and understanding first, preparation
and modelling in the middle, evaluation at the end.

It loops
----------

The stages are **not** a one-way street. Exploring the data often sends you back to collect more;
modelling reveals a preparation step you missed; a failed evaluation returns you to the drawing
board. Good analysis **iterates** — circling back as each stage teaches you something the last one
could not. The next lesson gives this loop a formal name and shape: **CRISP-DM**.
"""

CONTENT["CRISP-DM for Data Science"] = r"""
A shared blueprint
--------------------

**CRISP-DM** — the **Cross-Industry Standard Process for Data Mining** — is the most widely used
framework for structuring a data project. Introduced in the late 1990s and deliberately
**industry-independent**, it turns the loose workflow of the last lesson into six named phases that
any team can share as a common plan.

The six phases
----------------

The phases run:

1. **Business Understanding** — pin down the objective and turn it into a data-mining goal;
2. **Data Understanding** — collect the data, describe it, explore it, and check its quality;
3. **Data Preparation** — select, clean, construct, integrate and format the modelling dataset;
4. **Modeling** — choose techniques, build models, and tune them;
5. **Evaluation** — judge the results **against the business objective**, not just the metrics;
6. **Deployment** — put the model to work, with monitoring and a plan to maintain it.

Where the time goes
---------------------

One phase dominates the calendar: **Data Preparation** is generally the **most time-consuming** part
of the whole project — routinely cited as the bulk of the effort. It is also, not coincidentally, the
subject this course is named for. Getting the data right is most of the work, and most of the payoff.

A cycle, not a line
---------------------

CRISP-DM is drawn as a **cycle**, not a checklist. Arrows link the phases both ways — modelling sends
you back to preparation, evaluation back to business understanding — and an outer loop returns the
finished project to the start as new questions emerge. The framework's real lesson is that data work
is **iterative**: you revisit earlier phases as later ones teach you what you missed.
"""


MINDMAP.update({
    "Why Do We Analyze Data?": [
        "The Process of Data Analysis", "CRISP-DM for Data Science",
        "Assessing the Quality of Prediction Models", "The First Step in Knowing Your Data",
    ],
    "The Process of Data Analysis": [
        "Why Do We Analyze Data?", "CRISP-DM for Data Science",
        "The First Step in Knowing Your Data", "Measuring Associations in Data",
    ],
    "CRISP-DM for Data Science": [
        "The Process of Data Analysis", "Why Do We Analyze Data?",
        "The First Step in Knowing Your Data",
        "Partitioning Observations to Train Objective Models",
    ],
})


# ----------------------------------------------------------------------
# Stage 1 — Foundations (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Big Data: Definition, Characteristics, Evolution, and Business Impact"] = r"""
What makes data 'big'
-----------------------

"Big data" is not just "a lot of data" — it is data whose **scale, speed, or messiness** overwhelms
the traditional tools built for tidy tables. The standard way to pin the idea down is a short list of
characteristics, each beginning with **V**.

The five Vs
-------------

The canonical **five Vs**:

* **Volume** — the sheer **quantity**, now measured in terabytes, petabytes and beyond;
* **Velocity** — the **speed** at which data arrives, often as real-time streams;
* **Variety** — the mix of **types**: structured tables, semi-structured logs, and unstructured text, images and audio;
* **Veracity** — the **trustworthiness**, since large, fast, varied data is often noisy, biased or incomplete;
* **Value** — the **usefulness**, the reminder that data is only worth collecting if it can become insight.

How it grew
-------------

The list **grew over time**. The term surfaced in the late 1990s, but the analyst **Doug Laney** fixed
the first three in a 2001 note on data's growing **Volume, Velocity and Variety** — the "**3 Vs**". As
data quality and business worth became pressing, the industry added **Veracity** and **Value**, giving
today's **5 Vs**. (Some lists go further still.)

Why it matters
----------------

The practical impact is that big data broke the **old toolchain**. Fitting a spreadsheet or a single
database no longer suffices; distributed storage and processing, streaming systems and new modelling
methods appear precisely because volume, velocity and variety demand them. For analysts, the lesson is
humility about **veracity**: more data is not automatically better data, and the preparation stage
ahead exists to earn that trust.
"""

CONTENT["The First Step in Knowing Your Data"] = r"""
Look before you model
-----------------------

Before fitting anything, **look at the data**. This is CRISP-DM's **Data Understanding** phase, and
skipping it is how projects quietly go wrong — a mis-typed column, a hidden pile of missing values, or
an outlier that wrecks a model. The first step is always **profiling** what you actually have.

What to check
---------------

A good first pass answers a handful of questions:

* **Shape** — how many rows and columns?
* **Types** — which columns are numeric, categorical, or dates?
* **Ranges and distributions** — the min, max, centre and spread of each numeric field;
* **Missingness** — how many values are absent, and where?
* **Cardinality** — how many distinct values do categorical fields take?
* **Duplicates and oddities** — repeated rows, impossible values, surprising codes.

A first pass in pandas
------------------------

In practice a few ``pandas`` calls cover most of it:

.. code-block:: python

   df.shape            # (rows, columns)
   df.info()           # column names, dtypes, non-null counts
   df.describe()       # count, mean, std, min, quartiles, max (numeric)
   df.isnull().sum()   # missing values per column
   df.nunique()        # distinct values per column

Understanding, not just numbers
---------------------------------

Numbers alone are not understanding. The real goal is to know **what each variable means** — its
units, how it was collected, what a missing value signifies — so that later choices (which features to
use, how to handle gaps) are informed rather than mechanical. Every stage that follows, from
association measures to model evaluation, rests on this first honest look.
"""

CONTENT["IEEE 754 Floating-Point Standard"] = r"""
Storing real numbers
----------------------

Computers store real numbers in a **finite** number of bits, and the near-universal scheme for doing
so is the **IEEE 754** standard. Understanding it explains a whole class of surprises — why sums do
not quite add up, why you should never test two floats for exact equality — that otherwise look like
bugs.

Sign, exponent, mantissa
--------------------------

A floating-point number is stored in three parts, like scientific notation in binary: a **sign** bit,
an **exponent** (which scales the value), and a **mantissa** (the significant digits). The two common
sizes are **single precision** (32 bits: 1 sign, 8 exponent, 23 mantissa) and **double precision**
(64 bits: 1 sign, 11 exponent, 52 mantissa) — the ``float64`` that ``numpy`` and ``pandas`` use by
default. More mantissa bits mean more precision.

Why 0.1 + 0.2 ≠ 0.3
---------------------

With finite mantissa bits, most decimal fractions **cannot be represented exactly** — :math:`0.1` in
binary is a repeating fraction, rounded to fit. The rounding errors accumulate, so the famous result
is

.. math::

   0.1 + 0.2 = 0.30000000000000004 \neq 0.3.

It is not a language bug; it is the unavoidable cost of squeezing infinite decimals into 64 bits.

What it means for data work
-----------------------------

Three habits follow. **Never test floats for exact equality** — compare within a tolerance
(``numpy.isclose``) instead. **Beware accumulated error** when summing many values, and prefer stable
formulations. And know the **special values** the standard defines — positive and negative infinity,
and ``NaN`` (not-a-number) — because ``NaN`` in particular is how missing or undefined numeric results
surface throughout ``pandas``.
"""


MINDMAP.update({
    "Big Data: Definition, Characteristics, Evolution, and Business Impact": [
        "Why Do We Analyze Data?", "The First Step in Knowing Your Data",
        "How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach",
        "CRISP-DM for Data Science",
    ],
    "The First Step in Knowing Your Data": [
        "The Process of Data Analysis", "CRISP-DM for Data Science",
        "Measuring Associations in Data", "IEEE 754 Floating-Point Standard",
    ],
    "IEEE 754 Floating-Point Standard": [
        "The First Step in Knowing Your Data",
        "Big Data: Definition, Characteristics, Evolution, and Business Impact",
        "Least Squares Regression",
        "Correlation Coefficients in Python (Pearson, Spearman, Kendall)",
    ],
})


# ----------------------------------------------------------------------
# Stage 2 — Associations & Correlation
# ----------------------------------------------------------------------

CONTENT["Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022)"] = r"""
Things that move together
---------------------------

Much of data analysis begins with a simple question: **do two things move together?** When ice-cream
sales rise with temperature, or umbrella sales with rainfall, we sense an **association** — a tendency
for two variables to vary in step. Making that intuition precise, and measuring how strong it is, is
the work of this stage.

Why it's useful
-----------------

Associations are useful in two ways. They **explain** — revealing which factors relate to an outcome
you care about — and they **predict** — if two variables move together, knowing one helps you guess
the other. Spotting the right associations is often the first real insight a dataset yields, and it
points to which variables are worth modelling later.

The running example
---------------------

This stage grounds the idea in a real dataset used throughout: the **City of Chicago taxi trips** from
September 2022. It is an intuitive place to look for associations — does the **fare** rise with the
**distance** travelled? with the **time** taken? The next lesson introduces the dataset in detail; for
now, the point is that a familiar, everyday relationship becomes a measurable one once it is in data.

Association is not cause
--------------------------

One caution to carry from the start: **association is not causation**. Two variables can move together
because one drives the other, because a third factor drives both, or by pure coincidence. Measuring a
strong association tells you the variables are **related**, not **why** — a distinction that matters
the moment anyone tries to act on the finding.
"""

CONTENT["Taxi Trips \u2013 2022 dataset from the City of Chicago open data portal"] = r"""
An open, real dataset
-----------------------

The running dataset is the **Taxi Trips** file from the **City of Chicago open data portal** — a real,
public record of taxi journeys the city collects as a regulator. It is a favourite teaching set
because it is large, genuinely messy, and full of intuitive relationships to explore.

What's in a row
-----------------

Each **row is one trip**, with a unique ID and a timestamp, described by around two dozen fields. The
most useful for association work are the numeric ones — **Trip Miles** (distance), **Trip Seconds**
(duration), **Fare**, **Tips**, **Tolls** and **Trip Total** — alongside categorical fields like
**Payment Type** and **Company**, and location fields (**pickup / dropoff community area**, census
tract, and centroid latitude / longitude). Most columns load as floats or text.

Scale and loading
-------------------

The full record runs to **hundreds of millions** of trips over the years; a single week of September
2022 is already a workable subset of tens of thousands. It is published through the city's **Socrata**
portal, so you can download a CSV and read it with ``pandas.read_csv`` (or pull a filtered slice via
the Socrata API) rather than loading everything at once.

It needs cleaning
-------------------

Being real, it needs the **preparation** this course is about. Records appear with trip end **before**
start, absent durations, impossible distances (over 100 miles), fares below the city's base charge,
and missing community areas for trips outside Chicago. Filtering these out is a prerequisite before
any association or model is trustworthy — a concrete instance of why data preparation dominates the
workflow.
"""

CONTENT["Objective Selection of the Bin Width for a Time Histogram"] = r"""
The bin-width problem
-----------------------

A **histogram** summarises a distribution by counting values into **bins**, but the picture depends
entirely on the **bin width** you choose. Too **wide** and you smooth away real structure; too
**narrow** and random noise looks like signal. For a **time histogram** — counting events (like taxi
pickups) into time intervals — the width sets whether you see a genuine daily rhythm or a jagged mess.
Choosing it should not be guesswork.

Rules of thumb
----------------

Several **rules** pick a width from the data automatically. **Sturges' rule** sets the *number* of
bins from :math:`\lceil \log_2 n \rceil + 1`, assuming roughly normal data. **Scott's rule** chooses
width :math:`h = 3.49\,\hat{\sigma}\,n^{-1/3}`, and the **Freedman–Diaconis** rule
:math:`h = 2\,\mathrm{IQR}\,n^{-1/3}`, which uses the interquartile range and so resists outliers.
Each aims to balance detail against noise.

An objective criterion
------------------------

For a **time histogram** specifically, the **Shimazaki–Shinomoto** method turns the choice into an
**optimisation**. It picks the width :math:`\Delta` that minimises a cost estimating the error between
the histogram and the true underlying rate:

.. math::

   C(\Delta) = \frac{2\bar{k} - v}{\Delta^2},

where :math:`\bar{k}` is the mean and :math:`v` the variance of the bin counts. Sweeping
:math:`\Delta` and taking the minimum gives a width **derived from the data**, not chosen by eye.

Why it matters
----------------

The theme is **objectivity**. A histogram is one of the first plots an analyst makes, and an arbitrary
bin width can quietly manufacture or hide patterns. A principled rule makes the picture
**reproducible** — the same data yields the same histogram for everyone — which is exactly the standard
the rest of this course holds its methods to.
"""


MINDMAP.update({
    "Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022)": [
        "Taxi Trips \u2013 2022 dataset from the City of Chicago open data portal",
        "Measuring Associations in Data", "Measuring Associations Between Two Continuous Variables",
        "Correlation Coefficients in Python (Pearson, Spearman, Kendall)",
    ],
    "Taxi Trips \u2013 2022 dataset from the City of Chicago open data portal": [
        "Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022)",
        "The First Step in Knowing Your Data",
        "Measuring Associations Between Two Continuous Variables",
        "Objective Selection of the Bin Width for a Time Histogram",
    ],
    "Objective Selection of the Bin Width for a Time Histogram": [
        "Taxi Trips \u2013 2022 dataset from the City of Chicago open data portal",
        "The First Step in Knowing Your Data", "Measuring Associations in Data",
        "Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022)",
    ],
})


# ----------------------------------------------------------------------
# Stage 2 — Associations & Correlation (cont.)
# ----------------------------------------------------------------------

CONTENT["Measuring Associations in Data"] = r"""
One idea, many measures
-------------------------

To move from "these two variables seem related" to a **number**, you need an **association measure** —
a single value capturing how strongly, and often in which direction, two variables move together.
There is no one measure for all cases; the right choice depends on **what kind of variables** you have.

It depends on the types
-------------------------

Variables come in two broad flavours — **continuous** (numbers on a scale, like fare or distance) and
**categorical** (labels, like payment type or company). The pairing decides the tool: comparing two
numbers is a different problem from comparing two labels, or a number against a label.

The taxonomy
--------------

The map for this stage:

* **continuous ↔ continuous** — **correlation** (Pearson, Spearman, Kendall);
* **categorical ↔ categorical** — the **chi-square** test and **Cramér's V**;
* **continuous ↔ categorical** — **ANOVA** and its effect size **eta-squared** (:math:`\eta^2`).

The lessons ahead take these in turn.

Strength and direction
------------------------

Two properties matter. **Strength** — how tightly the variables track, usually scaled so that 0 means
"no association" and 1 (or :math:`\pm 1`) means "perfect"; and **direction** — whether they rise
together or move oppositely, which only makes sense for **ordered** variables. A good measure reports
strength on a comparable scale, so associations across different variable pairs can be ranked.
"""

CONTENT["Measuring Associations Between Two Continuous Variables"] = r"""
Covariance: direction
-----------------------

The starting point for two continuous variables is **covariance**, which measures whether they vary
in the **same direction**:

.. math::

   \operatorname{cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}).

When above-average :math:`x` tends to pair with above-average :math:`y`, the products are positive and
covariance is **positive**; when high :math:`x` pairs with low :math:`y`, it is **negative**; near
zero means no linear tendency.

The problem with covariance
-----------------------------

Covariance has a flaw as a **strength** measure: its size depends on the variables' **units**.
Covariance of fare and distance changes if you switch miles to kilometres, so its magnitude is **not
comparable** across variable pairs — it ranges without bound. You can read its **sign**, but not judge
"how strong" from its value.

Pearson correlation
---------------------

The fix is to **standardise** covariance by the two standard deviations, giving the **Pearson
correlation coefficient**:

.. math::

   r = \frac{\operatorname{cov}(X, Y)}{\sigma_X \, \sigma_Y}
     = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}
            {\sqrt{\sum (x_i - \bar{x})^2}\,\sqrt{\sum (y_i - \bar{y})^2}}.

Dividing out the units confines :math:`r` to the range :math:`[-1, 1]`, making it **comparable**
everywhere.

Reading r
-----------

On that scale, :math:`r = +1` is a **perfect positive** linear relationship, :math:`r = -1` a
**perfect negative** one, and :math:`r = 0` **no linear** relationship. The crucial caveat: Pearson
measures **linear** association only. A strong curved relationship can still give :math:`r \approx 0`,
and :math:`r` is **sensitive to outliers** — reasons the next lesson reaches for rank-based
alternatives.
"""

CONTENT["Correlation Coefficients in Python (Pearson, Spearman, Kendall)"] = r"""
Three coefficients
--------------------

Pearson is one of **three** standard correlation coefficients, and the other two fix its blind spots.
All three run from :math:`-1` to :math:`+1` and share the same sign convention, but they measure
**different notions** of "moving together".

Pearson, Spearman, Kendall
----------------------------

The trio:

* **Pearson** :math:`r` — the **linear** coefficient from the last lesson; assumes a straight-line relationship and is sensitive to outliers.
* **Spearman** :math:`\rho` — Pearson applied to the **ranks** of the data. It captures any **monotonic** relationship (always rising or always falling, even if curved), and, working on ranks, it is **robust to outliers** and fine for **ordinal** data:

  .. math::

     \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}, \qquad d_i = \operatorname{rank}(x_i) - \operatorname{rank}(y_i).

* **Kendall** :math:`\tau` — based on counting **concordant** versus **discordant** pairs:

  .. math::

     \tau = \frac{n_c - n_d}{\tfrac{1}{2}\,n(n-1)},

  also a monotonic measure, often preferred for **small samples** and data with many **ties**.

Which to use
--------------

A rule of thumb: reach for **Pearson** when the relationship looks **linear** and the data are
well-behaved; switch to **Spearman** or **Kendall** when it is **monotonic but curved**, when there
are **outliers**, or when a variable is **ordinal** rather than numeric. Spearman and Kendall usually
agree; Kendall is the more conservative on small or tie-heavy data.

In Python
-----------

All three are one call away. A whole correlation matrix comes from ``df.corr(method="pearson")`` (or
``"spearman"`` / ``"kendall"``); for a single pair with a **p-value**, use ``scipy.stats.pearsonr``,
``spearmanr`` or ``kendalltau``. On the taxi data, fare against distance shows a strong positive
correlation by any of the three.
"""


MINDMAP.update({
    "Measuring Associations in Data": [
        "Measuring Associations Between Two Continuous Variables",
        "Correlation Coefficients in Python (Pearson, Spearman, Kendall)",
        "What Are Statistical Tests?", "Eta Squared (\u03b7\u00b2): Effect Size in ANOVA",
    ],
    "Measuring Associations Between Two Continuous Variables": [
        "Measuring Associations in Data",
        "Correlation Coefficients in Python (Pearson, Spearman, Kendall)",
        "Karl Pearson", "Least Squares Regression",
    ],
    "Correlation Coefficients in Python (Pearson, Spearman, Kendall)": [
        "Measuring Associations Between Two Continuous Variables",
        "Measuring Associations in Data", "Karl Pearson",
        "Feature Importance in Linear Regression",
    ],
})


# ----------------------------------------------------------------------
# Stage 2 — Associations & Correlation (cont.)
# ----------------------------------------------------------------------

CONTENT["Karl Pearson"] = r"""
The founder of the field
--------------------------

**Karl Pearson** (1857–1936) is, more than anyone, the **founder of modern mathematical statistics**.
An English mathematician at **University College London**, he spent the 1890s and 1900s turning
statistics from a collection of tricks into a discipline — and established **Britain's first degree
course** in the subject. The correlation coefficient of the last few lessons is his.

Tools we still use
--------------------

A remarkable share of everyday statistics traces to Pearson. He gave the **product-moment correlation
coefficient** :math:`r` its modern form in 1896 (building on Galton and Bravais), and around 1900
introduced the **chi-square test** for goodness of fit and for association in contingency tables —
still the standard tool for categorical data, and the basis of Cramér's V in the next lesson. He also
devised the **method of moments**, the **Pearson family of distributions**, and coined terms as basic
as **standard deviation** and **histogram**.

Biometrika and the school
---------------------------

Pearson built an **institution**, not just a toolkit. He founded and ran laboratories at UCL,
gathered collaborators into a **"biometric school"**, and in 1901 co-founded **Biometrika** — the
first journal of modern statistics — with Francis Galton and Walter Weldon. Much of the field's early
literature ran through it.

A complicated legacy
----------------------

Pearson's record also carries a darker strand: he was a committed advocate of **eugenics**, promoting
views on race and class that are rightly rejected today. It is worth holding both facts at once — that
the mathematical machinery is foundational and still in daily use, and that the ideology some of it
was built to serve was harmful. The tools outlived the purpose.
"""

CONTENT["Harald Cram\u00e9r"] = r"""
A Swedish statistician
------------------------

**Harald Cramér** (1893–1985) was a Swedish mathematician and statistician whose name attaches to
several cornerstones of the field. Where Pearson built the discipline's foundations, Cramér gave much
of it its **rigorous mathematical footing**, bridging probability theory and statistics. Two of his
contributions appear directly in this course.

Cramér's V
------------

The first is **Cramér's V**, the standard measure of association between **two categorical (nominal)
variables** — the categorical counterpart to Pearson's :math:`r`. Built on Pearson's **chi-square
statistic**, it rescales it to a clean range of :math:`0` (no association) to :math:`1` (perfect):

.. math::

   V = \sqrt{\frac{\chi^2}{n \,\min(r - 1,\, c - 1)}},

where :math:`n` is the sample size and :math:`r, c` the numbers of rows and columns in the
contingency table. It fills the "categorical ↔ categorical" slot of the association taxonomy.

The Cramér–Rao bound
----------------------

The second is the **Cramér–Rao bound**, a deep result in **estimation** theory: it sets a hard
**lower limit** on the variance any unbiased estimator can achieve. An estimator that reaches it is as
precise as possible — a benchmark that underlies the **maximum-likelihood** methods appearing later in
this course.

A foundational text
---------------------

Cramér's influence also came through teaching. His 1946 book **"Mathematical Methods of Statistics"**
was among the first to put statistics on a firm measure-theoretic basis, and it trained a generation
of statisticians. Alongside the V and the bound, his broader work spanned probability, actuarial
mathematics and number theory.
"""

CONTENT["What Are Statistical Tests?"] = r"""
Is it real or chance?
-----------------------

When you measure an association in a sample, one question always lurks: **is it real, or could it be
chance?** A correlation of 0.2 in a handful of taxi trips might vanish in the next batch. **Statistical
tests** answer this — they quantify how likely your finding is to have arisen by luck alone.

Null and alternative
----------------------

A test pits two hypotheses against each other. The **null hypothesis** (:math:`H_0`) is the sceptic's
position: **no effect, no association** — any pattern seen is chance. The **alternative**
(:math:`H_1`) is the claim you are investigating: a real effect exists. The test computes a **test
statistic** from the data, measuring how far the sample departs from what :math:`H_0` would predict.

The p-value
-------------

That departure is summarised in a **p-value**: the probability of seeing data **at least as extreme**
as yours **if the null hypothesis were true**. A **small** p-value means your result would be
surprising under "pure chance", so chance is an unconvincing explanation; a **large** one means the
data is unremarkable and the null stands. You compare it to a chosen threshold, the **significance
level** :math:`\alpha` (commonly 0.05), and **reject** :math:`H_0` when :math:`p < \alpha`.

Reading the result
--------------------

Two cautions make tests trustworthy. Rejecting a **true** null is a **false positive** (a Type I
error, at rate :math:`\alpha`); failing to detect a **real** effect is a **false negative** (Type II).
And **significance is not importance** — with enough data a trivially small effect becomes
"significant", which is why the next lesson pairs tests with a measure of **effect size**. The
chi-square, t-test and ANOVA F-test are all instances of this one logic.
"""


MINDMAP.update({
    "Karl Pearson": [
        "Measuring Associations Between Two Continuous Variables",
        "Correlation Coefficients in Python (Pearson, Spearman, Kendall)",
        "What Are Statistical Tests?", "Harald Cram\u00e9r",
    ],
    "Harald Cram\u00e9r": [
        "Karl Pearson", "What Are Statistical Tests?",
        "Maximum Likelihood (MLE): Fitting a Distribution to Observed Data",
        "Eta Squared (\u03b7\u00b2): Effect Size in ANOVA",
    ],
    "What Are Statistical Tests?": [
        "Measuring Associations in Data", "Eta Squared (\u03b7\u00b2): Effect Size in ANOVA",
        "Karl Pearson", "Forward Selection with Nested Models and Deviance Tests",
    ],
})


# ----------------------------------------------------------------------
# Stage 2 — Associations & Correlation (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Eta Squared (\u03b7\u00b2): Effect Size in ANOVA"] = r"""
How big, not just whether
---------------------------

A statistical test tells you **whether** a categorical variable relates to a numeric one — do taxi
fares **differ** by payment type? But "significant" does not mean "large". **Eta squared**
(:math:`\eta^2`) answers the other half: **how big** is the effect? It is the **effect size** that
partners the ANOVA test, filling the "continuous ↔ categorical" slot of the association taxonomy.

A number to a category
------------------------

The setting is **ANOVA** (analysis of variance): one **categorical** variable splits the data into
groups, and you ask how much of the spread in a **continuous** variable is explained by which group a
point falls in. ANOVA works by splitting the **total** variability into a **between-group** part
(differences among the group means) and a **within-group** part (scatter inside each group).

The formula
-------------

Eta squared is simply the **between-group** share of the total:

.. math::

   \eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}, \qquad
   SS_{\text{total}} = SS_{\text{between}} + SS_{\text{within}}.

It ranges from :math:`0` (the grouping explains nothing) to :math:`1` (the grouping explains
everything). An :math:`\eta^2` of 0.30 means **30%** of the variance in the numeric variable is
accounted for by the category.

The cousin of r²
------------------

Eta squared is the categorical twin of a familiar quantity: it is analogous to :math:`r^2`, the
**coefficient of determination**. Both report the **proportion of variance explained** in a continuous
variable — :math:`r^2` when the predictor is also continuous (as in regression, Stage 5),
:math:`\eta^2` when the predictor is a category. Seen this way, correlation and ANOVA are two faces of
the same question: **how much does one variable tell you about another?** (In multi-factor models, a
variant called *partial* eta squared isolates one factor's contribution.)
"""


# ----------------------------------------------------------------------
# Stage 3 — Market Basket & Association Rules
# ----------------------------------------------------------------------

CONTENT["Understanding Market Baskets and Ideal Customers"] = r"""
What's in the basket
----------------------

A **market basket** is just the set of items a customer buys **together** in one transaction — the
contents of a single shopping cart or receipt. **Market basket analysis** is the study of these
baskets across many customers, looking for the products that tend to **appear together**. It is one of
the oldest and most intuitive forms of data mining, born in retail.

Why baskets matter
--------------------

Knowing what goes with what is directly **actionable**. If bread and butter sell together, a shop can
place them nearby, bundle them in a promotion, or recommend one when the other is added to a cart. The
patterns hidden in baskets drive **product placement, recommendations, promotions and cross-selling**
— turning a pile of receipts into merchandising decisions.

The ideal customer
--------------------

Basket analysis also sharpens the idea of an **"ideal customer"**. By seeing which combinations of
purchases mark high-value or loyal shoppers, a business can recognise and target customers who look
like its best ones. What people **buy together** becomes a signature of **who they are** and what they
might want next.

From baskets to rules
-----------------------

To act on baskets at scale, the co-occurrence patterns are written as **association rules** — precise
"if this, then that" statements — and mined automatically from transaction data. The next lessons make
that idea exact: what a rule is, how its strength is measured, and how the **Apriori** algorithm finds
the good ones among astronomically many possibilities.
"""

CONTENT["What Can Association Rules Tell Us?"] = r"""
If this, then that
--------------------

An **association rule** is an "if-then" statement about items in a basket: **if** a customer buys some
set of items, **then** they are likely to buy another. Written
:math:`\{\text{bread}, \text{butter}\} \rightarrow \{\text{milk}\}`, it reads "baskets with bread and
butter tend also to contain milk". The left side is the **antecedent**, the right the **consequent**.

What a rule says
------------------

A rule captures a **regularity** in the data — a combination that shows up together more than you
might expect. On its own the arrow is just a candidate pattern; its **usefulness** depends on how often
it holds and how reliable it is, which the next lesson measures with **support**, **confidence** and
**lift**. For now, the point is the *shape* of the knowledge: compact, readable statements about what
accompanies what.

What they're good for
-----------------------

Rules turn into **decisions**. "Customers who buy X also buy Y" suggests **recommendations** ("you
might also like…"), **bundles**, **store layout**, and **targeted promotions**. This is the engine
behind **cross-selling** (Stage 3's closing lesson): using a known purchase to suggest a complementary
one, lifting basket size and revenue.

A familiar caution
--------------------

The caution from the start of this stage returns: a rule reports **association, not causation**. That
bread and butter travel with milk does not mean one **causes** the other — both may simply reflect a
weekly grocery run. Rules are superb at spotting **what** goes together and useful for acting on it,
but they do not, by themselves, explain **why**.
"""


MINDMAP.update({
    "Eta Squared (\u03b7\u00b2): Effect Size in ANOVA": [
        "What Are Statistical Tests?", "Measuring Associations in Data",
        "Harald Cram\u00e9r", "Feature Importance in Linear Regression",
    ],
    "Understanding Market Baskets and Ideal Customers": [
        "What Can Association Rules Tell Us?",
        "How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach",
        "Cross-Selling", "Creating Segments of Observations for Business Reasons (RFM)",
    ],
    "What Can Association Rules Tell Us?": [
        "Understanding Market Baskets and Ideal Customers",
        "How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach",
        "Apriori: Frequent Itemsets via the Apriori Algorithm", "Cross-Selling",
    ],
})


# ----------------------------------------------------------------------
# Stage 3 — Market Basket & Association Rules (cont.)
# ----------------------------------------------------------------------

CONTENT["How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach"] = r"""
Three measures of a rule
--------------------------

A rule like :math:`X \rightarrow Y` is only worth keeping if it is both **common** and **reliable**.
Three numbers, all computed from how often itemsets appear in the transactions, make "worth keeping"
precise: **support**, **confidence** and **lift**.

Support, confidence, lift
---------------------------

Each captures a different facet:

* **Support** — how **common** the combination is, the fraction of all transactions containing the whole itemset:

  .. math::

     \operatorname{support}(X \rightarrow Y) =
       \frac{\text{transactions containing } X \cup Y}{\text{all transactions}}.

* **Confidence** — how **reliable** the rule is, the share of :math:`X`-baskets that also contain :math:`Y`, estimating :math:`P(Y \mid X)`:

  .. math::

     \operatorname{confidence}(X \rightarrow Y) =
       \frac{\operatorname{support}(X \cup Y)}{\operatorname{support}(X)}.

* **Lift** — how **surprising** it is, confidence compared with :math:`Y`'s baseline frequency:

  .. math::

     \operatorname{lift}(X \rightarrow Y) =
       \frac{\operatorname{confidence}(X \rightarrow Y)}{\operatorname{support}(Y)}.

  Lift :math:`> 1` means :math:`X` and :math:`Y` occur together **more** than chance (positive
  association); :math:`= 1` means independent; :math:`< 1` means they repel.

The scale problem
-------------------

Why not simply score every possible rule? Because there are **too many**. With :math:`d` distinct
items there are :math:`2^d` possible itemsets and more possible rules still — for a supermarket with
thousands of products, an astronomically large number. Counting the support of every candidate by
brute force is hopeless.

The Apriori idea
------------------

The escape is one simple observation, the **Apriori principle** (also called **downward closure**):
*if an itemset is frequent, all of its subsets must be frequent too* — and, turned around, *if an
itemset is infrequent, every superset of it is infrequent as well*. That lets an algorithm **prune**
vast regions of the search: once the pair **{milk, caviar}** is rare, nothing containing it can be
common, so none of its extensions need be checked. The next lesson builds an algorithm around exactly
this.
"""

CONTENT["Apriori: Frequent Itemsets via the Apriori Algorithm"] = r"""
Prior knowledge
-----------------

The **Apriori algorithm**, introduced by Agrawal and Srikant in **1994**, is the classic method for
finding all **frequent itemsets** — the groups of items whose support clears a chosen **minimum**. Its
name comes from the *a priori* (prior) knowledge it exploits: it uses the frequent itemsets already
found at one size to decide what is worth checking at the next.

The level-wise search
-----------------------

The algorithm walks the itemsets **level by level**, from small to large, applying downward closure at
every step. The key move is **candidate generation**: a :math:`k`-item candidate is formed only if
**all** of its :math:`(k{-}1)`-item subsets are already known to be frequent. Any candidate with an
infrequent subset is discarded **before** its support is ever counted — that is where the saving comes
from.

One level at a time
---------------------

Concretely, each level repeats three steps:

1. **Generate** candidate :math:`k`-itemsets by combining frequent :math:`(k{-}1)`-itemsets, then **prune** those with any infrequent subset;
2. **Count** each surviving candidate's support with a single **scan** of the transaction database;
3. **Keep** those meeting minimum support as the frequent :math:`k`-itemsets.

Start with frequent single items (:math:`k = 1`) and repeat, increasing :math:`k`, until no new
candidates survive. The cost is that dense data can still spawn many candidates and repeated database
scans — Apriori is simple and correct, not always the fastest.

Then the rules
----------------

Apriori delivers frequent **itemsets**; turning them into **rules** is the easy second half. For each
frequent itemset, split it into an antecedent and consequent every way possible and keep the splits
whose **confidence** (or **lift**) clears a threshold. Because both sides come from a frequent itemset,
the rule's support is already known — which is exactly what the next lesson automates in Python.
"""

CONTENT["association_rules: Generating Association Rules from Frequent Itemsets (mlxtend)"] = r"""
From itemsets to rules in code
--------------------------------

The Python library **mlxtend** implements the whole pipeline of the last two lessons in a few lines:
encode the transactions, mine frequent itemsets with **apriori**, then turn them into ranked rules with
**association_rules**. It is the standard tool for market-basket analysis in the scientific-Python
stack.

One-hot transactions
----------------------

The algorithms expect a **one-hot encoded** table — one row per transaction, one boolean column per
item, ``True`` where the item is in the basket. mlxtend's **TransactionEncoder** builds it from raw
lists of items:

.. code-block:: python

   from mlxtend.preprocessing import TransactionEncoder
   import pandas as pd

   transactions = [["bread", "milk", "eggs"], ["bread", "butter"], ["milk", "butter"]]
   te = TransactionEncoder()
   df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

Two functions
---------------

With the table ready, two calls do the work:

.. code-block:: python

   from mlxtend.frequent_patterns import apriori, association_rules

   items = apriori(df, min_support=0.5, use_colnames=True)
   rules = association_rules(items, metric="confidence", min_threshold=0.6)

**apriori** returns the frequent itemsets and their support; **association_rules** expands them into
rules and filters by the metric you choose (``"confidence"``, ``"lift"``, and others).

Reading the output
--------------------

The result is a tidy ``DataFrame``: each row a rule, with columns for **antecedents**, **consequents**,
**support**, **confidence** and **lift** (plus leverage and conviction). Sorting by **lift** surfaces
the most surprising, actionable pairings — the rules a shop would actually act on. The final lesson of
this stage puts them to use: **cross-selling**.
"""


MINDMAP.update({
    "How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach": [
        "What Can Association Rules Tell Us?",
        "Apriori: Frequent Itemsets via the Apriori Algorithm",
        "association_rules: Generating Association Rules from Frequent Itemsets (mlxtend)",
        "Cross-Selling",
    ],
    "Apriori: Frequent Itemsets via the Apriori Algorithm": [
        "How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach",
        "association_rules: Generating Association Rules from Frequent Itemsets (mlxtend)",
        "Understanding Market Baskets and Ideal Customers", "Cross-Selling",
    ],
    "association_rules: Generating Association Rules from Frequent Itemsets (mlxtend)": [
        "Apriori: Frequent Itemsets via the Apriori Algorithm",
        "How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach",
        "What Can Association Rules Tell Us?", "Cross-Selling",
    ],
})


# ----------------------------------------------------------------------
# Stage 3 — Market Basket & Association Rules (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Cross-Selling"] = r"""
Selling the complement
------------------------

**Cross-selling** is the practice of suggesting **complementary** products to a customer based on what
they are already buying — a bag and a lens for someone buying a camera, buns for someone buying hot
dogs. It is the direct **payoff** of the association rules this stage has built: the rules say what
goes with what, and cross-selling acts on it.

Cross-sell vs up-sell
-----------------------

It is worth distinguishing two moves. **Cross-selling** offers a **different, complementary** item
("customers who bought this also bought…"). **Up-selling** offers a **better version of the same**
item — a premium model or a larger size. Cross-selling widens the basket; up-selling deepens a single
choice. Association rules speak most directly to the former.

Rules in action
-----------------

The mechanics are exactly the mined rules. A rule :math:`\{\text{camera}\} \rightarrow \{\text{tripod}\}`
with high **confidence** and **lift** becomes a recommendation shown at checkout or on the product
page. Ranking candidate rules by lift surfaces the pairings that are **surprisingly** common — the
ones most likely to be genuine complements rather than two independently popular items.

Why it pays
-------------

The business case is simple: a relevant complement raises the **average order value** and improves the
customer's experience by anticipating a real need. Done well, cross-selling turns a single purchase
into a larger, more useful basket — which is why recommendation engines built on association rules are
everywhere in retail. It also sets up the next stage's question: not just *what* customers buy
together, but *which customers* to treat differently.
"""


# ----------------------------------------------------------------------
# Stage 4 — Sampling, Partitioning & Segmentation
# ----------------------------------------------------------------------

CONTENT["Stratified Random Sampling"] = r"""
Representative by design
--------------------------

A sample is only useful if it **resembles** the population it is drawn from. **Stratified random
sampling** guarantees that resemblance for the characteristics you care about, by sampling **within**
subgroups rather than trusting chance to balance them. It is a workhorse of survey design and, in this
course, of splitting data for modelling.

Strata
--------

The method starts by dividing the population into **strata** — mutually exclusive, exhaustive
subgroups that share a characteristic (gender, region, customer type). Strata are chosen to be
**internally homogeneous**: alike within, different between. A **simple random sample** is then drawn
independently from **each** stratum, and the pieces combined. Under **proportional allocation**, each
stratum contributes in proportion to its share of the population, so the sample mirrors the whole.

Why not simple random?
------------------------

Plain random sampling can, by luck, **under-represent** a small but important group — draw 100
customers at random and a rare segment might barely appear. Stratifying **removes** that luck: every
subgroup is present by construction, in the right proportion. The result is **greater precision**
(lower sampling variability) than a simple random sample of the same size, especially when the strata
differ from one another.

In machine learning
---------------------

The same idea is essential when splitting data. A **stratified** train/test split keeps the **class
proportions** identical in both parts — vital for **imbalanced** problems, where a naive split might
leave too few positive cases in the test set. In scikit-learn it is one argument:
``train_test_split(..., stratify=y)``, or ``StratifiedKFold`` for cross-validation. The next lessons
need this discipline, because honest model evaluation depends on representative partitions.
"""

CONTENT["Linear Congruential Random Number Generator (LCG)"] = r"""
Randomness you can repeat
---------------------------

The sampling of the last lesson needs a source of **randomness** — but for science it must also be
**reproducible**, so that a colleague running the same code gets the same split. Computers square this
circle with **pseudo-random** number generators: deterministic recipes that produce sequences which
*look* random. The **linear congruential generator (LCG)** is the classic, and the simplest to
understand.

The recurrence
----------------

An LCG produces each number from the previous one by a single formula:

.. math::

   X_{n+1} = (a\,X_n + c) \bmod m,

where :math:`m` is the **modulus**, :math:`a` the **multiplier**, :math:`c` the **increment**, and
:math:`X_0` the starting value. Each :math:`X_n` is an integer in :math:`[0, m)`; dividing by
:math:`m` rescales it to a fraction in :math:`[0, 1)`. From three constants and a start value, an
endless stream of "random" numbers follows.

The seed and the period
-------------------------

The starting value :math:`X_0` is the **seed**. Because the formula is deterministic, **the same seed
always yields the same sequence** — precisely what makes results reproducible (``random.seed`` /
``numpy.random.seed`` set it). Being finite, the sequence must eventually **repeat**; the length
before it does is the **period**, at most :math:`m`. Careful choices of :math:`a`, :math:`c` and
:math:`m` (the Hull–Dobell conditions) achieve the full period.

Simple, but dated
-------------------

The LCG is prized for being **fast, tiny and easy to reason about**, and it makes seeded
reproducibility concrete. But its statistical quality is **limited** — successive values fall on
detectable lattice patterns, and the low-order bits are weakly random. Modern libraries therefore
default to stronger generators (numpy now uses a **PCG64** generator by default), while keeping the
same crucial habit: **set a seed** so your sampling, splitting and modelling can be reproduced exactly.
"""


MINDMAP.update({
    "Cross-Selling": [
        "What Can Association Rules Tell Us?",
        "How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach",
        "Understanding Market Baskets and Ideal Customers",
        "Recency, Frequency, and Monetary Value (RFM)",
    ],
    "Stratified Random Sampling": [
        "Partitioning Observations to Train Objective Models",
        "Linear Congruential Random Number Generator (LCG)",
        "Creating Segments of Observations for Business Reasons (RFM)",
        "Assessing the Quality of Prediction Models",
    ],
    "Linear Congruential Random Number Generator (LCG)": [
        "Stratified Random Sampling",
        "Partitioning Observations to Train Objective Models",
        "IEEE 754 Floating-Point Standard", "Clustering",
    ],
})


# ----------------------------------------------------------------------
# Stage 4 — Sampling, Partitioning & Segmentation (cont.)
# ----------------------------------------------------------------------

CONTENT["Partitioning Observations to Train Objective Models"] = r"""
The temptation to cheat
-------------------------

How well does a model perform? The tempting answer — measure it on the **same data** you trained it on
— is also the **wrong** one. A model can score brilliantly on data it has already seen and still fail
on anything new. To judge a model **objectively**, you must test it on data it has **never
encountered**. That is what partitioning provides.

Train and test
----------------

The basic move is to split the observations into two disjoint sets. The **training set** is used to
**fit** the model; the **test set** (or holdout) is set aside and used **once**, at the end, to
estimate how the model will do on **unseen** data. Common splits give training the larger share —
80/20 or 70/30, sometimes 90/10 when data is plentiful. A third **validation** set (or
**cross-validation**) is used when tuning, so the test set stays untouched until the final verdict.

Overfitting
-------------

The problem the split guards against is **overfitting**: a model so flexible it memorises the training
data's **noise** along with its signal. Such a model looks excellent in training and disappoints in
deployment, because it learned the sample rather than the pattern. A held-out test set exposes this at
once — training accuracy soars while test accuracy stalls. A related danger is **data leakage**, where
information from the test set seeps into training and silently **inflates** the score.

Doing it right
----------------

In practice the split is one line — ``train_test_split`` in scikit-learn, ideally **stratified**
(previous lessons) so both parts stay representative. The discipline is non-negotiable and echoes this
course's standing warning: **report performance on the test set, never the training set**. Every
evaluation metric in Stage 8 assumes the model is being judged on data it has never seen.
"""

CONTENT["Putting Similar Observations into Clusters"] = r"""
Grouping without labels
-------------------------

Sometimes there is no outcome to predict — just a pile of observations and a hunch that they fall into
**natural groups**. **Clustering** is the task of finding those groups: partitioning observations so
that each group gathers items that are **alike**. It is **unsupervised** — unlike the classification
and regression of later stages, there are no labels to learn from, only the structure of the data
itself.

Similar within, different between
-----------------------------------

A good clustering has a simple signature: **high similarity within** each cluster and **low similarity
between** clusters. Members of a group should resemble one another; members of different groups should
not. This dual goal — tight, well-separated clusters — is what every clustering algorithm chases, and
what quality measures like the silhouette score reward.

Distance as similarity
------------------------

To make "alike" computable, similarity is usually expressed as **distance**: two observations are
similar if they are **close** in the feature space, typically by **Euclidean** distance. This is why
clustering is sensitive to **scale** — a feature measured in large units can dominate the distance —
so features are commonly **standardised** first. Close points cluster together; distant ones fall into
different groups.

Why segment?
--------------

The payoff is **segmentation**: dividing customers, products or observations into meaningful groups
that can be understood and treated differently. A business might discover a cluster of high-value
frequent buyers and another of occasional bargain-hunters, then tailor its approach to each. The next
lesson turns this idea into concrete algorithms; the RFM lessons that follow apply it to real customer
data.
"""

CONTENT["Clustering"] = r"""
Algorithms for groups
-----------------------

Turning the idea of clustering into practice means choosing an **algorithm** — a procedure that
actually finds the groups. Many exist, differing in how they define a cluster and how they search. The
most widely used, and the natural starting point, is **k-means**.

k-means
---------

**k-means** partitions the data into a pre-chosen number :math:`k` of clusters, each summarised by its
**centroid** (the mean of its members). It seeks to minimise the **within-cluster sum of squares** —
the total squared distance from points to their centroids — through a simple, repeating two-step loop:

1. **Assign** each observation to the **nearest** centroid;
2. **Update** each centroid to the mean of the points now assigned to it.

Repeat until assignments stop changing. In scikit-learn this is ``KMeans(n_clusters=k)``. It is fast
and intuitive, though it assumes roughly round, similarly-sized clusters and needs :math:`k` chosen in
advance.

Choosing k
------------

Since :math:`k` is an input, how many clusters should there be? Two common guides: the **elbow
method** plots the within-cluster sum of squares against :math:`k` and looks for the "elbow" where
adding clusters stops helping much; the **silhouette score** measures how well each point sits in its
cluster versus the nearest other one, rewarding tight, well-separated groups. Neither is automatic —
the "right" :math:`k` often depends on what is **useful** for the business.

Other approaches
------------------

k-means is not the only option. **Hierarchical** clustering builds a tree (dendrogram) of nested
groups, needing no :math:`k` up front and revealing structure at every scale. **Density-based**
methods like **DBSCAN** grow clusters from dense regions, handling odd shapes and marking outliers as
noise. Each embodies a different notion of what a cluster *is* — but all serve the same goal: similar
together, different apart.
"""


MINDMAP.update({
    "Partitioning Observations to Train Objective Models": [
        "Stratified Random Sampling", "Assessing the Quality of Prediction Models",
        "Linear Congruential Random Number Generator (LCG)",
        "Binary Classification Model Evaluation and Threshold Optimization",
    ],
    "Putting Similar Observations into Clusters": [
        "Clustering", "Creating Segments of Observations for Business Reasons (RFM)",
        "Cluster Profiling Using Decision Trees",
        "Measuring Associations Between Two Continuous Variables",
    ],
    "Clustering": [
        "Putting Similar Observations into Clusters",
        "Recency, Frequency, and Monetary Value (RFM)", "RFM Analysis",
        "Using Decision Trees to Explain Clustering Results",
    ],
})


# ----------------------------------------------------------------------
# Stage 4 — Sampling, Partitioning & Segmentation (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Recency, Frequency, and Monetary Value (RFM)"] = r"""
Three questions about a customer
----------------------------------

How valuable is a customer? **RFM** answers with three simple questions, each read straight from
transaction history: **how recently** did they buy, **how often** do they buy, and **how much** do
they spend? These three numbers — **Recency, Frequency and Monetary value** — summarise a customer's
behaviour compactly enough to rank an entire database.

The three dimensions
----------------------

Each dimension is one measurement per customer:

* **Recency** — days since their **last** purchase. Fewer days is better: recent buyers are far likelier to buy again.
* **Frequency** — the **number** of purchases in a chosen window. More is better: repeat buying signals habit and loyalty.
* **Monetary** — the **total spend** over that window. More is better: it captures the customer's economic value.

Why all three
---------------

No single dimension tells the whole story. A **big spender** who has not bought in two years is a
**churn risk**, not a star; a **frequent** buyer with tiny orders is loyal but low-margin. Combined,
the three give a **holistic** view that any one alone would distort. (Of the three, **recency** tends
to predict future behaviour best, and monetary least.)

Simple and proven
-------------------

RFM's great virtue is **simplicity**. It needs only data every business already has — an order history
— and no elaborate modelling. The technique dates back to **direct-mail** marketing in the last
century, and it endures because it works: it reliably surfaces the roughly 20% of customers who drive
most of the revenue. The next lesson turns these three raw numbers into scores.
"""

CONTENT["RFM Analysis"] = r"""
From raw values to scores
---------------------------

RFM analysis turns the three raw numbers into a **comparable score**. Raw values are awkward to
compare directly — is 30 days "recent"? is $500 "high"? — because the answer depends on your business.
The fix is to score each customer **relative to the others**, so the numbers become ranks on a common
1–5 scale.

Quintile scoring
------------------

The standard method is **quintiles**. For each dimension, sort all customers and split them into **five
equal groups**, then assign a score from **1 to 5**. The top 20% on **Frequency** and **Monetary**
score 5; for **Recency** the scale is **inverted** — the most recent 20% (fewest days) score 5.
Because quintiles are defined by your **own** data distribution, the scores adapt automatically to any
business, with each score level holding about 20% of customers.

The RFM code
--------------

The three scores are combined into a single **RFM code** — usually just concatenated, like
:math:`R{=}5,\ F{=}4,\ M{=}5 \rightarrow` "**545**". This gives :math:`5 \times 5 \times 5 = 125`
possible codes, from **555** (the best — recent, frequent, high-spending) down to **111** (the worst).
Sorting customers by their code, or by a **weighted** combination if one dimension matters more, ranks
the whole base at a glance. Because behaviour drifts, the scores are **recomputed** regularly.

In Python
-----------

The computation is a short ``pandas`` routine: group the orders **by customer**, then aggregate to the
three values — recency from the latest ``InvoiceDate``, frequency from a count of orders, monetary
from the sum of spend. ``pandas.qcut`` cuts each into quintile scores in one call. The result is a tidy
table, one row per customer with an R, F and M score — ready to be grouped into the segments of the
next lesson.
"""

CONTENT["Creating Segments of Observations for Business Reasons (RFM)"] = r"""
From codes to segments
------------------------

125 RFM codes are too many to act on. The final step is to group them into a handful of **named
segments** — typically **six to ten** — each describing a recognisable kind of customer and, crucially,
each calling for a **different business response**. This is where scores become **decisions**.

A common taxonomy
-------------------

A widely used starting taxonomy names segments by their RFM profile:

* **Champions** — high on all three (555, 554): the best customers.
* **Loyal Customers** — buy consistently and often.
* **Potential Loyalists** — recent buyers with growing frequency.
* **New Customers** — recent, but few purchases so far.
* **At Risk** — once frequent and high-spending, but **lapsing** (low recency).
* **Can't Lose Them** — high past value, gone quiet.
* **Hibernating / Lost** — low on everything, long inactive.

Each segment, an action
-------------------------

The point of naming segments is the **"so what"**. **Champions** get rewards, early access and
referral asks — **not** blanket discounts that erode margin. **At Risk** and **Can't Lose Them** get
**win-back** campaigns and personal outreach, ideally *before* they fully churn. **Hibernating / Lost**
get a re-permission push or are **suppressed** to save budget. One message for a new buyer and a
ten-year loyalist would waste both; segmentation lets each be treated for **who they are**.

Segments as clusters
----------------------

Notice this is the **clustering** idea from earlier in the stage, made concrete: customers are grouped
by similarity — here, similarity in RFM space — so that each group can be understood and served
differently. Whether the groups come from a rule-based RFM taxonomy or from an algorithm like k-means,
the goal is the one this stage began with: turn a mass of observations into **meaningful segments** a
business can act on. With customers understood, the course turns next to **predicting** outcomes —
starting with regression.
"""


MINDMAP.update({
    "Recency, Frequency, and Monetary Value (RFM)": [
        "RFM Analysis", "Creating Segments of Observations for Business Reasons (RFM)",
        "Cross-Selling", "Clustering",
    ],
    "RFM Analysis": [
        "Recency, Frequency, and Monetary Value (RFM)",
        "Creating Segments of Observations for Business Reasons (RFM)",
        "Stratified Random Sampling", "Putting Similar Observations into Clusters",
    ],
    "Creating Segments of Observations for Business Reasons (RFM)": [
        "Recency, Frequency, and Monetary Value (RFM)", "RFM Analysis",
        "Clustering", "Cluster Profiling Using Decision Trees",
    ],
})


# ----------------------------------------------------------------------
# Stage 5 — Regression
# ----------------------------------------------------------------------

CONTENT["Least Squares Regression"] = r"""
Fitting a line
----------------

The simplest predictive model draws a **straight line** through a cloud of points — predicting an
outcome :math:`y` from a feature :math:`x` as

.. math::

   \hat{y} = \beta_0 + \beta_1 x,

where :math:`\beta_0` is the intercept and :math:`\beta_1` the slope. On the taxi data, this is the
line predicting **fare** from **distance**. The question is: of all possible lines, which one **fits
best**?

Residuals
-----------

"Best" is defined through **residuals** — the vertical gaps between each observed point and the line's
prediction:

.. math::

   e_i = y_i - \hat{y}_i.

A residual is how wrong the model is for one point: positive when the point sits above the line,
negative below. A good line makes these gaps **small** overall. But small how? Summing them directly
is useless — positive and negative residuals cancel.

Least squares
---------------

The answer is to **square** each residual before adding, and choose the line that makes the total as
small as possible. This is the **least squares** criterion, minimising the **residual sum of squares**:

.. math::

   \text{minimise} \quad \sum_{i=1}^{n} (y_i - \hat{y}_i)^2.

Squaring removes the sign, so gaps cannot cancel, and the line that minimises this sum is the **line
of best fit**. For a linear model it has a neat **closed-form** solution — exact formulas for
:math:`\beta_0` and :math:`\beta_1`, no searching required.

Why squared?
--------------

Why squares rather than, say, absolute values? Squaring **penalises large errors far more** than small
ones — a residual of 4 counts sixteen times as much as a residual of 1 — so the fitted line works hard
to avoid big misses. This also makes the objective smooth and gives that unique closed-form solution,
and it coincides with **maximum likelihood** when the errors are normally distributed. The trade-off
is **sensitivity to outliers**: one extreme point can pull the line noticeably — a theme the
evaluation stage revisits with residual diagnostics.
"""

CONTENT["Multiple Linear Regression"] = r"""
More than one predictor
-------------------------

Real outcomes rarely depend on a single cause. A taxi fare responds to **distance and duration**; a
house price to **size, location and age**. **Multiple linear regression** extends the line to several
features at once:

.. math::

   \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p.

Geometrically this is no longer a line but a **hyperplane**, yet the fitting principle is unchanged:
choose the coefficients that minimise the **sum of squared residuals**.

Holding others constant
-------------------------

The coefficients gain a subtle, powerful meaning. Each :math:`\beta_j` is the change in :math:`y` for
a one-unit increase in :math:`x_j` **while holding all other features fixed**. That "holding constant"
is what lets regression **disentangle** overlapping influences — estimating the effect of duration on
fare *after accounting for* distance, rather than confounding the two. It is why multiple regression
is a workhorse for **interpretation**, not just prediction.

The matrix solution
---------------------

With many features, the tidy way to write and solve the problem is **matrix algebra**. Stacking the
data into a matrix :math:`X` (with a column of ones for the intercept) and outcomes into a vector
:math:`y`, least squares has the closed-form solution

.. math::

   \hat{\beta} = (X^{\top} X)^{-1} X^{\top} y.

This single expression delivers all the coefficients at once. It rests on assumptions worth
remembering — a genuinely linear relationship, independent errors of constant variance — which the
evaluation stage checks with residual plots.

Fitting it in Python
----------------------

In practice you never invert the matrix by hand. **scikit-learn** fits the model in three lines —
``LinearRegression().fit(X, y)`` — exposing ``.coef_`` and ``.intercept_``; **statsmodels** ``OLS``
adds a full statistical summary with standard errors and p-values for each coefficient. The next
lesson uses those coefficients to ask which features matter **most**.
"""

CONTENT["Feature Importance in Linear Regression"] = r"""
Which features matter?
------------------------

Once a multiple regression is fitted, a natural question follows: **which predictors matter most?**
Which features do the heavy lifting in explaining the outcome, and which barely register? The
coefficients seem like the obvious answer — bigger coefficient, bigger effect — but read naively, they
**mislead**.

The units trap
----------------

The problem is **units**. A coefficient's size depends on the **scale** of its feature. Suppose fare
rises by **one dollar** per mile and by **one cent** per second travelled; the mile coefficient (1.0)
dwarfs the second coefficient (0.01), yet that reflects the **units**, not the importance — a mile is
simply a bigger step than a second. Comparing raw coefficients across features measured in different
units is comparing apples to oranges.

Standardized coefficients
---------------------------

The fix is to put every feature on the **same scale** before comparing. **Standardising** each
predictor — subtracting its mean and dividing by its standard deviation, so all are measured in
**standard-deviation units** — yields **standardised coefficients** (often called **beta
coefficients**). Now each answers a comparable question: how much does :math:`y` move for a
**one-standard-deviation** change in this feature? Their **magnitudes** can be ranked, giving a genuine
measure of importance. Statistical significance (the **t-statistic** and **p-value** from
``statsmodels``) tells a complementary story: whether an effect is distinguishable from zero at all.

Cautions
----------

Two caveats. **Multicollinearity** — predictors that are themselves strongly correlated — makes
individual coefficients **unstable** and their importances hard to trust, since the model cannot
cleanly separate overlapping features (recall the correlation lessons of Stage 2). And linear-model
importance only captures **linear** effects; a feature with a strong curved relationship may look
unimportant. For a model-agnostic alternative, **permutation importance** measures how much
performance drops when a feature is shuffled — an idea that generalises to the trees and other models
ahead. The next lessons turn to choosing *which* features to include at all.
"""


MINDMAP.update({
    "Least Squares Regression": [
        "Multiple Linear Regression", "Feature Importance in Linear Regression",
        "Measuring Associations Between Two Continuous Variables",
        "Identifying Outliers Using Residuals and Studentized Residuals",
    ],
    "Multiple Linear Regression": [
        "Least Squares Regression", "Feature Importance in Linear Regression",
        "Forward Selection and Model Interpretation in Linear Regression",
        "Assessing the Quality of Prediction Models",
    ],
    "Feature Importance in Linear Regression": [
        "Multiple Linear Regression", "Least Squares Regression",
        "How Shapley Values Work", "Understanding Forward and Backward Stepwise Regression",
    ],
})


# ----------------------------------------------------------------------
# Stage 5 — Regression (cont.)
# ----------------------------------------------------------------------

CONTENT["Forward Selection: Definition and Core Idea"] = r"""
Too many features
-------------------

A dataset often offers **more candidate predictors than you should use**. Throwing every feature into
a regression risks **overfitting**, obscures which variables truly matter, and can break down entirely
when features outnumber observations. **Feature selection** chooses a smaller, better subset — and
**forward selection** is the most intuitive way to do it.

Start empty, add the best
---------------------------

Forward selection **builds the model up from nothing**. Begin with the **null model** — no predictors,
just the intercept. Then, at each step, try adding **each** remaining feature in turn and keep the
**single one** that improves the model most, by a chosen **criterion** (the largest drop in a score
like **AIC**, or the most significant feature by **p-value**). Add it, then repeat: search the
features not yet in the model, and again admit the best.

When to stop
--------------

The process halts when **no remaining feature is worth adding** — when the best candidate fails to
clear the entry criterion (say, its p-value exceeds a threshold like 0.05, or it no longer lowers
AIC). Because it starts empty and grows, forward selection can handle **very wide** data — even more
candidate features than data points — since it never has to fit the full model at once.

Greedy, not exhaustive
------------------------

One honest limitation: forward selection is **greedy**. It makes the **locally** best choice at each
step and never reconsiders, so it does **not** guarantee finding the **globally** best subset — a
feature that shines only in combination with another might never be picked. Checking every possible
subset would be exact but explodes combinatorially (echoing the Apriori scale problem). Forward
selection trades that guarantee for speed and simplicity; the next lessons apply and extend it.
"""

CONTENT["Forward Selection and Model Interpretation in Linear Regression"] = r"""
Building the model
--------------------

Put forward selection to work on a real regression and it produces a **compact, fitted model** — a
handful of predictors, each earning its place. Starting from the intercept alone, the procedure admits
features one at a time until none of the leftovers improve the fit, leaving a **parsimonious** equation
that is far easier to reason about than one stuffed with every available column.

Reading the order of entry
----------------------------

The **order in which features enter** is itself informative. The first variable admitted is the
**single strongest** predictor of the outcome; the second adds the most **on top of** the first, and
so on. This sequence gives a rough **importance ranking** — though a subtle one, because each entry is
judged given those already in, so a feature's rank reflects its **added** value, not its value in
isolation. On the taxi data, distance might enter first, with duration adding power beyond it.

Comparing nested models
-------------------------

Each step yields a slightly larger model **nested** inside the next, which invites comparison. Because
adding any feature can only **increase** ordinary :math:`R^2`, that raw measure always favours the
bigger model and cannot judge whether an addition is worthwhile. Penalised criteria fix this:
**adjusted** :math:`R^2`, **AIC** and **BIC** all reward fit but **charge** for each extra parameter,
so they rise only when a feature earns its complexity. These are the yardsticks forward selection
actually optimises.

Interpret with care
---------------------

Interpret the result **cautiously**. Because selection is **greedy** and driven by the data, the
chosen model can be **unstable** — a slightly different sample might select different features — and
repeatedly testing many features inflates apparent significance, so p-values from the final model read
**optimistically**. The coefficients are still interpreted the usual way (effect per unit, holding
others fixed), but the honest test of the model is its performance on the **held-out** data, not the
selection statistics.
"""

CONTENT["Understanding Forward and Backward Stepwise Regression"] = r"""
Three directions
------------------

Forward selection is one of **three** stepwise strategies, distinguished by the **direction** they
move. **Forward** starts empty and **adds**; **backward elimination** starts full and **removes**;
**bidirectional** does **both** at every step. All three share the same goal — a parsimonious model —
and the same criteria (p-values, AIC, BIC, adjusted :math:`R^2`), differing only in how they search.

Backward elimination
----------------------

**Backward elimination** works in reverse. Begin with the **full model** containing **all** candidate
predictors, then repeatedly drop the **least useful** one — the feature with the **highest p-value**
(least significant), or whose removal most improves the criterion — until every remaining feature
earns its place. Its advantage is that it weighs all variables **together** from the start, which can
handle **correlated** predictors more gracefully than forward selection. Its cost: it must fit the full
model, so it needs **more observations than features**.

Bidirectional stepwise
------------------------

**Bidirectional** (or plain "stepwise") selection **combines** the two. At each step it can **add** a
promising feature the way forward does, but also **re-examine** features already included and **drop**
any that have become redundant now that others are present. This flexibility corrects a weakness of
pure forward selection, where a feature admitted early can never be removed even if later additions
make it unnecessary.

Use with caution
------------------

All three are **greedy** — they explore only a sliver of the possible models and offer **no
guarantee** of the best subset. And all carry real hazards: on small samples they **overfit**, they
produce **biased** coefficient estimates, and the selected model can be **non-reproducible** — a
different sample yields a different set. Use them as **exploratory** tools when candidates are many and
theory is thin, always confirming the final model on held-out data. When you can, methods that assess
a feature's contribution more fairly — like the **Shapley values** of the next lesson — sidestep some
of these pitfalls.
"""


MINDMAP.update({
    "Forward Selection: Definition and Core Idea": [
        "Forward Selection and Model Interpretation in Linear Regression",
        "Understanding Forward and Backward Stepwise Regression",
        "Feature Importance in Linear Regression",
        "Forward Selection with Nested Models and Deviance Tests",
    ],
    "Forward Selection and Model Interpretation in Linear Regression": [
        "Forward Selection: Definition and Core Idea",
        "Understanding Forward and Backward Stepwise Regression",
        "Multiple Linear Regression", "How Shapley Values Work",
    ],
    "Understanding Forward and Backward Stepwise Regression": [
        "Forward Selection: Definition and Core Idea",
        "Forward Selection and Model Interpretation in Linear Regression",
        "Feature Importance in Linear Regression",
        "Forward Selection with Nested Models and Deviance Tests",
    ],
})


# ----------------------------------------------------------------------
# Stage 5 — Regression (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["How Shapley Values Work"] = r"""
A fair division problem
-------------------------

The feature-importance question has a surprisingly deep answer borrowed from **economics**. Imagine
several players cooperating to earn a joint payout — how should the winnings be split **fairly**,
according to each player's real contribution? Lloyd **Shapley** solved this in 1953, and the **Shapley
value** he defined turns out to be exactly what is needed to attribute a model's **prediction** to its
features.

Marginal contributions
------------------------

The idea rests on **marginal contribution**. A player's contribution to a group (a **coalition**) is
how much the payout **grows** when they join it: :math:`v(S \cup \{i\}) - v(S)` for a coalition
:math:`S`. But that depends on **who is already there** — a player may add a lot to a small group and
little to a large one. The Shapley value resolves this by **averaging** a player's marginal
contribution over **every** possible coalition (equivalently, every order in which players could join).

The formula
-------------

Written out, the Shapley value of feature :math:`i` among :math:`n` features is that weighted average:

.. math::

   \phi_i = \sum_{S \subseteq N \setminus \{i\}}
            \frac{|S|!\,(n - |S| - 1)!}{n!}\,\bigl[v(S \cup \{i\}) - v(S)\bigr].

The weights count the orderings, so each coalition is credited correctly. What makes the Shapley value
special is that it is the **unique** scheme satisfying four fairness axioms: **efficiency** (the parts
sum to the whole), **symmetry** (equal contributors get equal shares), **dummy** (a feature that
changes nothing gets zero), and **additivity**.

Explaining predictions
------------------------

In machine learning the analogy is exact: the **features are the players** and the **prediction is the
payout**. The Shapley value of a feature is how much it pushed *this* prediction **above or below** the
average prediction — a fair, model-agnostic attribution that works for any model, regression or
classification. The popular **SHAP** framework (2017) builds on this, decomposing a prediction into
feature contributions that **sum** to the output. The catch is cost: exact values require all
:math:`2^n` coalitions, so in practice they are **approximated**. Unlike the greedy selection of
earlier lessons, Shapley values weigh every feature **fairly against all others**.
"""


# ----------------------------------------------------------------------
# Stage 6 — Classification & Logistic Regression
# ----------------------------------------------------------------------

CONTENT["Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds"] = r"""
When the outcome is yes or no
-------------------------------

Regression so far has predicted a **number** — a fare, a price. But many outcomes are **binary**: will
a customer **churn** or not? will a student **return** next year? Ordinary linear regression fails
here, because a straight line runs off to :math:`\pm\infty` and would predict "probabilities" **below
0 or above 1**. **Logistic regression** is the standard model for a **yes/no** outcome, and it works by
predicting a **probability** instead.

Odds and log-odds
-------------------

The trick is to transform the probability so a linear model fits. Start with the **odds** — the ratio
of the probability of the event to its complement, :math:`p / (1 - p)` — which stretches :math:`[0, 1]`
out to :math:`[0, \infty)`. Then take the logarithm, giving the **log-odds** or **logit**, which spans
**all** real numbers. Logistic regression makes *this* linear in the features:

.. math::

   \ln\!\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p.

The model is an ordinary linear equation — just on the **log-odds scale** rather than the probability
scale.

The logistic curve
--------------------

To get a probability back, invert the transform. Solving for :math:`p` gives the **logistic**
(sigmoid) function:

.. math::

   p = \frac{1}{1 + e^{-z}}, \qquad z = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p.

This S-shaped curve takes the linear combination :math:`z` — any real number — and squashes it
smoothly into a valid probability between 0 and 1. Large positive :math:`z` gives :math:`p` near 1,
large negative near 0, and :math:`z = 0` gives :math:`p = 0.5`.

Reading coefficients
----------------------

The coefficients read on the **odds** scale. A one-unit rise in :math:`x_j` adds :math:`\beta_j` to the
log-odds, which **multiplies** the odds by :math:`e^{\beta_j}` — the **odds ratio**. So
:math:`e^{\beta_j} > 1` means the feature raises the odds of the event, :math:`< 1` lowers them. In
Python it is ``LogisticRegression`` in scikit-learn, or ``Logit`` in statsmodels for the full
coefficient table. Unlike least squares, its coefficients have **no closed form** — they are found by
**maximum likelihood**, the subject of the next lesson.
"""

CONTENT["Maximum Likelihood (MLE): Fitting a Distribution to Observed Data"] = r"""
What parameters best explain the data?
----------------------------------------

How does logistic regression actually **choose** its coefficients, when there is no closed-form
formula? The answer is a principle general enough to fit almost any model: **maximum likelihood
estimation** (MLE). Its question is simple and intuitive — *of all possible parameter values, which
ones make the* **data I actually observed** *most probable?*

The likelihood
----------------

The key object is the **likelihood**. For a candidate set of parameters :math:`\theta`, the likelihood
:math:`L(\theta)` is the probability of the observed data **computed under those parameters** — but
read as a function of :math:`\theta`, with the data held fixed. A parameter value under which the
observed data would be very **improbable** has low likelihood; one under which the data looks
**typical** has high likelihood. Because independent observations multiply, the likelihood is a
**product** of per-observation probabilities, and it is usually easier to work with its logarithm, the
**log-likelihood**, which turns the product into a sum without moving the maximum.

Maximising it
---------------

**Maximum likelihood** simply picks the :math:`\theta` that makes :math:`L(\theta)` — or the
log-likelihood — as **large as possible**. For simple cases this has a tidy answer: the MLE of a normal
distribution's mean is just the **sample mean**, and of a coin's bias the **observed proportion** of
heads. For models like logistic regression there is **no formula**, so the maximum is found
**numerically**, by iterative optimisation that climbs the log-likelihood to its peak.

Why it matters here
---------------------

MLE ties much of this course together. Logistic regression's coefficients **are** the maximum-likelihood
estimates. Least squares is itself MLE in disguise — minimising squared errors is *exactly* maximising
likelihood when the errors are **normally distributed**. And the precision of these estimates is
governed by the **Cramér–Rao bound** from Stage 2, which sets the best variance any unbiased estimator
can reach. The next lessons use likelihood again — to **assess** how well a fitted logistic model fits.
"""


MINDMAP.update({
    "How Shapley Values Work": [
        "Feature Importance in Linear Regression",
        "Understanding Forward and Backward Stepwise Regression",
        "Multiple Linear Regression", "Assessing the Quality of Prediction Models",
    ],
    "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds": [
        "Maximum Likelihood (MLE): Fitting a Distribution to Observed Data",
        "Assessing Model Fit in Logistic Regression",
        "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics",
        "Complete and Quasi-Complete Separation in Logistic Regression",
    ],
    "Maximum Likelihood (MLE): Fitting a Distribution to Observed Data": [
        "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds",
        "Least Squares Regression", "Harald Cram\u00e9r",
        "Assessing Model Fit in Logistic Regression",
    ],
})


# ----------------------------------------------------------------------
# Stage 6 — Classification & Logistic Regression (cont.)
# ----------------------------------------------------------------------

CONTENT["Assessing Model Fit in Logistic Regression"] = r"""
No R² to lean on
------------------

In ordinary regression, :math:`R^2` gives a quick read of fit — the fraction of variance explained.
Logistic regression has **no such natural measure**, because it predicts probabilities, not a
continuous quantity with variance to partition. Assessing a logistic model's fit therefore takes a
different toolkit, built on the **likelihood** the model was fitted to maximise.

Likelihood-based fit
----------------------

The foundation is the **log-likelihood** of the fitted model — higher means the model assigns greater
probability to what was actually observed. A closely related quantity is the **deviance**, defined as
:math:`-2` times the log-likelihood; **lower** deviance means **better** fit. On their own these
numbers are not interpretable, but **compared** — fitted model against a baseline — they become the
basis of every fit measure below.

Pseudo-R²
-----------

To mimic the familiar 0-to-1 feel of :math:`R^2`, several **pseudo-**:math:`R^2` measures compare the
fitted model's likelihood to that of the **null** (intercept-only) model. The most widely recommended
is **McFadden's**:

.. math::

   R^2_{\text{McF}} = 1 - \frac{\ln L_{\text{model}}}{\ln L_{\text{null}}}.

It is 0 when the predictors add nothing and approaches 1 as the fitted model explains the data far
better. It is **not** the proportion of variance explained — McFadden values of **0.2–0.4** already
indicate a very good fit — so it is read as a *relative* measure among models, not on the OLS scale.
(Cox–Snell, Nagelkerke and Efron's are common alternatives.)

Comparing to the null
-----------------------

That comparison can be made a formal test. The **likelihood-ratio test** asks whether the fitted model
fits **significantly** better than the null, using the drop in deviance as a **chi-square** statistic;
a significant result means the predictors, taken together, genuinely contribute. And to balance fit
against complexity across candidate models, **AIC** and **BIC** combine the log-likelihood with a
penalty for the number of predictors — the same parsimony principle from the selection lessons. The
next lesson turns this deviance-comparison idea into a tool for **choosing** features.
"""

CONTENT["Complete and Quasi-Complete Separation in Logistic Regression"] = r"""
When the fit blows up
-----------------------

Occasionally a logistic regression **refuses to behave**: coefficients balloon to absurd sizes,
standard errors explode, and the software prints a **non-convergence** warning. The usual culprit is a
specific, well-understood pathology called **separation** — and understanding it explains both the
symptom and the cure.

Perfect separation
--------------------

**Complete separation** happens when a predictor (or combination of predictors) **perfectly predicts**
the outcome — when there is a boundary with **all** the 1s on one side and **all** the 0s on the other,
no exceptions. For instance, if every student who studied more than 50 hours passed and every one who
studied less failed, hours perfectly separates the classes. **Quasi-complete separation** is the same,
except a few points sit **exactly on** the boundary. Both are more common in **small samples** and with
**rare** outcomes.

Why MLE breaks
----------------

Why does this wreck the fit? Recall the model estimates coefficients by **maximum likelihood**. When
the data are perfectly separated, the likelihood can **always** be increased by making the coefficient
**larger** — pushing the predicted probabilities ever closer to a flawless 0 and 1. There is no value
at which the likelihood peaks; the estimate wants to run off to **infinity**. The maximum-likelihood
estimate simply **does not exist**, which is why the optimiser never converges and the coefficients
diverge.

Spotting and fixing it
------------------------

The **symptoms** are unmistakable: enormous coefficients with enormous standard errors, fitted
probabilities of exactly 0 or 1, and convergence failures. The **fixes** address the root cause. A
**penalised** logistic regression — most notably **Firth's method**, which nudges the estimates toward
zero — always yields **finite** coefficients, even under complete separation; ordinary
**regularisation** (an L2 penalty) does the same. Alternatively, the offending predictor can be
**combined or removed**, or a rare category merged. The lesson generalises: an estimate that runs to
infinity is the data telling you the model, as posed, is **not identifiable**.
"""

CONTENT["Forward Selection with Nested Models and Deviance Tests"] = r"""
Selecting features by deviance
--------------------------------

The **deviance** — that :math:`-2` log-likelihood measure of misfit — does more than grade a single
model. Its real power is in **comparing** models, and it is the natural tool for **feature selection**
in logistic regression, playing the role that the F-test and residual sum of squares play in linear
regression.

Nested models
---------------

The comparison requires the models to be **nested**: one model's predictors must be a **subset** of the
other's, so the smaller (reduced) model is a special case of the larger (full) one. Adding predictors
can only **lower** the deviance (improve the fit on the training data), so the full model **always** has
deviance less than or equal to the reduced model. The question is whether that improvement is **real**
or just the inevitable reward of extra parameters.

The deviance test
-------------------

The answer is the **likelihood-ratio (deviance) test**. The **difference** in deviance between the
reduced and full models is itself a statistic that, under the hypothesis that the extra features add
nothing, follows a **chi-square** distribution — with degrees of freedom equal to the **number of
added parameters**:

.. math::

   \Delta D = D_{\text{reduced}} - D_{\text{full}} \sim \chi^2_{\,k}.

A **large** deviance drop is unlikely by chance, so a significant test means the added features
**genuinely improve** the model; a small, non-significant drop means they can be dropped. This
procedure is often called an **analysis of deviance**, the logistic cousin of analysis of variance.

Forward selection, revisited
------------------------------

This gives forward selection a principled **entry rule** for logistic models. Start from the null
model and, at each step, consider adding each remaining feature; admit the one whose deviance drop is
**largest and significant** by the chi-square test. Stop when no candidate produces a significant
improvement. It is the same greedy search as before, now driven by **deviance** rather than an
F-statistic — and with the same caution that data-driven selection inflates significance, so the final
model earns its keep only on **held-out** data. The next lesson works a full example on real data.
"""


MINDMAP.update({
    "Assessing Model Fit in Logistic Regression": [
        "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds",
        "Maximum Likelihood (MLE): Fitting a Distribution to Observed Data",
        "Forward Selection with Nested Models and Deviance Tests",
        "Assessing the Quality of Prediction Models",
    ],
    "Complete and Quasi-Complete Separation in Logistic Regression": [
        "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds",
        "Maximum Likelihood (MLE): Fitting a Distribution to Observed Data",
        "Assessing Model Fit in Logistic Regression",
        "Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention",
    ],
    "Forward Selection with Nested Models and Deviance Tests": [
        "Assessing Model Fit in Logistic Regression",
        "Understanding Forward and Backward Stepwise Regression",
        "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds",
        "Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention",
    ],
})


# ----------------------------------------------------------------------
# Stage 6 — Classification & Logistic Regression (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention"] = r"""
A real prediction problem
---------------------------

To see the classification tools work together, take a problem colleges genuinely care about: **student
retention** — will an enrolled student **return** the following year, or drop out? The outcome is
**binary** (retained / not), making it a textbook job for **logistic regression**, and because
institutions want to **understand** the drivers (not just predict), the model's interpretability is as
valuable as its accuracy.

Building the model
--------------------

Start with a pool of candidate predictors an institution has on hand — **prior GPA**, first-term
credits and grades, **entrance-exam scores**, financial aid, and engagement measures. **Forward
selection** with the deviance test (previous lesson) admits predictors one at a time, keeping each only
if it produces a **significant** drop in deviance. The result is a **parsimonious** model — a handful
of variables that together explain retention, easier to act on than a model burdened with every field
in the database.

Reading the odds ratios
-------------------------

Interpretation runs through **odds ratios**, the :math:`e^{\beta}` from the logistic lessons. A
coefficient on GPA might give an odds ratio of, say, 2 — meaning each additional grade point roughly
**doubles** the odds of returning, holding the other predictors fixed. Predictors with odds ratios
**above 1** raise the odds of retention (protective factors); those **below 1** lower them (risk
factors). This is what makes the model **actionable**: it points to *which* students are at risk and
*why*, so support can be targeted — the prescriptive payoff the course opened with.

Does it fit?
--------------

Finally, **assess** the fitted model. **McFadden's pseudo-**:math:`R^2` and the overall
**likelihood-ratio test** gauge whether the predictors collectively explain retention; **classification
accuracy** and the **ROC / AUC** of the next stage measure how well it separates returners from
leavers — all judged on **held-out** students, never the training data. Studies of retention routinely
reach AUCs in the high 70s to high 80s. The result is a model that is both **interpretable** and
**validated** — the goal of this whole stage, and a natural bridge to the trees that follow, which
pursue the same predictions with a very different, rule-based structure.
"""


# ----------------------------------------------------------------------
# Stage 7 — Decision Trees
# ----------------------------------------------------------------------

CONTENT["Motivation of Decision Trees: An Incremental Model of Decision-Making"] = r"""
How people decide
-------------------

Humans often decide by asking a **sequence of simple questions**. To triage a patient: *Is there chest
pain? If so, is it severe? Radiating to the arm?* Each answer narrows the possibilities until a
decision is reached. A **decision tree** formalises exactly this — an **incremental** model that
reaches a prediction by asking one yes/no question at a time. Its appeal is that it thinks the way
people do.

A tree of questions
---------------------

Structurally, a decision tree is a **flowchart**. Each **internal node** poses a test on one feature
("distance > 3 miles?"); each **branch** is an answer that leads onward; and each **leaf** delivers a
**prediction** — a class for classification, a number for regression. To predict for a new
observation, you start at the **root** and follow the branches its feature values dictate until you
land in a leaf. The **path** from root to leaf reads as a plain chain of if-then rules.

What trees are good at
------------------------

This structure has real strengths. Trees capture **non-linear** relationships and **interactions**
between features **automatically** — a split on one feature can lead to different splits on another, so
the effect of one variable can depend on another with no special terms. They handle **numeric and
categorical** features side by side, need **no scaling** or standardisation, and are unbothered by the
**linearity** assumptions that constrain regression. They cope naturally with the messiness real data
brings.

White-box models
------------------

Above all, trees are **interpretable** — a "white-box" model whose every decision can be traced and
explained, in contrast to "black-box" methods like neural networks. You can **read** a tree, show it to
a domain expert, and check whether its logic makes sense. That transparency is why trees are a
favourite when a decision must be **justified**, not merely made — and it sets up their use later in the
stage to **explain** the clusters of Stage 4. The next lesson gives the algorithm that actually builds
a tree from data: **CART**.
"""

CONTENT["The CART Algorithm"] = r"""
Growing a tree from data
--------------------------

How is a tree actually built? The standard method is **CART** — **Classification And Regression
Trees** — introduced by Breiman and colleagues in **1984**. Given labelled data, CART grows the tree
**top-down** and **greedily**: starting from all the data at the root, it repeatedly finds the **best
single split** and divides the node, then recurses on each resulting piece.

Recursive binary splits
-------------------------

Every CART split is **binary** — a node divides into exactly **two** children. For a numeric feature
the split is a threshold ("distance ≤ 3 miles?"); for a categorical one, a grouping of categories. At
each node the algorithm searches **all features and all candidate split points** and picks the single
split that best **separates** the outcomes, then applies the same procedure to each child. This
**recursion** continues down every branch, growing the tree one split at a time.

Measuring impurity
--------------------

"Best" is defined by **impurity** — a measure of how **mixed** the outcomes are within a node. A node
of all-retained students is **pure** (impurity 0); a fifty-fifty mix is maximally impure. CART chooses
the split that most **reduces** impurity across the two children. For **classification** the usual
measure is the **Gini impurity** (or entropy); for **regression**, the node's **variance** (mean
squared error). Each split is the one that makes the children as **homogeneous** as possible, echoing
the "similar within" goal of clustering.

Knowing when to stop
----------------------

Left unchecked, CART will split until every leaf is pure — a tree that **memorises** the training data
and **overfits** badly, the danger from the partitioning lesson. Two remedies rein it in. **Stopping
rules** halt growth early (a maximum **depth**, or a minimum number of samples per leaf). More
principled is **pruning**: grow a large tree, then **cut back** the branches that add little predictive
value, using **cost-complexity pruning** to trade size against accuracy. In scikit-learn,
``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` implement CART with both controls. The next
lessons read what a fitted tree **means**.
"""


MINDMAP.update({
    "Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention": [
        "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds",
        "Forward Selection with Nested Models and Deviance Tests",
        "Assessing Model Fit in Logistic Regression",
        "AUC\u2013ROC Curve: Evaluating Classification Model Performance",
    ],
    "Motivation of Decision Trees: An Incremental Model of Decision-Making": [
        "The CART Algorithm",
        "Decision Trees as Piecewise Models and Their Predictive Structure",
        "How CART Decision Trees Model Interactions",
        "Cluster Profiling Using Decision Trees",
    ],
    "The CART Algorithm": [
        "Motivation of Decision Trees: An Incremental Model of Decision-Making",
        "Decision Trees as Piecewise Models and Their Predictive Structure",
        "How CART Decision Trees Model Interactions", "Clustering",
    ],
})


# ----------------------------------------------------------------------
# Stage 7 — Decision Trees (cont.)
# ----------------------------------------------------------------------

CONTENT["Decision Trees as Piecewise Models and Their Predictive Structure"] = r"""
Boxes in feature space
------------------------

A decision tree has a clean **geometric** meaning. Every split is a test on **one** feature against a
threshold ("distance ≤ 3?"), so each split is a straight cut **parallel to an axis** of the feature
space. A sequence of such cuts carves the space into **rectangular regions** — boxes (or, in higher
dimensions, hyper-rectangles). Crucially, **each leaf of the tree corresponds to exactly one
region**: the set of points whose feature values follow that root-to-leaf path.

A constant in each box
------------------------

Inside each region, the tree makes a **single, constant prediction**. For a **regression** tree it is
the **mean** of the training outcomes that fell in that box; for a **classification** tree, the
**majority class**. So predicting for a new point is just: find which box it lands in, and return that
box's constant. The tree is, in effect, a **lookup table** over a partition of the feature space.

A step-function model
-----------------------

This makes a tree a **piecewise-constant** model. A regression tree's prediction surface is not a
smooth line or plane but a series of **flat steps** — constant within each box, jumping at the
boundaries. Algebraically it is a sum over regions,
:math:`T(x) = \sum_{j} \gamma_j \, \mathbb{I}(x \in R_j)`, where each :math:`R_j` is a box and
:math:`\gamma_j` its constant prediction. This is a sharp contrast to linear regression's single
continuous slope.

Flexible despite steps
------------------------

A staircase might seem **too crude** to model a smooth relationship — but with enough boxes, a
piecewise-constant function can **approximate any shape**, including the **non-linear** and
**non-monotone** forms that linear or additive models cannot represent. That flexibility is the tree's
power, and also its peril: enough boxes to fit any curve is also enough to fit the **noise**, which is
why the depth controls and pruning of the last lesson matter. The next lesson shows a subtler
consequence of this box structure: **interactions**.
"""

CONTENT["How CART Decision Trees Model Interactions"] = r"""
What is an interaction?
-------------------------

An **interaction** occurs when the effect of one feature on the outcome **depends on the value of
another**. Airport surcharges might make **distance** matter more for **airport** trips than for
others; a drug might help one age group and harm another. In each case you cannot describe the effect
of one variable **without knowing** the other — the two **interact**.

The regression chore
----------------------

Ordinary linear regression **cannot see** interactions on its own. Its form is strictly **additive** —
each feature contributes its coefficient times its value, independently — so the effect of one feature
is the **same** regardless of the others. To model an interaction you must **manually add** a product
term (:math:`x_1 \times x_2`), and you have to **know in advance** which interactions to include. Miss
one, and the model is blind to it.

Trees get them free
---------------------

Decision trees capture interactions **automatically**, as a byproduct of their structure. Because
splits are **nested**, a split on one feature can be **followed by different splits** on another in
different branches — so the effect of the second feature genuinely **differs** depending on the first.
A tree might split on trip type, then split on distance **only** in the airport branch: exactly an
interaction between type and distance, discovered without anyone specifying it. Each split is an effect
**conditional** on all the splits above it.

Depth and order
-----------------

This is why tree **depth** matters for expressiveness. A one-split (depth-1) tree captures only a
single feature's main effect; each additional level lets the tree condition on **one more** feature, so
deep trees can represent **high-order** interactions among many variables. The first split naturally
falls on the feature with the strongest overall (main) effect, with interactions emerging **below** it.
This effortless interaction modelling — together with the piecewise structure of the last lesson — is
what lets a single tree describe complex, realistic patterns, and it powers the cluster-explanation
uses that close this stage.
"""

CONTENT["Cluster Profiling Using Decision Trees"] = r"""
Clusters without descriptions
-------------------------------

Clustering (Stage 4) hands you **groups**, but not their **meaning**. k-means labels each customer with
a cluster number, yet those numbers are **opaque** — cluster 3 is just "cluster 3". Before you can act
on segments, you need to **describe** them: what actually distinguishes cluster 3's members from
everyone else? This is **cluster profiling**, and decision trees are an elegant way to do it.

Turn labels into a target
---------------------------

The trick is to **turn the unsupervised result into a supervised problem**. Take the **cluster label**
each point received and treat it as the **target** to predict, using the original features as inputs.
Then **fit a decision tree** to predict cluster membership. The clustering supplied the "answers"; the
tree's job is to find the **rules** that reproduce them.

Rules that define a cluster
-----------------------------

Because a tree is a chain of if-then splits (this stage's opening lessons), the fitted tree **reads as
a description** of the clusters. The path to a leaf dominated by cluster 3 might say: *recency < 30
days* **and** *frequency > 10* — a plain-language **profile** of that segment. Each cluster gets a
compact set of defining conditions, turning anonymous group numbers into **interpretable**
characterisations a business can name and target — "recent frequent buyers", say.

Why it works
--------------

This works because trees bring exactly the right strengths: they are **interpretable** (the whole point
here), they handle **mixed** feature types and interactions without fuss, and they naturally identify
**which** features separate the groups — a built-in importance ranking. It is a recurring pattern in
machine learning: use a **transparent** model to **explain** the output of an opaque one. The final
lesson of this stage takes the idea one step further — using a tree as a **surrogate** to explain
clustering results in general.
"""


MINDMAP.update({
    "Decision Trees as Piecewise Models and Their Predictive Structure": [
        "The CART Algorithm", "How CART Decision Trees Model Interactions",
        "Motivation of Decision Trees: An Incremental Model of Decision-Making",
        "Multiple Linear Regression",
    ],
    "How CART Decision Trees Model Interactions": [
        "Decision Trees as Piecewise Models and Their Predictive Structure",
        "The CART Algorithm", "Multiple Linear Regression",
        "Cluster Profiling Using Decision Trees",
    ],
    "Cluster Profiling Using Decision Trees": [
        "Using Decision Trees to Explain Clustering Results", "Clustering",
        "Creating Segments of Observations for Business Reasons (RFM)",
        "The CART Algorithm",
    ],
})


# ----------------------------------------------------------------------
# Stage 7 — Decision Trees (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Using Decision Trees to Explain Clustering Results"] = r"""
Explaining a black box
------------------------

The last lesson **profiled** individual clusters; this one generalises the idea into one of machine
learning's most useful tools. Clustering algorithms — and many models besides — are **opaque**: k-means
or a neural network produces outputs, but not an **explanation** of its logic. A **surrogate model** is
a deliberately **simple, interpretable** model trained to **mimic** the opaque one, so that its
transparent logic can stand in for the black box's hidden one.

The surrogate idea
--------------------

The recipe is general. Take whatever the opaque method produced — here, the **cluster label** for each
observation — and treat those outputs as the **target**. Then fit an interpretable model, most
naturally a **decision tree**, to reproduce them from the features. The tree learns the **rules** that
best explain the clustering as a whole, and because it is a tree, those rules can be **read**: a
compact, global description of *how* the clustering carved up the data. This is a **global surrogate**
— it approximates the whole model's behaviour, not just one prediction — and it is **model-agnostic**,
needing no knowledge of how the clustering worked inside.

Fidelity
----------

The obvious question is: can you **trust** the surrogate? A tree that mimics the clustering only
loosely would give a misleading explanation. The measure of trust is **fidelity** — how **faithfully**
the surrogate reproduces the original's outputs, typically the surrogate's **accuracy** at predicting
the black box's labels. **High** fidelity means the tree's rules genuinely reflect the clustering;
**low** fidelity means the explanation is unreliable. There is a real **tension** here: a deeper tree
fits the clustering more faithfully but is harder to read, so surrogates trade **fidelity against
interpretability**, and a shallow, high-fidelity tree is the happy case.

A general pattern
-------------------

This surrogate trick is far bigger than clustering. The **same** move — fit an interpretable model to a
complex model's outputs, then check its fidelity — explains **black-box classifiers, regressors and
ensembles** too, and it complements the per-prediction attributions of **Shapley values** from Stage 5.
It closes the trees stage on a fitting note: the decision tree, prized for its transparency, becomes a
**lens** for seeing into models that have none. With the workflow's models built and understood, the
final stage asks the essential question — **how good are they?**
"""


# ----------------------------------------------------------------------
# Stage 8 — Model Evaluation
# ----------------------------------------------------------------------

CONTENT["Assessing the Quality of Prediction Models"] = r"""
The essential question
------------------------

A model has been built — a regression, a classifier, a tree. The question that decides whether anyone
should **trust** it is simple: **how good is it?** Model evaluation is the discipline of answering this
**rigorously**, with numbers rather than hope. It is the last stage of the workflow and, for
**scikit-plots** especially, its centre of gravity — most of the library's charts exist to
**visualise** exactly these answers.

It depends on the task
------------------------

There is **no single measure** of quality, because it depends on the **kind** of prediction. For
**regression** — predicting a number — quality means small errors, captured by measures like **mean
squared error** (MSE), its square-root **RMSE**, **mean absolute error** (MAE), and the familiar
:math:`R^2`. For **classification** — predicting a class — it means getting labels right, captured by
**accuracy**, **precision**, **recall** and the **AUC** of the lessons ahead. The right family of
metrics follows from the problem type.

On held-out data
------------------

One rule overrides all the metric choices: quality is measured on **held-out test data**, never the
data the model was trained on. A model's score on its **training** set flatters it — it may simply have
memorised those examples (the overfitting of Stage 4). Only performance on **unseen** data estimates
how the model will behave in the real world, which is the entire reason for the partitioning discipline
established earlier. This is the course's standing warning, and it applies to every metric below.

Match metric to goal
----------------------

Even within a task, the **best** metric depends on the **business goal**. High accuracy can hide
disastrous behaviour on a **rare** but crucial class; a fraud detector that flags nothing is 99.9%
accurate and useless. So the metric must mirror what actually **matters** — catching positives,
avoiding false alarms, ranking well — and results should be judged against a sensible **baseline**
rather than in the abstract. The remaining lessons build the concrete tools: the confusion matrix and
its metrics, ROC and lift curves, threshold tuning, and residual diagnostics.
"""

CONTENT["Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics"] = r"""
Four kinds of outcome
-----------------------

A binary classifier predicts one of two labels — **positive** or **negative** (churn / not, fraud /
not). Comparing its prediction to the truth, every case falls into one of **four** outcomes: a **true
positive** (predicted positive, really positive), a **true negative** (predicted negative, really
negative), a **false positive** (predicted positive, actually negative — a false alarm), and a **false
negative** (predicted negative, actually positive — a miss). These four counts are the raw material for
**every** binary-classification metric.

The confusion matrix
----------------------

Arranged in a 2×2 table — predicted versus actual — the four counts form the **confusion matrix**, the
single most useful summary of a classifier's behaviour. Its diagonal (**TP** and **TN**) holds the
**correct** predictions; its off-diagonal (**FP** and **FN**) the **errors**. Reading the matrix shows
not just **how often** the model is wrong, but **in which direction** — whether it tends to over-predict
or under-predict the positive class. scikit-plots draws it directly with a labelled confusion-matrix
plot.

Metrics from the matrix
-------------------------

The headline metrics are all **ratios** of these four counts, each answering a different question:

* **Accuracy** :math:`= \frac{TP + TN}{TP + TN + FP + FN}` — overall, what fraction is correct?
* **Precision** :math:`= \frac{TP}{TP + FP}` — of those **predicted** positive, how many truly are?
* **Recall** (sensitivity) :math:`= \frac{TP}{TP + FN}` — of the **actual** positives, how many were caught?
* **F1** :math:`= 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}` — the harmonic mean, balancing the two.

Precision and recall pull in **opposite** directions — catching more positives (higher recall) usually
means more false alarms (lower precision) — so F1 is a common single-number compromise.

Which error costs more?
-------------------------

The crucial judgement is that the two errors are **not equally bad**, and which matters more is
**domain-specific**. Missing a cancer diagnosis (a **false negative**) is far graver than a false
alarm; flagging a legitimate transaction as fraud (a **false positive**) annoys a customer but harms
less than letting fraud through. **Accuracy** alone hides this, and is especially **misleading** on
**imbalanced** data — which is why precision, recall and the threshold-based tools of the next lessons
exist. The choice of which count to minimise is where evaluation meets the **business goal**.
"""


MINDMAP.update({
    "Using Decision Trees to Explain Clustering Results": [
        "Cluster Profiling Using Decision Trees", "Clustering",
        "How Shapley Values Work", "The CART Algorithm",
    ],
    "Assessing the Quality of Prediction Models": [
        "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics",
        "Partitioning Observations to Train Objective Models",
        "AUC\u2013ROC Curve: Evaluating Classification Model Performance",
        "Assessing Model Fit in Logistic Regression",
    ],
    "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics": [
        "Assessing the Quality of Prediction Models",
        "Binary Classification Model Evaluation and Threshold Optimization",
        "AUC\u2013ROC Curve: Evaluating Classification Model Performance",
        "Nominal Classification Models: Model State and Evaluation Metrics",
    ],
})


# ----------------------------------------------------------------------
# Stage 8 — Model Evaluation (cont.)
# ----------------------------------------------------------------------

CONTENT["Nominal Classification Models: Model State and Evaluation Metrics"] = r"""
Beyond two classes
--------------------

Many classification problems have **more than two** unordered labels — payment type (cash / card /
mobile), a product category, a species. These are **nominal** (multi-class) problems: the classes have
**no natural order**, and every prediction must pick one of :math:`K` labels. Evaluation carries over
from the binary case, but with two twists worth understanding.

The K×K confusion matrix
--------------------------

The confusion matrix generalises directly: it becomes a :math:`K \times K` table, actual classes down
the rows and predicted classes across the columns. The **diagonal** still holds the correct
predictions, and every **off-diagonal** cell now names a **specific confusion** — how often class A was
mistaken for class B. This is the matrix's great virtue in multi-class work: it shows **which pairs**
of classes the model muddles (cash vs mobile, say), pointing directly at what to fix. scikit-plots
renders it as the same labelled heat-map plot, just larger.

Per-class metrics
-------------------

Precision, recall and F1 are **binary** notions, so for :math:`K` classes they are computed **per
class**, one-vs-rest: for class A, treat A as "positive" and everything else as "negative", and read
TP, FP and FN from the matrix's row and column for A. Each class gets its own precision (of the
predictions of A, how many were right?) and recall (of the true A's, how many were found?) —
scikit-learn's ``classification_report`` prints exactly this table, one row per class.

Averaging: macro, micro, weighted
-----------------------------------

To summarise the per-class scores in **one number**, three averages are standard, and they answer
**different questions**:

* **Macro** — the plain mean of the per-class scores: every **class** counts equally, however rare, so it exposes weakness on small classes;
* **Weighted** — the mean weighted by class **frequency**: large classes dominate, mirroring overall behaviour;
* **Micro** — pool all the TP / FP / FN counts first, then compute: every **instance** counts equally. For single-label problems micro-precision, micro-recall and accuracy all **coincide**.

On **imbalanced** data the choice matters: a model useless on a rare class can still post a high
weighted or micro score, while its **macro** score collapses — so pick the average that matches whether
classes or instances are what the business weighs equally.
"""

CONTENT["Binary Classification Model Evaluation and Threshold Optimization"] = r"""
The hidden dial
-----------------

A binary classifier rarely outputs a label directly — it outputs a **probability** (or score), like the
logistic model's :math:`p`. The label comes from a **threshold**: predict positive when the score
exceeds a **cutoff**, conventionally **0.5**. That cutoff is a **dial**, not a law of nature — and
turning it changes every metric of the last lessons without retraining anything. Choosing it well is
**threshold optimisation**.

Turning the dial
------------------

The trade-off is mechanical. **Lower** the threshold and more cases are called positive: **recall
rises** (fewer misses) but **precision falls** (more false alarms). **Raise** it and the reverse —
predictions become conservative, precision improves, recall drops. The confusion matrix **morphs**
continuously as the dial turns, trading false negatives for false positives. No single threshold is
"correct"; each is a different **operating point** on the same fitted model.

Cost decides
--------------

The right operating point comes from the **relative cost of the two errors** — the domain judgement
from the confusion-matrix lesson, now made operational. If a **miss** is expensive (fraud, disease
screening), push the threshold **down** and accept false alarms — a fraud team might demand recall
above 0.85 and tune precision after. If a **false alarm** is expensive (spam filters binning real
mail), push it **up**. When costs can be written down, the threshold can even be chosen to **minimise
expected cost** directly.

Choosing it in practice
-------------------------

Practically, you **sweep** the threshold on validation data and plot the metrics against it: precision
and recall curves crossing, F1 peaking, or expected cost bottoming out. Pick the threshold that
optimises the criterion you care about — the F1-maximising point, the smallest threshold hitting a
target recall, or the cost minimum. scikit-plots' evaluation charts exist for exactly this sweep, and
the next lessons introduce the two most important of them: the **ROC curve**, which displays *every*
operating point at once, and the **lift** chart, which ranks rather than cuts.
"""

CONTENT["Identifying Outliers Using Residuals and Studentized Residuals"] = r"""
Evaluating regression fits
----------------------------

For regression models, evaluation is not only an average error — it is also asking whether **individual
points** are being fit sensibly. The tool is the **residual**, :math:`e_i = y_i - \hat{y}_i`, familiar
from least squares. A point whose residual is **far larger** than the rest is an **outlier** in the
response — a case the model badly mispredicts — and finding these is a core diagnostic, one that
scikit-plots visualises with residual plots.

Raw residuals mislead
-----------------------

Raw residuals are an awkward yardstick, for two reasons. Their size depends on the **units** of
:math:`y`, so "large" has no absolute meaning. Worse, they do **not share a common variance**: the
variance of :math:`e_i` is :math:`\sigma^2 (1 - h_{ii})`, where :math:`h_{ii}` is the point's
**leverage** — how unusual its feature values are. A **high-leverage** point *pulls the fitted line
toward itself*, artificially **shrinking** its own residual. The very points most able to distort the
fit are the ones whose raw residuals look most innocent.

Standardising
---------------

The first fix is the **standardized** (internally studentized) residual — the raw residual divided by
its own estimated standard deviation:

.. math::

   r_i = \frac{e_i}{\hat{\sigma}\,\sqrt{1 - h_{ii}}}.

Now every point is on a **common scale** of standard-deviation units, comparable across observations
and datasets, with :math:`|r_i| > 3` a common flag for an outlier.

Studentized residuals
-----------------------

One subtlety remains: a truly extreme point **inflates** :math:`\hat{\sigma}` itself, partially
**masking** its own residual. The **studentized** (externally studentized, or *deleted*) residual
removes the circularity by estimating the error scale **without** observation :math:`i` — refit the
model leaving the point out, and scale by that :math:`\hat{\sigma}_{(i)}`:

.. math::

   t_i = \frac{e_i}{\hat{\sigma}_{(i)}\,\sqrt{1 - h_{ii}}}.

Under the usual assumptions :math:`t_i` follows a **t-distribution**, so the flag becomes a genuine
**statistical test**. In practice, :math:`|t_i| > 2` marks a point worth examining and
:math:`|t_i| > 3` a strong outlier — first check for a **data error**; if the value is real, consider a
robust fit or report it as a notable exception. And when **many** points flag at once, the message is
usually not "bad data" but a **misspecified model** — a missing curve or interaction. Residual
diagnostics evaluate the *model* as much as the points.
"""


MINDMAP.update({
    "Nominal Classification Models: Model State and Evaluation Metrics": [
        "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics",
        "Assessing the Quality of Prediction Models",
        "Binary Classification Model Evaluation and Threshold Optimization",
        "AUC\u2013ROC Curve: Evaluating Classification Model Performance",
    ],
    "Binary Classification Model Evaluation and Threshold Optimization": [
        "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics",
        "AUC\u2013ROC Curve: Evaluating Classification Model Performance",
        "Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value",
        "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds",
    ],
    "Identifying Outliers Using Residuals and Studentized Residuals": [
        "Least Squares Regression", "Multiple Linear Regression",
        "Assessing the Quality of Prediction Models",
        "The First Step in Knowing Your Data",
    ],
})


# ----------------------------------------------------------------------
# Stage 8 — Model Evaluation (cont.)
# ----------------------------------------------------------------------

CONTENT["Nominal Classification Models: Model State and Evaluation Metrics"] = r"""
Beyond two classes
--------------------

Many classification problems have **more than two** unordered labels — payment type (cash / card /
mobile), a product category, a species. These are **nominal** (multi-class) problems: the classes have
**no natural order**, and every prediction must pick one of :math:`K` labels. Evaluation carries over
from the binary case, but with two twists worth understanding.

The K×K confusion matrix
--------------------------

The confusion matrix generalises directly: it becomes a :math:`K \times K` table, actual classes down
the rows and predicted classes across the columns. The **diagonal** still holds the correct
predictions, and every **off-diagonal** cell now names a **specific confusion** — how often class A was
mistaken for class B. This is the matrix's great virtue in multi-class work: it shows **which pairs**
of classes the model muddles (cash vs mobile, say), pointing directly at what to fix. scikit-plots
renders it as the same labelled heat-map plot, just larger.

Per-class metrics
-------------------

Precision, recall and F1 are **binary** notions, so for :math:`K` classes they are computed **per
class**, one-vs-rest: for class A, treat A as "positive" and everything else as "negative", and read
TP, FP and FN from the matrix's row and column for A. Each class gets its own precision (of the
predictions of A, how many were right?) and recall (of the true A's, how many were found?) —
scikit-learn's ``classification_report`` prints exactly this table, one row per class.

Averaging: macro, micro, weighted
-----------------------------------

To summarise the per-class scores in **one number**, three averages are standard, and they answer
**different questions**:

* **Macro** — the plain mean of the per-class scores: every **class** counts equally, however rare, so it exposes weakness on small classes;
* **Weighted** — the mean weighted by class **frequency**: large classes dominate, mirroring overall behaviour;
* **Micro** — pool all the TP / FP / FN counts first, then compute: every **instance** counts equally. For single-label problems micro-precision, micro-recall and accuracy all **coincide**.

On **imbalanced** data the choice matters: a model useless on a rare class can still post a high
weighted or micro score, while its **macro** score collapses — so pick the average that matches whether
classes or instances are what the business weighs equally.
"""

CONTENT["Binary Classification Model Evaluation and Threshold Optimization"] = r"""
The hidden dial
-----------------

A binary classifier rarely outputs a label directly — it outputs a **probability** (or score), like the
logistic model's :math:`p`. The label comes from a **threshold**: predict positive when the score
exceeds a **cutoff**, conventionally **0.5**. That cutoff is a **dial**, not a law of nature — and
turning it changes every metric of the last lessons without retraining anything. Choosing it well is
**threshold optimisation**.

Turning the dial
------------------

The trade-off is mechanical. **Lower** the threshold and more cases are called positive: **recall
rises** (fewer misses) but **precision falls** (more false alarms). **Raise** it and the reverse —
predictions become conservative, precision improves, recall drops. The confusion matrix **morphs**
continuously as the dial turns, trading false negatives for false positives. No single threshold is
"correct"; each is a different **operating point** on the same fitted model.

Cost decides
--------------

The right operating point comes from the **relative cost of the two errors** — the domain judgement
from the confusion-matrix lesson, now made operational. If a **miss** is expensive (fraud, disease
screening), push the threshold **down** and accept false alarms — a fraud team might demand recall
above 0.85 and tune precision after. If a **false alarm** is expensive (spam filters binning real
mail), push it **up**. When costs can be written down, the threshold can even be chosen to **minimise
expected cost** directly.

Choosing it in practice
-------------------------

Practically, you **sweep** the threshold on validation data and plot the metrics against it: precision
and recall curves crossing, F1 peaking, or expected cost bottoming out. Pick the threshold that
optimises the criterion you care about — the F1-maximising point, the smallest threshold hitting a
target recall, or the cost minimum. scikit-plots' evaluation charts exist for exactly this sweep, and
the next lessons introduce the two most important of them: the **ROC curve**, which displays *every*
operating point at once, and the **lift** chart, which ranks rather than cuts.
"""

CONTENT["Identifying Outliers Using Residuals and Studentized Residuals"] = r"""
Evaluating regression fits
----------------------------

For regression models, evaluation is not only an average error — it is also asking whether **individual
points** are being fit sensibly. The tool is the **residual**, :math:`e_i = y_i - \hat{y}_i`, familiar
from least squares. A point whose residual is **far larger** than the rest is an **outlier** in the
response — a case the model badly mispredicts — and finding these is a core diagnostic, one that
scikit-plots visualises with residual plots.

Raw residuals mislead
-----------------------

Raw residuals are an awkward yardstick, for two reasons. Their size depends on the **units** of
:math:`y`, so "large" has no absolute meaning. Worse, they do **not share a common variance**: the
variance of :math:`e_i` is :math:`\sigma^2 (1 - h_{ii})`, where :math:`h_{ii}` is the point's
**leverage** — how unusual its feature values are. A **high-leverage** point *pulls the fitted line
toward itself*, artificially **shrinking** its own residual. The very points most able to distort the
fit are the ones whose raw residuals look most innocent.

Standardising
---------------

The first fix is the **standardized** (internally studentized) residual — the raw residual divided by
its own estimated standard deviation:

.. math::

   r_i = \frac{e_i}{\hat{\sigma}\,\sqrt{1 - h_{ii}}}.

Now every point is on a **common scale** of standard-deviation units, comparable across observations
and datasets, with :math:`|r_i| > 3` a common flag for an outlier.

Studentized residuals
-----------------------

One subtlety remains: a truly extreme point **inflates** :math:`\hat{\sigma}` itself, partially
**masking** its own residual. The **studentized** (externally studentized, or *deleted*) residual
removes the circularity by estimating the error scale **without** observation :math:`i` — refit the
model leaving the point out, and scale by that :math:`\hat{\sigma}_{(i)}`:

.. math::

   t_i = \frac{e_i}{\hat{\sigma}_{(i)}\,\sqrt{1 - h_{ii}}}.

Under the usual assumptions :math:`t_i` follows a **t-distribution**, so the flag becomes a genuine
**statistical test**. In practice, :math:`|t_i| > 2` marks a point worth examining and
:math:`|t_i| > 3` a strong outlier — first check for a **data error**; if the value is real, consider a
robust fit or report it as a notable exception. And when **many** points flag at once, the message is
usually not "bad data" but a **misspecified model** — a missing curve or interaction. Residual
diagnostics evaluate the *model* as much as the points.
"""


MINDMAP.update({
    "Nominal Classification Models: Model State and Evaluation Metrics": [
        "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics",
        "Assessing the Quality of Prediction Models",
        "Binary Classification Model Evaluation and Threshold Optimization",
        "AUC\u2013ROC Curve: Evaluating Classification Model Performance",
    ],
    "Binary Classification Model Evaluation and Threshold Optimization": [
        "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics",
        "AUC\u2013ROC Curve: Evaluating Classification Model Performance",
        "Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value",
        "Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds",
    ],
    "Identifying Outliers Using Residuals and Studentized Residuals": [
        "Least Squares Regression", "Multiple Linear Regression",
        "Assessing the Quality of Prediction Models",
        "The First Step in Knowing Your Data",
    ],
})


# ----------------------------------------------------------------------
# Stage 8 — Model Evaluation (cont.)  [completes the stage and the course]
# ----------------------------------------------------------------------

CONTENT["AUC\u2013ROC Curve: Evaluating Classification Model Performance"] = r"""
Every threshold at once
-------------------------

Threshold optimisation showed that a classifier is really a **family** of operating points, one per
cutoff. The **ROC curve** (Receiver Operating Characteristic) displays that **entire family in one
picture**. For every possible threshold it plots the **true positive rate** (recall — the share of
actual positives caught) against the **false positive rate** (the share of actual negatives wrongly
flagged). Sweeping the threshold from strictest to loosest traces the curve from the bottom-left
corner (nothing called positive) to the top-right (everything called positive).

Reading the curve
-------------------

The curve's **shape** grades the model. The **ideal** point is the **top-left** corner — all positives
caught, no false alarms — so the closer the curve bows toward it, the better the classifier. The
**diagonal** line is the signature of **random guessing**: any model whose curve hugs it has no
discriminating power, and a curve *below* the diagonal is systematically worse than chance (invert its
predictions and it becomes useful). Where curves for competing models are compared, the one bowing
further toward the corner wins — and scikit-plots draws exactly this chart, one line per model or
class.

The area underneath
---------------------

The curve compresses into one number: the **AUC**, the **area under the curve**, ranging from **0.5**
(random) to **1.0** (perfect). It has an elegant probabilistic meaning: the AUC is the probability
that a **randomly chosen positive** case receives a **higher score** than a randomly chosen negative
one — a pure measure of how well the model **ranks**. As rough benchmarks, 0.9+ is excellent, 0.8–0.9
good, 0.7–0.8 acceptable; the retention model of Stage 6 reported its quality in exactly these terms.

Strengths and cautions
------------------------

AUC's virtue is being **threshold-free**: it judges the scores themselves, before any cutoff is
chosen, which makes it ideal for **comparing models** and more informative than accuracy on
**imbalanced** data. Its caution is the mirror image: because it averages over **all** thresholds —
including ones you would never operate at — a model can post a fine AUC yet disappoint at *your*
operating point, and AUC says nothing about whether the probabilities are **calibrated**. Use it to
pick the ranker; use the threshold lesson to pick the cutoff; and, when only the top of the ranking
will ever be acted on, use the **lift** analysis of the final lesson.
"""

CONTENT["Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value"] = r"""
A budget, not a threshold
---------------------------

The course closes with the evaluation tool born in **direct mail** — the corner of marketing where
every contact costs real money. A campaign can afford to mail, say, only **20%** of the customer base.
The question is not "which cases are positive?" but "**whom should we contact first?**" — a **ranking**
problem with a budget. **Lift analysis** measures how much better a model's ranking is than mailing at
**random**.

Score, sort, slice
--------------------

The procedure is simple. **Score** every prospect with the model (their probability of responding),
**sort** the list from highest to lowest score, then **slice** it into ten equal groups — **deciles** —
so decile 1 holds the top 10% of prospects by model score. Mailing a past campaign's data through this
lens shows the **response rate per decile**: if the model ranks well, decile 1 responds far above
average, decile 2 next, and the bottom deciles barely respond at all.

Gains and lift
----------------

Two curves summarise the table. The **cumulative gains** curve shows, for each targeting depth, the
share of **all responders** captured: for example, the top decile alone might contain **28%** of all
responders, and the top two deciles together over half. The **baseline** is random targeting — contact
X% of customers, reach X% of responders. **Lift** is the ratio of the two:

.. math::

   \text{lift at depth } X =
   \frac{\%\ \text{of responders captured in the top } X\%}{X\,\%}.

A lift of **1** means the model adds nothing; a lift of **2** at 20% depth means the model finds
**twice** the responders random mailing would. The **further** the gains curve rises above the
baseline, the more valuable the model — and scikit-plots draws both the cumulative-gain and lift
curves directly from scores.

The business payoff
---------------------

Lift converts model quality into **money**. If the top four deciles capture most responders, the
campaign mails **40%** of the base, captures the bulk of the responses, and saves the cost of the
other 60% — a concrete, defensible decision drawn straight from the chart. It is a fitting end to the
course: the journey that began with *why we analyse data* ends with a model, honestly evaluated on
held-out data, telling a business **exactly what to do** — the prescriptive payoff the first lesson
promised.
"""


MINDMAP.update({
    "AUC\u2013ROC Curve: Evaluating Classification Model Performance": [
        "Binary Classification Model Evaluation and Threshold Optimization",
        "Binary Classification Models \u2013 Conceptual Framework and Evaluation Metrics",
        "Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value",
        "Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention",
    ],
    "Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value": [
        "AUC\u2013ROC Curve: Evaluating Classification Model Performance",
        "Binary Classification Model Evaluation and Threshold Optimization",
        "Recency, Frequency, and Monetary Value (RFM)",
        "Why Do We Analyze Data?",
    ],
})
