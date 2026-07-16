:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-010:
.. _da-3-prep-prep-010:

========================================================================
Common Types of Data Bias
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`⚖️ Bias & Data Ethics` :bdg-info:`Lesson 010`

◀ :doc:`Previous <009-sampling-bias-and-unbiased-data>` · :doc:`Next <011-identifying-good-data-sources-roccc-framework>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


A field guide to bias
------------------------

Sampling bias is the most common, but not the only, way analysis goes
systematically wrong. Analysts benefit from a field guide to the recurring
types, because naming a bias is the first step to guarding against it. The
standard set covers most of what you will meet.

The common types
------------------

- **Sampling bias** — the sample does not represent the population (the previous
  lesson). Enters at *collection*.
- **Observer bias** (also experimenter or measurement bias) — the tendency for
  different observers to see or record the same thing differently, often nudged
  by what they expect. Two people rating call quality, or reading an instrument
  near a threshold, may systematically differ. Enters at *measurement*.
- **Interpretation bias** — the tendency to interpret ambiguous results in the
  way that fits a preferred or expected conclusion, when the data genuinely
  admits more than one reading. Enters at *analysis*.
- **Confirmation bias** — the tendency to search for, favour, and recall
  information that confirms what you already believe, while discounting what
  contradicts it. The analyst runs the query that supports the hunch, scrutinises
  the disconfirming result harder, and remembers the hits. Enters at *the whole
  process*.

Two of these live in the *data* (sampling, and the measurement side of
observer bias) and two live in the *analyst* (interpretation, confirmation) —
which is why guarding against bias means auditing both the dataset and your own
reasoning.

Why they are hard to catch
----------------------------

Every one of these produces internally consistent, plausible results — the
hallmark of bias from the previous lesson. Confirmation bias is especially
insidious because it feels like *diligence*: scrutinising an inconvenient result
harder than a convenient one looks responsible while being exactly the
mechanism of the bias. The tell is asymmetry — applying more skepticism to
findings you dislike than to findings you like.

Guarding against them
-----------------------

Each type has a countermeasure. Sampling bias: examine collection and compare to
population facts. Observer bias: use clear, objective definitions and, where
possible, multiple independent observers or automated measurement.
Interpretation bias: state alternative readings explicitly and check which the
data actually supports. Confirmation bias: deliberately seek *disconfirming*
evidence — ask "what would make me wrong?" and run *that* query too. The
unifying practice is inviting others to challenge the work, because another
person's biases rarely align exactly with yours.

The caveat
------------

Bias cannot be fully eliminated — it can be recognised, reduced, and disclosed.
Over-correcting has its own hazard: bending over backward to avoid one bias can
introduce another, and reflexively distrusting every finding is not rigour but
paralysis. The aim is calibrated, even-handed skepticism applied equally to
welcome and unwelcome results — the same discipline the fairness thread has
asked for throughout. The next lessons turn from bias in general to a concrete
checklist for judging whether a *source* is trustworthy.

.. hint::

   - :doc:`Sampling Bias and Unbiased Data <009-sampling-bias-and-unbiased-data>`
   - :doc:`Understanding Bias in Data Analysis <008-understanding-bias-in-data-analysis>`
   - :doc:`Context and Bias in Data Analysis <../2_data_driven_decisions/018-context-and-bias-in-data-analysis>`
   - :doc:`Identifying Good Data Sources (ROCCC Framework) <011-identifying-good-data-sources-roccc-framework>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/common-types-of-data-bias/ <https://insightful-data-lab.com/2023/09/04/common-types-of-data-bias/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: bias_ethics
