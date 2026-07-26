.. _bda-graphical-posterior-predictive-checks:

========================================================================
Graphical posterior predictive checks
========================================================================

**Part 2 · Stage 6 · 🔍 Model Checking & Comparison**  ·  Lesson 043 of 144  ·  *intermediate*

:doc:`◀ Previous · Posterior predictive checking <042-posterior-predictive-checking>`   ·   :doc:`Next · Model checking for the educational testing example ▶ <044-model-checking-for-the-educational-testing-example>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Look before you compute
-------------------------

A :math:`p`-value collapses a comparison to one number and answers only the question its test quantity
encodes. A **plot** shows the shape of the misfit — and often reveals a failure nobody thought to test
for. Graphical checks are therefore the first move, not the last.

The three displays
--------------------

* **Density overlay.** Draw the density of each of ~50 replicated datasets in light grey, the observed
  data in bold. Systematic offsets appear as the real curve sitting outside the grey band: skew the
  model cannot make, a peak at zero it cannot produce, tails too thin.
* **Test-statistic histogram.** Histogram :math:`T(y^{\text{rep}})` over replications and mark
  :math:`T(y)` with a line. This is the picture of which the Bayesian :math:`p`-value is the caption.
* **Predictive intervals versus observations.** For structured data (time series, groups), plot each
  observation against its 50%/90% predictive interval. Roughly the right proportion should fall inside,
  and the misses should look **random** — a run of consecutive misses signals unmodelled structure.

.. code-block:: python

   import arviz as az
   az.plot_ppc(idata, num_pp_samples=50)                 # density overlay
   az.plot_bpv(idata, kind="t_stat", t_stat="std")       # test statistic vs replications
   az.plot_loo_pit(idata, y="y")                         # calibration: should be uniform

Calibration plots
-------------------

A sharper display asks where each observation falls **within** its own predictive distribution — its
PIT value. If the model is calibrated, those values are **uniform**; a U-shape means predictive
distributions are too narrow (overconfident), a hump in the middle means too wide. LOO-PIT does this
using **leave-one-out** predictions, avoiding the double-use of data that makes ordinary predictive
:math:`p`-values conservative.

Reading a failure
-------------------

The value of the visual grammar is that the *shape* of the discrepancy names the missing feature. Too
many zeros in the data and none in the replications: the likelihood needs a zero-inflation component.
Replicated tails too thin: swap normal errors for :math:`t`. Replications too smooth across groups: the
hierarchy is over-pooling. Each verdict points to a specific **model expansion** — which is the closing
theme of this stage. Choose the plot that would embarrass the model if the model deserved it.

.. hint::

   **Related lessons:** :doc:`Posterior predictive checking <042-posterior-predictive-checking>`  ·  :doc:`Model checking for the educational testing example <044-model-checking-for-the-educational-testing-example>`  ·  :doc:`Continuous model expansion <048-continuous-model-expansion>`  ·  :doc:`Measures of predictive accuracy <045-measures-of-predictive-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/graphical-posterior-predictive-checks/ <https://insightful-data-lab.com/2025/11/10/graphical-posterior-predictive-checks/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
