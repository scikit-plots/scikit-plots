.. _dpa-objective-selection-of-the-bin-width-for-a-time-histogram:

========================================================================
Objective Selection of the Bin Width for a Time Histogram
========================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 09 of 56  ·  *beginner*

:doc:`◀ Previous · Taxi Trips – 2022 dataset from the City of Chicago open data portal <08-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal>`   ·   :doc:`Next · Measuring Associations in Data ▶ <10-measuring-associations-in-data>`   ·   :doc:`↑ Section <index>`


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

.. hint::

   **Related lessons:** :doc:`Taxi Trips – 2022 dataset from the City of Chicago open data portal <08-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal>`  ·  :doc:`The First Step in Knowing Your Data <05-the-first-step-in-knowing-your-data>`  ·  :doc:`Measuring Associations in Data <10-measuring-associations-in-data>`  ·  :doc:`Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022) <07-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/objective-selection-of-the-bin-width-for-a-time-histogram/ <https://insightful-data-lab.com/2026/01/14/objective-selection-of-the-bin-width-for-a-time-histogram/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner
