.. _dpa-least-squares-regression:

========================================================================
Least Squares Regression
========================================================================

**Stage 5 · 📈 Regression**  ·  Lesson 31 of 56  ·  *intermediate*

:doc:`◀ Previous · Creating Segments of Observations for Business Reasons (RFM) <30-creating-segments-of-observations-for-business-reasons-rfm>`   ·   :doc:`Next · Multiple Linear Regression ▶ <32-multiple-linear-regression>`   ·   :doc:`↑ Section <index>`


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

.. hint::

   **Related lessons:** :doc:`Multiple Linear Regression <32-multiple-linear-regression>`  ·  :doc:`Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>`  ·  :doc:`Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>`  ·  :doc:`Identifying Outliers Using Residuals and Studentized Residuals <54-identifying-outliers-using-residuals-and-studentized-residuals>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/least-squares-regression/ <https://insightful-data-lab.com/2026/01/16/least-squares-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
