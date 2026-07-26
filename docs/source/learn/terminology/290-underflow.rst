:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-underflow:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Underflow</b></div>`

===========
Underflow
===========

*Numerical loss of precision when values become too small to represent.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Underflow** happens when a computation produces a number **too small** to represent in the floating-point
format — smaller than the tiniest positive value — so the computer **rounds it to zero**, destroying a real
nonzero result. It is the small-magnitude counterpart of **overflow**.

Why ML hits it
--------------

Machine learning multiplies **many small probabilities** — in Naive Bayes, HMMs, and likelihoods — and the
product of hundreds of values below 1 quickly drops below the representable floor, collapsing to **0** and
corrupting the result. Low-precision (**float16**) training underflows even sooner, showing up as **vanishing
gradients**.

The fix
-------

Compute in **log space**. Because :math:`\log(a \cdot b) = \log(a) + \log(b)`, a fragile **product** of tiny
probabilities becomes a stable **sum** of log-probabilities — the reason libraries use **log-likelihoods** and
the **LogSumExp** trick, and why scikit-learn's Naive Bayes works with logs internally.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Logits <420-logits>` · :doc:`Softmax Function <296-softmax-function>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Quantization <343-quantization>` · :doc:`Sigmoid Function <297-sigmoid-function>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Underflow <https://insightful-data-lab.com/2025/08/21/underflow/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
