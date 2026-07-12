.. _dl-why-deep-learning-is-taking-off:

========================================================================
Why Deep Learning is Taking Off
========================================================================

**Stage 1 · 🧠 Introduction to Deep Learning**  ·  Lesson 03 of 17  ·  *beginner*

:doc:`◀ Previous · Supervised Learning and Neural Networks <02-supervised-learning-and-neural-networks>`   ·   :doc:`Next · Geoffrey Hinton Interview ▶ <04-geoffrey-hinton-interview>`   ·   :doc:`↑ Section <index>`


An old idea, newly working
----------------------------

The mathematics of neural networks is **decades old**, so why the recent explosion? Not one
breakthrough but a **convergence** — the raw ingredients finally reached the scale where deep
networks **decisively outperform** the alternatives.

Scale drives performance
--------------------------

Ng summarises it with a single picture: plot **performance** against the **amount of labelled
data**. Traditional methods (logistic regression, SVMs) improve for a while, then **plateau**.
Neural networks keep climbing, and **bigger networks climb higher** — small < medium < large. The
pattern has two regimes: with **little** data, careful feature engineering and skill can matter more
than model size, so the ordering blurs; with **lots** of data, a **large network** wins clearly.

The three drivers
------------------

Three forces made that scale reachable. **Data** — a digitised world (phones, sensors, the web)
produces the huge labelled datasets networks feed on. **Computation** — GPUs, faster hardware and
distributed training make large models trainable in reasonable time. **Algorithms** — better design
speeds learning; the switch from the **sigmoid** to the **ReLU** activation is the classic example,
easing the **vanishing-gradient** problem so gradient descent converges much faster.

The virtuous cycle
--------------------

These drivers reinforce each other. Faster hardware and better algorithms shorten the
**idea → code → experiment** loop, so researchers iterate more quickly; better models attract more
**users**, who generate more **data**, which trains still-better models. That feedback loop is much
of why progress has felt so **fast** — and why the fundamentals in this course sit beneath so many
modern systems.

.. hint::

   **Related lessons:** :doc:`What is a Neural Network? <01-what-is-a-neural-network>`  ·  :doc:`Supervised Learning and Neural Networks <02-supervised-learning-and-neural-networks>`  ·  :doc:`Geoffrey Hinton Interview <04-geoffrey-hinton-interview>`  ·  :doc:`Vectorizing Logistic Regression <17-vectorizing-logistic-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/why-deep-learning-is-taking-off/ <https://insightful-data-lab.com/2025/04/07/why-deep-learning-is-taking-off/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: beginner
