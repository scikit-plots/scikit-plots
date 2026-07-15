:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _deep-learning-index:

:raw-html:`<div style="text-align:center"><strong>` 🧠 Deep Learning
|br| From a single neuron to deep networks
|br| |full_version| - |today|
:raw-html:`</strong></div>`

======================================================================
Deep Learning
======================================================================

**Deep learning** stacks simple, differentiable units into networks trained end-to-end by
**gradient descent**. This course follows the ground-up path the source corpus takes — view
**logistic regression as a single neuron**, learn the **computation graph** and
**backpropagation** that train it, then **vectorise** it into the efficient building block that
scales to deep networks — as an ordered, self-contained course of 17 lessons.

Read it at any depth:

* **newcomers** — what a neural network is, and a single neuron;
* **practitioners** — the sigmoid model, the cross-entropy cost, and gradient descent;
* **researchers / engineers** — the computation graph, backpropagation, and full vectorisation with ``numpy``.

.. warning::

   These lessons build the mathematics **from scratch** in ``numpy`` to show what a framework
   does under the hood. In production you would use a framework (**PyTorch** / **Keras**) — its
   automatic differentiation computes exactly these gradients for you.

.. note::

   Follow the lessons in order with **Next ▶**, or jump in by stage below. Snippets use
   real ``numpy`` / ``PyTorch`` / ``Keras`` calls and connect to scikit-plots' ``visualkeras``
   architecture views. This course pairs with the
   :ref:`Terminology reference <terminology-index>` (classification metrics for evaluating networks).

======================================================================

.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
          placeholder="&#128269;&nbsp; Type to filter 17 lessons &mdash; by title or keyword&hellip;"
          style="width:100%;max-width:100%;padding:0.55rem 1rem;font-size:1rem;
                 border:1px solid var(--pst-color-border,#ccc);border-radius:0.55rem;box-sizing:border-box;
                 background:transparent;color:inherit"/>
   <div id="term-filter-count" style="opacity:0.65;font-size:0.85rem;
        min-height:1.2em;margin-top:0.35rem"></div>
   </div>
   <script>
   document.addEventListener('DOMContentLoaded',function(){
     var inp=document.getElementById('term-filter');if(!inp){return;}
     var dds=[].slice.call(document.querySelectorAll('details.sd-dropdown'));
     var az=document.querySelector('details.term-az');
     var items=[];
     dds.forEach(function(d){[].slice.call(d.querySelectorAll('li')).forEach(
       function(li){items.push({li:li,d:d,t:li.textContent.toLowerCase()});});});
     var cnt=document.getElementById('term-filter-count');
     inp.addEventListener('input',function(){
       var q=inp.value.trim().toLowerCase();var n=0;
       dds.forEach(function(d){d.tHits=0;});
       items.forEach(function(it){
         var hit=!q||it.t.indexOf(q)!==-1;
         it.li.style.display=hit?'':'none';
         if(hit){it.d.tHits+=1;if(az&&it.d===az){n+=1;}}});
       dds.forEach(function(d){
         if(q){d.style.display=d.tHits?'':'none';d.open=d.tHits>0;}
         else{d.style.display='';d.open=false;}});
       if(cnt){cnt.textContent=(q&&az)?(n+' of 17 match'+(n===1?'':'s')):'';}
     });
   });
   </script>

.. _dl-stage-intro:

.. dropdown:: 🧠 Stage 1 — Introduction to Deep Learning  ·  4 lessons
   :animate: fade-in-slide-down

   *What a neural network is, where supervised deep learning applies, and why it took off.*  ·  *beginner*

   * :doc:`01 · What is a Neural Network? <01-what-is-a-neural-network>` — How stacked neurons map inputs to outputs, learning the mapping from examples by gradient descent.
   * :doc:`02 · Supervised Learning and Neural Networks <02-supervised-learning-and-neural-networks>` — Learning input→output mappings from labelled data, and which network suits structured, image or sequence inputs.
   * :doc:`03 · Why Deep Learning is Taking Off <03-why-deep-learning-is-taking-off>` — The three drivers — data scale, compute, and algorithms — that made deep networks practical.
   * :doc:`04 · Geoffrey Hinton Interview <04-geoffrey-hinton-interview>` — Perspectives from a pioneer of backpropagation on how the field's key ideas emerged.

.. _dl-stage-neuron:

.. dropdown:: 🔵 Stage 2 — Logistic Regression as a Neuron  ·  3 lessons
   :animate: fade-in-slide-down

   *A single sigmoid unit — the binary-classification model, its loss and its cost function.*  ·  *beginner*

   * :doc:`05 · Binary Classification and Logistic Regression (Neural Network Basics) <05-binary-classification-and-logistic-regression-neural-network-basics>` — Framing a yes/no prediction, the notation for one training example, and stacking data into matrices.
   * :doc:`06 · Logistic Regression (Binary Classification Model) <06-logistic-regression-binary-classification-model>` — A single neuron: a linear score passed through the sigmoid to output a probability.
   * :doc:`07 · Logistic Regression – Loss Function and Cost Function <07-logistic-regression-loss-function-and-cost-function>` — The per-example cross-entropy loss and the averaged cost that training minimises.

.. _dl-stage-calculus:

.. dropdown:: 📉 Stage 3 — Derivatives & the Computation Graph  ·  5 lessons
   :animate: fade-in-slide-down

   *The calculus that trains a neuron: gradients, the computation graph and the chain rule.*  ·  *intermediate*

   * :doc:`08 · Gradient Descent in Logistic Regression <08-gradient-descent-in-logistic-regression>` — Iteratively stepping the parameters downhill along the cost gradient toward a minimum.
   * :doc:`09 · Derivatives <09-derivatives>` — The slope of a function — the building block of every gradient used to train a network.
   * :doc:`10 · More Derivative Examples <10-more-derivative-examples>` — Worked derivatives that build intuition for how small input changes move the output.
   * :doc:`11 · Computation Graph <11-computation-graph>` — Breaking a computation into elementary steps — the structure backpropagation runs on.
   * :doc:`12 · Derivatives with a Computation Graph <12-derivatives-with-a-computation-graph>` — Applying the chain rule backward through the graph to get every partial derivative efficiently.

.. _dl-stage-training:

.. dropdown:: ⚙️ Stage 4 — Backprop & Vectorization  ·  5 lessons
   :animate: fade-in-slide-down

   *Backpropagation across a whole training set, then vectorising it for speed with numpy.*  ·  *intermediate*

   * :doc:`13 · Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>` — Backpropagation for one example: the derivatives of the loss w.r.t. each weight and the bias.
   * :doc:`14 · Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>` — Averaging the single-example gradients over the whole training set for one update step.
   * :doc:`15 · Vectorization in Logistic Regression <15-vectorization-in-logistic-regression>` — Replacing explicit loops with matrix operations so the whole batch computes at once.
   * :doc:`16 · More Vectorization Examples <16-more-vectorization-examples>` — Further numpy patterns — broadcasting and whole-array functions — that remove Python loops.
   * :doc:`17 · Vectorizing Logistic Regression <17-vectorizing-logistic-regression>` — A fully vectorised forward and backward pass over all examples — the gateway to deep networks.

🔤 Every lesson, A–Z index
---------------------------

.. dropdown:: 🔠 Open the full alphabetical index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`Binary Classification and Logistic Regression (Neural Network Basics) <05-binary-classification-and-logistic-regression-neural-network-basics>`
      * :doc:`Computation Graph <11-computation-graph>`
      * :doc:`Derivatives <09-derivatives>`
      * :doc:`Derivatives with a Computation Graph <12-derivatives-with-a-computation-graph>`
      * :doc:`Geoffrey Hinton Interview <04-geoffrey-hinton-interview>`
      * :doc:`Gradient Descent in Logistic Regression <08-gradient-descent-in-logistic-regression>`
      * :doc:`Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>`
      * :doc:`Logistic Regression (Binary Classification Model) <06-logistic-regression-binary-classification-model>`
      * :doc:`Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`
      * :doc:`Logistic Regression – Loss Function and Cost Function <07-logistic-regression-loss-function-and-cost-function>`
      * :doc:`More Derivative Examples <10-more-derivative-examples>`
      * :doc:`More Vectorization Examples <16-more-vectorization-examples>`
      * :doc:`Supervised Learning and Neural Networks <02-supervised-learning-and-neural-networks>`
      * :doc:`Vectorization in Logistic Regression <15-vectorization-in-logistic-regression>`
      * :doc:`Vectorizing Logistic Regression <17-vectorizing-logistic-regression>`
      * :doc:`What is a Neural Network? <01-what-is-a-neural-network>`
      * :doc:`Why Deep Learning is Taking Off <03-why-deep-learning-is-taking-off>`

.. toctree::
   :hidden:
   :maxdepth: 1

   01-what-is-a-neural-network
   02-supervised-learning-and-neural-networks
   03-why-deep-learning-is-taking-off
   04-geoffrey-hinton-interview
   05-binary-classification-and-logistic-regression-neural-network-basics
   06-logistic-regression-binary-classification-model
   07-logistic-regression-loss-function-and-cost-function
   08-gradient-descent-in-logistic-regression
   09-derivatives
   10-more-derivative-examples
   11-computation-graph
   12-derivatives-with-a-computation-graph
   13-logistic-regression-gradient-descent
   14-gradient-descent-on-m-training-examples
   15-vectorization-in-logistic-regression
   16-more-vectorization-examples
   17-vectorizing-logistic-regression

.. tags:: purpose: reference, topic: deep learning
