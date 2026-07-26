:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-llms-large-language-models:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>LLMs (Large Language Models)</b></div>`

==============================
LLMs (Large Language Models)
==============================

*Large neural networks trained on text to understand and generate language.*

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

**LLMs — large language models** — are ML models trained on **massive text corpora** to
understand and generate human language. Built on **transformer** architectures, they handle a
wide range of NLP tasks **without task-specific training**, via zero-shot and few-shot prompting.

How they work
-------------

The architecture is the **transformer** with **self-attention**, which lets the model weigh every
word in a sequence for context. Training runs over **billions to trillions of tokens** (books,
articles, code, the web) with a deceptively simple objective: **predict the next token**. From
that single objective emerge text generation, summarisation, translation, reasoning and code
generation.

The landscape and uses
----------------------

The major families include OpenAI's **GPT**, Google's **Gemini**, Meta's **LLaMA**, Anthropic's
**Claude** and Mistral's **Mixtral**. They power conversational assistants, content creation,
**code copilots**, semantic search and **retrieval-augmented generation (RAG)**, and
data-science tasks like query-to-SQL.

The limitations
---------------

The caveats are serious: **hallucinations** (fluent but wrong answers), **bias** inherited from
training data, **cost and energy** (very high OpEx to train and serve), **stale knowledge** that
lags real-time events, and limited **explainability**. A useful mental model is a **very powerful
autocomplete** — predict the most likely continuation — that, at sufficient scale, begins to show
reasoning-like behaviour.

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Embedding <173-embedding>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`FLOPs <156-flops>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Regression Models <309-regression-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Target Variable <236-target-variable>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `LLMs (Large Language Models) <https://insightful-data-lab.com/2025/08/24/llms-large-language-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
