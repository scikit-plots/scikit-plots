..
  docs/source/user_guide/cleanprompt/entity_detection.rst

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-entity-detection:

======================================================================
Names, organisations and places
======================================================================

No regular expression can recognise a person's name. That job belongs to an
entity engine, and CleanPrompt can use two. Both are optional; without either,
``doctor`` reports names, organisations and places as a *blind spot*, every
surface repeats the warning, and an empty result is never presented as a clean
one.

The two engines
---------------

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Mode
     - Engine
     - Character
   * - ``spacy``
     - statistical models
     - many languages, better quality, needs a model download
   * - ``nltk``
     - the classic chunker
     - English only, needs NLTK's data packages, lower recall
   * - ``both``
     - union of the two
     - widest cover; overlapping spans are merged
   * - ``auto``
     - spaCy if ready, else NLTK
     - the default; use it unless you have a reason
   * - ``none``
     - patterns only
     - explicit, so it cannot be mistaken for a failed engine

Turn one on with ``--ner`` (``ner=True`` in Python) and choose with
``--ner-engine`` (``engine=``). Whatever the engine, labels are normalised to
one vocabulary — ``PERSON``, ``ORG``, ``GPE``, ``LOC``, ``FAC``, ``NORP``,
``EVENT``, ``WORK_OF_ART``, ``PRODUCT``, ``LAW``, ``LANGUAGE``, ``MISC`` — so a
vault written with one engine restores under the other. Dates, times, numbers,
money and percentages are deliberately not redacted: they rarely identify
anyone and the model usually needs them.

Installed is not ready
----------------------

An engine needs **three** things before it can run, and the commonest failure
is having the first without the third:

1. **its package** — ``pip install "scikit-plots[cleanprompt-ner]"`` for spaCy,
   ``"scikit-plots[cleanprompt-nltk]"`` for NLTK;
2. **a language it can read** — NLTK reads English only;
3. **its data** — a spaCy *model* (``python -m spacy download en_core_web_sm``),
   or NLTK's *data packages*.

``doctor`` checks all three and, where one is missing, names the single
command that supplies it:

.. prompt:: bash $

   python -m scikitplot.cleanprompt doctor --ner --format json

.. code-block:: text

   detection.ner_requested                     true
   detection.ner_ready                         false
   detection.ner_remedy                        python -m spacy download en_core_web_sm
   entity_engines.selected                     ["spacy"]
   entity_engines.ready                        false
   entity_engines.engines.spacy.installed      true
   entity_engines.engines.spacy.assets_ready   false
   entity_engines.engines.spacy.reason         spaCy is installed but model 'en_core_web_sm' is not

The run and the diagnosis are the **same decision**: ``doctor``, ``inspect``,
``encode``, ``redact``, :func:`encode`, the file runtime and the web app all
build their engines through one function,
:func:`~scikitplot.cleanprompt._engines.build_detectors`. So:

* ``auto`` never selects an engine that is not ready. With spaCy installed but
  no model, and NLTK ready, ``auto`` runs NLTK.
* An explicit request that cannot be met — ``--ner`` with nothing ready,
  ``--ner-engine spacy`` without a model, ``--ner-engine both`` with one of
  the two missing its data — fails **before any text is read**, with exit
  status ``69`` and each engine's reason and remedy. It never runs quietly
  with less than you asked for.

.. code-block:: text

   $ cleanprompt inspect --ner "Ada Lovelace met Charles Babbage."
   error: entity detection was requested (--ner) with engine mode 'auto', but no engine is ready for language 'en':
     spacy: MISCONFIGURED — spaCy is installed but model 'en_core_web_sm' is not
         fix: python -m spacy download en_core_web_sm
     nltk: MISCONFIGURED — NLTK is installed but 4 data package(s) are missing or cannot be loaded by this NLTK: ...
         fix: python -c "import nltk; nltk.download('punkt_tab'); ..."
   Fix one, or pass --ner-engine none to proceed without entity detection.

How readiness is decided
^^^^^^^^^^^^^^^^^^^^^^^^

**spaCy.** The model this configuration would load is resolved first (from
``--lang`` and ``--model-size``, or ``--ner-model``), then looked for without
importing it: as an installed distribution, as an importable package, or as a
model directory on disk.

**NLTK.** The data packages are looked for, and then the detector's own
tagger and chunker are run once on a fixed sentence. Running them is the only
check that cannot disagree with the real run: NLTK 3.9 moved its tagger and
chunker to new data packages (``averaged_perceptron_tagger_eng``,
``maxent_ne_chunker_tab``), and data present under the older names is found
on disk but cannot be loaded. The remedy therefore names **every** package
that can satisfy a missing group — the current name and the older one — so it
is right on every NLTK release the tier supports. The check imports NLTK, so
it runs only where NLTK is about to be used anyway: an explicit request,
``doctor``, or the web app with entity detection on.

In Python
^^^^^^^^^

.. prompt:: python >>>

   from scikitplot.cleanprompt import describe_engines, engine_readiness

   engine_readiness("spacy", "en").remedy
   describe_engines("en", "auto", check_assets=True)["selected"]

:func:`~scikitplot.cleanprompt._engines.engine_readiness` returns an
``EngineReadiness`` with ``installed``, ``language_supported``,
``assets_ready``, ``ready``, ``status``, ``reason`` and ``remedy``.

Languages and model sizes
-------------------------

spaCy carries models for many languages; ``--lang`` selects one and
``--model-size`` chooses between ``sm`` (the default), ``md``, ``lg`` and
``trf``. The model is resolved before anything is loaded. A language spaCy
publishes no entity model for falls back to the multilingual
``xx_ent_wiki_sm``, which recognises only broad categories (``PER``, ``ORG``,
``LOC``, ``MISC``) — and that substitution is announced, so a thinner result is
read as a limitation of the model rather than as an empty document.

.. prompt:: bash $

   python -m scikitplot.cleanprompt doctor --lang de --format json

The ``languages`` section of that report names the resolved model, whether it
is installed, and the command that installs it.

When the engines disagree
-------------------------

On one sentence the two engines may label the same name differently — a
person to one, an organisation to the other — and under ``both`` the span
that wins the overlap decides the label. What matters is that the value left
the text either way, and that its label is stable within the vault, so
restoration puts back the right value whichever engine was right about the
category. An engine's span is also checked before it becomes a vault value: a
span that swallowed an unmatched bracket (``Atatürk[e``) is trimmed back to
the name.

Behind a proxy
--------------

Recent NLTK releases refuse to download through a proxy unless told the proxy
is trusted, and say so with a message naming ``NLTK_ALLOW_PROXIED_URLOPEN``.
That is NLTK's decision about your network, not CleanPrompt's; read its
message before setting it. spaCy models install with ``pip`` and follow your
``pip`` configuration.
