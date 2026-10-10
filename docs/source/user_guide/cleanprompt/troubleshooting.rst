..
  docs/source/user_guide/cleanprompt/troubleshooting.rst

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-troubleshooting:

======================================================================
Troubleshooting and questions
======================================================================

When output is surprising
-------------------------

Debug the boundary in this order:

1. Run ``doctor`` and resolve any capability or detector gap relevant to the
   data you expect to find. With ``--ner``, read ``detection.ner_ready`` and
   ``detection.ner_remedy``.
2. Run ``inspect`` on the exact input and confirm the expected value is found.
3. Run ``kinds`` to verify that the structural category you expect is enabled.
4. Add a precise ``--hide`` value, choose a broader profile, or enable the
   appropriate entity engine when the missing value is outside the structural
   pattern surface.
5. If restoration is incomplete, read the decode report. Try the normal
   lenient decoder first; use ``--exact`` only when exact placeholder spelling
   is required.
6. If a persisted vault is involved, confirm that encode and decode use the
   same vault and the same placeholder grammar (``doctor`` reports the path).

``doctor --format json`` and other JSON-output modes are preferable in CI or
machine-readable diagnostics; human-readable output is intended for terminal
use.

Questions
---------

**I installed spaCy and ``--ner`` still fails.**
    Installing the package is the first of two steps; the model is the second.
    The error names it — for English, ``python -m spacy download
    en_core_web_sm``. ``doctor --ner`` shows ``assets_ready: false`` until it
    is there.

**I downloaded NLTK's data and it still says a package is missing.**
    NLTK 3.9 moved its tagger and chunker to new data packages
    (``averaged_perceptron_tagger_eng``, ``maxent_ne_chunker_tab``). Older
    packages on disk are found but cannot be loaded by a newer NLTK. Run the
    exact command in the error; it names both the new and the old package for
    every missing group.

**NLTK's download fails with a "proxied fetch" message.**
    Recent NLTK releases refuse to download through a proxy unless told it is
    trusted, and their message names the setting. That decision is about your
    network; read NLTK's message before changing it.

**``auto`` chose NLTK although spaCy is installed.**
    ``auto`` picks the first engine that is *ready*. spaCy without its model is
    not, so a ready NLTK is chosen. ``doctor --ner`` shows both engines'
    readiness and which one ``auto`` would run.

**The redacted text looks unchanged.**
    Either there was nothing the active detectors recognise, or the relevant
    detector is off. The report says which: an empty result together with a
    high-severity blind spot is an *alert*, not a success.

**A value with an odd character in it was missed.**
    Invisible format characters, full-width letters and Unicode spaces or
    dashes are read through the detection view. Look-alike letters from
    another script, and a non-ASCII letter inside an email address, are not
    covered yet (:ref:`cleanprompt-security`); use ``--hide``.

**The model rewrote a placeholder and it was not restored.**
    ``decode`` repairs case, separator, Markdown escaping and line-wrap
    rewrites of labels the vault holds, and reports every repair. Anything
    further is listed as unresolved. ``--style surrogate`` avoids the problem
    for names and contact details by not sending a bracket token at all.

**``--hide -secret`` is refused.**
    A value that starts with a dash is written attached: ``--hide=-secret``.
    A text argument that starts with a dash needs ``--`` in front of it.

**``--form json`` is an error.**
    Long options are never abbreviated, so a script that works today keeps
    working when a later version adds an option with the same prefix.

**Can I put the vault next to my files?**
    You can with ``--vault PATH``, but the default deliberately avoids the
    working directory: a vault holds the removed values, and beside your files
    it is one ``git add .`` away from being committed.

**``flask --debug`` is refused in my container.**
    By design. The debugger runs code typed into the browser, and the page has
    no authentication. Use ``--debug`` on ``127.0.0.1`` only.

**A custom pattern made a run hang.**
    Python's regular expressions backtrack, and a pattern with nested
    repetition can take exponential time on a near-match. Such patterns are
    reported when they load; ``cleanprompt packs --pack-file FILE --check``
    lists them with rewrites (:ref:`cleanprompt-pack-trust`).

**A ``warning: ... nested-quantifier`` line appears when my pack loads.**
    The pack loaded and the run continues; the warning goes to standard
    error, so standard output is unchanged. Rewrite the pattern as the
    warning suggests. If you are sure its inputs are safe, mark it
    ``risk: accepted`` with a ``risk_reason`` in the pack. Use
    ``--pattern-risk ignore`` to silence the check for one run, or
    ``refuse`` to make findings stop the run.

**``--surrogates`` is refused.**
    It needs ``--style surrogate``; the set only changes which names are used.
    A set that lists ``EMAIL``, ``PHONE``, ``URL`` or a credential kind is
    refused because those forms are fixed (:ref:`cleanprompt-surrogate-sets`).

**Encoding says "Appending would mix two kinds of stand-in".**
    The vault was written with another ``--style`` or surrogate set; the
    message names it. Pass the same options again, or start a new vault with
    ``--vault-mode overwrite`` or another ``--vault``. Decoding never needs
    the set file.

Examples and API reference
--------------------------

The executable gallery is a learning path covering the base API, the command
line, notebooks and modules, policies, encryption, packs and formats, batch
workflows, agents, and recipes. Start at :ref:`cleanprompt_examples`.

For symbol-level signatures and return types, use the API reference for
:mod:`scikitplot.cleanprompt`. Useful entry points include:

* :func:`encode`, :func:`decode`, :class:`Session`, and :func:`session`;
* :class:`Redactor`, :func:`restore`, and :class:`RedactionPolicy`;
* :class:`DetectorRegistry`, :class:`RegexDetector`, and
  :class:`LiteralDetector`;
* :func:`capabilities`, :func:`diagnose` and :func:`engine_readiness`;
* :class:`FluentCleanPrompt`, :class:`CleanPlan`, and :class:`Guard`.
