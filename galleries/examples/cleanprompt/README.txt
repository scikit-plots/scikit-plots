.. _cleanprompt_examples:

Cleanprompt
===========

.. currentmodule:: scikitplot.cleanprompt

Examples for :py:mod:`~scikitplot.cleanprompt` are ordered as a learning path.
The submodule replaces sensitive values in a text with stable placeholders
before that text is sent to a language model, and puts the values back
afterwards:

.. code-block:: text

    your text
        ↓
    detect        regular expressions, and optionally named entities
        ↓
    assign        one stable label per distinct value
        ↓
    rewrite       a single pass over the original text
        ↓
    ── the model ──   the only hop this submodule does not control
        ↓
    restore       the labels become the values again

.. prompt:: bash $

    # The base tier needs nothing beyond the standard library
    pip install scikit-plots

    # Optional tiers, each independent of the others
    # spaCy entity detection, NLTK entity detection, local Flask interface, the Fernet vault cipher
    pip install scikit-plots[cleanprompt]

    # What is active on this machine, and what is blind
    python -m scikitplot.cleanprompt doctor

.. seealso::
    * https://github.com/takashiishida/cleanprompt — the upstream project this
      submodule was restructured from.
    * :ref:`mcp_examples` — the other submodule with a centrally registered CLI.

Start here
----------

1. **Basics** — hide one value, send the prompt, get the value back. The whole
   loop in four commands and four function calls.
2. **The command line** — every input route, the option grammar, exit codes,
   and the edge cases that bite: a pasted paragraph, a dash-leading value,
   ``--``, abbreviations, pipes, and a closed pipe.
3. **The Python API** — :func:`encode`/:func:`decode`, :class:`Session`,
   :class:`Redactor`, and what each one does when the input is degenerate.
4. **Notebooks and modules** — send a whole ``.ipynb`` or ``.py``: the schema
   renamed role by role, paths and rendered data removed, and the code still
   readable enough for a model to help with.
5. **Moderate** — policies and profiles, the vault lifecycle, append versus
   overwrite, entity engines, languages, and the surrogate style.
6. **Advanced** — custom detectors, overlap arbitration, vault encryption
   without a compiled dependency, the lossy model hop, and the nine invariants.
7. **Recipes** — a CI gate, a notebook helper, a logging filter, a batch
   pipeline, and an agent loop.
8. **Packs and formats** — records, configs and whole folders: YAML packs that
   read field names (``mrn``, ``DB_PASSWORD``, ``member_id``), formats from
   CSV to Word to zip, the fluent plan (``all``, one domain, or any
   combination), your own pack from a JSON or YAML file, and the corpus
   bridge.
9. **A gate for any model or agent** — :class:`Guard` in front of any client:
   checked outgoing text, chat messages, tool calls, streamed replies,
   ``ask --via`` for command-line models, the agent skill, the MCP server for
   agents without code, plan files a team pins, and audit logs that record what
   left without keeping what was removed.

Which surface should I use?
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Goal
     - Start with
   * - Paste a prompt into a chat window by hand
     - ``cleanprompt encode`` / ``cleanprompt decode``
   * - See what *would* be removed, changing nothing
     - ``cleanprompt inspect``
   * - Fail a build when a file contains sensitive values
     - ``cleanprompt scan``
   * - One call in an application
     - :func:`encode` and :func:`decode`
   * - A multi-turn conversation with one stable vault
     - :class:`Session`, usually as a ``with`` block
   * - Full control of detectors, overlap and limits
     - :class:`Redactor` with a :class:`RedactionPolicy`
   * - Add a detector of your own
     - :class:`RegexDetector` / :class:`LiteralDetector` in a
       :class:`DetectorRegistry`
   * - Send a notebook, a module or any analysis file
     - ``cleanprompt encode --in analysis.ipynb --out analysis.clean.ipynb``
   * - See a file's schema before sending it
     - ``cleanprompt inspect --in analysis.ipynb``
   * - Records, configs, Office files, a whole folder or a zip
     - :class:`FluentCleanPrompt` → :class:`Cleaner`, or
       ``cleanprompt batch SRC --out DST``
   * - Hide your own domain's fields
     - a pack file: ``cleanprompt batch --pack-file hr.yaml --pack hr``
   * - Make corpus documents safe before indexing
     - :func:`redact_documents`
   * - Put any model, SDK or agent behind one privacy gate
     - :class:`Guard` (``FluentCleanPrompt().guard()``)
   * - Guard a command-line model (``ollama``, ``llm``, ...)
     - ``cleanprompt ask --via "COMMAND"``
   * - Teach an AI assistant to use it
     - ``cleanprompt skill --write DIR``
   * - Give any MCP agent the gate, without code
     - ``cleanprompt mcp --root PROJECT``
   * - Pin a team's packs and rules, and fail CI when they drift
     - ``cleanprompt plan --write`` / ``--check``, and ``--plan FILE``

Capability matrix
-----------------

The base tier imports **no third-party package**, so the normal gallery path
runs on a bare installation. Optional capabilities are preflighted and reported
as a visible ``SKIP`` when unavailable, never silently omitted.

.. list-table::
   :header-rows: 1
   :widths: 30 20 22 28

   * - Example
     - Normal path
     - Optional capability
     - Behavior when unavailable
   * - Basics
     - standard library
     - none
     - not applicable
   * - Command line
     - standard library
     - ``click`` frontend
     - falls back to ``argparse``; parity is asserted either way
   * - Python API
     - standard library
     - none
     - not applicable
   * - Notebooks and modules
     - standard library
     - none
     - not applicable
   * - Moderate
     - standard library
     - spaCy / NLTK entity engines
     - specific ``SKIP`` per engine; the pattern half still runs
   * - Advanced
     - standard library
     - ``cryptography`` for ``--cipher fernet``
     - the ``portable`` cipher runs regardless; Fernet section ``SKIP``
   * - Recipes
     - standard library
     - Flask for the web recipe
     - configuration shown, server not started
   * - Packs and formats
     - standard library (packs are compiled JSON)
     - PyYAML for YAML packs; :mod:`scikitplot.corpus` for the bridge
     - JSON packs still shown; specific ``SKIP`` for each
   * - A gate for any model
     - standard library
     - none (the model is a stand-in function)
     - not applicable

Gallery reliability rule
------------------------

The examples distinguish optional capability absence from real defects:

``missing optional package, model or data package``
    Report a visible, specific ``SKIP`` and continue when the example can
    remain truthful.

``invalid public API / failed round trip / a value surviving into the redacted text``
    Fail visibly. The gallery must not convert a leak or a regression into a
    skip, because the one property this submodule sells is that the value did
    not go to the model.

Every example asserts its own round trip. An example that prints a redacted
prompt has already checked that the original value is absent from it.

Where the vault goes during a documentation build
-------------------------------------------------

A vault holds the **removed values in clear text**. Left to itself the CLI
writes one into the platform state directory —
``$XDG_STATE_HOME/cleanprompt/vault.json`` on Linux,
``%LOCALAPPDATA%\cleanprompt\`` on Windows — which is right for a person at a
terminal and wrong for a documentation builder.

Every script here therefore points ``CLEANPROMPT_VAULT`` at a
:class:`~tempfile.TemporaryDirectory` in its first cell and removes it in its
last. Nothing a gallery run produces outlives the build, and nothing is written
next to the source tree, where a ``git add .`` could commit it.

Reading the examples is not the same as running them
----------------------------------------------------

The redacted prompts printed by these examples contain no real values: the
addresses use the reserved ``example.com`` and ``example.invalid`` domains
(:rfc:`2606`), the telephone numbers the North American fiction block
``+1 555 0100``–``0199``, the addresses the documentation ranges
``192.0.2.0/24`` (:rfc:`5737`) and ``2001:db8::/32`` (:rfc:`3849`), and the
card number the publicly published test value.

Use the same discipline in your own examples and tests. A tutorial that leaks a
real value has undone the thing it was teaching.

Browser / WASM note
-------------------

The base tier is pure Python and has no filesystem requirement beyond the vault
path, so pattern detection, the API and the placeholder round trip are strong
JupyterLite candidates. Subprocess execution, the platform state directory,
``getpass`` prompts, native spaCy models, NLTK data downloads and a listening
Flask server should not be assumed available in a browser runtime.

The examples that shell out to the CLI say so, and the equivalent Python call
is shown beside every one of them.
