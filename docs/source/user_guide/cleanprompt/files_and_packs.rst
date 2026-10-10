..
  docs/source/user_guide/cleanprompt/files_and_packs.rst

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-files-and-packs:

======================================================================
Files, records, packs and plans
======================================================================

A record says what its values are in its **keys**, and a pattern cannot read a
key:

.. code-block:: text

   {"mrn": "00412345"}          an eight-digit number, to a pattern
   DB_PASSWORD=pw-1             four characters, to a pattern
   member_id,M-00412            nothing at all, to a pattern

Packs and formats are how CleanPrompt reads those keys.

Packs and formats
-----------------

A **pack** says *what to hide*: field names and their kinds, patterns with
executed examples, code vocabulary. A **format** says *how a file is read* —
CSV, JSON, JSON Lines, ``.env``, YAML, notebooks, Python modules, Word and
other Office documents, zip archives — and which packs ``auto`` gives it.

.. prompt:: bash $

   python -m scikitplot.cleanprompt packs              # list packs and formats
   python -m scikitplot.cleanprompt packs --show patient
   python -m scikitplot.cleanprompt packs --check      # definitions load and agree

Both are data. A pack never carries code: a check such as a checksum is a
named function in a fixed registry, and a pack can only name it. The built-in
definitions are compiled into a JSON file the base tier reads with the
standard library alone.

A plan for a set of files
-------------------------

:class:`FluentCleanPrompt` builds an immutable :class:`CleanPlan`, validated
before any file is touched, then materializes a :class:`Cleaner`:

.. prompt:: python >>>

   from scikitplot.cleanprompt import FluentCleanPrompt

   cleaner = (
       FluentCleanPrompt()
       .packs("patient", "pandas")
       .formats("ipynb", "csv", ".env")
       .style("placeholder")
       .infer_roles()
       .materialize()
   )

   for item in cleaner.encode_tree("project/", "safe-project/"):
       print(item.status, item.relative)
   # encoded .env
   # encoded data.csv

Every file a selected format reads is encoded into a mirror folder under one
vault. A file no format reads is skipped, and one that cannot be read safely is
refused; neither is written, so nothing unexamined leaves. A value hidden in
one file keeps its label in every other file.

From the command line, ``batch`` does the same for a folder or a zip:

.. prompt:: bash $

   python -m scikitplot.cleanprompt batch project/ --dry-run
   python -m scikitplot.cleanprompt batch project/ --out safe-project/
   python -m scikitplot.cleanprompt batch safe-project/ --decode --out restored/

``--dry-run`` is the real walk with nowhere to write (so it takes no
``--out``): it reports what would be read, skipped or refused, and which kinds
were found, without writing a file or touching the vault.
:meth:`Cleaner.decode_tree` is the Python counterpart of ``--decode``.

Notebooks and modules
^^^^^^^^^^^^^^^^^^^^^

A notebook or module is rewritten in place of its values without being
re-serialised: cell structure, ids and outputs come back byte-identical, and
the redacted code still parses. Column names discovered syntactically are
hidden whether or not their role is known; ``--infer-roles`` is the opt-in that
chooses role-preserving stand-ins from evidence. The gallery's notebooks
example shows it end to end.

Pinning a team's configuration
------------------------------

A plan can be saved and checked. Its fingerprint hashes the resolved
definitions, so any change to what a pack means changes the fingerprint, and a
stale plan is refused rather than silently drifting:

.. prompt:: bash $

   python -m scikitplot.cleanprompt plan --write team.plan.json --pack patient
   python -m scikitplot.cleanprompt plan --check team.plan.json
   python -m scikitplot.cleanprompt batch project/ --out safe/ --plan team.plan.json

Your own pack
-------------

.. code-block:: yaml

   # hr_pack.yaml
   name: hr
   version: 1
   summary: Our HR system's identifiers.
   requires: [personal]
   fields:
     - names: [employee_number, badge_id]
       kind: EMPLOYEE
       role: id
   patterns:
     - kind: EMPLOYEE
       pattern: '\bEMP-\d{6}\b'
       intent: An employee number in our format.
       examples_yes: ['EMP-004121']
       examples_no: ['EMP-12']

.. prompt:: bash $

   python -m scikitplot.cleanprompt packs --pack-file hr_pack.yaml --check
   python -m scikitplot.cleanprompt batch hr/ --out hr-safe/ --pack-file hr_pack.yaml --pack hr

``--pack-file`` is accepted by the commands that read files or records —
``packs``, ``batch``, ``plan``, ``ask`` and ``mcp`` — and by
:meth:`FluentCleanPrompt.custom` in Python. A badge number in a roster column
named ``badge_id`` and the same number in prose elsewhere get one label,
``[EMPLOYEE-1]``.

A custom definition gets no special trust: unknown keys are refused, its
patterns' examples are executed before it is used, it can only *name* a
validator from the fixed registry, YAML is read with ``safe_load``, and a file
larger than a megabyte is refused unread. JSON needs nothing installed; YAML
needs PyYAML and says so when it is missing. Redefining a built-in pack is an
error unless ``--replace-builtins`` asks for it.

.. _cleanprompt-pack-trust:

A pack file is trusted like code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validation proves that a pattern means what its examples say. It cannot prove
that the pattern finishes quickly on every input. Python's :mod:`re` engine
backtracks, so a pattern with nested repetition — ``^(a+)+$``, ``(x*)*``,
``(x|xx)+`` — can take time that doubles with each extra character of a
near-match, and a single document can stall a run.

**Every custom pattern is checked for those shapes when it loads.** The check
reads the pattern and reports three of them:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Finding
     - Example
     - Rewrite
   * - ``nested-quantifier``
     - ``(\w+\s?)*``
     - end each repetition with a required separator, ``(\w+\s)*\w*``; on
       Python 3.11+, make the inner part possessive, ``(\w++\s?)*``
   * - ``overlapping-alternation``
     - ``(a|ab)+``
     - factor the shared start, ``(?:ab?)+``
   * - ``adjacent-quantifiers``
     - ``\d+\d+``
     - merge them, ``\d{2,}``

By default it **warns**, and the pack still loads. The warning goes to
standard error and names the file, the pack and the pattern. It also lists
the rewrites and the ways to proceed::

    warning: tickets.yaml: pack tickets, pattern TICKET: nested-quantifier (high) in `(?:[A-Z]+\d*)+`: ...
      - drop the outer repetition when the group adds nothing: `(x+)+` matches exactly what `x+` matches
      - end each repetition with a required separator the inner part cannot match, ...
      - accept this pattern in its pack: add `risk: accepted` and a `risk_reason:` ...
      - silence the check for this run: --pattern-risk ignore ...
      - make findings fatal: --pattern-risk refuse ...

You decide what happens next, at the level that fits:

* **for one pattern** — accept it in its pack, with the reason next to it.
  Both keys are required, and an accepted pattern is never warned about or
  refused:

  .. code-block:: yaml

     patterns:
       - kind: TICKET
         pattern: '\b(?:[A-Z]+\d*)+-\d+\b'
         intent: A ticket id.
         examples_yes: ['ABC-12']
         risk: accepted
         risk_reason: inputs are single ticket ids, never prose

* **for one run** — ``--pattern-risk ignore`` (load quietly), ``warn``, or
  ``refuse`` (stop before any text is read). It works on ``packs``, ``batch``,
  ``plan``, ``ask`` and ``mcp``;
* **for a team** — ``pattern_risk: refuse`` saved in a plan file. A plan
  fixes every choice, so ``--pattern-risk`` cannot be combined with
  ``--plan``;
* **for a machine or a CI job** — ``CLEANPROMPT_PATTERN_RISK=refuse``. It
  applies when neither a flag nor a plan sets the mode.

``cleanprompt packs --pack-file tickets.yaml --check`` lists every finding
under ``pattern_risk``, accepted ones included with their reason, so a review
can see what was accepted.

In Python, findings are :class:`PatternRiskWarning`, a :class:`UserWarning`.
``warnings.simplefilter("error", PatternRiskWarning)`` makes them fatal in a
test suite, and ``load_custom(paths, pattern_risk="refuse")`` does the same
for one call. :func:`analyse_pattern` checks a single pattern.

The check reads the pattern the way Python runs it. Verbose patterns
(``flags: [VERBOSE]`` or ``(?x)``) are checked without their whitespace and
comments. Inline ``(?i)`` and ``(?s)`` and escaped characters such as
``\x41`` are taken into account. A group repeated with more than three choices
(``{1,40}``) counts as repeated, and a fixed count (``{4}``) does not. Inside
a repeated group, any part that can vary in length (``\w{1,3}``, ``\.?``)
can trade characters with the next repetition, and a group whose parts are
all optional separates nothing.
Possessive and atomic forms (``\w++``, ``(?>\w+)``, Python 3.11+) give
nothing back, so they are not reported.

The check is a warning, not a proof. It reports shapes known to backtrack
badly; it can report a shape that is harmless on the inputs a pattern will
really see (that is what ``risk: accepted`` is for); and it cannot prove that
a pattern is fast. Two runs separated only by an optional character
(``\s*:?\s*``) are not reported: they cost at most quadratic time, and
policy limits already bound every span. A pattern the check cannot read
(a scoped ``(?x:...)`` group, a conditional group) is reported as
``not-analysed``, never passed as safe. The built-in patterns produce no
findings, and every verdict above was checked against measured run time.
A random-pattern fuzz found no pattern the check passes that was slow; of
the patterns it reports, roughly a third to a half were measurably slow on
generic near-miss inputs. The rest may be harmless, or slow only on inputs
shaped for them. That uncertainty is why the default is *warn*.

So keep treating a pack file like code you would run:

* load pack files only from people you would accept code from;
* prefer bounded forms (``\d{6}`` rather than ``\d+``) wherever an identifier
  has a fixed shape — they are also more precise;
* keep the ``intent`` and examples honest, because they are what a reviewer
  reads.

Before sending a record to a model
----------------------------------

CleanPrompt can also redact documents produced by :mod:`scikitplot.corpus`
before they are embedded, indexed, or prompted. See :ref:`corpus-index` for
the corpus side of that workflow.
