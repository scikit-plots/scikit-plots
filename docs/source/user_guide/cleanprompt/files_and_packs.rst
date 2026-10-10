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
``(x|x)+`` — can take time that doubles with each extra character of a
near-match, and a single document can stall a run. A length limit on pattern
source exists, and it is not a defence against this.

So treat a pack file like code you would run:

* load pack files only from people you would accept code from;
* review every new pattern for nested repetition;
* prefer bounded forms (``\d{6}`` rather than ``\d+``) wherever an identifier
  has a fixed shape — they are also more precise;
* keep the ``intent`` and examples honest, because they are what a reviewer
  reads.

Before sending a record to a model
----------------------------------

CleanPrompt can also redact documents produced by :mod:`scikitplot.corpus`
before they are embedded, indexed, or prompted. See :ref:`corpus-index` for
the corpus side of that workflow.
