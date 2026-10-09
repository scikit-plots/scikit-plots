..
  docs/source/user_guide/cleanprompt/index.rst

..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-index:

======================================================================
CleanPrompt User Guide
======================================================================

CleanPrompt is the local privacy boundary in :mod:`scikitplot.cleanprompt`.
It replaces values that its active detectors identify with stable placeholders
(or, when explicitly requested, selected surrogate values) before text leaves
your process, then restores the original values after a reply comes back.

The base tier is implemented with the Python standard library. Optional tiers
add named-entity detection, a local web interface, and the Fernet vault cipher.
The base package can therefore be imported and used without loading those
third-party packages.

.. important::

   CleanPrompt protects values that it **detects or that you explicitly tell it
   to hide**. It is not a proof that arbitrary text contains no sensitive
   information. Run ``doctor`` to see which detection surfaces are active and
   ``inspect`` when you need to review what a particular input would expose.

The security boundary
---------------------

The central design rule is simple:

.. code-block:: text

   original text
       |
       v
   detect -> assign stable labels -> rewrite
       |                              |
       |                              +---- safe text ----> model / tool / agent
       |
       +---- handle / vault stays local
                                      |
                           model reply v
                                 restore
                                      |
                                      v
                               local clear text

For the high-level Python API, :func:`encode` returns two different kinds of
state:

``EncodedPrompt.text``
    The redacted text intended to cross the model boundary.

``EncodedPrompt.handle``
    The local restoration authority. It owns a :class:`Vault` containing the
    removed values and **must not be sent with the prompt**.

``EncodedPrompt.report``
    Local diagnostics describing what happened and which detection surfaces
    were active. Reports can include suggested terms derived from the input, so
    review them before logging, attaching, or transmitting them.

Treat a :class:`Handle` or :class:`Vault` like a secret. Their representations
avoid printing stored values, but exporting either is an explicit operation
that can contain those values.

Choosing the right surface
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 39 61

   * - Goal
     - Start with
   * - One prompt/reply exchange in Python
     - :func:`encode` and :func:`decode`
   * - A multi-turn conversation with stable labels
     - :class:`Session` or :func:`session`
   * - Full control over policy and detectors
     - :class:`Redactor`, :class:`RedactionPolicy`, and
       :class:`DetectorRegistry`
   * - Paste text into a chat by hand
     - ``cleanprompt encode`` and ``cleanprompt decode``
   * - Preview what would be removed
     - ``cleanprompt inspect``
   * - Fail CI when findings are present
     - ``cleanprompt scan``
   * - Files, records, folders, Office documents, or archives
     - :class:`FluentCleanPrompt` / ``cleanprompt batch``
   * - Put a privacy gate in front of a model client
     - :class:`Guard`
   * - Guard an arbitrary command-line model
     - ``cleanprompt ask --via "COMMAND"``
   * - Give an MCP agent a local redaction boundary
     - ``cleanprompt mcp``
   * - Pin team configuration and detect definition drift
     - ``cleanprompt plan``
   * - Check installation and optional capabilities
     - :func:`capabilities` / ``cleanprompt doctor``

Installation and optional capabilities
--------------------------------------

The normal installation already includes the base CleanPrompt implementation:

.. prompt:: bash $

   pip install scikit-plots

Install the complete optional CleanPrompt bundle when you want all optional
integrations:

.. prompt:: bash $

   pip install "scikit-plots[cleanprompt]"

Or install only the tier you need:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Extra
     - Main dependency
     - Purpose
   * - ``cleanprompt-ner``
     - spaCy
     - named-entity detection
   * - ``cleanprompt-nltk``
     - NLTK
     - English named-entity detection without spaCy
   * - ``cleanprompt-web``
     - Flask
     - local browser interface
   * - ``cleanprompt-crypto``
     - cryptography
     - optional Fernet vault cipher

A lightweight partial distribution is also available as
``scikit-plots-cleanprompt``. It composes with ``scikit-plots-skinny`` and is
intended for installations that do not need the full ``scikit-plots``
distribution. Do not install the partial distribution beside full
``scikit-plots`` because both distributions own the same CleanPrompt package
files.

Before relying on an optional capability, inspect the installation instead of
assuming that importing a dependency means it is usable:

.. prompt:: python >>>

   from scikitplot.cleanprompt import capabilities
   reports = capabilities()
   reports["ner"].status

:func:`capabilities` reports the ``ner``, ``nltk``, ``web``, and ``crypto``
tiers without importing their third-party packages. Status is represented by
:class:`CapabilityStatus`, which distinguishes states such as ``ABSENT``,
``BROKEN``, ``INCOMPATIBLE``, and ``AVAILABLE``.

The command-line equivalent is:

.. prompt:: bash $

   python -m scikitplot.cleanprompt doctor

Python quick start
------------------

One exchange: ``encode`` and ``decode``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`encode` when you want the smallest safe boundary around one prompt:

.. prompt:: python >>>

   from scikitplot.cleanprompt import encode, decode

   encoded = encode(
       "Mail ada@example.com about the Acme renewal",
       hide=["Acme"],
   )
   encoded.text
   # 'Mail [EMAIL-1] about the [CUSTOM-1] renewal'

Only ``encoded.text`` goes to the model. Keep ``encoded.handle`` local:

.. prompt:: python >>>

   model_reply = "I will contact [EMAIL-1] about [CUSTOM-1]."
   decode(model_reply, encoded.handle)
   # 'I will contact ada@example.com about Acme.'

You can unpack :class:`EncodedPrompt` directly:

.. prompt:: python >>>

   safe_text, handle = encode("Write to ada@example.com")

Call :meth:`Handle.clear` when the restoration authority is no longer needed.

Multi-turn conversations: ``Session``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A conversation needs one vault across turns so that the same value keeps the
same label. :class:`Session` owns that lifetime deliberately:

.. prompt:: python >>>

   from scikitplot.cleanprompt import session

   with session() as chat:
       first = chat.encode("Mail ada@example.com")
       second = chat.encode("Remind ada@example.com tomorrow")
       # both turns use the same EMAIL label
       restored = chat.decode("I reminded [EMAIL-1].")

Leaving the context manager clears the session's vault. Use one session per
conversation; the class documents itself as a single-conversation object, not
a shared multi-thread conversation bus.

For model clients that already accept a callable, :meth:`Session.roundtrip`
keeps the boundary narrow:

.. prompt:: python >>>

   with session() as chat:
       reply = chat.roundtrip(
           "Mail ada@example.com",
           send=lambda safe: "Sent to " + safe.split()[-1],
       )

The callable receives only the redacted prompt and returns the model reply.

Full control: ``Redactor`` and policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :class:`Redactor` when you need explicit detector or overlap control:

.. prompt:: python >>>

   from scikitplot.cleanprompt import Redactor, restore

   redactor = Redactor()
   result = redactor.redact(
       "Mail ada@example.com about Project North",
       extra_terms=["Project North"],
   )
   result.text
   # 'Mail [EMAIL-1] about [CUSTOM-1]'

   restored = restore(result.text, result.vault)
   restored.text
   # 'Mail ada@example.com about Project North'

A :class:`Redactor` is stateless between documents. State that must survive
turns belongs in :class:`Session`, :class:`Guard`, or a persisted vault rather
than hidden process globals.

Detection, policy, and diagnostics
----------------------------------

Built-in structural detectors cover categories such as email addresses, phone
numbers, URLs, network addresses, payment/identity identifiers, and credential
shapes. The exact active library is inspectable with ``kinds`` and should be
preferred over copying a hard-coded list into automation:

.. prompt:: bash $

   python -m scikitplot.cleanprompt kinds

Named policy profiles provide three starting points:

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Profile
     - Intended starting point
     - Trade-off
   * - ``minimal``
     - email, phone, and URL contact details
     - narrowest built-in pattern surface
   * - ``balanced``
     - the default high-precision structural patterns
     - normal general-purpose starting point
   * - ``strict``
     - the default set plus title-case detection, with case folding
     - broader coverage and more false positives

Use the ``profile=`` keyword on the high-level Python APIs or ``--profile``
on the CLI to select one. More specific controls are available when the profile
is not enough:

``hide`` / ``--hide``
    Exact additional values that must be replaced.

``allow`` / ``--allow``
    Values that must not be replaced by the active policy.

``kinds``
    Restrict the structural pattern categories considered.

``ner=True`` / ``--ner``
    Add a named-entity engine. This is optional capability, not part of the
    standard-library base detector set.

For a source-grounded view of what the current policy can and cannot inspect,
use :func:`diagnose` or ``doctor``. For a particular piece of text, use
``inspect`` before sending anything:

.. prompt:: bash $

   python -m scikitplot.cleanprompt inspect "Ada will mail ada@example.com"

An empty redaction result is therefore not automatically evidence that the text
contains no sensitive information: it can also mean that a relevant detector
is disabled or unavailable.

Placeholder and surrogate styles
--------------------------------

``placeholder`` is the default style. Stable tokens such as ``[EMAIL-1]`` make
it explicit that the text was transformed and are the safest representation
for credentials or other values that must not be confused with real data.

``surrogate`` replaces selected human-readable categories with deterministic
stand-ins intended to make natural-language prompts easier for a model to read.
Credential-like categories remain placeholders instead of being turned into
plausible credentials.

.. prompt:: bash $

   python -m scikitplot.cleanprompt encode --style surrogate \
       "Mail ada@example.com about this request"

Restoration is tolerant of a bounded set of common model rewrites to issued
placeholder tokens (for example case, separator, Markdown escaping, or line
wrapping), and reports repairs. Use ``decode --exact`` when only the exact
issued spelling should match.

Vault lifetime and encryption
-----------------------------

The vault is the sensitive half of the operation. It contains the values that
were removed and is the authority used to restore them.

The two common CLI workflows intentionally have different state semantics:

``encode``
    Uses the default state-directory vault and appends by default, which keeps
    labels stable across conversational turns.

``redact``
    Is the explicit/script-oriented form and uses overwrite semantics by
    default unless another vault mode is requested.

At the end of a conversation, inspect and remove the persisted vault:

.. prompt:: bash $

   python -m scikitplot.cleanprompt forget
   python -m scikitplot.cleanprompt forget --force

The first command is a dry run. The second removes the vault file.
Unlinking a file is not a promise of physical secure erasure on journalling
filesystems, SSDs, or backups.

For values that must not be stored as clear text, use vault encryption. The
portable cipher is available in the base tier; Fernet is an optional
``cryptography``-backed alternative:

.. prompt:: bash $

   python -m scikitplot.cleanprompt doctor --new-key
   python -m scikitplot.cleanprompt encode --encrypt "Mail ada@example.com"
   python -m scikitplot.cleanprompt encode --encrypt --cipher fernet \
       "Mail ada@example.com"

Keep the passphrase outside the vault. Losing it makes the encrypted vault
unrecoverable.

Command-line guide
------------------

Two entry points reach the same CleanPrompt command surface:

.. prompt:: bash $

   python -m scikitplot.cleanprompt doctor
   scikitplot cleanprompt doctor

The current commands are grouped below by intent. Aliases are conveniences;
automation should prefer the canonical command names.

.. list-table::
   :header-rows: 1
   :widths: 19 25 56

   * - Command
     - Category
     - Purpose
   * - ``encode``
     - prompt boundary
     - redact text for a conversation; aliases: ``clean``, ``prompt``
   * - ``decode``
     - prompt boundary
     - restore a reply from the vault; alias: ``restore``
   * - ``redact``
     - explicit pipeline
     - redact text and write a named/default vault with a report
   * - ``forget``
     - vault lifecycle
     - preview or delete a vault; alias: ``clear-vault``
   * - ``roundtrip``
     - learning/debugging
     - show redact -> reply -> restore without a persistent vault; alias: ``demo``
   * - ``inspect``
     - diagnosis
     - dry-run one input and report findings/blind spots
   * - ``scan``
     - automation
     - CI-friendly detection gate
   * - ``doctor``
     - diagnosis
     - report installation, detectors, and capability gaps; alias: ``capabilities``
   * - ``kinds``
     - discovery
     - list built-in detection kinds
   * - ``cli``
     - interactive
     - terminal session; aliases: ``session``, ``repl``
   * - ``flask``
     - interactive
     - local browser interface; aliases: ``web``, ``serve``
   * - ``packs``
     - structured data
     - list/show/check pack and format definitions
   * - ``batch``
     - structured data
     - encode or decode a folder/archive copy
   * - ``ask``
     - model boundary
     - guard one prompt through a command-line model
   * - ``mcp``
     - agent boundary
     - expose local guarded tools over MCP stdio
   * - ``plan``
     - reproducibility
     - save/show/check a fingerprinted team plan
   * - ``skill``
     - agent setup
     - print or install CleanPrompt agent instructions
   * - ``docker``
     - deployment
     - emit container files for the local web interface

Use command-specific help for exact options; the command schema is rendered
from one internal definition for both supported CLI frontends:

.. prompt:: bash $

   python -m scikitplot.cleanprompt encode --help
   python -m scikitplot.cleanprompt batch --help

Exit status is part of the scripting contract:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Status
     - Meaning
   * - ``0``
     - command completed successfully
   * - ``1``
     - handled runtime error, such as an unreadable file or unusable vault
   * - ``2``
     - command-line usage error
   * - ``3``
     - ``scan`` found more sensitive values than allowed
   * - ``69``
     - a required optional capability is unavailable
   * - ``130``
     - interrupted
   * - ``141``
     - downstream pipe closed early

Structured files, packs, and reproducible plans
-----------------------------------------------

Prompt strings are only one input shape. :class:`FluentCleanPrompt` builds an
immutable :class:`CleanPlan` for files and structured data, then materializes a
:class:`Cleaner`:

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

   for item in cleaner.encode_tree("project/"):
       print(item.status, item.relative)

Packs describe domain-specific field semantics; formats describe how supported
file types are read and rewritten. A plan validates selections before a
filesystem operation begins and fingerprints the resolved definitions so team
configuration can be checked for drift.

Use ``packs`` to inspect definitions and ``batch`` to transform a directory or
archive from the CLI. Custom JSON definitions work without a YAML parser;
custom YAML definitions require the YAML capability used by that loader.

CleanPrompt can also redact documents produced by :mod:`scikitplot.corpus`
before they are embedded, indexed, or prompted. See :ref:`corpus-index` for
the corpus side of that workflow.

Guarding models, tools, and agents
---------------------------------

:class:`Guard` is the boundary for integrations where text should be checked
immediately before it crosses into a model/tool and decoded immediately after
it comes back.

A guard can encode/decode plain text and structured objects, process streamed
replies, and wrap callables through methods such as :meth:`Guard.ask` and
:meth:`Guard.chat`.

.. prompt:: python >>>

   from scikitplot.cleanprompt import FluentCleanPrompt

   guard = FluentCleanPrompt().profile("strict").guard()
   reply = guard.ask(
       "Mail ada@example.com",
       call=lambda safe: "Queued " + safe.split()[-1],
   )

For command-line model clients, ``ask`` performs the same boundary without a
shell and keeps the values in memory:

.. prompt:: bash $

   python -m scikitplot.cleanprompt ask \
       --via "ollama run llama3" \
       "Summarize the request from ada@example.com"

For agents that speak Model Context Protocol, ``mcp`` serves guarded file tools
over standard input/output. File access is constrained to configured roots:

.. prompt:: bash $

   python -m scikitplot.cleanprompt mcp --root ./project

``skill`` emits agent-facing instructions that describe the same boundary, and
``plan`` lets a team pin the packs/rules those integrations are expected to
use.

Reliability and failure behavior
--------------------------------

A privacy boundary should fail visibly rather than quietly reinterpret an
ambiguous state. Important behaviors include:

* :func:`decode` can run in strict mode and reject placeholders that the handle
  never issued.
* :func:`restore` reports restored, repaired, unknown, and unused labels rather
  than reducing the result to one success boolean.
* :class:`RedactionPolicy` carries explicit resource limits; exceeding a limit
  raises instead of silently truncating work.
* overlap behavior is explicit through :class:`OverlapStrategy`; strict mode
  can turn overlapping detector claims into an error.
* optional capability checks distinguish "not installed" from
  "installed but broken".
* :class:`Session` and :class:`Guard` own state explicitly so independent
  conversations do not depend on hidden module-global vault state.

The implementation also maintains regression coverage for round-trip behavior,
no detected value surviving the rewritten text, deterministic labeling,
stateless :class:`Redactor` behavior, idempotent re-redaction, and bounded
resource handling. These are guarantees about the transformation pipeline for
**detected values**; they do not make an inactive detector detect new kinds of
information.

Troubleshooting workflow
------------------------

When output is surprising, debug the boundary in this order:

1. Run ``doctor`` and resolve any capability or detector gap relevant to the
   data you expect to find.
2. Run ``inspect`` on the exact input and confirm the expected value is found.
3. Run ``kinds`` to verify that the structural category you expect is enabled.
4. Add a precise ``--hide`` value, choose a broader profile, or enable the
   appropriate NER tier when the missing value is outside the structural
   pattern surface.
5. If restoration is incomplete, inspect the decode report. Try the normal
   lenient decoder first; use ``--exact`` only when exact placeholder spelling
   is required.
6. If a persisted vault is involved, confirm that encode and decode use the
   same vault and placeholder grammar.

``doctor --format json`` and other JSON-output modes are preferable in CI or
machine-readable diagnostics; human-readable output is intended for terminal
use.

Examples and API reference
--------------------------

The executable CleanPrompt gallery is a learning path covering the base API,
CLI edge cases, notebooks/modules, policies, encryption, packs/formats, batch
workflows, agents, and recipes. Start at :ref:`cleanprompt_examples`.

For symbol-level signatures and return types, use the API reference for
:mod:`scikitplot.cleanprompt` and the docstrings of the specific public object.
Useful entry points include:

* :func:`encode`, :func:`decode`, :class:`Session`, and :func:`session`;
* :class:`Redactor`, :func:`restore`, and :class:`RedactionPolicy`;
* :class:`DetectorRegistry`, :class:`RegexDetector`, and
  :class:`LiteralDetector`;
* :func:`capabilities` and :func:`diagnose`;
* :class:`FluentCleanPrompt`, :class:`CleanPlan`, and :class:`Guard`.

.. seealso::

   * :ref:`cleanprompt_examples`
   * :ref:`corpus-index`
   * :mod:`scikitplot.cleanprompt`
