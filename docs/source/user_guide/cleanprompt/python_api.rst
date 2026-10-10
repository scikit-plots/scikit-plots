..
  docs/source/user_guide/cleanprompt/python_api.rst

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-python-api:

======================================================================
The Python API
======================================================================

Three levels, from the smallest boundary to full control. Pick the highest one
that does the job.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Level
     - Use it when
   * - :func:`encode` / :func:`decode`
     - one prompt, one reply
   * - :class:`Session`
     - a conversation: the same value keeps the same label across turns
   * - :class:`Redactor` / :func:`restore`
     - you choose the policy, the detectors and the overlap rule yourself

One exchange: ``encode`` and ``decode``
---------------------------------------

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

:func:`encode` returns an :class:`EncodedPrompt` with three parts:

``EncodedPrompt.text``
    The redacted text intended to cross the model boundary.

``EncodedPrompt.handle``
    The local restoration authority. It owns a :class:`Vault` containing the
    removed values and **must not be sent with the prompt**.

``EncodedPrompt.report``
    Local diagnostics describing what happened and which detection surfaces
    were active. Reports can include suggested terms derived from the input, so
    review them before logging, attaching, or transmitting them.

It unpacks as a pair:

.. prompt:: python >>>

   safe_text, handle = encode("Write to ada@example.com")

Call :meth:`Handle.clear` when the restoration authority is no longer needed.
A :class:`Handle` is frozen and its representation shows counts, never
values. :meth:`Handle.export` and :meth:`Handle.load` make it JSON-portable,
which is an explicit operation whose output contains the values — store it as
you would a password.

``decode(reply, handle, strict=True)`` refuses a reply that contains a
placeholder the handle never issued, instead of leaving it in place.

What ``encode`` can be told
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``profile=``
    ``"minimal"``, ``"balanced"`` (the default) or ``"strict"`` — see
    *Profiles* below.
``hide=``
    Exact additional values that must be replaced, for example a project code
    name. ``word_boundary=True`` matches them only as whole words.
``allow=``
    Values that must **not** be replaced even when a detector finds them.
``kinds=``
    Restrict the structural pattern categories considered. Narrow with care:
    switching ``CREDIT_CARD`` off does not leave a card alone, it lets a
    shorter pattern match a *fragment* of it. Prefer ``allow=`` for a
    specific surface.
``ignore_case=``
    Match exact terms regardless of case.
``ner=True``
    Add a named-entity engine; ``engine=`` (``"auto"``, ``"spacy"``,
    ``"nltk"``, ``"both"``, ``"none"``), ``language=``, ``size=`` and
    ``model=`` select it. A request that cannot be met raises
    :class:`CapabilityError` with the remedy — see
    :ref:`cleanprompt-entity-detection`.
``policy=`` / ``registry=``
    A full :class:`RedactionPolicy` or :class:`DetectorRegistry`, for the
    cases the keywords do not cover.

Profiles
^^^^^^^^

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

A conversation: ``Session``
---------------------------

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
conversation.

:meth:`Session.encode` returns a plain string — unlike the module-level
:func:`encode` — because the session owns the one handle. Handing back a
per-turn copy would invite decoding turn three with turn one's handle, which
silently restores the wrong values.

For model clients that already accept a callable, :meth:`Session.roundtrip`
keeps the boundary narrow:

.. prompt:: python >>>

   with session() as chat:
       reply = chat.roundtrip(
           "Mail ada@example.com",
           send=lambda safe: "Sent to " + safe.split()[-1],
       )

The callable receives only the redacted prompt and returns the model reply.
That callable is the whole integration seam: any function taking a string and
returning a string works, which is why no provider needs to be known here.

:meth:`Session.decode_report` returns the full restoration report;
:attr:`Session.turns` and :meth:`Session.report` describe the conversation so
far without showing a value.

Full control: ``Redactor`` and ``restore``
------------------------------------------

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
than hidden process globals. To continue numbering from an earlier pass, pass
its entries explicitly: ``redactor.redact(text, seed=previous.entries)``.

:func:`restore` returns a :class:`RestorationResult` rather than a bare string:
``text``, the labels it restored, the rewritten spellings it ``repaired``, the
``unknown`` placeholders it could not resolve, and vault entries left unused.

A policy is data
^^^^^^^^^^^^^^^^

:class:`RedactionPolicy` is a frozen dataclass; :meth:`RedactionPolicy.evolve`
returns a new one. Two policies with the same fields have the same
fingerprint in any process, which is what makes a redaction reproducible.

.. prompt:: python >>>

   from scikitplot.cleanprompt import DEFAULT_POLICY, TagStyle

   surrogate = DEFAULT_POLICY.evolve(tag_style=TagStyle(style="surrogate"))
   Redactor(policy=surrogate).redact("Mail ada@example.com").text

For your own invented names, load a surrogate set
(:ref:`cleanprompt-surrogate-sets`) and attach it to the grammar, or name it
in a plan:

.. code-block:: python

   from scikitplot.cleanprompt import FluentCleanPrompt, TagStyle, load_surrogate_set

   names = load_surrogate_set("nordic.yaml")  # validated; names.identity is recorded
   style = TagStyle(style="surrogate", surrogate_set=names)
   cleaner = FluentCleanPrompt().style("surrogate").surrogates("nordic.yaml").materialize()

Its fields are ``kinds``, ``allow``, ``tag_style`` (the placeholder grammar and
style), ``overlap`` (an :class:`OverlapStrategy`), ``limits`` (input size,
span and term bounds; exceeding one raises), ``case_insensitive``,
``min_confidence`` and ``preserve_placeholders``.

Your own detector
^^^^^^^^^^^^^^^^^

A detector is an object with a ``name``, a ``kind``, a ``priority`` and a
``detect(text, policy)`` method yielding :class:`Span` objects over the text it
was given. :class:`RegexDetector` and :class:`LiteralDetector` cover most needs:

.. prompt:: python >>>

   from scikitplot.cleanprompt import DetectorRegistry, LiteralDetector, default_registry

   registry = default_registry()
   registry.add(LiteralDetector(["Project North", "Northwind"], kind="PROJECT"))
   Redactor(registry=registry).redact("Northwind ships Project North").text
   # '[PROJECT-1] ships [PROJECT-2]'

:class:`RegexDetector` and :class:`LiteralDetector` also read the detection
view (:ref:`cleanprompt-how-it-works`), so a term written with an invisible
character inside it is still found. A detector of your own reads it only if
it sets the class attribute ``reads_view = True`` — which is correct only when
every offset it yields indexes the string passed to ``detect``.

Diagnosis in Python
-------------------

:func:`diagnose` reports what a policy and registry can and cannot see —
active kinds, inactive kinds, tier status and blind spots — without importing
an optional dependency:

.. prompt:: python >>>

   from scikitplot.cleanprompt import diagnose

   report = diagnose()
   report.healthy, report.headline()

An empty redaction result is not evidence that the text contains no sensitive
information: it can also mean that a relevant detector is disabled or
unavailable. That is why every surface reports the result *together with* what
was looking.
