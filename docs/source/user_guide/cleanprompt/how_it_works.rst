..
  docs/source/user_guide/cleanprompt/how_it_works.rst

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-how-it-works:

======================================================================
How it works
======================================================================

A redaction is four steps, and the last one happens after the model has
answered:

.. code-block:: text

   detect ── resolve ── assign ── rewrite ──▶ [ the model ] ──▶ restore

Each step is shaped by one question: *can this step return successfully while
having hidden less than it claims?* A privacy tool that silently does less is
worse than one that refuses, because the caller sends the result.

1. Detect
---------

Every detector reads the **original text** and reports *spans*: a start, an
end, a category (``EMAIL``, ``PHONE``, ``PERSON`` …) and the matched value.
No detector ever sees text another stage has already rewritten; that is how a
placeholder never ends up inside another placeholder.

Three kinds of detector contribute:

**Structural patterns.** Curated regular expressions, each with a written
intent, positive and negative examples that the test suite executes, and — where
a shape is not enough — a small validator: a card number must also pass the
Luhn check, an IBAN the mod-97 check, an IPv4 address must have octets in
range. ``python -m scikitplot.cleanprompt kinds`` lists them.

**Exact terms.** Anything you name with ``hide=`` / ``--hide`` is found
wherever it occurs.

**Entity engines** (optional). spaCy or NLTK recognise names, organisations,
places and similar categories no regular expression can.
:ref:`cleanprompt-entity-detection` covers them.

A second reading: the detection view
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A value can be written so it reads the same to a person — and to a language
model — while a pattern no longer matches it: an invisible zero-width space
inside an address, full-width letters, a non-breaking space or a non-breaking
hyphen inside a telephone number.

.. code-block:: text

   written     mail ada<ZWSP>@example.com      (an invisible character inside)
   view        mail ada@example.com            (format characters removed)
   found       EMAIL at view offsets 5..20
   mapped      EMAIL at original offsets 5..21  (the invisible character included)
   sent        mail [EMAIL-1]
   restored    mail ada<ZWSP>@example.com      (exactly as written)

Structural and exact-term detectors therefore read the text twice: as written,
and through a **view** in which every Unicode *format* character (zero-width
spaces and joiners, the word joiner, the byte-order mark, the soft hyphen,
bidirectional controls, tag characters) is removed and every other non-ASCII
character is folded to its single-character compatibility form (full-width
letters and digits, mathematical letters, Unicode spaces, dashes and
apostrophes). ASCII is read exactly as written, so line breaks stay line
breaks.

Spans found in the view are mapped back onto the original text before they
join the others. The text itself is never changed: the vault keeps the value
exactly as it was written, invisible characters included, and restoration puts
exactly that back. The view can only add detections.

Only detectors whose results depend on nothing but the text they are given
read the view — structural patterns, pack patterns and exact terms. Entity
engines read the original only, and so do the detectors a structured format
builds from a document's layout (fields, regions, JSON tokens), because their
positions were computed from the original. Field *names* are read through the
same view, so a header written ``n<ZWSP>ame`` still names the ``name`` field.

2. Resolve
----------

Detectors disagree, and their spans overlap. When two spans overlap only
partly, they are **merged into one span over their full extent**. Dropping the
loser would leave the characters only it covered in the output, which is a
disclosure; merging is coarser — two values can become one placeholder — but it
cannot leak, and it round-trips exactly. When one span contains another, the
outer one wins. :class:`OverlapStrategy` makes the arbitration explicit, and
its strict mode turns any overlap into an error.

3. Assign
---------

Each distinct value gets one label, numbered per category in order of first
appearance: ``[EMAIL-1]``, ``[EMAIL-2]``, ``[PERSON-1]``. The same value gets
the same label every time it occurs, and two different values never share
one. Within a conversation — :class:`Session`, or the command line's
``encode`` in its default append mode — a value keeps its label across turns,
so a reply that quotes an earlier placeholder still restores to the right
value.

Entity labels are normalised to one vocabulary before they are assigned:
spaCy says ``ORG`` where NLTK says ``ORGANIZATION``, and without the
normalisation a vault written with one engine would not restore under the
other.

4. Rewrite
----------

One pass over the original text replaces every resolved span with its label.
The result is a :class:`RedactionResult` whose ``text`` is safe to send and
whose ``vault`` is not. They are different types precisely so that no
serializer can emit both by accident.

5. Restore
----------

The hop to the model is the one step this library does not control, and a
language model rewrites what it is given. Replies come back with
``[email-1]``, ``[EMAIL_1]``, ``[EMAIL 1]``, Markdown-escaped brackets, or a
label wrapped across a line.

:func:`restore` and :func:`decode` therefore recognise a bounded set of such
rewrites — and act on one **only** when it resolves to a label the vault
actually holds, so ordinary bracketed text such as ``[note 2]`` is left
alone. Every repair is reported. ``decode --exact`` on the command line, or
``lenient=False`` in Python, restores only the exact issued spelling. A
restoration that resolves nothing while the vault is not empty says so rather
than reporting a quiet zero.

Placeholders or surrogates
--------------------------

``placeholder`` is the default style: stable bracket tokens such as
``[EMAIL-1]`` make it obvious that the text was transformed, and they cannot be
mistaken for real data.

``surrogate`` (``--style surrogate``) puts an invented but ordinary-looking
value where a proper noun is the natural replacement — ``PERSON``, ``ORG``,
``GPE``, ``LOC``, ``FAC``, ``EMAIL``, ``PHONE``, ``URL`` — so the text reads as
prose and a model has nothing to normalise. Contact details use reserved forms
(``example.invalid`` addresses and links, the ``+1 555 0100``–``0199``
fiction block for telephone numbers). Credentials and identifiers —
``CREDIT_CARD``, ``IBAN``, ``SSN_US``, ``AWS_ACCESS_KEY``, ``JWT``,
``PRIVATE_KEY``, ``MAC``, ``IPV4``, ``IPV6`` — and ``NORP`` keep their
placeholders: a plausible card number or key can be mistaken for real, acted
on, or by chance *be* real.

A stand-in is never one that already occurs in the text, one already issued,
or one that contains a value the conversation holds; after a bounded search it
falls back to a placeholder. The style is part of the placeholder grammar, so
a vault written in one style is refused, not half-read, by the other.

.. _cleanprompt-surrogate-sets:

Your own invented names
^^^^^^^^^^^^^^^^^^^^^^^

The built-in names are English-sounding. ``Marion Holt`` in a Finnish or
Turkish paragraph reads oddly, and a model may comment on it. A **surrogate
set** is a small file with your own lists:

.. code-block:: yaml

   # nordic.yaml
   name: nordic
   version: 1
   summary: Nordic-sounding invented names.
   kinds:
     PERSON: {first: [Aino, Eero, Liv], last: [Halvorsen, Lindgren, Virtanen]}
     ORG:    {first: [Fjord, Norrsken], second: [Data, Logistik]}
     GPE:    [Granvik, Solberga]

.. code-block:: bash

   cleanprompt encode --ner --style surrogate --surrogates nordic.yaml \
     "Ada Lovelace met Grace Hopper at the Acme Corporation in Paris"
   # Aino Halvorsen met Eero Virtanen at Fjord Data in Granvik

* ``PERSON`` and ``ORG`` take two lists, which are combined so that
  consecutive people do not share a surname. ``GPE``, ``LOC`` and ``FAC``
  take one list. A kind the set leaves out uses the built-in names.
* E-mail addresses, telephone numbers and links are always built by
  CleanPrompt in their reserved forms, and credentials and identifiers always
  keep placeholders. A set that lists ``EMAIL``, ``PHONE``, ``URL`` or any
  other kind is refused, with the reason.
* Each entry must read as a name, and no two entries may look alike:
  letters, spaces and ``' - .`` and the typographic apostrophe, 1–64
  characters, starting and ending with a letter. Combining marks may follow
  a letter, at most two in a row. Full-width, circled and other
  drawn-differently forms of a letter are refused, an entry must be in one
  script (so a Cyrillic ``\u0410`` cannot pass for a Latin ``A``), and
  spellings that differ only in how a letter and its marks were typed are
  stored once, composed. Digits, ``@``, ``/``, brackets and invisible
  characters (zero-width characters, variation selectors, fillers) are
  refused. So is an entry that a default core pattern or a built-in pack
  pattern would detect, since the next encode would hide the stand-in
  itself. Combined names (``Aino Halvorsen``) are checked again when they
  are issued.
* **Choose names that are not ordinary words.** ``decode`` restores a
  stand-in wherever it appears in the reply. An entry such as ``The`` or
  ``Ash`` would turn every ``The`` in the model's answer into your value.
  The built-in names are chosen to be unusual for this reason.
* Every rule above still applies: no stand-in already in the text, none
  issued twice, none containing a value the conversation holds. When a set
  runs out of names, the remaining values get placeholders.
* ``--surrogates`` without ``--style surrogate`` is refused. The flag never
  switches the style on by itself.
* ``decode`` needs no set file. The vault records which set wrote it, as
  ``name@version#digest``, and the digest changes when any entry changes. To
  continue an existing vault with ``--vault-mode append`` (``encode``'s
  default), pass the same ``--style`` and set again: appending with another
  style or set is refused, so one vault never mixes two kinds of stand-in.

The guarantees, in plain words
------------------------------

Nine core runtime invariants hold for every redaction, and the test suite and
probes measure them on randomised documents:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Invariant
     - What it means for you
   * - Round trip
     - restoring the redacted text gives back exactly the original, as long as
       the original contained no literal placeholder of the active grammar
   * - No leakage
     - no detected value survives in the redacted text
   * - Determinism
     - the same text and policy give byte-identical output in any process
   * - Statelessness
     - one :class:`Redactor` can process many documents; they never affect
       each other's numbering
   * - Disjointedness
     - resolved spans never overlap
   * - Detector purity
     - no detector reads rewritten text (the detection view is a reading of
       the original, mapped back, not a rewrite)
   * - Idempotence
     - redacting already-redacted text finds nothing new
   * - Bounded
     - input size, span count and pattern count are limited by the policy;
       exceeding a limit raises, it never truncates silently
   * - Value identity
     - equal values get one label; different values never share one

These are guarantees about **detected values**. They do not make an inactive
detector detect anything; see :ref:`cleanprompt-security`.
