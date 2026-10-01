"""
Cleanprompt from Python: Single Calls and Edge Cases
====================================================

.. currentmodule:: scikitplot.cleanprompt

A reference for the Python surface, organised as *the smallest call that does
the job, then what happens when the input is degenerate*.

There are three entry points, and picking the right one is most of the design
work:

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Entry point
     - Scope
     - Use when
   * - :func:`encode` / :func:`decode`
     - one exchange
     - a single request/response, stateless, nothing to close
   * - :class:`Session`
     - a conversation
     - several turns share one vault, so a value keeps its label
   * - :class:`Redactor`
     - one pass, fully controlled
     - you supply the policy and the detector registry yourself

All three run on the base tier. ``import scikitplot.cleanprompt`` imports no
third-party package at all, so none of this changes the import cost of any
other submodule.

Every value below is reserved and cannot reach anybody.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import io
import json
import logging

# %%
# Part one — the ordinary forms
# -----------------------------
#
# 1. One exchange: :func:`encode` and :func:`decode`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :func:`encode` returns an :class:`EncodedPrompt`.  Three attributes matter,
# and the split between them is the security boundary:
#
# ``.text``
#     safe to transmit — this is what goes to the model.
# ``.handle``
#     **not** safe to transmit — it holds the values that were removed.
# ``.report``
#     what happened, including what was *not* looked for.

from scikitplot.cleanprompt import decode, encode  # noqa: E402

PROMPT = "Mail ada@example.com about the outage on 192.0.2.10."

prompt = encode(PROMPT)

print("send   :", prompt.text)
print("keep   :", dict(prompt.handle.vault.items()))
print("report :", prompt.report["headline"])

# %%
# The model's reply comes back still in placeholders, because that is all it
# ever saw.  :func:`decode` closes the loop.

REPLY = "I have written to [EMAIL-1] and checked [IPV4-1]."

print(decode(REPLY, prompt.handle))

# %%
# The property worth asserting, rather than describing: a reply that is exactly
# the redacted prompt restores to exactly the original text.

assert decode(prompt.text, prompt.handle) == PROMPT
print("round trip exact:", True)

# %%
# 2. A conversation: :class:`Session`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :func:`encode` is stateless, so two calls give the same value two different
# vaults and the same label can mean two different things.  A conversation
# needs one vault across turns, which is what :class:`Session` is.
#
# Use it as a context manager.  Leaving the block clears the vault, so the
# values do not outlive the conversation by accident.

from scikitplot.cleanprompt import session  # noqa: E402

with session() as chat:
    first = chat.encode("Mail ada@example.com about the outage.")
    print("turn 1 :", first)

    print("reply  :", chat.decode("I wrote to [EMAIL-1]."))

    second = chat.encode("Copy bob@example.com, and ada@example.com again.")
    print("turn 2 :", second)

    print()
    print("ada keeps her label across turns:", "[EMAIL-1]" in second)
    print("turns:", chat.turns)
    print("labels:", chat.report()["labels"])

    vault = chat.handle.vault

print()
print("vault closed on leaving the block:", vault.closed)

# %%
# :meth:`Session.roundtrip` takes a callable that stands for the model.  This is
# the shape to reach for when wiring an actual client: the callable receives the
# redacted prompt and returns the reply, and nothing else in your code ever sees
# a value.


def pretend_model(sent: str) -> str:
    """Stand in for a language model. It only ever sees placeholders."""
    assert "ada@example.com" not in sent
    return "Acknowledged: {0}".format(sent)


with session() as chat:
    print(chat.roundtrip("Mail ada@example.com", pretend_model))

# %%
# 3. Full control: :class:`Redactor`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`Redactor` takes a :class:`RedactionPolicy` and a
# :class:`DetectorRegistry` and does one pass.  Everything that changes *what*
# the pipeline does is in the policy, as data: no stage reads an environment
# variable or a module global.

from scikitplot.cleanprompt import DEFAULT_POLICY, Redactor, restore  # noqa: E402

policy = DEFAULT_POLICY.evolve(kinds=("EMAIL", "IPV4"), case_insensitive=True)

result = Redactor(policy=policy).redact(PROMPT, extra_terms=["Acme Ltd"])

print("text     :", result.text)
print("entries  :", [(e.label, e.kind, e.original) for e in result.entries])
print("stats    :", result.stats.by_kind)
print("policy   :", result.policy_fingerprint)

# %%
# Restoration takes the same policy.  Passing a different *grammar* is refused
# rather than guessed at, because the alternative is returning somebody else's
# value.

print(restore(result.text, result.vault, policy=policy).text)

# %%
# Part two — the edge cases
# -------------------------
#
# 4. Repeated, overlapping and adjacent values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# One value, however many times it appears, gets **one** label.  That is what
# makes the placeholder useful to the model: it can tell that two mentions are
# the same thing.

repeated = encode("Mail ada@example.com, then ada@example.com again, cc bob@example.com")

print(repeated.text)
print("entries:", len(repeated.result.entries))

# %%
# When two detections overlap, the longer one wins by default.  This is the
# safe direction: covering more of the text cannot disclose more than covering
# less.  ``OverlapStrategy.STRICT`` turns an overlap into an error, which is
# what a pipeline that must prove its detector set is disjoint should use.

from scikitplot.cleanprompt import OverlapStrategy, OverlapError  # noqa: E402

print("strategies:", [one.value for one in OverlapStrategy])

strict = DEFAULT_POLICY.evolve(overlap=OverlapStrategy.STRICT)
try:
    Redactor(policy=strict).redact("Mail ada@example.com", extra_terms=["ada@example.com"])
    print("no overlap to report")
except OverlapError as exc:
    print("STRICT refused:", str(exc)[:88])

# %%
# 5. Text that is already redacted
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Invariant ``I6``: running the pipeline on its own output changes nothing.
# Without it, a placeholder would be re-detected as a value and wrapped again,
# and the vault would no longer describe the text.

once = encode(PROMPT).text
twice = encode(once).text

print("once :", once)
print("twice:", twice)
print("idempotent:", once == twice)
assert once == twice

# %%
# 6. Degenerate input
# ^^^^^^^^^^^^^^^^^^^
# Empty text, whitespace, and text with no values at all.  None is an error;
# all three must produce an empty vault and an unchanged text.

for label, text in [("empty", ""), ("spaces", "   "), ("clean", "nothing here")]:
    outcome = encode(text)
    print(
        "{0:<7} text={1!r:<16} entries={2} unchanged={3}".format(
            label, outcome.text, len(outcome.result.entries), outcome.text == text
        )
    )

# %%
# 7. A reply the model rewrote
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The one hop this submodule does not control is the model, and a model
# rewrites tokens.  :func:`restore` accepts a bounded set of rewrites and
# **reports every repair it made**, which is the part that keeps the leniency
# honest.

pair = encode("Mail ada@example.com or bob@example.com")

rewritten = restore(
    "I wrote to [email_1] and [EMAIL 2]; see [note 2] for the rest.",
    pair.handle.vault,
)

print(rewritten.text)
print("repaired :", rewritten.repaired)
print("unknown  :", rewritten.unknown)
print("prose untouched:", "[note 2]" in rewritten.text)

# %%
# ``lenient=False`` restores only what was issued verbatim, for a caller who
# depends on the strict reading.  ``strict=True`` turns an unknown placeholder
# into an error instead of leaving it in the text.

from scikitplot.cleanprompt import RestorationError  # noqa: E402

exact = restore("I wrote to [email_1].", pair.handle.vault, lenient=False)
print("lenient=False restored:", len(exact.restored), "unknown:", exact.unknown)

try:
    restore("what about [EMAIL-9]?", pair.handle.vault, strict=True)
except RestorationError as exc:
    print("strict=True refused:", exc)

# %%
# 8. A vault written under a different grammar
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The vault records the placeholder grammar it was issued under.  Reading it
# back under another one is refused: silently proceeding would return the wrong
# value for the right-looking label, which is the worst failure this submodule
# has.

from scikitplot.cleanprompt import PolicyError, TagStyle  # noqa: E402

angled = DEFAULT_POLICY.evolve(tag_style=TagStyle(prefix="<<", suffix=">>"))
issued = Redactor(policy=angled).redact("Mail ada@example.com")

print("issued under <<>>:", issued.text)
try:
    restore(issued.text, issued.vault, policy=DEFAULT_POLICY)
except PolicyError as exc:
    print("refused:", str(exc)[:96])

# %%
# 9. Limits, and why they are refusals rather than truncations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Every limit is part of the policy and every breach raises.  A redaction tool
# that silently processed the first megabyte of a larger document would report
# success over text it never looked at.

from scikitplot.cleanprompt import LimitExceededError, Limits  # noqa: E402

print("defaults:", DEFAULT_POLICY.limits)

small = DEFAULT_POLICY.evolve(limits=Limits(max_input_chars=32))
try:
    Redactor(policy=small).redact("x" * 64)
except LimitExceededError as exc:
    print("refused:", str(exc)[:96])

# %%
# 10. A detector of your own
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# A :class:`PatternSpec` carries the pattern *and* the examples that prove it,
# so a detector added here is checked by the same class-level tests as the
# built-in ones — including whether it survives at the end of a sentence.

from scikitplot.cleanprompt import (  # noqa: E402
    PatternSpec,
    RegexDetector,
    default_registry,
)

TICKET = PatternSpec(
    kind="TICKET",
    pattern=r"\bJIRA-\d{2,6}\b",
    intent="An internal issue identifier, which links a prompt to a customer.",
    priority=90,
    examples_yes=("JIRA-4821", "JIRA-77"),
    examples_no=("JIRA-", "JIRA-1"),
)

registry = default_registry(kinds=("EMAIL",))
registry.add(RegexDetector(TICKET))

custom = DEFAULT_POLICY.evolve(kinds=("EMAIL", "TICKET"))
found = Redactor(policy=custom, registry=registry).redact(
    "See JIRA-4821, raised by ada@example.com."
)

print("detectors:", [one.name for one in registry])
print("redacted :", found.text)
print("restored :", restore(found.text, found.vault, policy=custom).text)

# %%
# 11. What is *not* being detected
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :func:`diagnose` is the programmatic form of the ``doctor`` command.  It
# answers the question a redaction tool must never leave unanswered: not "what
# did you find" but "what were you unable to look for".

from scikitplot.cleanprompt import diagnose  # noqa: E402

diagnosis = diagnose(policy=DEFAULT_POLICY)

print("healthy:", diagnosis.healthy)
for blind_spot in diagnosis.blind_spots:
    print("  - [{0}] {1}".format(blind_spot.severity, blind_spot.category))

# %%
# :func:`suggest_terms` is the other half: candidates a regular expression can
# never recognise, offered for you to confirm rather than redacted on a guess.

from scikitplot.cleanprompt import suggest_terms  # noqa: E402

for suggestion in suggest_terms("Marion Holt at Northwind Logistics called.")[:4]:
    print("  suggest:", suggestion.text)

# %%
# 12. Keeping values out of your logs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :func:`redacting` installs a filter that replaces known literal secrets in
# every record passing through a logger.  It is a last line of defence for
# strings you hold — an API token, a passphrase — and it is deliberately
# literal: it does not pattern-match, so it cannot be relied on for values you
# have not named.

from scikitplot.cleanprompt import redacting  # noqa: E402

buffer = io.StringIO()
demo = logging.getLogger("cleanprompt.gallery")
demo.handlers = [logging.StreamHandler(buffer)]
demo.setLevel(logging.INFO)

with redacting(secrets=["hunter2"], logger=demo):
    demo.info("connecting with token hunter2")

print("logged:", buffer.getvalue().strip())
print("the named secret is gone:", "hunter2" not in buffer.getvalue())

# %%
# 13. Reports as data
# ^^^^^^^^^^^^^^^^^^^
# :func:`as_dict` renders any result object as JSON-safe data, which is what
# the ``--format json`` output of the CLI is built from.  Note what it does
# *not* contain: the values.

from scikitplot.cleanprompt import as_dict  # noqa: E402

payload = as_dict(encode(PROMPT).result.stats)

print(json.dumps(payload, indent=2, sort_keys=True))

# %%
# 14. Cleanup
# ^^^^^^^^^^^
# Nothing in this example wrote to disk: :func:`encode` and :class:`Session`
# keep the vault in memory.  Clearing it is still worth doing explicitly when a
# vault outlives a ``with`` block.

prompt.handle.vault.clear()
pair.handle.vault.clear()

print("vault cleared:", prompt.handle.vault.closed)

# %%
#
# .. tags::
#
#    model-workflow: cleanprompt
#    plot-type: text
#    level: intermediate
#    purpose: reference
