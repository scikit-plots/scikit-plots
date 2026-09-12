# R173T33 — Share redaction was correct, and unguarded

Status: COMPLETE

Base: R173T32 export format parity, same package.

## The finding is that there was no defect

Five global-share exports of one conversation — JSON, HTML, text, YAML, TOML —
were extracted and compared. All five are correctly redacted:

- `session.id`, `page_url`, `exported_at`, `exported_at_iso`: null everywhere;
- model id, name and provider: absent everywhere;
- timestamps: absent everywhere;
- **no resources manifest and no attachment excerpts at all**.

A UUID appears in every file and is a Chrome MHTML `cid:` content id, not a
session identifier. The `"id"` hits in JSON and YAML are serialized nulls; TOML
omits null keys and text/HTML omit empty fields, which is format-idiomatic
rather than a discrepancy.

Nothing was changed in the export behaviour, because nothing was wrong with it.

## What was missing was the gate

The redaction is correct *by construction*, and no test asserted it. A sixth
format, or a session field added later and rendered unconditionally by one
builder, would leak with nothing failing — and the leak would be a published
artifact, not a local file.

`test_ai_assistant__share_redaction_parity.mjs` drives **every registered
format builder** with a redacted snapshot and asserts none of the removed
values appears in its output. Three properties make it hard to pass vacuously:

- the builder list is checked against `_EXPORT_FORMATS`, so a new format that
  nobody adds here fails the count assertion;
- surviving content carries sentinels, so a builder that emits nothing cannot
  pass;
- a builder that cannot be driven is **reported, not skipped** — an uncovered
  builder is an unguarded one.

Dependencies are the real functions, resolved as a closure from the source
(47 of them) rather than listed by hand, and module constants are read from
their declarations. Stubbing the serializers would have defeated the point:
the serializers are exactly where a redacted value would reappear.

## Two of my own assertions were wrong first

`_buildConvTomlString never prints a bare null` failed against a correct
builder: the TOML export documents its own convention — *omitted optional
values represent null* — and the assertion matched that sentence. Comments are
now excluded and the check targets `key = null` assignments, which is what
would actually be wrong.

Three builders also failed to run until their real dependencies were supplied.
Skipping them was the tempting fix and the wrong one.

## Verification

- browser wrapper gate: **148/148** (one new harness, 46 assertions);
- architecture gates: **485/485**, one new mutant caught:
  `share-export-leaks-the-source-page`, which makes the text builder emit a
  page URL the share policy nulled.

## Still open

The activity timeline is still absent from every export format, and the
decision it needs is unchanged: publishable, or local-only like the session id?
This checkpoint's gate is where the answer will be enforced either way.
