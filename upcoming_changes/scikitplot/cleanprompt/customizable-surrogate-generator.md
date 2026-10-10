---
title: "Let users customise surrogate stand-ins without weakening their safety rules"
status: promoted
kind: "api"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 25 planning (maintainer request: generator customisation)"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "feature"
towncrier_fragment: "docs/source/whats_new/upcoming_changes/scikitplot.cleanprompt/864.feature.rst"
---

# Let users customise surrogate stand-ins without weakening their safety rules

## Summary

`--style surrogate` uses fixed, built-in pools (`_surrogates.py`: `_FIRST`,
`_LAST`, `_ORG_FIRST`, `_ORG_SECOND`, `_PLACE`, `_FEATURE`, `_FACILITY`) for
eight kinds. A user cannot supply their own names (a locale, a domain's
vocabulary, a team's fictional cast), cannot choose which kinds are
surrogated, and cannot plug in a generator. This note designs that, keeping
every safety rule the current generator enforces.

## Why it matters

Surrogates are the prevention half of `CP-042`/`CP-043` (a model rewrites
bracket tokens; it does not rewrite names). English-only pools make
non-English prompts read oddly — `Marion Holt` in a Turkish paragraph — which
reintroduces the "model comments on the redaction" problem surrogates exist
to remove.

## Current evidence

- `_surrogates.surrogate_for(kind, ordinal, avoid, source, forbidden)` is the
  only generator; `_engine.py` calls it once per new label.
- `SURROGATE_KINDS` is fixed; credentials, identifiers and `NORP` always keep
  placeholders.
- `TagStyle.style` is part of the grammar fingerprint; pools are not (they
  are constants), so a vault does not record which pools produced it — which
  is fine today because restoration reads stand-ins from the vault, never
  from the pools.

## Root cause / current understanding

Not a defect: the generator was built closed on purpose. The design question
is how to open it without letting a custom generator produce something the
built-in one never would (a plausible card number, a real domain, a value the
text already contains).

## Expected behavior

1. **Data first (no code).** A pack-like YAML/JSON "surrogate set":

   ```yaml
   name: tr
   version: 1
   summary: Turkish stand-ins.
   kinds:
     PERSON: {first: [Deniz, Ece, Mert], last: [Aksoy, Kaya, Yıldız]}
     ORG:    {first: [Kuzey, Mavi], second: [Lojistik, Yazılım]}
     GPE:    [Ilgaz, Sapanca]
   ```

   validated like a pack (unknown keys refused, pools non-empty, entries
   unique, no entry matching any structural pattern or any `secrets`-pack
   pattern — invariant `I14`), selected with `--surrogates tr` /
   `TagStyle(style="surrogate", surrogates="tr")`.
2. **Code second (Python only).** A provider protocol:

   ```python
   class SurrogateProvider(Protocol):
       name: str  # recorded, versioned
       version: str

       def candidate(self, kind: str, index: int) -> str | None: ...
   ```

   The core keeps the loop: it asks for `candidate(kind, index + attempt)`
   and applies every existing rule itself — the kind allow-list, uniqueness
   against issued stand-ins, absence from the source, the `forbidden` check
   against held values (`CP-071`), the bounded search with placeholder
   fallback. A provider can only *propose*.
3. **Fixed safety floor, not configurable:**
   - credentials and identifiers never get a surrogate, whatever a set or a
     provider says (`CREDIT_CARD`, `IBAN`, `SSN_US`, `AWS_ACCESS_KEY`, `JWT`,
     `PRIVATE_KEY`, `MAC`, `IPV4`, `IPV6`, and anything from the `secrets`
     pack);
   - `EMAIL` and `URL` candidates must end in a reserved domain
     (`example.invalid` etc.), `PHONE` in the fiction block; a candidate that
     does not is rejected by the core;
   - semantic categories with meaning a model needs (medication, diagnosis,
     religion, politics) are never naturalistically substituted — opaque
     placeholders only (see the external review's decision ledger).
4. **Recorded and reproducible:** the set's name and fingerprint (or the
   provider's name and version) join the grammar fingerprint, so a vault and
   a plan say which generator produced them, and `plan --check` refuses
   drift. Restoration is unchanged: it reads stand-ins from the vault.

## Affected paths and ownership

`_surrogates.py` (provider loop, safety floor), `_policy.py` (`TagStyle`
field + fingerprint payload, keeping the default-valued omission so existing
vaults keep their digest), `_catalog.py`/`_custom.py` (loading sets),
`_cli.py` (`--surrogates`, both frontends, parity cases), `_plan.py`.

## Constraints and non-goals

- Base tier only; no faker-style dependency.
- No network, no randomness without a declared seed; output deterministic for
  a given text, policy and set.
- Do not change the default style (`placeholder`) or the built-in pools'
  output (existing vaults and tests depend on it).
- No general fake-data generator.

## Edge cases to cover

A set entry equal to a word in the text; a set too small for the number of
values (fallback to placeholder, reported); a provider returning a real
domain (rejected); a provider raising (fail loudly, naming the provider); a
provider returning a held value written differently (`CP-071` via
`_canonical`); non-Latin scripts; two conversations with different sets
sharing one vault (refused by the grammar fingerprint).

## Proposed direction

Slice A: data sets + safety floor + fingerprint + CLI option + docs/gallery.
Slice B: Python provider protocol, behind the same core loop.

## Verification / acceptance criteria

Every edge case as a test; the round-trip, no-leakage and determinism
invariants over randomised documents with a custom set; frontend parity for
`--surrogates`; a vault written with a set restores only under the same set.

## Documentation impact

`how_it_works.rst` (placeholders or surrogates), `python_api.rst`, a new
section in the moderate gallery example.

## Release-note promotion

Required: a `feature` fragment under `scikitplot.cleanprompt`.

## Promotion record (2026-10-10, round 26)

**Slice A is implemented**; the design is now
`maintenances/cleanprompt/_maintenance/GENERATOR_DESIGN.md` (invariants
G1–G6, the safety floor, slices, locale, growth paths).

- `scikitplot/cleanprompt/_surrogate_sets.py`: `SurrogateSet`,
  `surrogate_set_from_document`, `load_surrogate_set`, `entry_problem`.
  Kinds a set may define: `PERSON`, `ORG` (two lists each), `GPE`, `LOC`,
  `FAC`. `EMAIL`/`PHONE`/`URL` are refused (their reserved forms are the
  core's), and so is every other kind.
- `TagStyle.surrogates` (identity `name@version#digest16`, recorded, in the
  fingerprint only when set) and `TagStyle.surrogate_set` (the object; not
  compared, not serialised). Default digests unchanged (pinned in
  `test__surrogate_sets.TestIdentity`).
- `--surrogates FILE` (refused without `--style surrogate`) on `redact`,
  `encode`, `roundtrip`, `batch`, `ask`, `mcp`, `plan`;
  `FluentCleanPrompt().surrogates(path)`; `CleanPlan.surrogates` (identity in
  the plan fingerprint, so editing the set makes a saved plan stale).
- The round's independent review hardened the floor before release:
  default-ignorable code points, mark runs, full-width and other look-alike forms and mixed
  scripts are refused (entries could look identical); combined names are
  checked when issued; opt-in `TITLE_CASE` is not a detection rule; appending
  with another grammar is refused (`CP-106`). Ordinary-word entries remain a
  documented limit:
  `upcoming_changes/scikitplot/cleanprompt/surrogate-stand-ins-that-are-ordinary-words.md`.
- Deviations from the proposal: sets are selected by **path**, not by a
  registered name (no catalog of sets exists yet; growth path 3), and the
  set does not supply `EMAIL` local parts (floor rule 2).

**Slice B (provider protocol) is not implemented**; it moved to
`upcoming_changes/scikitplot/cleanprompt/surrogate-provider-protocol.md`.
Remove this note once PR 864 is merged.
