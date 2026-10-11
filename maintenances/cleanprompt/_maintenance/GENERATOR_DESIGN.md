# Stand-in generator: design and growth plan

This file is the design for how cleanprompt makes **stand-ins**: the
replacement text written where a sensitive value was. It covers what exists,
the rules that may never change, what round 26 adds (custom name sets), and
how the generator can grow later without weakening anything.

Read `DESIGN.md` for the whole subsystem. Read this file before changing
`_surrogates.py`, `_surrogate_sets.py`, `TagStyle` in `_policy.py`, or the
part of `_engine.py` that issues labels.

Status: section 5 (custom name sets) is **implemented in round 26**. Every
section after it is a plan.

---

## 1. What the generator is for

A stand-in has two jobs, and they pull in different directions:

| Job | Placeholder `[PERSON-1]` | Surrogate `Marion Holt` |
|---|---|---|
| A reader sees at once that a value was removed | yes | no |
| A model reads the sentence normally | no | yes |
| A model copies the stand-in back unchanged | often not (`[person-1]`, `[PERSON_1]`) | yes |
| Could be taken for a real value | never | only if the rules below fail |

`placeholder` stays the default, because being obvious is the safe default
for text that people read and act on. `surrogate` is the better choice for a
prompt, because the model neither comments on it nor rewrites it (`CP-042`,
`CP-043`).

Restoring a value never depends on the generator. The vault maps each
stand-in to its value, and `decode` reads that map. So the generator can
change without breaking any vault, as long as the grammar fingerprint records
what changed (section 4).

## 2. Inputs, output, invariants

**Inputs:** the kind (for example `PERSON`), the ordinal (the n-th distinct
value of that kind), the stand-ins already issued, the source text, and the
values the conversation holds.

**Output:** a stand-in string, or `None`, which means "use the placeholder".

**Invariants.** Each one has a test, named in brackets. One limit is not an
invariant and cannot be: `decode` restores a stand-in wherever it appears in
the model's reply, so a stand-in that is also an ordinary word (`The`, `Ash`)
rewrites the reply's own words. The built-in names are chosen to be unusual;
a set's author has to do the same. No word list is shipped to enforce it,
because any list would be a guess about someone else's language.

- **G1 Deterministic.** The same text, policy and set always give the same
  stand-ins. No clock, no randomness, no environment lookups.
  (`test__surrogates`, `test__surrogate_sets.TestDeterminism`)
- **G2 Unique.** Two different values never share a stand-in in one vault.
  Otherwise restoring could not tell them apart. (`CP-001` from the other
  side)
- **G3 Absent from the source.** A stand-in that already occurs in the text
  is never issued. Otherwise restoring would also replace the user's own
  occurrence of it.
- **G4 Holds no real value.** A stand-in never contains a value the
  conversation holds, whatever its case, Unicode form or word separators
  (`Marion Holt`, `marion.holt`, `Marion-Holt` and `MarionHolt` are one
  value). (`CP-071`, `CP-105`; `_engine._held_forms` and `_shows_held`)
- **G5 Bounded.** At most 100 attempts per value, then the placeholder. The
  result is worse to read, but still correct.
- **G6 Recorded.** Anything that changes how stand-ins are spelled enters the
  grammar fingerprint, and the default spelling keeps the old digest. So no
  vault written earlier becomes unreadable.

## 3. The safety floor

No set, plan, flag or future provider can relax these rules.

1. **Credentials and identifiers never get a surrogate.** Only these kinds
   do: `PERSON`, `ORG`, `GPE`, `LOC`, `FAC`, `EMAIL`, `PHONE` and `URL`
   (`SURROGATE_KINDS`). Card numbers, IBANs, social-security numbers, keys,
   tokens, MAC and IP addresses, and every kind from a pack, keep their
   placeholders. A realistic card number is a hazard: a person or a system
   could act on it, and by chance it could even be real.
2. **Contact forms are made by the core, never by a set.** `EMAIL` and `URL`
   always end in `example.invalid` (`.invalid` is reserved for ever by RFC
   2606). `PHONE` stays in the `+1 555 0100`–`0199` range, which the North
   American numbering plan keeps for fiction. A set cannot provide these
   kinds; it is refused if it tries.
3. **Name entries look like names, and no two look alike.** Each character
   is a letter, a space, one of `' - . ’`, or a combining mark after a letter
   (at most two in a row). That rules out digits, `@`, `/`, `:`, brackets
   and control characters. Default-ignorable code points are refused
   (zero-width characters, the combining grapheme joiner, variation
   selectors, Hangul fillers), drawn-differently compatibility forms are
   refused (full-width, circled, superscript, Arabic presentation forms —
   but not plain `<compat>`, which ordinary Thai and other letters carry),
   entries are stored in NFC so canonically equivalent spellings are one
   name, and all its letters must be in one script (Han and kana count as
   one). Requiring NFKC was tried and refused ordinary Thai and Arabic
   names; that is why the rule names the forms instead. An entry is 1–64 characters, starts and ends with
   a letter (or a letter and its marks), and has no doubled spaces. The
   round-26 review showed why each rule exists: without them, `Aino`,
   `Ai` + U+034F + `no` and `Ai` + U+3164 + `no` loaded as three entries that
   look the same, and decoding turned every mention of one into another.
4. **No stand-in looks like a detected value.** No entry, and no combined
   two-part name when it is issued, may match a default core pattern or any
   built-in pack pattern (with its validator), in the same spirit as `I14`.
   If one did, the stand-in would be detected again the next time the text
   is encoded. Opt-in core patterns are not used: `TITLE_CASE` matches every
   two-word name, the built-in ones included.
5. **Kinds whose meaning matters stay placeholders.** `NORP` (nationality,
   religion, politics), and anything medical or legal that a pack adds, is
   never replaced by an invented natural word. "A Catholic" turned into an
   invented demonym changes what the text means.

## 4. Determinism and the fingerprint

The grammar fingerprint (`TagStyle.fingerprint`) is computed from
`TagStyle.as_dict()`. It leaves out fields that still have their default
value, so the default grammar keeps the digest it had before those fields
existed:

- `style` is left out when it is `placeholder` (round 9 rule);
- `surrogates` is left out when it is `None` (round 26).

A set is recorded by its **identity**, `name@version#digest16`. The digest is
a SHA-256 of the set document in canonical form, so changing any entry
changes the identity, and with it the vault's grammar. Restoring needs only
the identity, never the entries, so `decode` works without the set file. The
vault writes the identity as part of `tag_style`. Encoding needs the entries
themselves: a grammar that names a set the engine was not given is refused.
The engine never quietly falls back to the built-in pools, because those
would issue stand-ins the fingerprint does not describe. For the same reason
the command line refuses to append to a vault written under another grammar
(another style or set) before anything is encoded; until the round-26 review
it mixed the two and re-stamped the vault.

A plan (`CleanPlan.surrogates`) saves the set's *path*. It also adds the
set's identity to the plan fingerprint (`I12`), so editing the set file makes
a saved plan stale. `load_plan` then refuses it until someone reviews the
change.

## 5. Slice A — custom name sets (implemented, round 26)

A **surrogate set** is a small YAML or JSON file of names:

```yaml
name: nordic
version: 1
summary: Nordic-sounding invented names.
kinds:
  PERSON: {first: [Aino, Eero, Liv], last: [Halvorsen, Lindgren, Virtanen]}
  ORG:    {first: [Fjord, Norrsken], second: [Data, Logistik]}
  GPE:    [Granvik, Solberga]
  LOC:    [the Tunturi Fells]
  FAC:    [Granvik Station]
```

- `PERSON` and `ORG` take two lists, which are combined so that consecutive
  people do not share a surname (`_pair`). `GPE`, `LOC` and `FAC` take one
  list.
- A kind the set leaves out uses the built-in list. A kind the set must not
  define is refused with the reason (floor rules 1 and 2).
- Validation collects every problem in one pass, as packs do: unknown keys,
  empty or duplicate entries (compared in canonical form), entries that break
  rule 3 or 4, and lists over 256 entries.
- It is selected with `TagStyle(style="surrogate", surrogate_set=...)`,
  `FluentCleanPrompt().style("surrogate").surrogates("nordic.yaml")`, or
  `--style surrogate --surrogates nordic.yaml`. `--surrogates` without
  `--style surrogate` is refused: the flag never switches the style on by
  itself.
- If the set is too small for the number of values, the extra values get
  placeholders (G5). Nothing loops and nothing is
  made up.

**Why data, and not code, first.** A data file can be validated completely
before anything runs, reviewed in a pull request, and shared between teams
without running anyone's code. Most requests ("names in our language", "our
fictional cast") are data.

## 6. Slice B — provider protocol (planned)

The core loop is already written so that code can plug in. A provider only
**proposes** candidates. The core keeps every rule:

```python
class SurrogateProvider(Protocol):
    identity: str                                  # recorded, versioned
    def candidate(self, kind: str, index: int) -> str | None: ...
```

`SurrogateSet` implements exactly this interface today. Slice B will:

- accept any object with that interface in `TagStyle.surrogate_set`;
- run the floor check (rule 3) on every candidate at run time, not only at
  load time. A failing candidate is skipped; a provider that raises an error
  stops the pass with an error that names the provider;
- keep `EMAIL`, `PHONE` and `URL` in the core, as slice A does;
- require `identity` to change whenever the output changes. This cannot be
  enforced, so the documentation says it, and a test checks that two runs
  produce the same output.

The open question for the maintainer: is a Python-only surface worth having
when sets already cover the stated needs? Recommendation: wait until a
concrete request needs something a data file cannot express.

## 7. Locale

A set has no locale field. The set *is* the locale: a team picks one that
fits its text. Planned refinements, each optional:

- `language:` in the set file, checked against `_languages` so that `doctor`
  can say "the Turkish set is active";
- choosing a set per language when `--ner-language` is given, through a plan
  mapping (`surrogates: {tr: tr.yaml, de: de.yaml}`);
- ASCII folding for e-mail local parts made from non-Latin names, if the core
  ever builds them from set names. (Today it builds them from the built-in
  lists, so the result is always ASCII.)

## 8. Per-kind actions

`per-kind-action-policy.md` plans a choice per kind: `placeholder`,
`surrogate`, `mask` or `drop`. It fits this design without changes. The
action decides *whether* the generator is asked. The floor (rule 1) still
decides whether a surrogate is *allowed*. A per-kind `surrogate` request for
`CREDIT_CARD` is refused when the policy is built, never ignored at run time.

## 9. Growth paths

These are ordered by value for the risk they carry. None has been started.

1. **Coverage report for sets.** `packs --check`-style output for a set: how
   many distinct stand-ins each kind can produce, and which of them collide
   with built-in pack patterns. Cheap, and helps people write good sets.
2. **Gender- and script-consistent pairs.** Optional `group:` tags on
   first-name entries, so a pair never mixes scripts. Stays deterministic.
3. **Shared sets as artifacts of their own.** Sets distributed like packs
   (`--surrogates` given a name from a catalog), once more than one team
   shares them.
4. **Provider protocol** (section 6), behind the same core loop.
5. **Consistency across a vault.** One person's stand-in kept the same
   across conversations. This already holds within one vault through
   `remember`; a cross-vault version needs a keyed mapping and is out of
   scope until asked for.

## 10. Rejected alternatives

- **A faker-style dependency.** It would bring in a third-party package,
  produce realistic contact data (which breaks floor rule 2), and its output
  changes between versions (which breaks G1).
- **Random stand-ins with a seed.** Same output as deterministic indexing,
  but harder to reproduce and review, and it gains nothing over indexing.
- **Letting a set provide `EMAIL`/`PHONE`/`URL`.** One typo in a domain
  (`example.in`) gives an address that resolves. The core owns these forms.
- **Recording the set's entries in the vault.** It would make vaults larger,
  and if the set file held something sensitive, it would leak into every
  vault. The identity is enough to restore.
