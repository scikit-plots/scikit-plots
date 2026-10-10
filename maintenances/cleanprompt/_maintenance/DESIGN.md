# `scikitplot.cleanprompt` — design and code guide

Big picture, decided before any code was written. Every defect cited below was
reproduced against the upstream source in `_maintenance/evidence/upstream-defects.log`;
nothing here is asserted from reading alone unless it is marked `by-construction`.

Upstream: <https://github.com/takashiishida/cleanprompt> (`cleanprompt.py`, `app.py`).

---

## 1. Purpose, inputs, outputs

CleanPrompt replaces sensitive substrings of a text with stable placeholders
before that text leaves the user's machine, and restores them afterwards from the
LLM's reply.

```text
input    str  (untrusted, arbitrary length, arbitrary unicode)
         RedactionPolicy  (declarative, hashable, serializable)
output   RedactionResult  (redacted text + ordered entries + stats + policy fingerprint)
         Vault            (label -> original secret; held separately, never in the result text)
inverse  restore(llm_reply, vault) -> RestorationResult
```

The unit of value is the **pair** `(RedactionResult.text, Vault)`. The redacted
text is safe to transmit; the vault must not be. Keeping them in two objects is
the single most important structural decision in this rewrite: upstream returned
them entangled in one `dict` that the web tier then round-tripped through the
user's browser cookie.

`_api.py` states that pair in the shape a caller sending prompts to a model
wants it: `encode(text, ...)` returns an `EncodedPrompt`, iterable so that
`safe, handle = encode(...)` works, and `decode(reply, handle, strict=False)`
is the inverse. The `Handle` is frozen, reprs without secrets, and is
JSON-portable through `export()`/`load()`. A `Session` (or the `session()`
context manager) holds one handle across turns and clears it on exit; its
`encode` returns a plain `str` rather than a pair, because the session owns the
handle and a per-turn copy would invite decoding turn 3 with turn 1's handle.
The only integration seam is `send`, a callable taking a string and returning
one, so no provider is named anywhere in this submodule.

## 2. Data flow

```text
                 ┌── detectors see the ORIGINAL text only ──┐
text ──► detect ─┤ regex · literal · spacy/nltk (optional)  ├──► [Span, ...]
                 └──────────────────────────────────────────┘
      ──► resolve   deterministic overlap arbitration        ──► disjoint, sorted [Span, ...]
      ──► assign    value-identity -> stable label ordinals   ──► [Entry, ...]
      ──► rewrite   one left-to-right pass, O(n)              ──► text', Vault
```

Restoration is the mirror image: scan `text'` for the **placeholder grammar**,
look each hit up in the vault, rebuild in one pass. Since `CP-042` the scan is
lenient by default — it also recognises the bounded set of rewrites a language
model actually performs on a bracket token, and acts on one only when it
resolves to a label the vault holds. Under the `surrogate` style the same
single pass additionally collects literal stand-in matches through
`LiteralDetector`, over the original reply text, before rewriting once.

Entity detection has two possible engines, spaCy and NLTK, selected by
`_engines.resolve_engine` over `ENGINE_MODES = ("auto", "spacy", "nltk",
"both", "none")` with `DEFAULT_ENGINE = "auto"`. Whichever runs, its labels are
normalised into `CANONICAL_LABELS` by `canonical_label` before the span leaves
the detector. That is a correctness requirement, not tidiness: spaCy says `ORG`
where NLTK says `ORGANIZATION` and `LOC` where NLTK says `LOCATION`, so raw
labels would make the placeholder depend on which engine happened to be
installed, and a vault written under one engine would not restore under the
other because its placeholders would no longer match anything.

Which spaCy model serves a language is `_languages.resolve_model(language,
size, explicit)`, over 24 `LANGUAGES` and `MODEL_SIZES = ("sm", "md", "lg",
"trf")`. It returns the model **and a note**: a language with no model of its
own falls back to `MULTILINGUAL_MODEL` (`xx_ent_wiki_sm`), which recognises
only the coarse categories, and every caller announces the substitution. A user
who asked for Turkish and silently received a lower-resolution model would read
the thinner result as "there was nothing to find", which is `CP-018` reached by
another route.

## 3. Invariants

These are the contract. Each has a named regression test.

| # | Invariant | Enforced by |
|---|---|---|
| I1 | **Round trip.** `restore(redact(t).text, vault).text == t` whenever `t` contains no literal placeholder of the active grammar. | `_engine`, tested incl. fuzz |
| I2 | **No leakage.** No resolved span's surface text survives in `RedactionResult.text`. | single-pass rewrite |
| I3 | **Determinism.** Same `(text, policy)` gives byte-identical output in any process; no reliance on set/hash order. | sorted resolution, ordinal assignment |
| I4 | **Statelessness.** `Redactor` carries no per-document mutable state; N documents through one instance number independently. | all counters live in the per-call result |
| I5 | **Disjointedness.** Resolved spans are pairwise non-overlapping and ascending. | `resolve_spans` |
| I6 | **Detector purity.** No detector ever observes rewritten text. | detection precedes rewrite, by-construction |
| I7 | **Idempotence.** Re-redacting an already-redacted text yields no new entries. | placeholder grammar is masked from detection |
| I8 | **Bounded.** Input size, span count and pattern count are bounded by policy; exceeding a bound raises, never truncates silently. | `_policy` limits + `LimitExceededError` |
| I9 | **Value identity.** Equal (optionally case-folded) surfaces get one label; distinct surfaces never share a label. | `assign` |

## 4. Failure modes

Every failure is an explicit typed exception. No silent degradation, no
best-effort partial output.

```text
CleanPromptError
├── PolicyError          invalid/contradictory policy
├── PatternError         uncompilable or unsafe pattern
├── DetectorError        a detector raised; carries the detector name
├── OverlapError         STRICT overlap strategy hit a genuine conflict
├── LimitExceededError   input/span/pattern bound exceeded; carries limit and actual
├── RestorationError     unknown placeholder under strict restore
└── CapabilityError      optional tier unavailable; carries actionable install string
```

`CapabilityError` carries the tier's `status` as well as its remedy, and two
round-three findings turn on that. An optional tier that is installed but
unimportable — a half-finished upgrade, a wheel built for another interpreter,
a shadowing local file — satisfies metadata-based probing and then fails at
import; it is reported as `status="BROKEN"` with a `--force-reinstall` hint
rather than escaping as a generic detector crash (`CP-026`). And an engine that
was *asked for* and cannot be provided raises here too:
`_engines.build_detectors(required=True)` refuses rather than returning an
empty detector list, naming each engine's own status and remedy (`CP-024`).
Without `required`, `auto` still degrades quietly, which is what keeps the base
tier usable, and `mode="none"` is exempt.

`CapabilityError` must be raisable **without** importing the missing dependency.
This is the `MCP-M02-01` lesson recorded in this repository: an actionable error
placed *after* a module-scope third-party import can never fire, because the
import raises first. Capability probing therefore lives in `_capabilities.py` and
uses `importlib.metadata.version`, never `find_spec` presence — `find_spec`
cannot distinguish installed-and-usable from installed-and-broken.

## 5. Confirmed upstream defects this design retires

Evidence: `_maintenance/evidence/upstream-defects.log`.

| ID | Severity | Upstream behaviour (reproduced) | Root cause | Retired by |
|---|---|---|---|---|
| `CP-001` | critical | `replace_custom("Ann met Anna", ["Ann","Anna"])` → `"[ADDITIONAL-1] met [ADDITIONAL-1]a"` — the `a` of `Anna` leaks and the round trip is unrecoverable | rewriting with `str.replace` per mapping entry, so a shorter secret corrupts a longer one | span-based single-pass rewrite + longest-wins resolution (I1, I2, I5) |
| `CP-002` | critical | `cleanprompt.py:167` calls `replace_custom(text)` with no `additional_texts`; the returned mapping is `{}` and the email survives the whole "additional text" branch | the additional-text branch re-runs the wrong method, dropping the email/phone/URL stage entirely | one pipeline, one entry point; the detector set is policy data, not a call-site choice |
| `CP-003` | high | two documents through one `PromptCleaner` number continuously (`a@x.com`→`EMAIL-1`, then `b@x.com`→`EMAIL-2`); `app.py` shares one instance across every HTTP request | `counters`/`found_items` are instance attributes that are never reset | I4 — counters live in the per-call result; `Redactor` is immutable and reusable |
| `CP-004` | medium | URL class `[$-_@.&+]` is the **range** `$`(0x24)–`_`(0x5F); `5`, `A`, `<` all match | `-` between `$` and `_` was meant literally | curated pattern library, each entry with a stated intent and negative tests |
| `CP-005` | medium | email TLD class `[A-Z\|a-z]` admits a literal `\|`; `a@b.a\|b` matches | `\|` inside a character class is a literal, not alternation | same |
| `CP-006` | critical | NER runs over already-tagged text: `"Contact [EMAIL-1] now"` → `"Contact [[ORG-1]-1] now"`; `EMAIL-1` can no longer be restored | the pipeline feeds stage *n+1* the output of stage *n* | I6 — every detector receives the original text; stages compose over spans, not over strings |
| `CP-007` | critical | `app.py:59,67` `eval()`s a string reconstructed from the session cookie | `str(dict)` + `eval` used as a serializer | vault never leaves the server; transport is JSON with a typed schema |
| `CP-008` | low | `replace_custom(self, text, additional_texts=[])` — mutable default | — | frozen dataclasses, `None` defaults |
| `CP-009` | high | `app.secret_key` and the Fernet key are generated at import, per process | multi-worker deployments silently invalidate each other's sessions, and the PII map travels to the browser | server-side vault store keyed by an opaque token; key comes from configuration with an explicit error when absent |
| `CP-012` | medium | `revert_text(..., color=True)` returns `'hi \x1b[92mAda\x1b[0m'` — ANSI escapes embedded in the data | presentation mixed into the transform | rendering is a separate, optional presentation layer over `RestorationResult` |
| `CP-013` | medium | phone pattern matches `2024-01-15` and `12345678` | an over-broad pattern with no anchoring or validation hook | patterns carry an optional `validate` hook and word-boundary anchoring; phone detection is opt-in with a documented precision/recall trade-off |
| `CP-014` | high | POST forms have no CSRF token | — | per-session CSRF token, constant-time compared |

### Two defects found by this rewrite's own verification

Neither is inherited from upstream. Both were found by the scale probe in
`evidence/negative-probes.log`, not by the unit suite, and both are leaks — which
is why the scale probe exists as a separate verification lane.

| ID | Severity | Behaviour | Root cause | Retired by |
|---|---|---|---|---|
| `CP-015` | high | `"123-45-6789 4242 4242 4242 4242"` left `4242 4242` in the clear; 105 of 3,000 random documents leaked a detected value | `re.finditer` consumes a match *before* the validator can reject it, so a rejected candidate swallows every shorter alternative starting inside it | validated patterns are scanned from an explicit cursor for the longest **accepted** match at each start, shortening the window before advancing the start |
| `CP-016` | high | a candidate could end mid-value, redacting `555 010 447` and leaving `7` | `endpos` makes the engine treat that offset as end-of-string, so a trailing `\b` or `(?!\w)` succeeds at a boundary the real text does not have | each candidate ending before end-of-text is re-matched with one further character visible and kept only if the engine still ends at the same offset |

`CP-016`'s probe is exact for single-character trailing assertions — `\b`,
`(?!\w)`, `(?![\w.])` — which is what every pattern in the library uses. A
pattern needing wider trailing context must widen the probe with it; the
docstring on `RegexDetector` says so, and
`test_regressions.TestCP016WindowBoundary` is the gate.

A documented consequence of the merge rule surfaced here and is worth stating:
when two partially overlapping detections survive, the whole region becomes one
placeholder rather than two. That is coarser but it cannot disclose, and
`TestCP015ValidatorShadowing.test_shorter_candidate_is_found_after_a_rejection`
asserts the guarantee rather than a particular decomposition.

### Six defects found by round three's verification

Also not inherited. Each was found by a lane rather than by reading, and each
is closed with a named regression test in `tests/test_regressions.py`.

| ID | Severity | Behaviour | Root cause | Retired by |
|---|---|---|---|---|
| `CP-024` | high | an explicit `--ner` that could not be met exited 0 having run no engine, reporting that the text had been scanned for names when nothing had looked at it | the `auto` fallback rule — correct for a default — was applied to an explicit request, so `build_detectors` returned an empty list | `build_detectors(required=True)` raises `CapabilityError` naming every engine's own status and remedy; `auto` without `required` still degrades quietly and `mode="none"` stays exempt |
| `CP-025` | medium | a registry actively detecting names with NLTK was reported as the high-severity blind spot "names are NOT being detected" | `diagnose()` identified entity detectors by the name prefix `"ner:"`, which is spaCy's naming scheme | the test is `detector.kind == "NE"`, which both engines declare; the same review found the remedy hard-coding `en_core_web_lg` after the default moved to `sm`, so the remedy is now computed through `resolve_model` |
| `CP-026` | medium | installed-but-unimportable spaCy or NLTK arrived as `detector 'ner:…' failed: ImportError: …`, sending the reader to audit this submodule for a fault in their environment | metadata-based probing cannot see an import failure, and nothing mapped onto `BROKEN`, which existed for exactly this state | `CapabilityError(status="BROKEN")` with a `--force-reinstall` hint |
| `CP-027` | high | NLTK redaction took roughly 460 ms per sentence | `nltk.ne_chunk` constructs a fresh chunker on every call and the detector called it once per sentence | one chunker cached per process; 50 documents 23.58s -> 0.09s, a 12-sentence paragraph ~5.6s -> 0.015s |
| `CP-028` | critical | a sentence-final IP address was not detected at all and went to the model in the clear | `IPV4`'s guard `(?![\w.])` rejects the longer dotted run `1.2.3.4.5` and also rejects `192.168.1.10.` at the end of a sentence; `IPV6` carried the same hole | the guard is narrowed to what it was for: a trailing dot disqualifies the match only when a digit follows it |
| `CP-029` | medium | `see https://example.com/x.` stored `https://example.com/x.` as the secret, so the value in the vault was not the URL and the model received a sentence with no end to it | `.`, `,` and `)` are all legal inside a URL, so the tail classes admitted the sentence's punctuation; the text still round-tripped, which is why no test caught it | each tail segment must end on a character from `_URL_TAIL`, which excludes sentence punctuation |

`CP-027` is not only a speed finding. The lighter engine exists for the machine
that cannot take spaCy, so being far slower than the heavy one inverts the
reason to choose it, and in the web interface a multi-second redaction reads as
a hang — which invites a second submission of the same unredacted text.

`CP-029` leaves one cost in place deliberately: a URL genuinely ending in a
bracket now has its closing `)` left in the clear. That is the safe direction,
because it discloses nothing.

`CP-028` and `CP-029` were both found by the same new class-level test, which
runs every pattern's own `examples_yes` through twelve ordinary sentence
positions. The suite had never done that: each pattern had only ever been
tested against its examples in isolation, and in isolation there is no
punctuation next to a value. Because the check is class-level rather than a
list of cases, a pattern added next year is checked the same way without anyone
remembering to.

### Two defects found by a user in round four

Neither was found by a lane. Both were reported by someone trying to use the
command line, and neither is a leak — which does not make them small, because a
redaction tool that is awkward to feed gets bypassed, and the text goes to the
model unredacted instead.

| ID | Severity | Behaviour | Root cause | Retired by |
|---|---|---|---|---|
| `CP-030` | high | `cleanprompt inspect --ner` followed by a pasted paragraph produced three errors in a row: `bash: syntax error near unexpected token '('` from the shell, then `Error: No such option '-M'`, then `Error: Got unexpected extra argument` — the last one ours, naming the user's own text as the problem and offering no alternative. Run bare at a terminal, the same commands blocked on standard input with no output at all, which reads as a hang | the text commands accepted only `--in PATH` and standard input, so words on the command line had nowhere to go, and nothing announced that stdin was being read | a variadic `TEXT` positional on `redact`, `restore`, `inspect`, `scan` and `roundtrip`; one `_resolve_text` helper resolving positional → `--in` → stdin; and an interactive hint on stderr when the fallthrough is a terminal |
| `CP-031` | medium | the restoration half of the pipeline was never demonstrated: seeing a value come back required two commands, a vault path, and knowing what a vault is, and every README example used `--in prompt.txt` | `redact` and `restore` are deliberately separate, with a file between them — correct for real work, where the reply may arrive days later in another process, but it hides the half that makes the tool trustworthy | a `roundtrip` command (alias `demo`) showing all five stages in one shot with nothing written to disk, and `--format json` emitting the same stages with `round_trip_exact` and `unresolved_placeholders` |

**One helper resolves text, and it refuses ambiguity.** Every handler that takes
text calls `_resolve_text(args, stdin, stderr, command)`, which tries the
positional, then `--in`, then standard input. Five handlers resolving their own
text would drift into five different orders, and the difference would only show
up when a user supplied two sources at once. That case raises `PolicyError`
rather than silently preferring one: the input that would be dropped is exactly
the text the person meant to redact, and a redaction tool that quietly processes
the other one has returned a confident answer about the wrong document.

The positional is variadic (`Param(dest="text_args", kind="argument",
multiple=True)`) because a shell splits `Ada Lovelace` into two arguments before
the program is reached. Requiring one quoted argument would enforce a
distinction the user never made, and the error for getting it wrong is the one
`CP-030` was about.

The interactive hint goes to **stderr**, never stdout, because stdout carries
the redacted text in a pipe and a note mixed into it would be sent to the model.
It is printed only when `stdin.isatty()`, and that call is guarded, so a
`StringIO` in a test and a pipe in a script stay silent.

The shell error is not ours to fix and the design does not pretend otherwise. A
shell consumes `(`, `$` and a backtick before this program is executed. The
documented answer is a heredoc with a quoted delimiter, `<<'END'`, which
disables every substitution so arbitrary prose arrives intact — a workaround for
a layer below us, not a repair.

`roundtrip`'s stage 4 is a fixed stand-in string, labelled "NOT from a model" on
every run, with `--reply TEXT` / `--reply-in PATH` to substitute a real answer.
Generating plausible text and presenting it as a model's reply would be a lie
told by a tool whose entire value is that its output can be trusted.

### Two defects found in the option grammar in round five

Also not inherited. Both came out of a user's question — whether the CLI handled
`--`, the POSIX end-of-options delimiter, and long options properly — answered
by running the same option grammar through both frontends. `--` itself already
behaved identically. These two did not.

| ID | Severity | Behaviour | Root cause | Retired by |
|---|---|---|---|---|
| `CP-032` | medium | an abbreviated long option — `--form` for `--format` — was accepted under argparse and exited 2 under click, so the same command line worked or failed depending on which library happened to be installed | argparse accepts any unambiguous prefix of a long option by default; click accepts none | `allow_abbrev=False` on the root parser **and on every subparser** — subparsers do not inherit it, and the subparsers are where every declared option lives |
| `CP-033` | high | `inspect "--secret is a@b.co"` was accepted as text under argparse and quietly processed, while click rejected it as an unknown option; a mistyped option name became the user's prompt and the option they meant was never applied, with nothing saying so | argparse classifies any token containing a space as a positional, whatever character it begins with | `_reject_stray_options` in `_frontends.py`, a post-parse check refusing a dash-leading positional unless a `--` delimiter authorised it, with a message naming the delimiter |

`CP-032` is `CP-021`'s class of divergence, and it is resolved in the same
direction: argparse conforms to click, because the strict behaviour is the one
that can be relied on. An abbreviation is a latent break on a single machine as
well as across two. It is unambiguous only until some later option shares its
prefix, so a script that worked for a year fails on an upgrade that added a
feature the script never used.

`CP-033`'s divergence matters less than its direction. A redaction tool that
decides what leaves the machine must not silently do something other than what
was asked, and treating a mistyped flag as the text to redact is exactly that.
The fix is a check rather than an override because argparse makes this decision
inside `_parse_optional`, which is private and has moved between releases; the
declared dependency range has to keep working, so the behaviour is corrected
after parsing rather than by reaching into the parser. Nothing is inspected past
a `--` — that is precisely what the delimiter means — and a lone `-` stays
valid, since it conventionally names standard input.

**The IR promises only what both frontends render identically, and that promise
covers the grammar, not just the set.** `_spec.py` was already constrained to
parameters argparse and click can both express (`CP-021`); round five extended
the same rule to how a token is *read* — abbreviation, `--` handling, short
aliases, and the spelling of a dash-leading option value. Anything the two
libraries read differently is either made to agree or written down. Two pieces
follow from that. Short aliases (`-h/--help`, `-V/--version`, `-i/--in`,
`-o/--out`, `-f/--format`, `-k/--kinds`, `-q/--quiet`) are declared once on the
`Param` and rendered by both. And `_Parser`, a thin `argparse.ArgumentParser`
subclass, rewrites one message: argparse's "expected one argument", which fires
on `--hide -secret`, now also names the two forms that work,
`--hide=-your-value` and `--` for text. It appends rather than replaces, so a
message nobody anticipated still reaches the user intact.

One split is documented rather than eliminated. The attached form
`--hide=-secret` is accepted by both frontends and is the spelling the README
gives; the separated `--hide -secret` is accepted by click and refused by
argparse. Closing that would mean overriding the same private argparse
internals, so it is stated instead of hidden.

### Three defects found by a user in round six

Also not inherited, and none of them a leak in the pipeline. All three came out
of one reported session, and all three are about the vault being a file the
user had to manage by hand — which is the same class as `CP-030` and `CP-031`,
because a tool whose safe path is laborious gets stepped around.

| ID | Severity | Behaviour | Root cause | Retired by |
|---|---|---|---|---|
| `CP-034` | medium | `--vault` was declared required on both `redact` and `restore`, so the shortest useful command carried a path the user had to invent and then keep identical across runs, with no default to fall back on | the vault was modelled purely as a caller-supplied artifact; nothing in the CLI owned the question of where a vault should live | `default_vault_path()` resolving `CLEANPROMPT_VAULT`, else the platform state directory — `$XDG_STATE_HOME/cleanprompt/vault.json`, falling back to `~/.local/state/cleanprompt/`, and `%LOCALAPPDATA%\cleanprompt\` on Windows — with the directory created `0700` and the file `0600`, a one-line stderr note naming the resolved path on every write, and `doctor` reporting `configuration.vault_path` / `vault_exists` |
| `CP-035` | high | a vault could not be continued across the turns of one conversation: each run rewrote it, so numbering restarted and the same address could be `[EMAIL-1]` in one turn and `[EMAIL-2]` in another. A reply quoting an earlier placeholder restored to the wrong value, or to nothing | the CLI wrote a fresh vault per invocation and never seeded the redactor from an existing one, although the engine had taken a `seed` argument since round one for the `Session` API | `--vault-mode overwrite\|append`; append seeds the redactor from the existing vault through that same `seed` argument and then merges, so a value keeps the label it already had and a new one is numbered after the old ones. The vault document gained an `index` array of `{label, kind, ordinal}` and the format moved from 1 to 2; the build reads both |
| `CP-036` | medium | there was no command for "just give me the clean prompt". The pieces existed — `redact --vault v.json 2>/dev/null` produced exactly that — but the reported session shows a user reaching for `inspect`, whose entire purpose is the report, and then asking how to get the clean prompt with no additional information around it | an affordance reachable only as a combination of three flags and a shell redirection is not an affordance | a `clean` command (alias `prompt`): the redacted text alone on stdout, the vault at the default location in append mode, and one stderr line naming the vault plus a warning when a high-severity detection gap means the text has not actually been checked for names; `-q` silences both |

**The default vault location is a security decision, and the working directory
is the one place it must not be.** A vault holds the removed values in clear
text, and this tool is used inside checkouts — the reported session ran from
`/work/.git_clones/learn/docs` on a checked-out branch. A vault defaulting
there is one `git add .` from committing the exact values the user was
redacting in order not to send them anywhere, in a repository whose history is
then pushed. The state directory is chosen instead because it is per-user, not
per-project, and nothing sweeps it into a commit. `--vault PATH` overrides it
for one run and `CLEANPROMPT_VAULT` for a whole shell, so the explicit,
scripted workflow is unchanged.

The `0700` directory and `0600` file modes are requested **at creation** rather
than applied with a `chmod` afterwards. A chmod after the fact leaves a window
in which the file exists with whatever the process umask allowed, and on a
shared machine that window is long enough to matter for a file of this kind.

The path note goes to **stderr**, for the same reason as `CP-030`'s interactive
hint: stdout carries the redacted text into a pipe, and a line mixed into it
would be sent to the model.

**Appending onto a format-1 vault is refused rather than guessed.** A vault
written before the `index` existed records no label-to-category mapping, so a
merge would have to renumber from `[KIND-1]` while the old vault already uses
that label for a different value. The result is one placeholder standing for
two values, and restoration then hands the user someone else's data — the worst
outcome available to this subsystem, and strictly worse than the error message
that replaces it. Refusing costs a user one `--vault-mode overwrite`. Format-1
vaults still restore normally; only appending is refused.

The `index` deliberately carries no secrets. A label and its category are
already present in the text that was sent to the model, so recording them adds
no disclosure; the values stay in `entries`, which is what vault encryption
covers. Nothing that would widen that should be moved into the index to make
merging cheaper.

**`clean` appends and `redact` overwrites, on purpose.** They are used for
different things. Append is what a conversation wants — the same address keeps
the same placeholder in turn one and turn nine, and the vault accumulates, so a
reply quoting any earlier turn still restores. Overwrite is right for `redact`'s
scripted, file-based workflow, which keeps behaving exactly as it always did.
The asymmetry is the point; making both defaults the same would break one of
the two uses.

### Two defects found by a user in round seven, one of them ours

`CP-037` is inherited in the usual sense: the names were wrong before this
round. `CP-038` is not, and not in the way the earlier "found by our own
verification" sections mean either — it was **introduced by round six**. This
register otherwise reads as a list of other people's mistakes, and it should
not, because the change that caused `CP-038` was two decisions this plane made
on purpose and would make again.

| ID | Severity | Behaviour | Root cause | Retired by |
|---|---|---|---|---|
| `CP-037` | medium | `encode` and `decode` named a pair they did not form. `encode` was an alias of `redact`, which writes a vault to a named path and prints a table of what it removed; `decode` was an alias of `restore`, which prints the restored text and little else. Two names that read as the two halves of one operation behaved nothing like it — one verbose and file-oriented, the other minimal. The user asked for "encode to clean, decode that AI chat answer decoded", which is the symmetry the names were already promising | the aliases were attached to whichever existing command was nearest in mechanism rather than to the command each name describes | the names moved onto the commands that mean them. `encode` is the canonical name of the minimal half — text in, a pasteable prompt on stdout alone, the default vault in append mode — with `clean` and `prompt` as aliases; `decode` is the canonical name of the other half, with `restore` as its alias and a summary saying it puts the values back into the model's answer. `redact` keeps its behaviour under its own name and loses only the alias. Each half's help names the other, and the module docstring leads with the pair rather than with ten subcommands |
| `CP-038` | high | the removed values accumulated with no way to delete them. `CP-034` put the vault at a fixed default path and `CP-035` made the conversational command append to it, so a user who ran it a few times had a growing clear-text file of everything they had redacted, at a location they had never chosen, and no command removed it | not an oversight in either change on its own: each is correct, and together they mean the tool starts retaining personal data by default. Nothing in the command table owned the end of a conversation | a `forget` command (alias `clear-vault`) that reports and stops, deleting only under `--force`; the report names how many values and of which categories, never a value; the summary read catches `ValueError`, `OSError` and `CleanPromptError` so an unparsable vault is still deleted; and the deletion message states that unlinking is not shredding and points at `--encrypt` |

**A feature that retains personal data by default must ship the command that
deletes it.** That is the rule `CP-038` leaves behind, and it is stated as a
rule because the failure was not visible from either contributing change. A
default vault path is a good decision; appending is a good decision; the
combination is a tool that begins collecting personal data on the user's behalf
and offers no way to stop, which is worse than one that never collected it. The
question to ask of any new default is not "is this convenient?" but "what does
this now keep, where, and what removes it?".

**`forget` reports before it acts, and never by showing a value.** Deleting the
vault is the one action in this subsystem that cannot be undone from inside it:
afterwards `decode` cannot restore that conversation, which is the point and
also the cost, so the default is a description and `--force` is the deletion.
The description gives counts and categories — the same information already
present in the text that was sent to the model — because a user deciding
whether to delete their own secrets should not have those secrets put back on
their screen by the act of deciding.

**An unreadable vault is still deleted.** `_cmd_forget` wraps the summary read
in `except (ValueError, OSError, CleanPromptError)`: a truncated write leaves
invalid JSON, a permission or device problem raises `OSError`, and a malformed
document raises our own error. Only the summary is lost. Refusing to delete a
file because it cannot be parsed would block exactly the user who most wants it
gone, and the failure that produced the unparsable file — an interrupted
write — is one this tool can cause.

**Unlinking is not shredding, and the message says so.** On a journalling or
copy-on-write filesystem, on an SSD with wear levelling, or where a backup ran
between the write and the delete, the bytes may outlive the unlink. `forget`
states that limit and names `--encrypt` with a key the user holds as the answer
for values that must not be recoverable — a file that was never readable on
disk does not depend on the delete having worked. Claiming a secure erase this
layer cannot deliver would be security theatre, in a tool whose only value is
that its claims can be believed.

### Two defects found by a user in round eight

Neither is a leak, and neither was found by a lane. Both are about the same
thing from two directions: encryption was easier to recommend than to use.

| ID | Severity | Behaviour | Root cause | Retired by |
|---|---|---|---|---|
| `CP-039` | medium | `encode` had no `--encrypt` flag, while `forget`'s deletion note ended by telling the reader to "use `--encrypt` with a key you hold for values that must not be recoverable". `encode` is the command that writes the default vault in append mode, so its users accumulate the most removed values, and the one piece of advice the tool gives about that accumulation named an option their command did not have | `--encrypt` had been attached to `redact`, the explicit file-based command, when the conversational command was the one whose vault grows | `--encrypt` and `--cipher` are shared `Param` objects declared once and attached to both `encode` and `redact`, rendered from that single declaration by both frontends |
| `CP-040` | high | `--encrypt` required the `crypto` tier — the `cryptography` distribution, with compiled extensions — so the one security feature in the subsystem was the only part of the base workflow needing an installed package, and a vault written with it could not be opened on a machine without it | encryption was modelled as an optional capability like entity detection, when it is the mitigation the tool itself recommends for its own retained data | `_vaultcrypt.py`, a base-tier module building an authenticated construction from `hashlib`, `hmac` and `secrets` alone, with `--cipher portable\|fernet\|auto` defaulting to `portable` and the Fernet backend kept behind `--cipher fernet` |

**A security feature must not depend on an optional tier.** That is the rule
`CP-040` leaves behind. This submodule's central promise is that the base tier
is pure standard library and therefore works anywhere; making the one
protective feature the single exception inverted that guarantee at the point it
mattered most, and did it twice over — the user who most needs to encrypt is
the one least likely to be able to install a package, and a vault they did
manage to encrypt became a file only certain machines could open.

**The construction, stated so it can be reviewed.** `_vaultcrypt.py` does four
things, in this order:

1. **Key derivation.** `hashlib.scrypt` with a random 16-byte salt, `n=2**14`,
   `r=8`, `p=1`, 64 bytes out. Where the interpreter cannot run scrypt, which
   needs OpenSSL 1.1, `hashlib.pbkdf2_hmac` with SHA-256 and 240,000 rounds.
   Presence of the `scrypt` attribute is not availability, so
   `_scrypt_is_usable()` runs it once with throwaway parameters and returns a
   decision rather than inferring one.
2. **Key separation.** The 64 bytes split into a 32-byte encryption key and a
   32-byte MAC key. One secret never serves two purposes.
3. **Encryption.** HMAC-SHA256 in counter mode — `HMAC(k_enc, nonce ||
   counter)` — XORed with the plaintext, with a fresh 16-byte nonce per value
   from `secrets`, so no keystream is ever reused across values or runs.
4. **Authentication.** Encrypt-then-MAC: HMAC-SHA256 over the cipher name, the
   label, the nonce and the ciphertext, compared with `hmac.compare_digest`
   before any byte is decrypted. The **label** is bound into the tag because
   without it someone with write access to the vault could swap two tokens and
   make `[EMAIL-1]` restore to a different person's address, with every
   integrity check still passing and no key needed — restoration returning
   someone else's data, which is this subsystem's worst outcome, reached
   without breaking anything.

The vault document records `cipher`, the versioned construction name
`cleanprompt-hmac-v1` rather than the `--cipher` word the user typed, and the
full `kdf` parameters — name, salt, and cost. Decryption therefore dispatches
on what the document says instead of guessing from what this build happens to
do, and a future construction can be introduced under a new name without
stranding the vaults written today.

**The honest limit.** This is a standard composition, not a reviewed
implementation of AES. Fernet is the better primitive where it can be had, and
`--cipher fernet` keeps it one flag away, with `auto` preferring it when the
tier is present. `portable` is the default because a vault that cannot be
opened is a worse outcome than a difference between two authenticated
constructions that both rest on standard assumptions, and the reason to encrypt
a vault at all is usually that it will outlive the moment. The module sets the
construction out step by step so it can be checked rather than taken on trust,
and the tests assert the properties it must deliver — a wrong passphrase, a bit
flipped anywhere in the token, a swapped label, a truncated token, nonce
freshness across values — rather than claiming to prove the construction
secure, which no test suite can do. Nothing written about this module should
go further than that.

**The key never has to live in the environment.** It is read from
`CLEANPROMPT_VAULT_KEY` if set, and otherwise from a `getpass` prompt at a
terminal. Without the prompt the only way to encrypt is the environment
variable, and on most shells that leaves a plain-text copy of the key in the
history file next to the vault it protects — the key and the ciphertext on one
disk, which is the failure encryption was supposed to prevent.
`doctor --new-key` emits a passphrase from the standard library alone: four
groups of five characters from a 30-symbol alphabet with look-alikes removed,
just under 98 bits. A key generator that needs an optional package is no use to
the person being told to encrypt.

The `crypto` tier's declared purpose narrowed accordingly, from "authenticated
vault encryption" to "the Fernet vault cipher (`--cipher fernet`); encryption
itself needs no tier". That is a much smaller claim than the row used to make,
and the accurate one now that the base tier encrypts.

One note on how `CP-040` was built rather than what it does. The first draft of
the scrypt probe swallowed its failure in `except Exception: pass`; the
project's own architecture test `test_no_bare_except_or_silent_pass` and the
contract checker rule `CP-SAFE-002` both rejected it, and it became
`_scrypt_is_usable()` returning a decision. A silent `pass` inside a capability
probe is how a tier comes to report itself unusable for a reason nobody can
reconstruct afterwards, which is the `BROKEN`-is-not-`ABSENT` distinction in
section 8 arriving by another route.

### Three defects found in round nine, all from one root cause

The pipeline is `detect → assign → rewrite → [language model] → restore`, and
every invariant in section 3 holds across the parts this submodule controls.
Restoration was the exception. The design treated the hop between redaction and
restoration as a lossless channel — a pipe that returns what was put into it.
It is a language model, and a language model rewrites tokens. All three
findings below are that assumption failing in a different place, which is why
the three responses are deliberately complementary rather than three attempts
at the same fix: `CP-041` prevents, `CP-042` mitigates and detects, `CP-043`
removes the invitation. That is the same defence-in-depth pattern `_logging.py`
already uses, where the rule holds by construction at every call site and
`SecretFilter` scrubs the records anyway.

| ID | Severity | Behaviour | Root cause | Retired by |
|---|---|---|---|---|
| `CP-041` | high | on `Mustafa Kemal Atatürk[e] (c. 1881)` spaCy returns the person as `'Mustafa Kemal Atatürk[e'` — it swallows the opening bracket and the footnote letter and leaves the closing one behind. Visibly this produced a malformed `[PERSON-1]]` in the prompt; silently it recorded the person's name in the vault **as** `Mustafa Kemal Atatürk[e`, so restoring into any other text would have produced that string as somebody's name | an asymmetry: every structural pattern carries a validator — Luhn, mod-97, octet ranges — while an entity span, which arrives from a third-party model, was taken verbatim with nothing checked | `trim_entity_span()` in `_engines.py`, applied by both engines: a span is truncated at its first unmatched opener and started after its last unmatched closer. Balanced brackets inside a span (`Acme (Europe) Ltd`) are left alone, and only brackets are trimmed |
| `CP-042` | critical | a placeholder the model rewrote was not restored, and the report barely said so. Measured against realistic replies, `[EMAIL-1]` comes back lower-cased, with an underscore or a space for the hyphen, with a Unicode dash, with the brackets escaped for Markdown, and wrapped across a line. Exact matching restored the first spelling and missed every other; the report then read `restored 0 placeholder(s); 2 vault entr(y/ies) unused` with exit status 0, and `-q` silenced even that, so a half-restored answer could be pasted onward unnoticed | restoration scanned for the placeholder grammar exactly as issued, because the reply was assumed to carry the tokens back unchanged | `TagStyle.lenient_pattern()` and `TagStyle.normalize()` recognising that bounded set of rewrites and nothing wider; `restore()` acting on a lenient match only when it resolves to a label the vault holds; every repair listed in the new `RestorationResult.repaired` field and named in the note; and an explicit message when nothing resolved while the vault is non-empty. `--exact` (CLI) / `lenient=False` (API) restores the previous behaviour |
| `CP-043` | medium | `[PERSON-1] emailed [EMAIL-1] about [ORG-1]` is not a sentence: the tokens carry no grammatical number and no animacy, they break the sentence, and they invite the model to comment on the redaction instead of answering. They are also exactly the sort of token a model normalises, which is what produces `CP-042`'s damage in the first place | the stand-in was designed to be unambiguous to *this* code and nothing was asked about how it reads to the model that receives it | a `surrogate` tag style (`--style surrogate`, `_surrogates.py`) substituting consistent invented values for the kinds where a proper noun is the natural replacement — `PERSON`, `ORG`, `GPE`, `LOC`, `FAC`, `EMAIL`, `PHONE`, `URL` — so the text reads as prose and there is nothing to normalise |

**Nothing from a third party becomes a vault value unchecked.** That is the
rule `CP-041` leaves behind, and it is worth stating because the register up to
here is about untrusted *text*. Here the untrusted input was a dependency's
output. A pattern in `_patterns.py` is written in this repository and validated
here; an entity span is produced by a model nobody in this project controls,
and it had no check of any kind. The check is deliberately narrow — brackets
only. A trailing full stop is legitimate in `Inc.`, so trimming it would be a
guess, and this subsystem does not guess about what a value is.

**Leniency is bounded by the vault, not by the pattern.** `restore()` resolves
a lenient match only when the normalised label is one the vault actually holds.
Without that condition every bracketed word in a model's answer becomes a
restoration candidate: `[note 2]` would be rewritten or, at best, reported as
an unknown placeholder in a text where it is ordinary prose. The pattern is
bounded too — the category must start with a letter, so `[1]` is never a
candidate — but the vault-membership test is what makes the leniency safe, and
it must stay.

**The `style` field is in the grammar fingerprint, but a default style is
omitted from the payload.** `style` lives on `TagStyle`, so it participates in
the fingerprint and a vault written in one style cannot be read as the other —
which is correct, because half-restoring a reply is worse than refusing it.
`TagStyle._fingerprint_payload()` nevertheless omits the field when it holds
its default value, so a placeholder-style grammar keeps the digest it has
always had and every vault written before the field existed stays readable.
Adding the field naively would have made every existing vault unreadable in the
name of a distinction those vaults could not be on the wrong side of.

**Surrogate reversal is one merged pass, and it must stay one.** Reversal
reuses `LiteralDetector` and collects grammar matches and literal surrogate
matches over the *original* reply text, then rewrites once. A second pass over
the first pass's output would let an already-restored value be re-matched as
another key — `CP-006` reappearing on the restoration side, where it would
corrupt the model's answer instead of the prompt. The structural reason is the
same one invariant I6 states for detection: stages compose over spans, never
over each other's strings.

**Credentials keep their placeholders, on purpose.** `CREDIT_CARD`, `IBAN`,
`SSN_US`, `AWS_ACCESS_KEY`, `JWT`, `PRIVATE_KEY`, `MAC`, `IPV4`, `IPV6` and
`NORP` are never surrogated. A plausible-looking card number or access key is a
hazard rather than a convenience: a person can mistake it for real, a system
can act on it, and by chance it could *be* real. `NORP` is excluded for a
different reason — it is adjectival, and an invented demonym reads as nonsense.
Where a surrogate is generated it uses a reserved form if one exists:
`example.invalid` (RFC 2606) for addresses and links, and the `+1 555
0100`–`0199` fiction block for telephone numbers. Every stand-in is checked
against both the source text and the ones already issued, because a collision
on either side restores two values as one.

**The honest limit.** A surrogate is ordinary text, so it gives up the one
property a bracket label has for free: being obviously not part of the
document. A reader of the redacted prompt cannot see which words were
substituted, and a surrogate that escapes into something a person acts on is
indistinguishable from a fact. `placeholder` therefore remains the default, and
`surrogate` is the choice to make for text going to a model rather than for
text a person will read and act on.

### Investigated and rejected

Recorded so they are not "re-found" later:

- Placeholder prefix collision (`[PERSON-1]` vs `[PERSON-11]`) does **not**
  occur: the closing delimiter makes `[PERSON-1]` not a prefix of `[PERSON-11]`.
  Measured, not assumed. The rewrite keeps a closing delimiter mandatory and
  tests it (`test__engine.py::test_ordinal_ten_does_not_collide`).
- `/reset` rendering the template without context is **not** a failure: Jinja's
  default `Undefined` is falsy and renders empty.

## 6. Module map

`scikitplot/cleanprompt/` — one responsibility per module; `test_<module>.py`
mirrors each source name.

```text
__init__.py        public facade: base-tier __all__, PEP 562 __getattr__, non-resolving __dir__
_exceptions.py     the exception tree above
_capabilities.py   CapabilityStatus (7 states) + probes; no third-party imports
_types.py          Span, Entry, RedactionResult, RestorationResult — frozen dataclasses
_vault.py          Vault: label -> secret, redacting repr, explicit export/clear
_policy.py         TagStyle, Limits, RedactionPolicy — declarative, hashable, serializable
_patterns.py       curated regex library, each with intent + positive/negative examples
_detectors.py      Detector protocol, RegexDetector, LiteralDetector, DetectorRegistry
_engine.py         resolve -> assign -> rewrite; Redactor, restore()
_engines.py        engine selection and the canonical entity vocabulary
_languages.py      language -> spaCy model, with an announced fallback
_surrogates.py     consistent invented stand-ins for the `surrogate` tag style
_logging.py        logger, SecretFilter, JsonFormatter, configure_logging
_api.py            encode / decode / Handle / Session — the LLM-facing surface
_vaultcrypt.py     vault encryption from hashlib/hmac/secrets; base tier, no dependency
_diagnostics.py    describe_outcome and suggest_terms; one story for all surfaces
_spec.py           framework-neutral Param/Command IR for the command surface
_frontends.py      argparse and click renderers over that IR
_ner.py            optional spaCy detector (tier: ner)
_nltk.py           optional NLTK detector (tier: nltk); offsets carried, not searched
_crypto.py         optional Fernet vault cipher (tier: crypto); --cipher fernet only
_render.py         optional ANSI/HTML presentation over results
_cli.py            the command handlers, stdin/stdout safe; one _resolve_text for every text command
_session.py        interactive terminal session
__main__.py        python -m scikitplot.cleanprompt
_app.py            optional Flask app (tier: web); server-side vault store
_serve.py          starting the web interface, and the files to run it elsewhere
```

`_nltk.py` deserves the one-line reason it is written the way it is:
`nltk.ne_chunk` returns tokens with no character offsets, and this pipeline is
built entirely on spans over the original string. Recovering an offset by
searching for the entity text would collapse a name's second occurrence onto
its first, and NLTK's tokenizer rewrites some characters, so the token text is
not always a substring of the source at all. The offsets are therefore carried
through `PunktSentenceTokenizer.span_tokenize` and
`TreebankWordTokenizer.span_tokenize`, and the guarantee — checked as its own
verification lane, for every engine — is that `text[span.start:span.end]` is
the entity exactly as it appears in the source.

`_logging.py` has one rule: a removed value never reaches a log record. Records
carry counts, kinds, labels, offsets, durations and capability decisions, never
surfaces. A redaction tool that logs what it removed has defeated itself — the
data leaves through the log rather than the prompt, in a more durable form,
because log lines are shipped off the machine, indexed and retained long after
the prompt is gone. The rule holds by construction at every call site, and
`SecretFilter` scrubs records as defence in depth. A `NullHandler` is installed
at import under `LOGGER_NAME`, so nothing is emitted until `configure_logging`
(idempotent) is called.

## 7. Tier and lazy-import contract

`import scikitplot.cleanprompt` must import **no third-party package at all** —
not `spacy`, not `flask`, not `cryptography`, not `numpy`. The base tier is
pure standard library so that adding this submodule cannot change the import
cost or the failure surface of any other `scikitplot` submodule.

| Tier | Requires | Surface |
|---|---|---|
| `base` | stdlib only | `Redactor`, `restore`, policy/types/patterns/detectors, `Vault`, `encode`/`decode`/`Session`, engine and language tables, logging, CLI, vault encryption (`_vaultcrypt.py`), surrogate stand-ins (`_surrogates.py`) |
| `ner` | `spacy` + a model | `NerDetector`, `spacy_detector()` |
| `nltk` | `nltk` + corpora | `NltkDetector`, `nltk_detector()`, `corpora_status()` |
| `web` | `flask` | `create_app()`, `SessionStore` |
| `crypto` | `cryptography` | the Fernet vault cipher (`--cipher fernet`); encryption itself needs no tier |

`_engines.py`, `_languages.py`, `_logging.py` and `_api.py` are all base tier:
they name engines, models and log formats without importing any of them, which
is what lets `doctor` report on a tier that is not installed.

The `nltk` tier is honest about what it is. It is English-only, and its chunker
is of 1990s vintage, so its recall is below spaCy's. It exists for the machine
that cannot take spaCy — and `CP-027` is the reminder that a lighter engine
which is slower than the heavy one has no reason to exist.

Facade rules, following this repository's `MCP-D04` decision:

- `__all__` contains **base-tier names only**. `__all__` *is* the star-import
  surface; listing an optional name there makes `from … import *` resolve it and
  defeats the lazy tier at the one operation that touches every entry.
- Optional-tier names resolve through `__getattr__` (PEP 562) on first attribute
  access, and each raises `CapabilityError` with an install instruction when the
  tier is unavailable — raised *before* the heavy import is attempted.
- `__dir__` unions `globals()` with the optional names **as strings** and never
  resolves them, so tab-completion and Sphinx stay base-safe.

## 8. Independence

Per this repository's standing rule, `scikitplot.cleanprompt` imports no other
`scikitplot` submodule — not even a shared `_utils`. The 7-state
`CapabilityStatus` vocabulary is the project's canonical one (owned by
`scikitplot.corpus`); it is consumed **by value** as a local `str` enum copy,
which is exactly how `scikitplot.mcp` was directed to consume it. `BROKEN`
(installed and failing) must never collapse into `ABSENT` (not installed).

## 9. Dependency policy

No `==` pins. The extras in `pyproject.toml` are version-free, like every other
requirement of the project: the installer is never told a range. The ranges
below are the ones the submodule supports and checks at run time
(`_capabilities.py`: `TIERS`), reporting a tier outside its range as
unavailable with the reason; an upper bound exists only where the
distribution's major number marks breaking changes:

```text
cleanprompt        (none — stdlib only)
cleanprompt-ner    spacy>=3.4,<5
cleanprompt-nltk   nltk>=3.6,<4
cleanprompt-web    flask>=2.2,<4
cleanprompt-crypto cryptography>=41       (the Fernet cipher only, since CP-040; no ceiling, CP-092)
```

Upper bounds exist because each has a history of breaking changes at the major
boundary (`spacy` 2→3 changed the model API, `flask` 1→2 changed the app
factory and async surface, and `nltk` 3.8.2 split several of its data
packages, so both the legacy and the `_tab` resource names are probed and
either one satisfies the lookup). Compatibility is asserted across the declared
floor and ceiling, not at one pin.

`cryptography` has a floor and no ceiling (`CP-092`). Its API stability policy
says the major number "is incremented on any feature release"; four majors
appeared between April and July 2026. A ceiling at a major therefore carries no
compatibility information and expires within weeks, and when it expires the
tier reports `INCOMPATIBLE` for every fresh install although nothing changed.
The same policy keeps code that runs without warnings working for two further
majors and announces a removal with `CryptographyDeprecationWarning`, which the
project's test configuration turns into a failure. The floor and the newest
release are both run: 41.0.0 and 50.0.2 this round.

A dependency that is installed and refused must not pass unnoticed:
`test__capabilities.TestInstalledTiers` fails in that state, and every skip
reason carries the probe's status and detail (`tests/_tiers.py`).

## 10. Verification ladder

Distinct lanes; a missing lane is `UNAVAILABLE`, never `PASS`.

`VERIFICATION.md` holds the authoritative table and what each lane proves; this
is the shape of it.

1. maintenance contract (static/mutation)
2. focused unit tests, collected from the checked-in files with no harness edits
3. invariant/property tests (round trip, determinism, disjointedness, idempotence)
4. import-isolation probe under an `__import__` blocker for `spacy`, `nltk`,
   `flask`, `cryptography`, `numpy` — including a full encrypt/decrypt round
   trip while `cryptography` is blocked
5. star-import base-safety probe
6. negative probes, one per retired defect `CP-001 … CP-045`
7. CLI end-to-end, frontend parity, and global CLI delegation
8. entity engines live — real spaCy and real NLTK, both measured
9. engine interchangeability: a vault written under one engine restores under
   the other
10. offset alignment: every span indexes the original text, for every engine
11. every pattern's own examples in ordinary sentence positions
12. `web` tier live, and the vault ciphers live — the portable construction
    with no dependency installed, and Fernet under the `crypto` tier
13. `encode`/`decode` round trip, and that the handle discloses nothing
14. option grammar through both frontends — `--` passthrough, refused
    abbreviations, stray dash-leading positionals, full/short/attached spellings
15. vault persistence — the resolved default path and its modes, append
    continuing a conversation's numbering, and the format-1 append refusal
16. restoration through a lossy channel — the ten measured rewrite shapes of a
    placeholder all restore, prose lookalikes are left untouched, and `--exact`
    still restores only what was issued
17. published gallery examples — every one executes, confines itself to its own
    workspace, and prints no value that could reach anybody
18. supported Python/platform matrix — `UNAVAILABLE`

Every lane is `PASS` except the last. The platform matrix needs a CI matrix:
the code targets Python 3.8 and avoids every construct newer than that, but
"avoids" is not "verified", and this lane says so rather than implying
otherwise — one interpreter, Linux only.

Lane 8 was `UNAVAILABLE` for two rounds and is now real: spaCy 3.8.16 with
`en_core_web_sm` and NLTK 3.10.3 with its data packages, both measured over 120
fuzzed documents per engine mode. `tests/test__ner.py` still runs a stub, and a
stub proves this module's translation of entity offsets into spans and nothing
about recognition quality; it was never recorded as this lane. Holding the lane
open rather than closing it early is what surfaced `CP-027`, because the probe
timed out where it should have taken seconds.


## 11. Documentation surface

Added in round ten, because the submodule had a thorough `README.md` inside the
package and **no presence at all** in the rendered documentation: the gallery
folder's `README.txt` was checked in empty, so Sphinx-Gallery skipped it and
the build succeeded having published nothing.

### D-11.1 — difficulty is the spine, surface is the axis

The request named six documents: a readme, CLI usage with edge cases, Python
usage with edge cases, and tours at basics, moderate and advanced levels. Read
literally that is a 2×3 grid — every concept taught once per surface — which
would put the same material in two places and let the two drift.

They are two axes instead:

- **basics / moderate / advanced** are a narrative learning path. They teach
  *concepts*, and use whichever surface makes the point clearest.
- **command line / Python API** are references organised by *surface*, each
  with the ordinary form first and then the edge cases.

That the request asked for "single and complex edge cases" on the two surface
documents and not on the three levels is the shape confirming itself. A concept
is taught in exactly one place; the exhaustive forms live in exactly one place.

A seventh, **recipes**, holds the complete patterns — a CI gate, a pre-commit
hook, a notebook wrapper, a batch pass, an agent loop, log hygiene — because a
worked pattern is not a concept and does not belong in a tour.

### D-11.2 — a gallery example is code, and is verified as code

Sphinx-Gallery *executes* these scripts during a documentation build. They are
therefore held to the same standard as the rest of the tree, with three rules
that have nothing to do with prose quality:

**Hermetic.** A vault holds removed values in clear text, and the CLI resolves
one in the platform state directory by default. Every script points
`CLEANPROMPT_VAULT` at a `TemporaryDirectory` in its first cell and cleans it
up in its last. Lane 23 runs each script with `HOME` and `XDG_STATE_HOME`
redirected into a sandbox and asserts the sandbox is **empty** afterwards.

**Self-asserting.** An example that prints a redacted prompt has already
checked that the original value is absent from it. A tutorial that demonstrates
a leak while claiming success is worse than no tutorial.

**Reserved values only.** A gallery page is published. Addresses come from the
`example.*` domains of RFC 2606, network addresses from RFC 5737 and RFC 3849,
telephone numbers from the NANP fiction block. Lane 23 greps every captured
stream and fails on anything else.

### D-11.3 — an absent capability is a skip, a defect is a failure

The repository's existing gallery convention, kept: a missing optional package,
model or data package reports a visible, specific `SKIP` and the example
continues where it can stay truthful; an invalid public API, a failed round
trip or a value surviving into the redacted text fails visibly.

The distinction is load-bearing here rather than stylistic, and the sandbox in
lane 23 is what keeps it honest. Redirecting `HOME` hides `nltk_data`, which
reproduces the ordinary reader's machine — `nltk` imports and `build_detectors`
succeeds, and the missing data packages only surface when a sentence is
tokenised. An example that guards construction instead of detection passes on
a developer's machine and crashes on a fresh one. That is exactly what happened
on the lane's first run.

## 12. Structured artefacts: notebooks, modules, and the schema they carry

Added in round eleven. Everything before this section assumes the input is
prose. A data-science artefact is not prose, and the difference is not
cosmetic: it changes what is sensitive, where it lives, and what a stand-in is
allowed to be.

### D-12.1 — the measured gap (`CP-046`)

A realistic churn notebook and its feature module, through the round-ten
pipeline:

```
redacted 3 value(s): EMAIL=1, PHONE=1, SSN_US=1

  customer_ssn             still present: True
  acme                     still present: True
  marion.holt              still present: True    (from /home/marion.holt/...)
  internal_risk_score_v3   still present: True
  AC-NORTH                 still present: True
  /mnt/prod/exports/...    still present: True
```

and, on the module, `blind_spots: []`.

The last line is the finding. Three values were removed, and the report said
the installation was doing its job, on a file that discloses the organisation,
the analyst's name, the production data path, the fact that the dataset
contains social-security numbers, the proprietary feature set, and the
categorical levels of a segment. `CP-023` established that a redaction tool
which finds nothing must never look like a clean bill of health; this is the
same failure reached through a door nobody had opened, because the blind-spot
vocabulary has no entry for *"this is source code, and these identifiers are a
schema"*.

### D-12.2 — what is sensitive in an artefact, by stage

The five stages a user named, and what each leaks that prose redaction misses:

| Stage | Artefact | Leaks prose redaction misses |
|---|---|---|
| data analysis | exploratory notebook | rendered `head()` rows, column names, source path |
| data analytics | reporting notebook, SQL | table and column names, categorical levels, small counts |
| modeling | training script | feature names, target name, hyper-parameters tuned to a client |
| model analysis | evaluation notebook | feature-importance labels, class names, segment names |
| any stage | tracebacks, configs, logs | absolute paths, home directories, hostnames, buckets |

Three classes run through all five, and none is reachable from the text
pipeline as it stands:

**Identifiers, not values.** `customer_ssn` contains no social-security number.
It discloses that the dataset does, which is often the more sensitive fact.

**Structure, not content.** A file path names the organisation, the
environment, the quarter and the analyst. `/home/marion.holt/work/acme-churn/`
identifies a person and a project in nine tokens.

**Rendered data.** The code can be clean while the *output* underneath it holds
two hundred real rows. This is the leak people are least aware of, because they
are reading the code.

### D-12.3 — the design tension, and why a placeholder is the wrong stand-in here

The reason to send a notebook to a model is to get help with the code. That
help depends on the schema's *semantics*:

```
df['acct_balance_usd']  →  numeric, skewed, needs a log
df['region_code']       →  categorical, needs encoding
df['signup_date']       →  temporal, needs care about leakage
df['customer_ssn']      →  an identifier, must not be a feature at all
```

Replace all four with `[COLUMN-1..4]` and the model can only give generic
advice, because the information its advice depends on was the information that
was removed. The redaction succeeds and the task fails.

So the stand-in must be **role-preserving**: it keeps the semantic class and
discards the specific one.

```
customer_ssn            →  id_1
acct_balance_usd        →  amount_1
signup_date             →  date_1
region_code             →  category_1
internal_risk_score_v3  →  score_1
churned                 →  target
```

"Log-transform `amount_1`, one-hot `category_1`, and drop `id_1` — it is an
identifier, not a feature" is correct advice, transfers back verbatim on
decode, and discloses nothing. This is the surrogate idea of `CP-043` applied
to schema rather than to people, and it reuses the same machinery: a stand-in
bank, a grammar fingerprint, and a vault.

### D-12.4 — how a role is established, and what happens when it is not

Rule 4 of this project forbids heuristics in logic. A name is not evidence of a
dtype: `count` could be anything, and `date_added` could be a string. So role
assignment has three tiers, and the third is not a guess but an admission.

1. **Declared.** The user says so, on the command line or in
   `.cleanprompt.toml`. Always wins, never questioned.
2. **Observed.** The artefact contains evidence: a dtype in a `df.info()`
   output, a `dtype=` argument, a parsed value in a rendered table, a
   `parse_dates=` list. This is a fact read out of the document.
3. **Neutral.** Everything else becomes `field_N`, and the report *says* the
   role was not established.

Name-based classification exists, is **off by default**, and when enabled is
reported as inferred — the same contract `suggest_terms` already has, where a
candidate is offered for confirmation rather than acted upon.

The split that makes this sound: **hiding is deductive, role assignment is
evidential.** A discovered column name is hidden whether or not its role is
known. Only the *choice of stand-in* depends on evidence, and an unestablished
role costs the model some context — it never costs the user a leak.

### D-12.5 — how a column name is discovered

Syntax, not vocabulary. In Python source a column name occupies positions where
nothing else is a sensible reading, and an `ast` walk finds them
deterministically:

```python
df["X"]              df.loc[:, "X"]       df[["X", "Y"]]
columns=["X"]        usecols=["X"]        on="X"    by="X"
subset=["X"]         rename(columns={"X": "Y"})     parse_dates=["X"]
```

In a notebook the rendered output cross-confirms: the header row of a printed
DataFrame names its columns.

Attribute access (`df.X`) is deliberately **not** a discovery site, because it
cannot be told from a method call without knowing the object's type. It is
*rewritten* once `X` is known from a subscript, which is the asymmetry that
keeps discovery conservative and rewriting complete.

### D-12.6 — regions, roles, and why the file is edited in place

An artefact is parsed into **regions**: half-open intervals of the *raw file
text*, each tagged with a role — `code`, `markdown`, `output`, `traceback`,
`metadata`, `path`. Detectors are selected per role, so a base64 image is never
scanned and a code cell is never treated as prose.

Offsets index the original bytes, never a re-serialisation. A notebook that
round-trips through `json.loads`/`json.dumps` comes back with different
whitespace, reordered keys and lost cell ids, which makes the diff unreadable
and the artefact untrustworthy. The adapter therefore locates regions with a
position-tracking scanner and the engine performs its usual single pass over
the original text, which also preserves every invariant `I1`–`I9` unchanged:
regions are disjoint by construction, so span resolution is unaffected.

### D-12.7 — failure modes this layer must refuse rather than absorb

- A column name that is also a Python keyword, a builtin, or a one-letter name.
  Rewriting every `id`, `type`, `count` or `x` in a notebook breaks the code.
  Such names are **refused with a message**, not silently skipped.
- A column name that is a substring of another (`age` in `average`): `CP-001`
  on the schema side, resolved by the existing longest-wins arbitration plus
  identifier boundaries.
- A notebook whose JSON is malformed: refused, because a half-parsed notebook
  cannot be reassembled.
- An output cell holding megabytes of base64: never scanned, and **dropped by
  default with a report**, because a rendered figure can *show* the data that
  the redaction just removed from the table above it.
- A model reply that returns the notebook as fenced Markdown rather than JSON:
  restoration is literal substitution of stand-ins, so it works on whatever
  shape comes back.

## 13. Packs, formats and the fluent plan

Added in round twelve.

### D-13.1 — the measured root cause (`CP-048`)

Four record-style files that people actually paste into a chat, through the
round-eleven pipeline:

```
addressbook.csv  names, streets, cities, member ids sent; 2 phones caught
patient.json     {"mrn": "00412345", "dob": "1984-03-02", "name": ..., "diagnosis": "E11.9"}
                 -> nothing redacted at all
deploy.sh        export DB_PASSWORD=hunter2; API_TOKEN="sk_live_..." -> sent
app.ini          db_password = hunter2 -> sent
```

One cause explains all four. **Detection was keyed on the shape of a value,
and record-style data is identified by the name of its field.** `00412345` is
an eight-digit number, indistinguishable from any other; the key `mrn` is what
says it is a medical record number. Every structured format carries that name —
a CSV header, a JSON key, an environment variable, an INI key — and the pipeline
threw it away before looking.

The second cause is structural and is why the first could not be fixed by
adding one more regular expression. Every detection rule lived in one Python
module, so covering a new domain meant editing code, and a clinical vocabulary,
a secrets vocabulary and a pandas vocabulary had nowhere to live except next to
each other.

### D-13.2 — the vocabulary

**Pack.** A named, versioned bundle of detection rules for one domain:
`personal`, `addressbook`, `patient`, `finance`, `secrets`, `records`, `email`,
`cloud`, `pandas`, `numpy`, `sklearn`. A pack declares any of:

- `fields` — names whose *value* is sensitive in a structured record, with a
  kind and a schema role. This is the answer to `D-13.1`.
- `patterns` — value shapes, each carrying `examples_yes` and `examples_no`
  that are **executed when the pack loads**. A pattern that does not match its
  own positives, or matches its own negatives, is refused. A pack cannot ship
  an untested regular expression.
- `code` — the column-naming keyword arguments, methods and dtype spellings of
  a library, which extend `_code.py`'s discovery for notebooks and modules.
- `requires` — other packs it builds on.

**Format.** How a file is split into regions: extensions, a splitter
(`text`, `python`, `notebook`, `script`, `json`, `delimited`, `keyvalue`,
`corpus`), and whether it round-trips byte-exactly.

**Plan.** A frozen selection — packs, formats, policy, style, keep-set, roles —
with a fingerprint, built fluently and validated before it runs. Modes are
selections, not flags: `packs("all")`, `packs("addressbook")`,
`packs("pandas", "patient")`.

### D-13.3 — data in YAML, behaviour in named Python, never code in YAML

The user-facing ask was `_config/*.yaml` beside a `_pandas.py` for each domain.
The split kept is the one underneath that ask: **what** a domain is — names,
patterns, examples — is data and lives in YAML; **how** to check something a
regular expression cannot, such as a Luhn checksum or an IBAN mod-97, is
behaviour and lives in Python. A pack refers to behaviour by name
(`validate: luhn`), resolved against a fixed registry in `_hooks.py`.

One module per domain was not built, because measured against the packs it
would have held nothing: the pandas vocabulary is entirely data, and so is the
clinical one. A module that exists to be empty is a place for the next person
to put code that belongs in data. The rule that replaces it: a domain needs
Python only when it needs a *validator*, and a validator is a named function
in one registry, reviewable in one place.

That same rule is the security boundary. YAML is read with `safe_load`; no
field is ever evaluated, imported or used as a module path; a custom pack can
name a validator but cannot supply one.

### D-13.4 — YAML without a third-party import on the base tier

`import scikitplot.cleanprompt` loads no third-party package, and PyYAML is
third-party. The built-in packs are therefore authored as YAML and **shipped
compiled** to `_config/_compiled.json`, which the base tier reads with the
standard library. A test re-compiles the YAML and fails if the two differ,
naming the command that fixes it. No second YAML parser exists anywhere.

Custom packs may be YAML (which needs PyYAML, and says so) or JSON (which does
not). Both go through the same validator as the built-ins.

### D-13.5 — fields: values found by the name of their key

A structured splitter yields **field-bound regions**: an interval of the raw
file plus the name of the key it is the value of. A field detector hides the
whole value when the name matches a selected pack's `fields`, whatever the
value looks like. Matching is on a normalised name — lower-cased, camelCase
split, separators folded — so `DateOfBirth`, `date_of_birth` and
`date-of-birth` are one field.

Values are hidden **whole**. A field detector never tries to find the
sensitive part inside a value: the key already said the whole value is the
sensitive part.

The byte-exact rule of `D-12.6` holds for every text-native format: regions
index the raw file, and only the interval of a value is rewritten.

### D-13.6 — corpus: partially dependent, never entangled

`corpus` reads PDF (and images, audio, XML and more) with capability probing
already built. Reviewed rather than assumed: it has **no** reader for `.docx`,
`.xlsx` or `.pptx`. So Office files are read by `_office.py`, a
standard-library reader with size limits and DTD refusal, and `_corpus.py`
borrows corpus's readers — PDF today — **lazily, inside functions, in that one
module**, reporting a specific capability error when corpus is not importable
(`ABSENT`) or cannot be imported in this interpreter (`BROKEN`, measured on
3.8). In the other direction, `register_corpus_readers()` gives corpus the
Office reader it lacked, on request, never on import. Neither submodule
imports the other at module scope, and corpus never imports cleanprompt.

The contract rule `CP-INDEP-001` is amended to say exactly that, rather than
worked around: a sibling import is permitted only as `scikitplot.corpus`, only
in `_corpus.py`, and only inside a function.

Three corpus properties found in review shape the adapter:

- `PipelineHooks` **fails open** — an exception in a hook is logged and the
  document passes unchanged. A redaction step there would send the original
  text whenever it failed. The adapter never registers a hook; it redacts
  documents and **raises** on any failure.
- `CustomNormalizer` writes only `normalized_text`, while embedding falls back
  to `text` and several fields carry copies of the original. The adapter
  rewrites **every** text-bearing field and every string in `metadata`.
- `doc_id` and `content_hash` are digests of the original text and survive
  `replace()`. The adapter recomputes both from the redacted text, so no
  identifier derived from a removed value leaves with the document.

The output of a binary format is text, not a rebuilt `.docx`: extract, redact,
send. That is stated in the format table rather than implied by it.

### D-13.7 — invariants added

- `I10` A built-in pack's compiled form equals its YAML source.
- `I11` Every pattern in every loaded pack matches all of its `examples_yes`
  and none of its `examples_no`, at load time.
- `I12` A plan's fingerprint depends on its selection and nothing else, and the
  same plan redacts the same input identically across processes.
- `I13` A field value, once matched, is hidden whole or not at all.
- `I14` (added in round twenty-two, `D-23.1`) No file in the package holds, as
  stored, a whole value that a pattern of a credential pack accepts.

### D-13.8 — what implementation settled that design had left open

**JSON stays JSON.** A placeholder written over a JSON number or literal gives
a file no parser opens, and a model given invalid JSON "repairs" it — usually
by quoting the placeholder, which then restores as a string, changing a type.
So in JSON and JSON Lines every detector's span is first clipped to the scalar
token it lies in: inside a string it never crosses the quotes and is widened
so it never splits an escape sequence; on a number or literal it becomes the
whole token; over structure it is dropped, because JSON puts only punctuation
there. A value that lands on a number or literal is re-issued, in a second
pass over identical spans, as a **sentinel**: `-99` and eight digits, fixed
width so no sentinel is a prefix of another, checked against the document like
a surrogate, and *reserved* — a sentinel in the input is never re-hidden, which
keeps re-encoding idempotent (`I7`). The output is parsed before it is
returned, and a document that no longer parses is an error, never a result.

**Hide the value, not its label (`CP-051`).** The vault is keyed on the value.
`MRN: 00412345` and `00412345` are different values, so a labelled pattern
that replaced its label gave one patient two labels across two files. A pack
pattern may name a group `value`; only the group is replaced, and every
positive example must fill it (`I11` extended). The prose `Key: value` reader
leaves sentence punctuation outside the value for the same reason.

**`auto` protects by default.** Measured on the four files of `D-13.1`, format
defaults of `personal` and `records` still sent the MRN, the diagnosis and the
insurer id of `patient.json`. Field rules are precise — they fire on a key
that says what the value is — so the cost of a wider default is small and the
cost of a narrow one is the leak this round exists to close. Data formats now
default to `personal, records, patient, finance, secrets`; prose formats to
`personal, patient, finance, secrets`; configuration formats add `personal` to
`secrets, cloud`. Code formats do not get `personal`: `name = "model_v2"` is
not a person.

**One field, one meaning, checked.** Two packs may not give one field name two
kinds. The check runs whenever packs are combined, and a test asserts the
whole built-in catalogue combines cleanly — it caught `username` defined as
both `PERSON` and `RECORD_ID` during this round.

**The shared vault is the cleaner's, not the engine's (`I4`).** The engine
stays stateless; a `Cleaner` holds the entries issued so far and passes them to
each call as a `seed`. Positions are per document and are not kept. A literal
stand-in (a sentinel, a column stand-in, a surrogate) that already occurs in a
later file is **reported** in that file's `preexisting_labels`: the encoded
text is still safe to send, but decoding that file would also replace the
author's copy.

**Refuse, never guess.** A file that is not UTF-8, JSON that does not parse,
an archive member whose name is absolute or climbs out with `..`, an encrypted
member, a nested archive, a PDF inside an archive, a PDF with no text layer:
each is refused or skipped with a reason and **never written**. There is no
"read it as plain text instead" fallback anywhere.

**Folders.** Symbolic links are not followed, `.git`, `.hg`, `.svn`,
`__pycache__` and `.ipynb_checkpoints` are never entered, the target may not be
inside the source, and output archives are deterministic (sorted members, a
fixed timestamp) and written through a temporary name.

### D-13.9 — known limits

- A field rule reads a key. A name in running prose has no key and needs the
  entity tier or `--hide`.
- In prose, a `Key: value` line's value runs to the end of the line, so
  `MRN: 00412345 email x` hides the remainder under one label: more than
  necessary, never less.
- *(Closed in rounds 13–14.)* A value found in one file is propagated to
  every other by `remember`; folders, archives and file lists are read twice
  so the result does not depend on file order (`CP-063`).
- The core `EMAIL` pattern requires an ASCII boundary, so an address glued to a
  preceding accented letter is not found. Pre-existing; unchanged here.
- A custom pack's regular expression runs as written. Its examples are
  executed, which catches a wrong pattern but not a slow one; a catastrophically
  backtracking pattern in a user's own pack is that pack's cost.
- Two archive members that map to one output name (`a.docx` and `a.docx.txt`)
  cannot both be written; the second in name order is refused and reported.

## 14. A gate in front of any model

Added in round thirteen.

### D-14.1 — the question the whole submodule answers

"What text leaves this machine?" Every earlier round answered it for one input
at a time. An application, an agent or a person at a shell sends many texts
through many clients, and the answer has to hold across all of them. So the
unit this round adds is not a detector but a **gate**: one object every
outgoing text passes through and every reply comes back through.

### D-14.2 — the gate owns no network

`Guard` takes a callable. It imports no vendor SDK, reads no key and opens no
socket; `_bridge.py` moves bytes through the pipes of a command the user names,
with `shell=False`. That is why it works with every model — hosted, local,
agent framework, command-line client — and why the submodule stays
standard-library only. It is also a security property: the gate cannot be the
component that leaks, because it never holds a connection.

### D-14.3 — encode, then check with a different mechanism

Encoding decides spans with detectors and arbitration. The check that follows
(`Cleaner.leaks`) searches the finished text for the removed values
themselves, as whole tokens. Two mechanisms, so one defect cannot pass both;
and a finding raises `LeakError` before the caller's function is invoked. The
check covers exactly the kinds `remember` repeats, so with `remember` on it can
only fire on a defect, and with it off it fires on every recurrence no rule
caught — the fail-closed behaviour a user who turned memory off is choosing.

### D-14.4 — remember: a value hidden once is hidden everywhere

A name found by a field rule in a CSV has no pattern; the same name in a later
note would otherwise be sent. The cleaner's own history is the evidence, so it
is used: one literal detector per kind over the values hidden so far, bounded
by "no word character on either side", seeded so a recurrence gets the label
the value already has. Column, figure and output stand-ins are excluded — they
rename identifiers and blobs, and repeating a column called `age` across prose
would hide an English word. The output depends on what was encoded before, by
design; `remember(False)` makes each input independent again.

### D-14.5 — a prose value ends where its format says

`Key: value` lines in running text used to run to the end of the line, which
put `ann@example.com` inside an MRN (`CP-062`). A field now declares its prose
`span` in data — `line`, `clause` or `token` — chosen by the identifier's
format: `line` for anything that may contain a comma, `token` only for formats
with no whitespace. The default is `line`, which hides more and never less.

### D-14.6 — streaming is decoding, cut at a safe point

`StreamDecoder` emits decoded text up to the earliest point where a label could
still be forming — an unclosed opening delimiter, a tail that is a prefix of a
literal stand-in — and never inside a restoration candidate found in the
buffer. The property checked is the only one that matters: every chunking
decodes exactly like the whole reply (`CP-060`).

### D-14.7 — logging: scrub the namespace, audit the facts

A logger's filters see only that logger's records, never its children's, so a
scrubbing filter must sit on every logger in the namespace (`CP-059`). Each
`Session`, `Cleaner` and `Guard` owns a `VaultScrubber` that scrubs its values
from messages, `extra=` fields and tracebacks for exactly as long as it holds
them, released by `clear()` or a weak-reference finalizer. Audit events carry
counts, kinds, the plan fingerprint and a digest of the output: enough to prove
what left, nothing that was removed. File names are not logged; a file name can
be personal data.

### D-14.8 — known limits

- A name in running prose with no key is found only by the entity tier,
  `--hide`, or `remember` after it was hidden elsewhere.
- `remember` is case-sensitive unless the policy folds case.
- Scrubbing ignores surfaces under four characters (`SecretFilter.MIN_LENGTH`).
- A bracket label a model rewrites across more than 64 characters is not held
  back by the stream decoder; decoding the whole reply still restores it.

## 15. Agents without code, and plans a team can pin

Added in round fourteen.

### D-15.1 — MCP: what a tool returns is what the model reads

An MCP server's tool results go straight into the model's context. That one
fact decides the tool list. A `decode` tool would hand the model the values
the gate withholds, so there is none. Reading is `cleanprompt_read_file`: the
agent gets the file *through* cleanprompt, encoded, instead of reading it raw.
Writing is `cleanprompt_write_file`: the model's text is restored locally,
written to disk, and the tool answers with a relative path and counts. Paths
are shown relative to their root, because an absolute path usually carries a
home directory — a user name. The test that holds this calls every tool,
successes and failures, and asserts no result contains a value or the root.

### D-15.2 — a protocol implementation in the standard library

The server speaks JSON-RPC 2.0 over newline-delimited stdio and negotiates the
protocol version (2025-06-18, 2025-03-26, 2024-11-05; anything else is
answered with the newest). It needs no SDK, so it runs wherever the base tier
does, including Python 3.8 where the SDK does not. `McpServer.handle` is a
pure function of one message and the server's state, so conformance is tested
without a subprocess; `serve` only moves lines, keeps running after a
malformed one, and clears the vault on exit however it exits.

### D-15.3 — confinement

Tools read and write only under the `--root` folders, after resolving symbolic
links, and never replace an existing file unless asked. This is the same rule
as the tree walk — nothing is followed out — applied to a caller that is a
model, which is the least trusted caller this submodule has.

### D-15.4 — learning before writing

`remember` in one pass made a folder's protection depend on its file names
(`CP-063`). A folder, an archive or a file list is therefore encoded twice when
`remember` is on: a learning pass that fills the vault and discards the text,
then the writing pass. Labels are issued in the learning pass, in sorted order,
so output is still deterministic. A conversation cannot do this — it does not
have the future — and remembers only what came before, by construction.

### D-15.5 — plan files: approval that a run can verify

A plan's fingerprint covers its selection and the content of every pack and
format it resolves to (`I12`). Saved beside the plan, it turns "the team
approved this plan" into a check each run performs: an upgrade that changes a
pack, or an edit to a custom pack, makes every `--plan` run refuse until the
plan is saved again. A plan cannot be combined with ad-hoc selection flags; it
fixes every choice or none.


## 16. Scale without a second behaviour

Added in round fifteen. Each change here makes something larger work *the same
way* as the small case, rather than adding a mode that behaves differently.

### D-16.1 — record files in pieces, identical to one pass

A CSV, TSV or JSON Lines file above the document limit was refused; the only
advice was "split it", which puts the cut, the header and label continuity on
the user. The file is now encoded in pieces, and the design is the proof that
the pieces equal one pass:

- *cuts only between records* — `record_starts` comes from the same scan that
  locates cells, so a quoted newline is never a cut, and no field value spans
  a cut;
- *field names from the whole file* — regions are located once and shifted
  into each piece, because the header is only in the first;
- *one vault, in order* — each piece is seeded with every entry issued so far;
- *memory frozen per file* — the known-value detectors are taken once before
  the first piece, so a piece cannot know more than a single pass would.

A property test compares both over random cut sizes (1 character to larger
than the file); a record longer than the limit still meets the limit's own
error. The per-call bounds `max_spans` and `max_entries` apply per piece, so a
file one piece long that is dense enough to exceed them is refused with their
own actionable errors; `Cleaner(chunk_chars=...)` sets smaller pieces. Formats without independent records (JSON, prose) are not cut: no cut
point in them is safe by construction.

### D-16.2 — a limit bounds one call's work, not a history (`CP-064`)

`max_entries` counted the seed, so every conversation and every chunked file
had a lifetime budget. Only values first found in the current text count now.

### D-16.3 — one counted log filter whose cost follows the record (`CP-065`)

A filter per holder made each log record cost one scan per live session. The
shared filter counts holders per value — a value two sessions share stays
scrubbed until both let go — and is attached once. Adding and releasing are
O(1): nothing is copied or compiled per add. A compiled alternation was
measured and rejected (quadratic in sessions), and so was the first shared
version, which copied its value list per add and tested every held value
against every record: in a process holding 126 000 values, 2000 records took
10.9 s. Every held value is at least four characters, so values are indexed
by their first four with the lengths held under each prefix; a record is read
once, and at each position whose prefix is held one lookup per recorded
length decides membership exactly — 0.06 s for the same 2000 records. Below
256 held values a substring test each is cheaper (measured) and is used; a
property test holds both equal to brute force.

### D-16.4 — a dry run is the real walk with nowhere to write

`survey_tree` runs the same `_walk_tree` as `encode_tree`, including the
learning pass, on a scratch cleaner with the same plan, and writes nothing.
It is not a separate estimate, so its statuses and counts are exactly what
`batch` would produce — a test asserts that — and the caller's vault is
untouched. `batch --dry-run` and the MCP `cleanprompt_inspect` on a folder both
use it, and both total kinds with one function, `kind_totals`.

### D-16.5 — rejected: finding names by salutation (`CP-066`)

"Mr./Dr. + a capitalised word" matches "Dr. Pepper". Packs hold formats, not
guesses; names in prose come from `remember`, `hide=`, or the NER tier.

## 17. Many callers at once

Added in round sixteen.

### D-17.1 — a vault owner is a monitor (`CP-067`)

Encoding is read-seed, issue-labels, store. Correctness of every placeholder
depends on those three being one step: if two callers read the same seed, the
next free label goes to two values, and a reply restores the wrong one. That
is worse than a leak to the model — it writes one person's data into another
person's text, locally, where nothing checks it. So `Cleaner` and `Session`
each hold a re-entrant lock, and one decorator, `_synchronized`, puts every
encode, the learning pass and every vault read under it. Re-entrant, because
public methods call each other. Not on generators, because a lock held across
`yield` would block every other caller until the consumer finished; a tree
walk locks per file, and its learning pass whole, which is what keeps the
numbering deterministic.

`Guard.outgoing` holds the cleaner's lock across encode *and* check. Without
that, a value another thread learns between the two would turn a correct
encode into a refusal — safe, but a spurious one.

The lock costs nothing measurable: encoding is CPU-bound Python under the GIL,
so it was serial already; what the lock removes is only the interleaving.

### D-17.2 — async is the same gate, not a second one

`aask`, `achat` and `adecode_stream` await what the caller passes and share
every step with `ask`, `chat` and `decode_stream` — encoding, the check, reply
decoding — through the same helpers. No event-loop library is imported:
`asyncio` is only used by the caller, and the tests drive the methods with
`asyncio.run`. A plain function passed to `aask` is refused with an error that
names `ask`, rather than failing later on `await` of a string. Encoding stays
synchronous in the calling thread — it is fast for a prompt, and for a large
document `asyncio.to_thread(guard.outgoing, text)` is safe because of D-17.1.

### D-17.3 — deferred: MCP prompts

Still no client in use needs `prompts/list`; the tool list covers every
operation. Recorded so it is not re-proposed without a client that needs it.

## 18. One vault file, many processes

Added in round seventeen. Section 17 made each vault owner a monitor inside
one process; the vault *file* is shared between processes, and had the same
read-issue-store sequence with nothing around it.

### D-18.1 — read-to-write under a lock between processes (`CP-068`)

Every command that reads a vault and writes it back holds an exclusive lock on
`<vault>.lock` from the read to the write; `forget` holds it around the
deletion, so an in-flight `encode` cannot write forgotten values back. The lock
is the kernel's (`fcntl.flock`, `msvcrt.locking`), because a lock that exists
as a file can outlive the process that made it, and a kernel lock cannot. The
lock file is never deleted: removing it while another process waits would let
a third lock a new file, and two would run at once. The lock is taken after
reading standard input, never across it, so a run waiting for a person to type
cannot block every other run. Waiting is bounded (600 s, enough for a folder
`batch`) and announced.

### D-18.2 — replace, never truncate (`CP-069`)

A vault is written beside itself under a random name, `0600` from creation,
flushed and synced, then renamed over the old one. A reader sees the old
vault or the new one; a crash leaves the old one. Readers therefore need no
lock at all, which keeps `decode` fast and unblockable.

## 19. The same value, however it is written

Added in round eighteen.

### D-19.1 — equivalence, not similarity (`CP-070`)

`remember` found a held value by its exact spelling, and text rarely repeats a
value exactly: capitals in a heading, a name broken across a line, a
non-breaking space from a word processor, a typographic apostrophe. Each of
those was sent in the clear. The fix is a definition, in one module
(`_canonical`), of when two writings are the *same string*: any whitespace
run for any other, apostrophe and dash variants for each other, a Unicode
compatibility form that is one character for that character, and case. That
is an equivalence — deterministic, symmetric, testable — and deliberately not
similarity: reordering, initials and partial names are guesses, and a guess
in either direction is a defect (a miss leaks; a false match hides text the
user needed).

The canonical form never changes length, so offsets found in it are the
original's. Each writing therefore becomes its own span with its own exact
surface, and its own label: the model sees `[PERSON-1]` and `[PERSON-3]` for
one person, and the reply restores each exactly as written. Merging them
would keep the identity and lose exact restoration (I1); privacy and
exactness both win this way, at the price of the model not knowing the two
are one person.

### D-19.2 — a stand-in is held to the same rule (`CP-071`)

A surrogate is a fixed-pool string, so it can coincide with a real value the
conversation holds. The engine now refuses any candidate that contains a held
value under the same equivalence, which is also exactly what the leak check
would refuse afterwards — the two can no longer disagree.

## 20. The way back uses the same equivalence

Added in round nineteen.

### D-20.1 — restore what the model rewrote, and say so (`CP-072`)

Section 19 made "the same value" an equivalence on the way out. The way back
had the same gap for surrogate stand-ins: a model re-cases headings and
reflows lines, and a stand-in written `DEAR MARION HOLT` was left in the reply
unrestored. With bracket placeholders an unrestored label is visible and is
reported; with a surrogate it is an invented name that reads as real, so the
miss was silent and misleading at once.

Restoration now adds, beside the exact matches, matches of every literal key
under the section-19 equivalence, and reports each as a repair. Three rules
keep it exact:

- *added, not substituted* — every exact match that restored before still
  does, and the longer candidate still wins;
- *unambiguous only* — keys that share a normal form cannot be told apart once
  rewritten, so neither is restored from a rewritten form;
- *bounded* — at most 16 whitespace characters between words, so a streamed
  reply knows how far back a stand-in can begin, and the stream decoder holds
  back normal-form prefixes up to that reach. Streaming still equals the whole.

### D-20.2 — build per key set, not per call

A stream asks for candidates on every chunk and a vault only grows, so the
matchers are built once per key set (`lru_cache` over the sorted keys, which
are labels and stand-ins, never values). Streaming a 30 000-character surrogate
reply in 5-character chunks went from 3.44 s (v18) to 0.53 s.

## 21. A tool call is an exit

Added in round twenty.

### D-21.1 — least privilege for tool arguments (`CP-073`)

The gate had one direction covered — nothing held reaches the model — and
assumed the other was safe because decoded text stays local. A tool call is
the exception: the model chooses its arguments, the application runs it with
decoded values, and the tool may send them anywhere. Prompt injection turns
that into exfiltration without the model ever seeing a value: it only has to
write the placeholder where the attacker wants the value.

So restoring a tool call's arguments is an authorisation decision, made per
tool and stated in code: `decode_tool_arguments(args, allow={...})`. `allow` is
required, because the safe default depends on the tool and cannot be guessed.
A call that names a kind the tool was not given is refused before anything is
restored, since a tool running on a half-restored call is rarely what anyone
wanted; `on_withheld="keep"` is there for callers who have decided otherwise.
The check and the decoding use one restoration, walked over one set of
strings, so they cannot disagree. `chat()` no longer decodes tool calls at all:
the one path that restored them implicitly was the one to close.

### D-21.2 — negative examples must assert the whole outcome (`CP-074`)

A pattern's declared negatives were checked with "no match, or the match is
not the whole example" — true for `</document>` even while `/document` was
being hidden. The new tests assert the text that leaves, which is the property
that matters.

## 22. A tool result is a document

Added in round twenty-one.

### D-22.1 — guard what will be sent, in the form it will be sent (`CP-076`, `CP-077`)

A tool result reaches the model as JSON text. Guarding it string by string
guarded something other than what was sent: keys and numbers were never read.
`encode_object` now serialises once and guards that document with the `json`
format — the same reader a `.json` file gets — so keys are tokens, a number
under a sensitive field gets a numeric stand-in, and the independent check
sees the whole document. The result is what `json.loads` reads back, which is
also exactly what the model will see.

### D-22.2 — a JSON string is text (`CP-078`)

Moving tool results onto the `json` format exposed a gap in that format: it
found fields by key only, so prose inside a string value — a note, a log line,
a ticket — was never read for `Key: value` fields. Each string value is now
read by the text format's prose reader over its *decoded* content, and the
regions are mapped back through the escapes. Deciding in decoded text is what
keeps a line-span value from running past an escaped newline; mapping at
character boundaries is what keeps a placeholder from splitting an escape.

## 23. What is committed is read by scanners

Added in round twenty-two.

### D-23.1 — an example of a credential is not stored whole (`CP-079`, `I14`)

A positive example for a key pattern is, by construction, a string that has
the shape of a key. `secrets.yaml` held eight of them, the compiler copied
them into `_compiled.json`, and a push carrying both files was refused by the
host's secret scanner. The scanner was right: it cannot know that a value is
an example, and neither can the next one.

Three ways out were considered. *Marking the files as ignored* fixes one
scanner at one host and nothing at the package index or in a downstream audit.
*Dropping the positive examples* gives up `I11` for exactly the patterns where
a silent mismatch is a leak. *Generating examples from the expression* needs a
regular-expression inverter, which is a larger and less certain thing than the
problem.

So the example stays, and stays tested, but is written in pieces: an entry of
`examples_yes` or `examples_no` may be a list of two or more fragments, which
`_packs._example_list` joins before `I11` runs. The compiler copies documents
as written, so the fragments are what `_compiled.json` stores; the whole value
exists only in memory. A one-fragment list and an empty fragment are refused,
because neither splits anything.

`I14` makes this a property instead of a habit. `_catalog.at_rest_findings`
runs the patterns of the packs named in `AT_REST_PACKS` over text as stored.
`compile_config` applies it to every YAML definition and to the JSON it is
about to write, and raises before writing; `check_compiled` applies it to the
stored JSON, for a file edited by hand; an architecture test applies it to
every text file under the package, tests and documentation included. The
definition of "credential-shaped" is the package's own, so a pattern added to
a credential pack is enforced on the files from the commit that adds it. A
finding names the file, the line and the kind, and never the value.

What `I14` does not claim: that a scanner will accept the tree. It claims the
tree is clean by this package's patterns. A scanner with a pattern the
`secrets` pack lacks can still object, and the answer is then to add that
pattern to the pack — which is also what a user of the pack would want.

### D-23.2 — a refusal reads the bytes the parser will read (`CP-080`)

The Office reader refuses any part containing a DTD before parsing it, which
is what makes the standard-library parser safe to use on untrusted files. The
refusal searched raw bytes for `<!DOCTYPE`. ECMA-376 allows a part to be
UTF-16, where that markup is stored with a NUL beside each character; the
search saw nothing and the parser expanded the entity. The search now runs on
the part with NUL bytes removed. Every other encoding the parser accepts keeps
XML's markup characters at their ASCII values, so two spellings is all there
are.

### D-23.3 — a linter's fix is a change, and is tested as one (`CP-081` to `CP-083`)

A lint pass after round twenty-one replaced the standard-library XML parser
with `defusedxml`, moved two `typing_extensions` imports back to module scope,
rewrote the corpus bridge's import as `from scikitplot import corpus`, and
left `Session.encode` calling a `Vault.values` that does not exist. Each
looked like an improvement in isolation and each broke something the suite
already asserted: the base tier's standard-library-only import, the bridge
rule, and every session turn. Nothing new was needed to catch them — only
running the suite after the pass. The comments at each site now say why the
code is written the way a linter dislikes, and the suppressions are standard
`noqa` codes.

### D-23.4 — a declared section has content (`CP-084`)

`fields: []` was read as "no fields". A key that is present and empty is an
unfinished pack; it is now a problem, reported with the others.


## 24. Every surface says what it does

Added in round twenty-five. The governing rule of the round, taken from the
internal review of 2026-10-10: do not add capability until the current
contract is internally truthful. Six of the nine findings are places where one
surface described one thing and the run did another.

### D-24.1 — installed is not ready, and one function decides (`CP-093`, `CP-100`)

An entity engine is *ready* when three conditions hold, checked in the order
their remedies must be applied: it can read the language; its package is
installed and inside the declared range; its data is present and loadable.
`_engines.engine_readiness` is the one decision. `auto` selects only a ready
engine. `build_detectors(required=True)` refuses any engine it would run that
is not ready, before any text is read, naming each engine's status, reason and
remedy. `doctor`, `inspect`, `encode`, `redact`, `encode()`, the file runtime
and the web app all build through that function, so the diagnosis and the run
are the same decision and cannot disagree.

spaCy's data is checked without importing anything: the resolved model as a
distribution, as an importable top-level package (`find_spec` locates without
executing), or as a directory. NLTK's data cannot be located without NLTK's
own search path, and copying that logic would be a guess about its internals,
so the NLTK check imports NLTK. It therefore runs only under `check_assets`,
which the callers set where NLTK is about to be loaded anyway; elsewhere the
report says `assets_checked: false` instead of pretending. Construction
without `required` still imports nothing (`test_construction_imports_nothing`).

`CP-100` is why the NLTK check is a *run*, not a lookup. NLTK 3.9 moved the
tagger and chunker to new data packages; the path lookup accepted either name
of a group, so data present only under the old names was reported ready and
failed at the first sentence. `_nltk.missing_data` runs the path lookup and
then, only if nothing is absent, the detector's own tagger and chunker once on
a fixed sentence — each step independently, so both failures are named at
once. `LookupError` is the only exception read as missing data. Success caches
the very machinery the detector uses (the `CP-027` chunker), so the check is
free the first time and absent afterwards; failure is not cached, so data
downloaded mid-process is believed. The remedy names every package that can
satisfy a group, current and older, which is right on every release in the
declared range rather than the one it was written against.

### D-24.2 — generated files are derived, never written beside the code (`CP-095`, `CP-096`)

The container files were hand-written text next to the code they deploy, and
drifted twice: the image installed `en_core_web_lg` after the default moved to
`sm` (the `CP-025` drift, generated), and the `docker run` line published on
every interface while the compose file beside it used loopback. The model is
now `resolve_model(language, model_size)` and is passed back explicitly
(`--ner-engine spacy --ner-model <model>`), so a future default change cannot
separate what is installed from what is requested. Every launch line is
`127.0.0.1:PORT:PORT`. A setting nothing reads (`CLEANPROMPT_NER`) is removed,
not documented: a dead setting that looks load-bearing is a false statement.

### D-24.3 — no acknowledgement unlocks code execution (`CP-097`)

`--allow-remote` and `--docker` acknowledge that the page is reachable. Flask's
debug mode serves a debugger that executes code typed into the browser; that
is a different exposure and no flag here acknowledges it. `resolve_bind`
refuses `debug` on any non-loopback host, container mode included.

### D-24.4 — a second reading, never a rewrite (`CP-098`)

Values written with an invisible format character inside them, in full-width
letters, or with a Unicode space or dash inside a telephone number read the
same to a person and to a model, and went out in the clear. The obvious fix —
normalise the text before detection — breaks `I6` and exact restoration at
once: detectors would see rewritten text, and the vault would hold a value the
user never wrote.

`_canonical.detection_view` is a second *reading*: Unicode `Cf` characters
removed and every non-ASCII character folded by the same one-to-one table
`canonical` already owns for "the same value" (ASCII untouched, so line breaks
stay line breaks and code stays code). The only offset change is a deletion,
so the view carries the deletion positions and maps a view index back with one
`bisect` — logarithmic, so a text salted with a zero-width character between
every letter costs no more per span. `Redactor._view_spans` runs structural
and literal detectors over the view, maps every span onto the original, and
adds it to the original-text spans before overlap resolution. Entity engines
never read the view: they are statistical models of natural text. Spans are
only added, and the resolver merges overlaps, so the view can widen redaction
and cannot narrow it. When the view equals the text — always for ASCII — there
is no second pass.

What it does not do, recorded so it is not over-claimed: fold look-alike
letters from other scripts (Unicode has no normalisation form for them; see
the ledger note), or rejoin a value split by ordinary spaces.

### D-24.5 — a documented command line is a claim that is tested (`CP-099`, `CP-101`)

A remedy string and a documentation example are both instructions, and both
drifted: the spaCy repair hint carried an old range (`CP-099`), and a module
docstring showed `encode --pack-file`, which never existed (`CP-101`). Hints
are now computed from the tier declaration; every `cleanprompt <command>
--option` line in the README, docstrings, skills, guide and gallery is checked
against the command table by `_maintenance/tests/test_documented_cli.py`.

### D-24.6 — the reviewer says whether its record is current

`review_subsystem.py` reported the recorded `PASS` while the checker failed on
a stale evidence fingerprint. It now reports `evidence_current` with both
fingerprints and exits 3 when the record describes another tree.

### D-24.7 — continuity is a file and a test

A round's state lives in `_maintenance/RESUME.md`: step log with evidence,
last verified numbers, next action, the open ledger and pending decisions.
`tests/test_resume.py` fails when its ledger and
`upcoming_changes/scikitplot/cleanprompt/` disagree, so a fresh session cannot
miss a note or redo a closed one.

### D-24.8 — only pure detectors read the view (`CP-102`, `CP-103`)

The view's first form ran every non-entity detector over it, and the round's
independent review found the flaw before delivery: field, region and
JSON-token detectors carry offsets computed from the original document, so
their spans were mapped twice and cut through newlines and separators. The
rule is now a declared contract, not an inference from a detector's kind:
`Detector.reads_view` defaults to `False`, and only detectors whose every
offset indexes the string passed to `detect` — `RegexDetector`,
`LiteralDetector`, `PackPatternDetector` — set it. A third-party detector
does not read the view unless it says it may.

Testing that contract found `CP-103`, older than the round: field names were
normalised as written, so one invisible character in a header hid a whole
column. `normalise_field` now reads names through the view. Splitters that
tokenise names with an identifier class first (`.env`, shell, code) and
token-bound JSON values are the recorded remainder; the proposed fix computes
regions on the view and maps their boundaries back before any detector runs.


## 25. The user decides, and the floor does not move

Added in round twenty-six. Two capabilities open the subsystem further — a
check on custom patterns and custom surrogate names — and both follow one
rule: the user chooses the policy, at whatever level fits (one pattern, one
run, a team, a machine), and the safety rules underneath cannot be changed by
any of those choices.

### D-25.1 — a static check, by public means (`CP-104`)

Custom patterns are checked for shapes that backtrack catastrophically when
they load (`_pattern_risk.py`). `sre_parse` / `re._parser` would give a parse
tree for free, but they are private, moved in 3.11, and warn on import there,
which the suite turns into an error. The source is therefore read by a small
parser that understands repetition (escapes, classes, every group kind,
alternation, every quantifier form). Whether two pieces can match the same
character is decided by Python's public `re`: each piece is compiled alone and
tried against a probe alphabet — every character the pattern mentions, its
neighbours, and fixed representatives.

Before parsing, the source is read the way Python runs it: a verbose
pattern (`VERBOSE` flag or a leading `(?x)`) loses its whitespace and
comments, inline `i` and `s` flags join the flags overlap is decided with
(wherever they appear — they can only widen matching, so applying a scoped
one to the whole pattern errs towards a report), and escaped characters
(`\x41`, `\u0430`, `\N{...}`) join the probe alphabet with their case
variants. A pattern that does not compile, or uses a scoped `(?x:...)` or a
conditional group, is `not-analysed`.

"Repeated" means more than three *choices* (`+`, `*`, `{1,40}`; not `{4}` or
`{1,3}`). Possessive quantifiers and atomic groups give nothing back and are
never the inner part of a finding. Three rules:

* `nested-quantifier` — inside a repeated group, an element that can vary
  in length (any quantifier with a range, an optional part, a group with
  branches or such parts), when, walking round from it — the rest of the
  alternative, the start of the next repetition, itself again — an element
  that can start with a character it consumes is met before an element
  that cannot be skipped. "Skipped" means *nullable*: a group whose parts
  are all optional separates nothing, whatever its count (the soundness
  fuzz, `probe_round26_fuzz.py`, found three misses of the first fix this
  way: `(?:\w?\.?)*`, `(?:(?:\d?){1,9} {2,})+`, `(?:\w{1,5}\s*(?:-?){1,9})+`). Elements are taken by position: `(?:\w+,\w+)+` holds two equal
  `\w+`, and an equality lookup found the first one and its separator.
  `(\.\w+)+` and `(?:(?:\w+)-)+` are linear; `(\w+,?)+` is not.
* `overlapping-alternation` — two branches of a repeated, non-atomic group
  that can start alike.
* `adjacent-quantifiers` — two repeated single units side by side that can
  trade characters.

Runs separated only by an optional element (`\s*:?\s*`) are *not*
reported: at most quadratic, everywhere in label patterns, and every span is
already bounded by the policy. `probe_round26.py` checks every verdict
against run time measured in a killable child, including the independent
review's cases (verbose, inline flags, escapes, equal units, bounded outer
repetition; separators one group up, fixed counts, possessive forms), and
the built-in patterns report nothing — a warning is always about something
a user wrote.

### D-25.2 — warn by default; the user tunes, at four levels (`CP-104`)

The maintainer's decision: always warn, with concrete rewrites and the ways
out on screen; never refuse by default. The mode comes from, highest first: a
per-pattern acceptance in the pack (`risk: accepted` with a non-empty
`risk_reason` — both or neither, and `accepted` is the only value, so a typo
cannot silence anything); then `--pattern-risk` or a plan's `pattern_risk`
(a plan fixes every choice, so the two are never combined); then
`CLEANPROMPT_PATTERN_RISK`; then `warn`. `refuse` is a plan validation
problem, so it stops before any text is read. The cleaner warns once per open
finding; validating and fingerprinting stay silent. Warnings are
`PatternRiskWarning` (`UserWarning`) at a fixed library line, so Python's
default filter shows each once per process, `-W error::` makes them fatal,
and the command line prints them as one `warning:` block on standard error
without touching any filter. `pattern_risk` is left out of `as_dict` while
unset, so every saved plan keeps its fingerprint.

### D-25.3 — custom names, by data, under a fixed floor (slice A)

`GENERATOR_DESIGN.md` is the full design. In short: a surrogate set
supplies name lists for `PERSON`, `ORG`, `GPE`, `LOC`, `FAC`; `EMAIL`,
`PHONE`, `URL` stay in the core's reserved forms and every other kind keeps
placeholders. Entries are whitelisted by character class, refused when they
hold a default-ignorable code point, more than two combining marks in a row,
a drawn-differently compatibility form (full-width, circled) or two
scripts, entries are stored in NFC, — so no two entries can look alike — and
checked against the default core patterns and every built-in pack pattern;
the combined two-part name is checked again when it is issued. A set only
*proposes*; the core loop keeps uniqueness, absence from the source, absence
of held values and the bounded search. One limit cannot be enforced
without guessing at someone's language: a stand-in that is also an ordinary
word is restored wherever the reply uses that word. The identity
`name@version#digest16` is derived from the validated content and enters the
grammar fingerprint only when a set is present, so default digests are
unchanged (pinned in a test) and `decode` needs only the identity. A grammar
that names a set it was not given refuses to encode rather than issue
built-in names the identity does not describe.

### D-25.4 — one value, whatever its separators (`CP-105`)

A stand-in must not show a held value however it is written (`CP-071`). The
held-value pattern joins a value's words with spaces, so `Marion Holt` was not
found in `marion.holt@example.invalid` and the real name reached the model.
Both sides are now word-split — separator runs and lower-to-upper case
changes become spaces — so `Marion Holt`, `marion.holt`, `Marion-Holt` and
`MarionHolt` are one value whichever is held and whichever is proposed (the
first version folded only the candidate; the independent review found the
reverse direction). Adding forms keeps the check monotone: it can reject
more stand-ins and never accept one the plain check rejects.

### D-25.5 — one vault, one grammar (round 26 review)

Appending to a vault with another style or surrogate set mixed two kinds of
stand-in and re-stamped the vault with the new grammar's fingerprint, so
earlier text could not be decoded as written. The command line now compares
the vault's recorded grammar with the run's before seeding and refuses,
naming the recorded style and set. A vault that records no grammar is not
checked.
