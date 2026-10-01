---
name: cleanprompt-maintainer
description: Maintain scikitplot.cleanprompt, the prompt redaction subsystem. Use for the detect/resolve/assign/rewrite pipeline, the placeholder grammar and restoration, the structural pattern library and its validators, the two named-entity engines and their canonical label vocabulary, the Vault secrecy contract, the encode/decode API and its session surface, lazy optional tiers (ner, nltk, web, crypto) and capability truth, the CLI and local web interface, and release verification.
---

# `scikitplot.cleanprompt` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/` and
`skills/`.

The runtime is a package, not a module:

```text
scikitplot/cleanprompt/
├── __init__.py        public facade: base-tier __all__, PEP 562 __getattr__, non-resolving __dir__
├── _engine.py         the pipeline — the only place a string is rebuilt
├── _policy.py         TagStyle, Limits, RedactionPolicy, the two fingerprints
├── _patterns.py       curated pattern library with intents, validators, examples
├── _detectors.py      Detector protocol, RegexDetector, LiteralDetector, registry
├── _types.py          frozen Span, Entry, Stats, results
├── _vault.py          the secret half of a result
├── _capabilities.py   CapabilityStatus (7 states) and tier probes
├── _exceptions.py     the typed error tree
├── _engines.py        engine selection and the canonical entity vocabulary
├── _languages.py      language -> spaCy model, with an announced fallback
├── _surrogates.py     consistent invented stand-ins for the `surrogate` tag style
├── _logging.py        logging that never records a removed value
├── _api.py            encode / decode / Session — the LLM-facing surface
├── _vaultcrypt.py     vault encryption from hashlib/hmac/secrets — no tier
├── _ner.py            optional spaCy detector      (tier: ner)
├── _nltk.py           optional NLTK detector       (tier: nltk)
├── _app.py            optional Flask app           (tier: web)
├── _render.py         presentation, kept out of the data
├── _cli.py / __main__.py
└── tests/             test_<module>.py per source module, plus test_regressions.py
```

## Read first

1. `maintenances/cleanprompt/_maintenance/DESIGN.md` — the invariants and the
   reasoning; everything else assumes it
2. `maintenances/cleanprompt/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/cleanprompt/REVIEW.json`
4. `maintenances/cleanprompt/_maintenance/VERIFICATION.md`
5. `maintenances/cleanprompt/_maintenance/STATE.json`

Then run:

```sh
python -B maintenances/cleanprompt/_maintenance/check_trackers.py --json
python -B maintenances/cleanprompt/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/cleanprompt/_maintenance/tests -q -p no:cacheprovider
python -B -m pytest scikitplot/cleanprompt/tests -q -p no:cacheprovider
```

## This is a privacy tool, so failure is not symmetric

A redaction that silently does less than it claims is worse than one that
raises, because the caller will transmit the result. Every stage therefore
fails loudly rather than degrading. When you are weighing a change, the
question is not "does this still work?" but "can this now return successfully
while having redacted less?".

Five of the closed findings are exactly that failure: `CP-002` (extra terms
disabled structural detection), `CP-015` (a rejected candidate hid shorter
ones), `CP-016` (a truncated match stranded a value's tail), `CP-024` (an
explicit `--ner` that could not be met exited 0 having run no engine at all)
and `CP-028` (an IP address at the end of a sentence was never detected). All
five returned a plausible-looking redacted string.

`CP-024` is the one worth keeping in mind when you add an option, because it
arrived from the opposite direction to the others: nothing was mis-detected,
the default degradation rule was simply applied to an explicit request. A
default may fall back quietly; a request may not. `build_detectors(required=True)`
is where that distinction is enforced, and it raises `CapabilityError` naming
every engine's own status and remedy rather than one summary line.

## Never break the import contract

`import scikitplot.cleanprompt` imports **no third-party package**. This is what
lets the submodule sit in `scikitplot` without changing any other submodule's
import cost or failure surface, and it is verified under an `__import__` blocker
in `_maintenance/evidence/probe_isolation.py`.

Rules:

- `__all__` holds **base-tier names only**. `__all__` *is* the star-import
  surface, and `from … import *` resolves every entry; adding an optional name
  there defeats the lazy tier at the one operation that touches all of them.
- Optional names resolve in `__getattr__`, which calls `require(tier)` **before**
  importing the owning module. Putting the capability check after the import
  makes the actionable message unreachable, because the dependency's own
  `ModuleNotFoundError` fires first.
- `__dir__` lists optional names as strings and never resolves them.
- A missing attribute raises `AttributeError` without importing anything, so
  `hasattr`, Sphinx and IDE introspection stay free.

## Detectors never see rewritten text

Composition is over **spans**, not strings. If you find yourself wanting to run
one stage over another's output, you are re-creating `CP-006`: upstream fed the
named-entity stage the already-tagged text, which turned
`"Contact [EMAIL-1] now"` into `"Contact [[ORG-1]-1] now"` and made `EMAIL-1`
permanently unrestorable.

`_engine.py` is the only module that rebuilds a string. Keep it that way.

## Adding a pattern

Every entry in `_patterns.py` carries a stated intent, positive examples and
negative examples, and the test suite executes all of them. Both reproduced
upstream pattern defects were typos that read as intentional — `[$-_@.&+]` is a
character *range*, and `[A-Z|a-z]` admits a literal pipe — which is why an
unexplained expression is not acceptable here.

Prefer a small pure validator over a cleverer expression: a card is sixteen
digits *and* passes Luhn; an IBAN is a shape *and* passes mod-97.

**If your pattern needs a validator, read the `RegexDetector` docstring first.**
Validated patterns are not scanned with `finditer`, and the reason is
load-bearing:

- `finditer` consumes a match before the validator can reject it, so a rejected
  candidate swallows every shorter alternative inside it (`CP-015`);
- limiting a retry with `endpos` makes the engine treat that offset as
  end-of-string, so a trailing `\b` or `(?!\w)` succeeds at a boundary the real
  text does not have (`CP-016`).

The scanner therefore looks for the longest **accepted** match at each start and
re-verifies every window boundary with one further character visible. That probe
is exact for single-character trailing assertions; a pattern needing wider
trailing context must widen the probe with it.

A pattern that recognises a credential carries its positive examples as
fragments — `examples_yes: [['sk_live_', '0123…']]` — which the loader joins
before testing. Invariant `I14`: no file under the package holds, as stored, a
whole value a `secrets`-pack pattern accepts, because committed files are read
by secret scanners that cannot tell an example from a leak (`CP-079`).
`packs --compile` refuses a whole one; a test that needs such a value builds
it at run time.

Your examples are also run in prose, not only in isolation. A class-level test
puts every `examples_yes` through twelve ordinary sentence positions, and it
exists because testing each pattern alone never puts punctuation next to a
value. That blind spot hid two defects pointing in opposite directions:
`CP-028`, where `IPV4`'s guard `(?![\w.])` rejected `192.168.1.10.` at the end
of a sentence so the address went to the model in the clear, and `CP-029`,
where a URL pulled the sentence's full stop into the placeholder and stored a
non-URL in the vault. Write a guard that says what you mean — for `IPV4` a
trailing dot now only disqualifies a match when a digit follows it — and let
the class-level test check the rest.

`CP-029` left one deliberate cost in place: a URL genuinely ending in `)`,
such as a bracketed citation, has that final `)` left in the clear. That
discloses nothing, which is the direction to err in.

## Overlaps merge, they do not drop

When detections overlap only partially, the cluster is merged into one span over
its full extent. Discarding the loser would leave the characters it alone
covered in the output, which is a disclosure. Merging is coarser — two values
can become one placeholder — but it cannot leak, and it round-trips exactly.

Do not "improve" this by trimming spans. A trimmed email is not an email, and
the trimmed remainder is a leak.

## Two entity engines, one vocabulary

`_engines.py` owns engine selection (`ENGINE_MODES` is `auto`, `spacy`, `nltk`,
`both`, `none`, defaulting to `auto`) and `CANONICAL_LABELS`, the vocabulary
both engines are normalised into by `canonical_label` before a span leaves the
detector.

That normalisation is not tidiness. spaCy says `ORG` where NLTK says
`ORGANIZATION`, and `LOC` where NLTK says `LOCATION`, so raw labels would make
the placeholder depend on which engine happened to be installed: the same value
becomes `[ORG-1]` on one machine and `[ORGANIZATION-1]` on another. A vault
records the label it issued, so a vault written under one engine would not
restore under the other — the placeholders in the text would no longer match
anything. Any new engine normalises at the boundary or it breaks that.

`_languages.py` answers the other half: 24 languages in `LANGUAGES`, sizes
`sm`/`md`/`lg`/`trf`, and `resolve_model(language, size, explicit)` returning
the model **and a note**. A language with no model of its own resolves to
`xx_ent_wiki_sm`, which recognises only the coarse categories, and the
substitution is announced by every caller. A user who asked for Turkish and
silently got a lower-resolution multilingual model would read the thinner
result as "there was nothing to find", which is `CP-018` again.

`_ner.py` now defaults to `en_core_web_sm`, not `lg`. If you write a remedy
string, compute it through `resolve_model` rather than hard-coding a model
name: `CP-025` shipped an instruction to download `en_core_web_lg`, a 560 MB
download that still left the tier misconfigured because the default had moved.

That same finding is worth stating as a rule. `diagnose()` identified entity
detectors by the name prefix `"ner:"`, which is spaCy's naming scheme, so an
NLTK registry that was actively finding names was reported as blind. The test
is now `detector.kind == "NE"`, which both engines declare. A false alarm is
not the harmless direction: the banner exists so that a real gap is believed,
and one that cries wolf whenever the lighter engine is used teaches people to
dismiss it.

Being installed is not the same as being importable. A half-finished upgrade, a
wheel built for another interpreter or a shadowing local file all satisfy
`require()` from distribution metadata and then fail at import. That used to
escape as a generic detector crash naming this submodule; it is now
`CapabilityError(status="BROKEN")` with a `--force-reinstall` hint (`CP-026`).
`BROKEN` existed in the vocabulary for exactly this state, and nothing had been
mapped onto it.

## Offsets are carried, never recovered

`_nltk.py` is the lighter engine: English only, and a chunker of 1990s vintage,
so expect lower recall than spaCy. Its difficult part is not entities but
positions. `nltk.ne_chunk` returns tokens with no character offsets, and this
pipeline is built entirely on spans over the original string.

Searching for the entity text to recover an offset is wrong in the ordinary
case, never mind the adversarial one: a name appearing twice collapses onto its
first occurrence, and the tokenizer rewrites some characters, so the token text
is not always a substring of the source. The offsets are therefore carried
through `PunktSentenceTokenizer.span_tokenize` and
`TreebankWordTokenizer.span_tokenize`. The guarantee is that
`text[span.start:span.end]` is the entity exactly as it appears in the source,
and a verification lane checks it for every engine.

One chunker is built per process and cached. `nltk.ne_chunk` constructs a fresh
model on every call, and this detector called it once per sentence, at 460 ms a
sentence (`CP-027`). Measured after the fix: 50 documents 23.58s -> 0.09s, a
12-sentence paragraph ~5.6s -> 0.015s. That was not only a speed problem — the
lighter engine exists for the machine that cannot take spaCy, so being slower
than the heavy one inverts the reason to choose it, and in the web interface a
multi-second redaction reads as a hang and invites a second submission of the
same unredacted text.

## Nothing that was removed may be logged

`_logging.py` installs a `NullHandler` at import under `LOGGER_NAME`
`"scikitplot.cleanprompt"`, so nothing is emitted until `configure_logging`
(idempotent) is called by `--log-level` / `--log-format` or by a caller.

Log records carry counts, kinds, labels, offsets, durations and capability
decisions — what happened, never what was found. A redaction tool that logs the
values it removed has defeated itself: the data leaves through the log instead
of the prompt, and in a more durable form, because log lines are shipped off
the machine, indexed and retained long after the prompt is gone.

The rule is enforced twice. It holds by construction at every call site, and
`SecretFilter` scrubs records as defence in depth. Keep both. If you add a log
statement, pass the label, never the surface.

## The vault is structurally separate, and stays that way

`RedactionResult.text` is safe to transmit; `RedactionResult.vault` is not. They
are different types so that no serializer can emit both by accident.
`_types.as_dict` is the only sanctioned serializer and deliberately omits
`Entry.original`; `dataclasses.asdict` would emit it, which is asserted in
`test__types.py`.

`Vault.__repr__` reports counts, never values. Keep it that way: reprs reach
tracebacks, debuggers, notebook cells and log aggregators.

## The API surface is a pair; a session is not

`_api.py` is what a caller sending prompts to a model actually touches.
`encode(text, ...)` returns an `EncodedPrompt` that is iterable, so
`safe, handle = encode(...)` works, and `decode(reply, handle, strict=False)`
puts the values back. `Handle` is frozen, has a secret-free `repr`, and
`export()`/`load()` make it JSON-portable.

`Session` (and the `session(**options)` context manager) keeps placeholders
consistent across turns and clears itself on exit; it offers `encode`,
`decode`, `decode_report`, `roundtrip(text, send)`, `handle`, `turns`,
`report` and `clear`. The integration seam is `send`: any callable taking a
string and returning a string, which is why no provider needs to be known here.

`Session.encode` returns a plain `str`, deliberately unlike the module-level
`encode`. The session owns the one handle; handing back a per-turn copy would
invite decoding turn 3 with turn 1's handle, which silently restores the wrong
values. Do not "fix" the asymmetry.

## The web tier's rules

No module-level app, cleaner or key — `create_app` is a factory. The vault stays
server-side in a bounded, expiring `SessionStore` keyed by an opaque token; the
cookie carries the token and nothing else. The secret key is configuration, and
an unset key refuses to start unless `ephemeral_secret_key=True` is passed
explicitly. Every form carries a CSRF token compared with
`hmac.compare_digest`. Nothing evaluates input.

Upstream put the encrypted mapping in the cookie and read it back with `eval`
(`CP-007`), and generated both keys at import (`CP-009`).

## Evidence discipline

Maintenance PASS, runtime PASS and release PASS are three separate facts, and
the checker reports them separately for that reason. A static checker cannot
prove behaviour; mark unavailable lanes `UNAVAILABLE` and never convert "not
run" into green.

One lane is `UNAVAILABLE` and says why:

- **Python/platform matrix** — 3.8 compatibility is targeted (no
  `dataclass(slots=)`, no `kw_only`, no runtime PEP 604 unions) but not
  verified: one interpreter, Linux only. "Avoids" is not "verified", and the
  lane says so rather than implying otherwise.

The entity-engines lane was `UNAVAILABLE` for two rounds and is now `PASS`:
spaCy 3.8.16 with `en_core_web_sm` and NLTK 3.10.3 with its corpora, both
measured over 120 fuzzed documents per engine mode. `tests/test__ner.py` still
uses a stub, and the stub still proves only this module's span translation —
never record it as that lane. Holding the lane open paid for itself: running it
for real is what surfaced `CP-027`, because the probe timed out where it should
have taken seconds.

## Repair workflow

1. reproduce the smallest failure, and save the reproduction;
2. decide which module owns the contract — see `_maintenance/FAMILY.md`;
3. change the focused test to state the **desired** behaviour first;
4. make one runtime change;
5. run the focused suite, then the scale probe in
   `_maintenance/evidence/probe_negative.py` — the two `CP-015`/`CP-016` leaks
   were found by the probe and not by the unit suite, so the probe is not
   optional;
6. run the import-isolation probe;
7. rerun the maintenance mutation tests;
8. refresh `REVIEW.json`, `STATE.json` and `EVIDENCE.json` from executable
   evidence only.

Never use `--update` to bless a structurally red runtime; the checker refuses.

## The CLI has two frontends and one command table

`_spec.py` holds a framework-neutral `Param`/`Command` IR; `_frontends.py`
renders it through argparse and through click. Neither frontend owns any
behaviour — both parse and hand the *same* `argparse.Namespace` to the same
handler, so there is nowhere for a difference to hide.

The IR expresses only what both can render faithfully. `CP-021` is what happens
when that slips: argparse rendered a repeatable option as `nargs="+"` and
accepted `--hide A B`, click rendered it as repeatable and rejected it, and the
same command line worked or failed depending on which library was installed.
When adding a parameter, check both frontends can express it, and add a case to
`test__frontends.TestParity`.

Round three added `--ner-engine`, `--lang`, `--model-size`, `--log-level` and
`--log-format` through that IR, and `doctor` now reports `entity_engines`,
`nltk_corpora` and `languages`: which engines are usable and which one `auto`
would pick, which NLTK corpora are present, and which models are installed for
which language.

## The option grammar is part of that contract

What the IR promises is not only the option *set* but the *grammar* — how a
token on the command line is read. Round five probed that grammar through both
frontends after a user asked whether `--`, the POSIX end-of-options delimiter,
worked. It already did, identically. Two other things did not.

`CP-032`: argparse accepts any unambiguous prefix of a long option by default,
so `--form` reached `--format`; click accepts none. The same command line
therefore succeeded on a machine without click and exited 2 on a machine with
it, which is `CP-021` in a new place. An abbreviation is a latent break on a
single machine too: it is unambiguous only until a new option shares its prefix,
so a script that worked for a year fails on an upgrade that added a feature it
never used. The fix is `allow_abbrev=False` on the root parser **and on every
subparser** — subparsers do not inherit it, and the subparsers are where every
declared option actually lives. argparse conforms to click, the same direction
as `CP-021`, because the strict behaviour is the one that can be relied on.

`CP-033`: argparse treats any token containing a space as a positional, whatever
it begins with, so `inspect "--secret is a@b.co"` was accepted as text and
quietly processed, while click rejected it as an unknown option. The direction
matters more than the divergence. A user who mistyped an option name would have
had the flag itself treated as their prompt, with the option they meant never
applied and nothing on screen saying so, and a tool that decides what leaves the
machine must not silently do something other than what was asked.
`_reject_stray_options` in `_frontends.py` now refuses a dash-leading positional
unless a `--` delimiter authorised it, and the message names the delimiter. It
is a post-parse check rather than an override because argparse decides this
inside `_parse_optional`, which is private and has moved between releases, and
the declared dependency range has to keep working. Nothing is checked past a
`--` — that is exactly what the delimiter means — and a lone `-` stays valid,
since it conventionally means standard input.

Two smaller pieces belong to the same grammar. `_Parser`, a thin
`argparse.ArgumentParser` subclass, rewrites one message: argparse's "expected
one argument", which fires on `--hide -secret`, now also names the two forms
that work — `--hide=-your-value`, and `--` when the dash-leading token is text.
It appends rather than replaces, so an unanticipated argparse message still
reaches the user intact. And both frontends render short aliases from the one
`Param` declaration: `-h/--help`, `-V/--version`, `-i/--in`, `-o/--out`,
`-f/--format`, `-k/--kinds`, `-q/--quiet`.

One split is documented rather than closed. The portable spelling for a
dash-leading option *value* is the attached form, `--hide=-secret`, which both
frontends accept; the separated `--hide -secret` is accepted by click and
refused by argparse. Eliminating that would mean overriding the argparse
internals named above, so the README states the attached form as the spelling to
use instead of pretending the difference is gone.

If you add an option, assert in **both** frontends that its abbreviation is
refused and that `--` shields a dash-leading value, in
`test__frontends.TestParity`. Sixteen option-grammar cases are compared across
the two frontends today; an option added without them is how the next `CP-021`
arrives.

## Every text command takes text three ways

`redact`, `restore`, `inspect`, `scan` and `roundtrip` each carry a variadic
`TEXT` positional (`Param(dest="text_args", kind="argument", multiple=True)`)
alongside `--in PATH` and standard input. Until round four only the last two
existed, and a user who ran `cleanprompt inspect --ner` followed by a pasted
paragraph hit three errors in a row: `bash: syntax error near unexpected token
'('` from their shell, then `Error: No such option '-M'`, then `Error: Got
unexpected extra argument` — ours, naming the person's own text as the problem
and saying nothing about what to do instead (`CP-030`).

The positional is variadic because a shell splits `Ada Lovelace` into two words
before the program sees it; rejecting the second word would be a distinction the
user never made.

All five handlers go through one helper, `_resolve_text(args, stdin, stderr,
command)`, in the order positional, `--in`, stdin. Do not resolve text anywhere
else, and do not let a handler pick a winner when both a positional and `--in`
are given — that raises `PolicyError`, because the input being ignored is
exactly the text the person cared about.

When a command falls through to standard input and `stdin.isatty()`, the helper
writes to **stderr** a note that it is reading standard input, that Ctrl-D on a
blank line ends the paste, and the three other ways in. Stderr, never stdout:
stdout carries the result in a pipe. The `isatty()` call is guarded, so a
`StringIO` in a test and a pipe in a script print nothing. Before this, a text
command run bare at a terminal blocked on stdin with no output at all, which
reads as a hang.

The first of those three errors is the shell's and no program can fix it — a
shell consumes `(` before this code runs. The documented answer is a heredoc
with a **quoted** delimiter, `<<'END'`, which disables every substitution so
arbitrary prose arrives intact. That is a workaround, not a cure; do not write
as though the program had solved it.

## `roundtrip` exists because the restoration half was invisible

`redact` and `restore` are separate commands with a vault file between them,
which is right for real work — the reply may arrive days later, in another
process — but it meant that seeing values come *back* took two commands, a file
path, and an understanding of what a vault is. Every README example used `--in
prompt.txt` (`CP-031`).

`roundtrip` (alias `demo`) shows five stages in one command with nothing written
to disk: your text, what gets sent, what was removed as a table with values
hidden unless `--reveal`, a reply, and the restored text with a `round trip
exact: yes/no` line. `--format json` emits the same five stages with
`round_trip_exact` and `unresolved_placeholders`.

Stage 4 is a fixed stand-in string, labelled "NOT from a model" on every run.
Keep that label wherever you touch this command: presenting generated text as a
model's answer would be a lie told by a tool whose whole purpose is trust.
`--reply TEXT` and `--reply-in PATH` substitute a real answer.

`load_runner()` is separate from `select_frontend()` on purpose: the first
answers "which will work", the second "which is wanted". They differ when click
is present but unimportable, which took the whole CLI down — including
`--help` — until the fallback existed (`CP-022`). This is the same
`BROKEN`-is-not-`ABSENT` distinction the capability vocabulary draws.

## The default vault path is a security decision, not a convenience

Until round six `--vault` was required on both `redact` and `restore`, so the
shortest useful command carried a path the user had to invent and then keep
identical across runs. `default_vault_path()` now resolves `CLEANPROMPT_VAULT`
if set, otherwise the platform state directory:
`$XDG_STATE_HOME/cleanprompt/vault.json`, falling back to
`~/.local/state/cleanprompt/`, and `%LOCALAPPDATA%\cleanprompt\` on Windows
(`CP-034`).

The obvious default — the working directory — is the one place it must not go.
A vault holds the removed values in clear text, and this tool is used inside
checkouts: the reported session ran from `/work/.git_clones/learn/docs` on a
checked-out branch. A vault written there is one `git add .` from committing
the exact values the user was redacting so as not to send them anywhere. If you
ever find yourself adding a "vault next to the input file" convenience, that is
the failure you are reintroducing.

The directory is created `0700` and the file `0600`, and both modes are
requested **at creation** rather than applied afterwards. A chmod after the
fact leaves a window in which the file exists with the process umask's
permissions, which on a shared machine is long enough. A one-line stderr note
names the resolved path on every write — stderr, because stdout carries the
redacted text into a pipe — and `doctor` reports it as
`configuration.vault_path` and `vault_exists`.

## A vault can be continued, and that needs an index

`CP-035`: each run rewrote the vault, so numbering restarted every turn. The
same address could be `[EMAIL-1]` in one turn and `[EMAIL-2]` in another, and a
reply quoting an earlier placeholder restored to the wrong value or to nothing
at all — restoration returning someone else's data is the worst outcome this
subsystem has.

`--vault-mode overwrite|append` is the fix. Append seeds the redactor from the
existing vault before detection, then merges the result, so a value keeps the
label it already had and a new value is numbered after the old ones. It seeds
through the engine's `seed` argument, which has existed since round one for the
`Session` API; the CLI had simply never used it. Prefer wiring a CLI path onto
the existing engine argument over adding a second mechanism — two ways to seed
numbering would drift.

The vault document gained an `index` array of `{label, kind, ordinal}` and the
format moved from 1 to 2; the build reads both. The index carries no secrets:
a label and its category are already present in the text that was sent, and the
values stay in `entries`, which is what encryption covers. Do not be tempted to
put a surface in the index to make merging easier.

**Appending onto a format-1 vault is refused, not guessed.** An old vault has
no index, so merging would mean renumbering from `[EMAIL-1]` while the old
vault already uses that label for a different value — one placeholder standing
for two values, and restoration then hands the user someone else's data.
Refusing costs a user one `--vault-mode overwrite`; guessing costs them a
disclosure. Format-1 vaults still restore normally, and that must stay true.

## `encode` exists because a combination of flags is not an affordance

`CP-036`. Nothing was missing: `redact --vault v.json 2>/dev/null` already
produced exactly "just the clean prompt". Nobody found it. The reported session
shows a user reaching for `inspect`, whose entire purpose is the report, and
then asking how to get the clean prompt with no additional information around
it. An affordance reachable only as a combination of three flags and a shell
redirection is not an affordance, and in this subsystem the cost of not finding
it is that the raw text gets pasted into the model instead.

That command puts the redacted text alone on stdout. The vault goes to the
default location in **append** mode, so placeholders stay stable across a
conversation and the restoring half needs no arguments. Stderr carries one line
naming the vault, plus a warning when a high-severity detection gap means the
text has not actually been checked for names — that warning is the `CP-018`
rule applying here, and it must not be moved to stdout to make the output
tidier. `-q` silences both.

It defaults to append and `redact` defaults to overwrite. That is not an
inconsistency to tidy up: they are used for different things, and `redact`'s
scripted, file-based workflow keeps behaving as it always did.

It was added in round six under the name `clean`; round seven made `encode` its
canonical name, keeping `clean` and `prompt` as aliases. The reasoning is in
the next section.

## `encode` and `decode` are a pair, so they are the canonical names

`CP-037`. The two names already promised a symmetry the commands did not have.
`encode` was an alias of `redact`, which writes a vault to a named path and
prints a table of what it removed, and `decode` was an alias of `restore`,
which prints the restored text and little else. So two names that read as the
two halves of one operation behaved nothing like one: one verbose and
file-oriented, the other minimal. The user's own words were "encode to clean,
decode that AI chat answer decoded" — the symmetry the names were already
claiming.

The fix was to move the names onto the commands that mean them rather than to
add a third command. `encode` is the canonical name of the minimal half — text
in, a pasteable prompt on stdout alone, the default vault in append mode — with
`clean` and `prompt` as aliases. `decode` is the canonical name of the other
half, with `restore` as its alias, and its summary now says what it is for:
putting the values back into the model's answer. `redact` keeps its explicit,
scripted behaviour under its own name and lost only the alias, so nothing that
used `redact` changed.

Each half's help text names the other, and the module docstring leads with the
pair rather than with a list of ten subcommands. That is `CP-036`'s lesson
applied to naming: a reader who finds one half and never learns the other half
exists does the restoration by hand, or not at all. If you add or move an
alias, check that the pair still reads as a pair from the help alone — a name
describing an operation the command does not perform is worse than no alias,
because the user believes it.

The whole loop needs no paths at all:

```sh
cleanprompt encode <<'END'
…
END
cleanprompt decode <<'END'
…
END
```

## A vault that fills by default obliges the command that empties it

`CP-038`, and unlike the rest of the register it was introduced here rather
than inherited. `CP-034` gave this command a vault at a fixed default path and
`CP-035` made it append. Both are right, and together they are what makes a
conversation work — and together they mean the removed values accumulate in one
clear-text file, indefinitely, with no command to clear it. A tool that begins
collecting personal data by default and offers no way to stop is worse than one
that never collected it, which is an uncomfortable thing for two improvements
to have produced.

`forget` (alias `clear-vault`) is the answer, and three things about it are
load-bearing:

- It reports and stops; `--force` is what deletes. Deleting the vault is the
  one action here that cannot be undone from inside the tool, because `decode`
  can no longer restore that conversation afterwards — which is the point, and
  also the cost.
- The report names how many values and of which categories, never a value.
  Someone deciding whether to delete should not have the data put back on their
  screen by the act of looking at it.
- It deletes a vault it cannot parse. The summary read is wrapped in `except
  (ValueError, OSError, CleanPromptError)` — a truncated write leaves invalid
  JSON, a permission or device problem raises `OSError`, a malformed document
  raises ours — and only the summary is lost. Refusing to delete an unreadable
  file of secrets would block exactly the user who most wants it gone.

It also says plainly that unlinking is not shredding. On a journalling or
copy-on-write filesystem, on an SSD with wear levelling, or where a backup ran
in between, the bytes may outlive the unlink, so the message points at
`--encrypt` with a key the user holds for values that must not be recoverable.
Do not upgrade that wording into a promise the filesystem does not make.

The general rule, and the one to apply to the next feature: if a change makes
the tool retain personal data by default, the command that deletes it ships in
the same change.

## Advice has to be followable from where it is given

`CP-039`. That deletion note named `--encrypt`, and `encode` had no such flag.
`encode` is the command that writes the default vault in append mode, so its
users are the ones who accumulate the most removed values, and the single piece
of advice the tool offers about that accumulation pointed at an option their
command did not have. Advice that cannot be acted on from where it is given is
worse than none, because it reads as reassurance instead of an instruction.

`--encrypt` and `--cipher` are now shared `Param` objects declared once and
attached to both `encode` and `redact`, rendered from that one declaration by
both frontends. If you write a message that names an option, check that the
command printing it can accept it.

## Encryption is base tier, and the construction is written down

`CP-040`. `--encrypt` required the `crypto` tier: the `cryptography`
distribution, with compiled extensions. So the one security feature here was
the only part of the base workflow that needed an installed package, and a
vault written with it could not be opened on a machine without it. For a
submodule whose central promise is that the base tier is pure standard library
and works anywhere, that inverted the guarantee exactly where it mattered most.

`_vaultcrypt.py` is base tier and builds an authenticated construction from
`hashlib`, `hmac` and `secrets` alone. Its four steps, stated so they can be
reviewed:

1. **Key derivation.** `hashlib.scrypt` with a random 16-byte salt, `n=2**14`,
   `r=8`, `p=1`, 64 bytes out; `hashlib.pbkdf2_hmac` with SHA-256 and 240,000
   rounds where the interpreter cannot run scrypt. Presence of the attribute is
   not availability — scrypt needs OpenSSL 1.1 — so `_scrypt_is_usable()` runs
   it once with throwaway parameters and returns a decision.
2. **Key separation.** Those 64 bytes split into a 32-byte encryption key and a
   32-byte MAC key. One secret never serves two purposes.
3. **Encryption.** HMAC-SHA256 in counter mode — `HMAC(k_enc, nonce ||
   counter)` — XORed with the plaintext, with a fresh 16-byte nonce per value
   from `secrets`, so no keystream is reused.
4. **Authentication.** Encrypt-then-MAC: HMAC-SHA256 over the cipher name,
   **the label**, the nonce and the ciphertext, verified with
   `hmac.compare_digest` before any byte is decrypted. The label is inside the
   tag because without it someone with write access to the file could swap two
   tokens and make `[EMAIL-1]` restore to a different person's address, with
   every check still passing and no key needed.

The vault document records `cipher` — the versioned construction name
`cleanprompt-hmac-v1`, not the `--cipher` word the user typed — and the full
`kdf` parameters, so decryption dispatches rather than guesses, and a future
construction can arrive without stranding the vaults written today.

**Be straight about what this is.** It is a standard composition, not a
reviewed implementation of AES. Fernet is the better primitive where it can be
had. `_vaultcrypt.py` sets the construction out step by step so it can be
checked rather than taken on trust, and the tests assert the properties it must
deliver — a wrong passphrase, a bit flipped anywhere in the token, a swapped
label, a truncated token, nonce freshness — rather than claiming to prove the
construction secure, which no test suite can. Do not write anything about this
module that goes further than that.

`--cipher portable|fernet|auto` defaults to `portable`, because a vault that
cannot be opened is a worse outcome than a difference between two authenticated
constructions that both rest on standard assumptions. `fernet` keeps the
reviewed AES backend and `auto` prefers it when the tier is present. The
`crypto` tier's declared purpose narrowed with it, from "authenticated vault
encryption" to "the Fernet vault cipher (`--cipher fernet`); encryption itself
needs no tier" — a much smaller claim than that row used to make, and the only
honest one now that the base tier can encrypt.

The key comes from `CLEANPROMPT_VAULT_KEY`, else a `getpass` prompt at a
terminal. The prompt is not a convenience: without it the only way to encrypt
is an environment variable, and on most shells that means a plain-text copy of
the key sitting in the shell history next to the vault it protects.
`doctor --new-key` emits a standard-library passphrase — four groups of five
characters from a 30-symbol alphabet with look-alikes removed, just under 98
bits — because a key generator that needs an optional package is no use to the
person being told to encrypt.

One process note from this round is worth keeping. The first draft of the
scrypt probe swallowed its failure in `except Exception: pass`, and the
project's own architecture test `test_no_bare_except_or_silent_pass` and the
contract checker rule `CP-SAFE-002` both caught it before it reached review. It
was restructured into `_scrypt_is_usable()`, which returns a decision rather
than hiding one. The rules on this plane are not ceremony: a silent `pass` in a
capability probe is how a tier reports itself absent for a reason nobody can
name afterwards.

## The hop to the model is not a lossless channel

The pipeline is `detect → assign → rewrite → [language model] → restore`. Every
invariant this submodule proves holds across the parts it controls: exact round
trips, spans that index the original text, no leakage, determinism. Restoration
was the exception, because the design treated the hop between redaction and
restoration as a pipe that returns what was put into it. It is a language
model, and a language model rewrites tokens.

Round nine's three findings are that one assumption failing in three different
places, and the three answers are deliberately complementary — prevention,
mitigation, detection. That is the same defence-in-depth shape `_logging.py`
already uses, where the rule holds by construction at every call site and
`SecretFilter` scrubs the records anyway.

**`CP-041` (high) — what becomes a vault value is checked, whoever produced
it.** On `Mustafa Kemal Atatürk[e] (c. 1881)` spaCy returns the person as
`'Mustafa Kemal Atatürk[e'`: it swallows the opening bracket and the footnote
letter and leaves the closing one behind. The visible symptom was a malformed
`[PERSON-1]]` in the prompt — untidy, and *more likely to be rewritten by the
model*, which is `CP-042`'s damage. The silent symptom was the worse one: the
vault recorded the person's name **as** `Mustafa Kemal Atatürk[e`, so restoring
into any other text would have produced that string as somebody's name.

The root cause is an asymmetry. Every structural pattern carries a validator —
Luhn for a card, mod-97 for an IBAN, octet ranges for an address — while an
entity span, which arrives from a third-party model, was taken verbatim with
nothing checked at all. `trim_entity_span()` in `_engines.py` closes that, and
both engines apply it: a span is truncated at its first unmatched opener and
started after its last unmatched closer. Balanced brackets inside a span are
left alone, so `Acme (Europe) Ltd` is untouched. Only brackets are trimmed — a
trailing full stop is legitimate in `Inc.`, and deciding otherwise would be a
guess this submodule does not make.

**`CP-042` (critical) — a placeholder the model rewrote was not restored, and
barely reported.** Measured against realistic replies, `[EMAIL-1]` comes back
lower-cased, with an underscore or a space for the hyphen, with a Unicode dash,
with the brackets escaped for Markdown, and wrapped across a line. Exact
matching restored the first spelling and missed every other one. The report
then said `restored 0 placeholder(s); 2 vault entr(y/ies) unused` with exit
status 0, and `-q` silenced even that — so a user could paste a half-restored
answer onward without noticing.

The fix has three parts, and they are complementary rather than redundant.
`TagStyle.lenient_pattern()` and `TagStyle.normalize()` recognise that bounded
set of rewrites and nothing wider: the category must start with a letter, so
`[1]` is never a candidate. `restore()` acts on a lenient match **only** when it
resolves to a label the vault actually holds, which is what keeps prose like
`[note 2]` untouched and stops it being reported as an unknown placeholder.
Every repair is listed in the new `RestorationResult.repaired` field and named
in the note, and a restoration that resolved nothing while the vault is
non-empty now says so explicitly instead of reporting a quiet zero. `--exact`
on the CLI, `lenient=False` in the API, restores the previous behaviour.

If you widen the lenient pattern, widen it against measured replies and keep
the vault-membership condition. Leniency without that condition turns every
bracketed word in a model's answer into a restoration candidate, which is how a
tool that puts values back starts putting them where they do not belong.

**`CP-043` (medium) — the stand-in itself invited the rewriting.**
`[PERSON-1] emailed [EMAIL-1] about [ORG-1]` is not a sentence. The tokens
carry no grammatical number and no animacy, they break the sentence, and they
invite the model to comment on the redaction rather than answer the question.
They are also exactly the sort of token a model normalises, which is what
produces `CP-042`'s damage in the first place.

`--style surrogate` substitutes consistent invented values for the kinds where
a proper noun is the natural replacement — `PERSON`, `ORG`, `GPE`, `LOC`,
`FAC`, `EMAIL`, `PHONE`, `URL` — so the text reads as prose and there is
nothing to normalise. Credentials keep their placeholders: `CREDIT_CARD`,
`IBAN`, `SSN_US`, `AWS_ACCESS_KEY`, `JWT`, `PRIVATE_KEY`, `MAC`, `IPV4`, `IPV6`
and `NORP`. A plausible-looking card number or access key can be mistaken for
real by a person, acted on by a system, or by chance *be* real; and `NORP` is
adjectival, where an invented demonym reads as nonsense. Where a surrogate is
generated it uses a reserved form if one exists — `example.invalid` (RFC 2606)
for addresses and links, the `+1 555 0100`–`0199` fiction block for telephone
numbers. Stand-ins are checked against both the source text and the ones
already issued, because a collision on either side restores two values as one.

Reversal reuses `LiteralDetector` and a **single merged pass** over the reply:
grammar matches and literal surrogate matches are collected over the *original*
text and the rewrite happens once. A second pass over the first pass's output
would let a restored value be re-matched as another key, which is `CP-006`
reappearing on the restoration side. If you add another stand-in mechanism,
collect it into the same pass rather than chaining another one.

Two details are easy to undo by accident.

The `style` field lives on `TagStyle`, so it is part of the **grammar
fingerprint** and a vault written in one style cannot be read as the other —
which is right, because half-restoring is worse than refusing. But
`TagStyle._fingerprint_payload()` deliberately omits a *default-valued* style,
so a placeholder-style grammar keeps the digest it always had and every vault
written before the field existed stays readable. Adding the field naively would
have made every existing vault unreadable.

And the honest limit for `CP-043`: a surrogate is ordinary text, so it gives up
the one property a bracket label has for free — being obviously not part of the
document. `placeholder` remains the default for that reason, and it should.

## Registration in the project-wide CLI

`scikitplot/_cli/registry.py` carries a delegated `CommandSpec` pointing at
`scikitplot.cleanprompt.__main__:main`, so `scikitplot cleanprompt <sub>` and
`python -m scikitplot.cleanprompt <sub>` produce identical output.

It deliberately declares **no** `capabilities=` gate, unlike `mcp`. The base
tier is pure standard library, so the command is always runnable; gating it on
an optional extra would make the whole command unavailable because a capability
nobody asked for is missing.

## Never let "nothing found" look like "nothing to find"

This is `CP-018`, and it is the defect that shaped the diagnosis surface. A user
pasted an encyclopaedia paragraph into the web interface and got it back
unchanged. The engine was right — no structural PII, and the `ner` tier absent —
but nothing said so, and a user reads silence as safety.

So every outcome goes through `_diagnostics.describe_outcome`, which reports the
result *together with* what was looking and what was not. An empty result plus a
high-severity blind spot is an **alert**, in all three surfaces. If you add an
output path, route it through there rather than printing a summary directly.

`suggest_terms` is the complement: high-recall candidates offered for a person
to choose. It never redacts. Guessing that a capitalised word is a name is a
heuristic and has no place in the pipeline; showing the candidates and letting
someone decide is inspection, not guessing.

## Packs and formats are data; checks are named code

Round twelve moved domain knowledge out of Python. A **pack**
(`_config/packs/*.yaml`) says what to hide — field names, patterns with
executed examples, code vocabulary; a **format** (`_config/formats/*.yaml`)
says how a file is read and which packs `auto` gives it. The base tier reads
the compiled `_config/_compiled.json` with the standard library.

- Edit YAML, then `python -m scikitplot.cleanprompt packs --compile`; the suite
  fails on drift (`I10`), and `packs --check` says so in one line.
- A pattern with a label to anchor it hides only `(?P<value>...)`, so one value
  keeps one label across files (`CP-051`).
- A new check is a function in `_hooks.VALIDATORS`. A pack never carries code,
  and a custom file is validated exactly like a built-in.
- JSON output must parse: spans are clipped to scalar tokens and hidden numbers
  become reserved sentinels (`-99` + eight digits). Do not add a path that
  writes a bracket label over a JSON number.
- A new round-trip format needs a sample in `tests/test__runtime.SAMPLES`.
- `_corpus.py` is the only file that may import `scikitplot.corpus`, inside a
  function; corpus never imports cleanprompt.
- Measure the base-tier claim, do not list it: `test_importing_the_package_
  loads_only_the_standard_library` is what caught `typing_extensions` (`CP-054`)
  after eleven rounds of a block list that did not.

## The gate: encode, check, then call

`Guard` (`_guard.py`) is the one path from user data to any model. Keep three
properties: the independent `leaks()` check runs after encoding and raises
`LeakError` before a caller's function is invoked; the gate never owns a
network connection (`_bridge.py` runs a user's command without a shell and
kills it on early exit); and `StreamDecoder` output equals whole-reply
decoding for every chunking. A value hidden once is hidden everywhere
(`remember`), except column, figure and output stand-ins.

## Logging: the namespace, not the root

A logger's filters never see its children's records (`CP-059`). Scrubbing is
attached to every logger in `scikitplot.cleanprompt.*`; always obtain loggers
through `get_logger`. Emit facts with `audit()` — counts, kinds, fingerprints,
digests — never a value and never a file name.

## MCP: whatever a tool returns, the model reads

`_mcp.py` is a standard-library MCP server. Never add a tool whose result
contains a removed value — there is deliberately no decode tool; reading goes
through the gate and writing restores on disk. Show paths relative to a root.
Keep `McpServer.handle` free of I/O so conformance stays testable.

## Plans are pinned by fingerprint

`save_plan`/`load_plan` and `--plan` refuse a plan whose definitions changed.
Any change to what a pack means must change the fingerprint (it does: it
hashes the definitions), so a team re-approves instead of drifting.

## Scale: larger inputs, the same answers

A record file above the document limit is encoded in pieces cut at
`record_starts`; the output must equal one pass (`TestChunkedRecords`,
`probe_scale.py`). Per-call limits bound one call, never a history (`CP-064`).
The shared log filter's add and release are O(1) and its cost per record
follows the record's length (`CP-065`) — measure it in a process that already
holds values, as `probe_scale.py` does. `survey_tree` (behind `batch --dry-run`
and MCP folder inspect) is the real walk with nowhere to write; totals come
from `kind_totals`.

## Many callers: every vault owner is a monitor

`Cleaner` and `Session` hold a re-entrant lock; `_synchronized` puts every
encode, the learning pass and every vault read under it, and `Guard.outgoing`
encodes and checks as one step (`CP-067`: threads sharing a guard gave two
values one label, and a reply restored the wrong person's email). Never lock a
generator. Async methods (`aask`, `achat`, `adecode_stream`) reuse the sync
helpers.

## One vault file, many processes

A command that reads a vault and writes it back holds `_vault_lock` (a kernel
lock on `<vault>.lock`) from the read to the write, and `forget` holds it too
(`CP-068`). The vault is replaced with `_files.atomic_write`, never truncated
(`CP-069`). Take the lock after reading stdin; never delete the lock file.

## The same value, however written

`_canonical` is the one definition of "the same value": canonical text keeps
length (whitespace, apostrophes, dashes, one-character NFKC forms) and
`value_pattern` matches held values in it case-insensitively. `remember`, the
leak check and surrogate choice all use it (`CP-070`, `CP-071`). Equivalences
only — never reorderings or partial names.

## The way back uses the same equivalence

Restoring literal stand-ins adds bounded (`RESTORE_MAX_GAP`), unambiguous
`value_pattern` matches beside the exact ones and reports each as a repair
(`CP-072`); the stream decoder's hold-back follows the same reach. Matchers are
cached per key set — keys only, never values.

## A tool call is an exit

Tool arguments are decoded only through `Guard.decode_tool_arguments(args,
allow=...)`, with the kinds that tool may receive; a call naming another kind
is refused (`CP-073`). `chat()` returns tool calls encoded. Assert a pattern's
negatives on the outgoing text (`CP-074`).

## A tool result is a document

`Guard.encode_object` guards the JSON document the model receives — keys,
numbers and strings — with the `json` format (`CP-076`, `CP-077`). JSON strings
are also read as prose over their decoded text, mapped through the escapes
(`CP-078`).

## Current findings

All `CP-001` … `CP-043` are **closed**, each with a named regression test in
`tests/test_regressions.py` and an executable negative probe. `CP-010` and
`CP-011` were investigated and rejected as non-defects and are recorded so they
are not re-reported.

`CP-024` … `CP-029` are the six found by round three's own verification rather
than inherited from upstream: by the suite (`CP-024`), by review of the
diagnosis surface (`CP-025`), by the import-isolation probe once its harness
was strengthened (`CP-026`), by the live-engine probe's runtime (`CP-027`), and
by exercising every pattern's own examples in ordinary sentence positions
(`CP-028`, `CP-029`).

`CP-030` and `CP-031` came from neither a lane nor a reading but from a person
trying to use the thing: text could not be given directly, and the restoration
half was never demonstrated. Neither is a leak, and both are worth the same
seriousness — a privacy tool that is hard to feed gets pasted into the model
instead.

`CP-032` and `CP-033` came from a user's question rather than a bug report —
whether `--` was handled — and were found by putting the same option grammar
through both frontends. `CP-032` is a divergence; `CP-033` is a divergence whose
argparse side was also the wrong answer on its own, because a mistyped option
was redacted as text.

`CP-034`, `CP-035` and `CP-036` came from a user's report of one session. None
is a leak in the pipeline, and all three are about the vault being a file the
user had to manage by hand: a required path on every invocation, numbering that
restarted each turn so an earlier placeholder restored to the wrong value, and
no command that simply printed the clean prompt. The last is the one to keep in
mind when reviewing a new capability — the pieces of the clean-prompt command
all existed, and that made no difference to the person who could not find them.

`CP-037` and `CP-038` came from a user's report in round seven, and they differ
in where the fault came from. `CP-037` is the older kind: `encode` and `decode`
were aliases of two commands that did not form a pair, so the names described
an operation the tool did not offer. `CP-038` is the one this maintenance plane
caused itself — `CP-034`'s default vault path and `CP-035`'s append mode are
both right and together left the removed values accumulating in a clear-text
file with nothing to delete it. Record it that way when you touch the register:
the list otherwise reads as an inventory of other people's mistakes, and this
round's more useful finding is ours.

`CP-039` and `CP-040` came from a user's report in round eight and are both
about encryption being advertised more readily than it could be used.
`CP-039` is the smaller one: `forget` told the reader to use `--encrypt`, and
the command that writes the default vault did not have it. `CP-040` is the
structural one: `--encrypt` needed a compiled optional dependency, so the one
security feature was the least portable part of a standard-library tool, and a
vault encrypted on one machine could not be opened on another. Both are of the
`CP-036` kind — nothing was mis-detected, and a protection nobody can reach is
a protection nobody has.

`CP-041`, `CP-042` and `CP-043` are round nine's, and they share one root
cause: the hop between redaction and restoration was treated as a lossless
channel when it is a language model. They are closed in three complementary
directions rather than one — `CP-041` prevents a malformed span from ever
reaching the vault, `CP-042` repairs and reports the rewrites that do happen,
and `CP-043` removes the invitation by not sending a token worth normalising.
`CP-041` is also the only finding so far where the untrusted input was a
dependency's output rather than a user's text, which is why it is stated as a
rule: a third-party span is validated before it becomes a vault value.

The suite stands at 1521 passing, 6 skipped, plus 44 maintenance tests; the
contract checker reports maintenance `PASS` and runtime `PASS`. A
`vault_persistence` verification lane covers the default path, the two modes
and the format-1 refusal, and is `PASS`. The `crypto` lane now covers both
ciphers, and the import-isolation probe runs a full encrypt/decrypt round trip
with `cryptography` blocked — 20 probes plus 6 CLI entry points, zero leaks.
Both frontends produce byte-identical output for every round-four, round-five,
round-six, round-seven and round-eight path, asserted in the tests.

Round twelve added `CP-047` to `CP-058`; round thirteen `CP-059` to `CP-062`
(logging scope, stream cut, early-exit hang, prose span); round fourteen
`CP-063` (file order under remember), and round fifteen's scale verification
found `CP-064` (a history counted against `max_entries`) and `CP-065` (log
cost that grew with sessions, then with held values); round sixteen's
concurrency probe found `CP-067` (one label, two values, under threads);
round seventeen's multi-process probe found `CP-068` and `CP-069` (the vault
file shared between processes, and truncated before rewriting); round
eighteen's variant probe found `CP-070` (remembered values matched only as
spelled) and, through its fix, `CP-071` (a surrogate containing a held value);
round nineteen's restoration probe found `CP-072` (a rewritten stand-in left
unrestored, silently); round twenty's agent-threat review found `CP-073`
(tool calls restored every value), `CP-074` (tags read as paths) and `CP-075`
(Windows paths kept the user name); round twenty-one's review of tool results
found `CP-076` (keys unread), `CP-077` (numbers unread) and `CP-078` (prose
inside JSON strings unread); round twenty-two began with a refused push and
found `CP-079` (whole key-shaped examples in committed files), `CP-080` (a DTD
in a UTF-16 Office part was not refused) and `CP-081` to `CP-084` (a lint pass
that broke the base-tier import, session turns and the corpus bridge; an empty
pack section accepted). `CP-048` is the design driver
(records leaked because keys were never read); `CP-049` to `CP-053` were found
while building and verifying it; `CP-054` to `CP-056` were found by running the
Python matrix, lane 13, for the first time; `CP-057` and `CP-058` by the
format fuzz (`probe_fuzz.py`). The suite stands at 2042 passing,
6 skipped, green on CPython 3.8 to 3.13 (2128 after round fourteen, 2210
after round fifteen, 2243 after round sixteen, 2258 after round seventeen, 2302 after round eighteen, 2354 after round nineteen, 2381 after round twenty, 2424 after round twenty-one, 2464 after round twenty-two).

Do not reopen any of them from source inspection alone, and do not close a new
one without both a regression test and a probe.
