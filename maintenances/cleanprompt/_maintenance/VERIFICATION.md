# Verification ladder

Each lane proves something different. A lane that was not run is `UNAVAILABLE`,
and stays `UNAVAILABLE` rather than being filled in with something adjacent.
The machine-readable record is `EVIDENCE.json`; this file is what it means.

| #  | Lane | What it proves | Status | Artifact |
|----|---|---|---|---|
| 1  | maintenance contract | this plane is consistent with the runtime tree | PASS | `evidence/maintenance-tests.log` |
| 2  | focused tests | the checked-in suite collects and passes with no harness edits | PASS | `evidence/focused-tests.log` |
| 3  | invariants at scale | I1, I2, I7, I8 over randomized documents | PASS | `evidence/negative-probes.log` |
| 4  | import isolation | no third-party package loads, under an `__import__` blocker — including a full vault encrypt/decrypt round trip with `cryptography` blocked | PASS | `evidence/import-isolation.log` |
| 5  | star-import safety | `from … import *` resolves on a base install | PASS | `evidence/import-isolation.log` |
| 6  | negative probes | one executable probe per `CP-` finding | PASS | `evidence/negative-probes.log` |
| 7  | CLI end to end | `--help`, `doctor`, `kinds`, `inspect`, `redact`, `restore` | PASS | `evidence/focused-tests.log` |
| 8  | determinism | byte-identical output across processes and hash seeds | PASS | `evidence/multi-run.log` |
| 9  | repeatability | five consecutive runs agree, plus shuffled module order | PASS | `evidence/multi-run.log` |
| 10 | `web` tier live | routes, CSRF, session isolation against real Flask | PASS | `evidence/focused-tests.log` |
| 11 | entity engines live | real spaCy model and real NLTK, both measured | PASS | `evidence/engines-live.log` |
| 12 | vault ciphers live | both ciphers: the portable construction with nothing installed, and Fernet under the `crypto` tier — tamper, wrong key, swapped label, truncation | PASS | `evidence/focused-tests.log` |
| 13 | Python matrix | the suite on CPython 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 and 3.14 with nothing optional installed (round 25), 3.13 with every tier, and PyYAML at its declared floor (5.1) and latest (6.0.3) (round 21) | PASS | `evidence/python-matrix.log` |
| 14 | frontend parity | argparse and click agree on status and output | PASS | `evidence/cli-entrypoints.log` |
| 15 | global CLI delegation | `scikitplot cleanprompt` reaches the same code | PASS | `evidence/cli-entrypoints.log` |
| 16 | engine interchangeability | a vault written under one engine restores under the other | PASS | `evidence/engines-live.log` |
| 17 | offset alignment | every span indexes the original text, for every engine | PASS | `evidence/engines-live.log` |
| 18 | pattern sentence positions | every pattern found, and not over-captured, in prose | PASS | `evidence/focused-tests.log` |
| 19 | API encode/decode | the LLM-facing surface round-trips and discloses nothing | PASS | `evidence/negative-probes.log` |
| 20 | `option_grammar` | both frontends read a command line the same way: `--`, abbreviations, stray options, spellings | PASS | `evidence/cli-entrypoints.log` |
| 21 | `vault_persistence` | where the default vault resolves, that append continues a conversation's numbering, and that a format-1 append is refused | PASS | `evidence/focused-tests.log` |
| 22 | `lossy_restoration` | the ten measured rewrite shapes of a placeholder all restore, prose lookalikes are left untouched, and `--exact` preserves the previous behaviour | PASS | `evidence/focused-tests.log` |
| 23 | `gallery_examples` | every published example executes, confines itself to its own workspace, and prints no value that could reach anybody | PASS | `evidence/gallery-examples.log` |
| 24 | `structured_artifacts` | a notebook or module is redacted, stays the kind of file it was, and restores byte for byte | PASS | `evidence/focused-tests.log` |
| 25 | `packs_and_formats` | the compiled catalogue equals its YAML (`I10`), every pattern's examples execute (`I11`), and every format that claims a round trip restores byte for byte under `all`, `auto` and `none` | PASS | `evidence/packs-formats.log` |
| 26 | `record_fields` | a value whose field name marks it sensitive — CSV header, JSON key, `KEY=value`, assignment — is absent from the output (`CP-048`), one value keeps one label across files, and JSON stays JSON | PASS | `evidence/packs-formats.log`, `evidence/negative-probes.log` |
| 27 | platform matrix | Windows and macOS | UNAVAILABLE | every run so far is Linux x86-64 |
| 28 | `randomized_formats` | 4000 randomized documents across nine formats keep every invariant: round trip, JSON validity, no leak, determinism, idempotence | PASS | `evidence/fuzz.log` |
| 29 | `model_gate` | outgoing text is encoded and then checked by an independent search; a finding stops the call before the model is invoked; chat messages, tool arguments and results, streamed replies (any chunking equals the whole), async clients (`aask`, `achat`, `adecode_stream`) and command-line models (no shell, no orphaned process on a closed pipe) | PASS | `evidence/focused-tests.log`, `evidence/negative-probes.log` |
| 30 | `logging_discipline` | no removed value reaches a log record from any logger in the namespace — message, `extra=` fields, tracebacks — while a Session, Cleaner or Guard holds it; audit events carry counts, kinds and digests only; the cost of a record follows the record, not the number of sessions or held values (`CP-065`) | PASS | `evidence/negative-probes.log`, `evidence/import-isolation.log`, `evidence/scale.log` |
| 31 | `mcp_server` | a standard-library MCP server negotiates the protocol, answers requests and not notifications, survives malformed lines, confines paths to its roots, and no tool result — success or failure — carries a removed value or the root path; runs on Python 3.8 | PASS | `evidence/focused-tests.log`, `evidence/negative-probes.log`, `evidence/python-matrix.log` |
| 32 | `plan_pinning` | a saved plan reloads as the same plan, and is refused when a pack or format it resolves to has changed; folders are order-independent under `remember` | PASS | `evidence/negative-probes.log` |
| 33 | `record_chunking` | a CSV or JSON Lines file above the document limit is encoded in record-aligned pieces whose output equals one pass, over random cut sizes and at 6 MB; time is linear in size; history never exhausts a per-call limit (`CP-064`) | PASS | `evidence/scale.log`, `evidence/negative-probes.log` |
| 34 | `folder_survey` | a dry run of a folder (`batch --dry-run`, MCP `cleanprompt_inspect`) reports the statuses and kind counts encoding would produce, writes nothing, leaves the caller's vault unchanged, and carries no value | PASS | `evidence/focused-tests.log`, `evidence/negative-probes.log` |
| 35 | `concurrency` | a Cleaner, a Guard and a Session each shared by 8 threads with a forced switch interval give every value exactly one label and round-trip every text; the probe fails with the lock removed (`CP-067`) | PASS | `evidence/negative-probes.log`, `evidence/focused-tests.log` |
| 36 | `vault_file` | eight processes encoding into one vault each get their own label and every reply decodes; a reader during 60 rewrites never sees a partial vault; a waiting run is told why and times out with a reason; a killed holder leaves no stale lock; `forget` takes the same lock (`CP-068`, `CP-069`) | PASS | `evidence/negative-probes.log`, `evidence/focused-tests.log` |
| 37 | `value_variants` | a remembered value is hidden in capitals, lower case, across a line, with a non-breaking space, tab or double space, in full-width letters and with a typographic apostrophe — each restoring exactly — and refused in each with `remember` off; rewordings are not claimed; `canonical` preserves length over random Unicode; no surrogate contains a held value (`CP-070`, `CP-071`) | PASS | `evidence/negative-probes.log`, `evidence/focused-tests.log` |
| 38 | `restored_variants` | a surrogate the model re-cased, re-wrapped or re-spaced (up to 16 whitespace characters) is restored and reported as a repair; reformatted or reworded text is left alone; keys that differ only in case restore only exactly; `lenient=False` is unchanged; any chunking of such a reply streams exactly like the whole (`CP-072`) | PASS | `evidence/negative-probes.log`, `evidence/scale.log`, `evidence/focused-tests.log` |
| 39 | `tool_privilege` | a tool call is decoded only with the kinds its tool is allowed: an injected call naming another kind is refused before anything is restored, `keep` leaves those placeholders, `"all"` must be explicit, and `chat()` returns OpenAI and Anthropic tool calls encoded; closing tags are not paths and Windows paths hide the user name (`CP-073`–`CP-075`) | PASS | `evidence/negative-probes.log`, `evidence/import-isolation.log`, `evidence/focused-tests.log` |
| 40 | `tool_results` | a tool result is guarded as the JSON document the model receives: keys, numbers under sensitive fields and `Key: value` prose inside strings are hidden and restore exactly; escape offsets agree with the parser on random strings; NaN, bytes and a plan without `json` are refused (`CP-076`–`CP-078`) | PASS | `evidence/negative-probes.log`, `evidence/fuzz.log`, `evidence/import-isolation.log` |
| 41 | `secrets_at_rest` | no text file committed for this subsystem — the runtime package, this plane, the skill, the gallery — holds a whole value a `secrets`-pack pattern accepts (`I14`, `CP-REST-001`); the compiler refuses to write one; an unarmed check is an error, not a pass | PASS | `evidence/negative-probes.log`, `evidence/maintenance-tests.log` |
| 42 | `entity_readiness` | an engine is ready only with its package, its language and its data; `doctor`, `inspect`, `encode` and the web app decide through one function and agree, measured live with no model, no data, and NLTK data under pre-3.9 names (`CP-093`, `CP-094`, `CP-100`) | PASS | `evidence/focused-tests.log`, `evidence/round25.log` |
| 43 | `detection_view` | values with invisible format characters, compatibility forms, Unicode spaces or dashes are found and restored byte for byte; every built-in pattern example salted; 1000 salted values at scale; document-bound detectors never read the view; record files keep their structure; field names read through the view (`CP-098`, `CP-102`, `CP-103`) | PASS | `evidence/negative-probes.log`, `evidence/focused-tests.log` |
| 44 | `generated_deployment` | the container image installs the model it runs with, every generated launch line publishes on loopback, and debug mode is refused on every non-loopback bind (`CP-095`–`CP-097`) | PASS | `evidence/round25.log`, `evidence/focused-tests.log` |
| 45 | `documented_commands` | every `cleanprompt <command> --option` line in the README, docstrings, skills, guide and gallery uses an option the command declares; the reviewer exits 3 on a stale record (`CP-101`) | PASS | `evidence/maintenance-tests.log` |

## Lane 11, and why it was held open for two rounds

`tests/test__ner.py` runs a stub pipeline. That stub proves *this module's*
translation of entity offsets into spans and its composition with the
structural detectors — and nothing about spaCy's recognition quality. It was
never recorded as lane 11, and it still is not.

Lane 11 is now real: spaCy 3.8.16 with `en_core_web_sm` and NLTK 3.10.3 with
its four data packages, both installed, both measured over 120 fuzzed
documents per engine mode. The fuzzer builds documents from the material that
breaks offset handling — tabs, newlines, directional quotation marks, accented
letters, astral-plane emoji, bracketed footnote markers, and names repeated so
that any implementation recovering offsets by search would collapse every
occurrence onto the first.

Holding the lane open turned out to be worth more than closing it early would
have been. Running it is what surfaced `CP-027`: the probe timed out where it
should have taken seconds, because `nltk.ne_chunk` rebuilds its model on every
call and this detector calls it once per sentence.

## Lane 18, and the blind spot it closed

Every pattern was tested against its own examples, in isolation. In isolation
there is no punctuation, so nothing ever checked what happens when a value ends
a sentence. Lane 18 runs each pattern's `examples_yes` through twelve ordinary
sentence positions and found two defects going in opposite directions:
`CP-028`, where a sentence-final IP address was not detected at all and went to
the model in the clear, and `CP-029`, where a URL swallowed the sentence's full
stop into the placeholder.

The lane is a class-level test rather than a list of cases, so a pattern added
next year is checked the same way without anyone remembering to.

## Lane 20, and what it proves

Lane 14 compares what the two frontends *do* once a command line has been
parsed. Lane 20 compares how they read the command line in the first place,
which is where `CP-032` and `CP-033` were hiding: argparse accepted `--form` for
`--format` where click exited 2, and argparse accepted `inspect "--secret is
a@b.co"` as text — quietly redacting a mistyped flag as the user's prompt —
where click rejected it as an unknown option.

Sixteen cases run through both frontends and their status and output are
compared: `--` passthrough, including a value of `-rf`; a literal `--` inside
the text, which must survive because only the first is consumed; text that is
exactly `--`; `--` shielding arguments that look like options; a stray
dash-leading positional with no `--`, which must be refused with the delimiter
named; abbreviated long options, which must be refused; the full, short and
attached spellings of the same option; `-V`; and `-h`. Zero divergences,
recorded in `evidence/cli-entrypoints.log`.

What the lane does *not* claim: the separated spelling of a dash-leading option
value, `--hide -secret`, still differs — click accepts it, argparse refuses it.
That is documented in `DESIGN.md` and in the README rather than asserted away,
because closing it would mean overriding `argparse._parse_optional`, which is
private and has moved between releases.

An option added without a case in this lane is how the next `CP-021` arrives,
so the rule is that every new option is asserted in both frontends for
abbreviation refusal and for `--` behaviour.

## Lane 21, and what it proves

Three things, one per round-six finding, and the first of them is as much a
security check as a functional one.

**Where the default resolves.** `default_vault_path()` is exercised with
`CLEANPROMPT_VAULT` set and unset, with `XDG_STATE_HOME` set and unset, so the
resolution order is asserted rather than assumed: the environment variable,
then `$XDG_STATE_HOME/cleanprompt/vault.json`, then
`~/.local/state/cleanprompt/`, and `%LOCALAPPDATA%\cleanprompt\` on Windows.
The lane also asserts the negative: the resolved path is never inside the
working directory. That is the case `CP-034` exists for — a vault is clear text
and this tool runs inside checkouts, so a default beside the input file is one
`git add .` from committing the values the user was trying not to send. The
`0700` directory and `0600` file modes are checked on the created artifacts.

**That append continues numbering.** The same value redacted across several
runs keeps the label it was first given, and a value seen for the first time in
a later run is numbered after the existing ones, with the vault accumulating so
that a reply quoting any earlier turn still restores. Overwrite is checked to
do the opposite, because it is `redact`'s default and the scripted workflow
depends on it.

**That a format-1 append is refused.** A vault with no `index` cannot be
appended to, and the lane asserts the refusal rather than any particular merge
result. Merging would renumber from `[KIND-1]` while the old vault already used
that label for something else, and restoration would then return someone else's
value — a wrong answer from the half of the tool whose whole job is putting the
right value back. The lane also asserts that such a vault still *restores*
normally, so that the refusal stays narrow.

## Lane 22, and what it proves

Every other lane measures a part of the pipeline this submodule controls. This
one measures the part it does not: the hop between redaction and restoration,
which the design had treated as a lossless channel and which is in fact a
language model that rewrites tokens. That is `CP-042`, and the lane exists
because an assumption about someone else's behaviour can only be checked by
reproducing that behaviour.

**The nine measured rewrite shapes restore.** `[EMAIL-1]` is put through the
forms a model actually returns — among them lower-cased, with an underscore for
the hyphen, with a space for the hyphen, with a Unicode dash, with the brackets
escaped for Markdown, and wrapped across a line — and each one must restore to
the same value and be named in `RestorationResult.repaired`. Before the fix,
exact matching restored the first spelling and missed every other, and the only
signal was `restored 0 placeholder(s); 2 vault entr(y/ies) unused` at exit
status 0, which `-q` silenced entirely. The lane therefore asserts the report
as well as the text: a restoration that resolved nothing while the vault is
non-empty has to say so.

**Prose lookalikes are untouched.** `[note 2]`, `[1]` and the other bracketed
things that occur in ordinary writing must come back byte-identical, and must
not be reported as unknown placeholders. This is the half of the lane that
keeps the leniency honest. The pattern is bounded — a category must start with
a letter — but the real guard is that a lenient match is acted on only when it
resolves to a label the vault actually holds, and that is what this half
measures. A leniency without it would rewrite a model's own prose, which is a
worse failure than the one it was added to fix.

**`--exact` preserves the old behaviour.** The same inputs under `--exact`
(`lenient=False` in the API) restore only what was issued verbatim and repair
nothing. A caller who depends on the strict reading has to be able to keep it,
and an option that quietly does something adjacent to what it names is not one.

## Lane 24, and the leak that looked like a clean bill of health

`CP-046`. A realistic churn notebook and its feature module, through the
round-ten pipeline: three values removed, and on the module `blind_spots: []`.
What went to the model was the organisation, the analyst's name via
`/home/marion.holt/work/acme-churn/`, the production data path, the column name
`customer_ssn` — which discloses that the dataset holds them without containing
one — the proprietary feature set, the categorical levels of a segment, and
every row of a rendered `df.head()`.

The lane exists because the fix could have been much worse than the defect. A
redaction that turns every column into `[COLUMN-n]` is perfectly safe and
completely useless, because a model's advice depends on knowing which column is
numeric and which is a date — exactly the information such a redaction removes.
So the lane asserts **four** things at once, and a change satisfying any three
is a plausible bug:

*Nothing discloses.* No column name, path, rendered row or figure payload
survives into the output.

*It is still the same kind of file.* The notebook parses, and `output_type`,
`execution_count`, `ename` and the cell ids come back byte-identical. An
earlier version of the role mapping redacted `"output_type": "execute_result"`
into `"[OUTPUT-1]"` — safe, and a file no tool could open.

*It restores exactly.* Byte for byte, including code the model wrote that did
not exist at encode time.

*It is still useful.* The redacted code parses as Python, and the roles a model
needs are present, so "drop `id_1`, log-transform `amount_1`, one-hot
`category_1`" is advice that decodes back into the user's own schema.

Two further defects were found by running the example rather than the tests, and
both are now asserted. A notebook stores a cell's source as an array of lines,
so parsing per region meant a `df[[...]]` split across two lines was never
parsed and its columns never discovered — with a *truthful* `NOT read` line in
the report that looked like a notebook quirk. And the artefact branch did not
thread `--hide` through, so hiding an organisation name silently did nothing on
exactly the files where it most needed to.

## Lane 22, and the shape the table was missing

`CP-042` enumerated the shapes a model returns a placeholder in and admitted
each. `CP-044` is the tenth, and the way it was found is the useful part: the
implementation had been read back against the docstring's table rather than
against its own behaviour, and the table was written in doubled escapes, which
made it easier to check against itself than against the pattern.

Two of the listed shapes compose. A model writing `[EMAIL_1]` inside Markdown
escapes the underscore, because an unescaped one opens emphasis — so
`[EMAIL\_1]` arrives, matches nothing, and restores nothing, with the same
silent `restored 0 placeholder(s)` at exit `0` that `CP-042` exists to prevent.

The fix is one optional backslash before the separator, the allowance the
delimiters already had. It was measured before it was made: it widens the set
of *labels* matched by exactly the four escaped spellings and widens the set of
*prose* matched by nothing. The table is now written literally, with the two
characters a source line cannot show spelled `<U+2011>` and `<NL>`, and a test
reads the table out of the docstring and asserts every row against the pattern.
That test is the lane's actual content: the previous table was true when
written and stopped being checked.

## Lane 23, and why a documentation example needs a lane at all

Sphinx-Gallery *executes* these scripts during a documentation build. They are
not prose; they are code that runs on somebody else's machine, in the process
that builds the docs, and three things can go wrong that no unit test sees.

**They stop working.** A renamed option or a changed default turns a tutorial
into a failing build, discovered at release time. Every script is run end to
end and its status asserted.

**They write somewhere real.** This is the one that justified the lane. A vault
holds the removed values in clear text, and the CLI's default location is the
platform state directory — correct for a person at a terminal, wrong for a
documentation builder, which would be left holding those values in its own home
directory. Every script points `CLEANPROMPT_VAULT` at a `TemporaryDirectory` in
its first cell; the probe runs each one with `HOME` and `XDG_STATE_HOME`
redirected into a sandbox and then asserts the sandbox is **empty**. A script
that forgets leaves a file there and fails the lane.

**They publish something that can reach a person.** A gallery page is published.
The probe greps every captured stream for addresses, telephone numbers and IP
addresses outside the reserved ranges — :rfc:`2606` domains, :rfc:`5737` and
:rfc:`3849` address blocks, the NANP fiction block — and fails on anything else.

The sandbox earned its keep immediately. NLTK's data packages live under
`$HOME/nltk_data`, so redirecting `HOME` reproduced the common reader's
machine: `nltk` imports, `build_detectors` succeeds, and the failure appears
only when a sentence is tokenised. The example had guarded construction rather
than detection and crashed the build. It now reports a specific `SKIP`, which
is what the gallery reliability rule requires, and the probe records two skips
on that run rather than a pass it did not earn.

## Lane 7, and the pipe that was reported as an error

`CP-045` is not in a lane of its own because it belongs to the CLI end-to-end
lane, but it is worth recording here because of what caused it.

`main` translated `OSError` into `error: ...` and status `1`, which is right for
a missing vault or an unreadable file. `BrokenPipeError` is an `OSError` and is
none of those: it means the reader went away, which is what `| head -1` does on
purpose. So a correct command line printed `error: [Errno 32] Broken pipe`, and
the interpreter's shutdown flush hit the same closed pipe and added
`Exception ignored in: <_io.TextIOWrapper ...>` after it.

Both halves are fixed together, because the second happens after every handler
has returned: descriptor `1` is pointed at `os.devnull` before returning, and
the status is `128 + SIGPIPE` so a pipeline cannot distinguish the case from a
process the kernel signalled.

The guard on that repair is the part worth keeping in mind. `main` takes
injectable streams so the tests drive the real entry point, and the first
version replaced descriptor `1` whenever a `BrokenPipeError` arrived — which
tore down the test harness's own descriptor. The condition is not "a broken
pipe happened" but "the object the interpreter will flush at shutdown is the
one that refused the write".

## Lane 12, and what it covers since round eight

The lane used to mean one thing: the `crypto` tier, exercised against a real
`cryptography` install. Since `CP-040` there are two ciphers and the lane runs
both, because the default is now the one that needs nothing installed and an
untested default is the worst kind.

**`portable`, with no optional package present.** The construction in
`_vaultcrypt.py` — scrypt (or PBKDF2-HMAC-SHA256 where the interpreter cannot
run scrypt), separate encryption and MAC keys, HMAC-SHA256 counter mode, and
encrypt-then-MAC over the cipher name, the label, the nonce and the ciphertext
— is exercised for the properties it has to deliver: a wrong passphrase is
refused, a bit flipped anywhere in the token is refused, a token moved from one
label to another is refused because the label is inside the tag, a truncated
token is refused, and two encryptions of the same value under the same key
differ, which is the nonce doing its job. The lane asserts those properties. It
does not assert that the construction is secure, and no lane in this ladder
could: what it proves is that the composition behaves as described and fails
closed when it is interfered with.

**`fernet`, under the `crypto` tier.** The reviewed AES backend is exercised as
it was before, so `--cipher fernet` and `auto` keep their evidence.

Lane 4 carries the portability half of the same finding. The import-isolation
probe writes an encrypted vault and reads it back while `cryptography` is
blocked by the `__import__` blocker, which is the claim `CP-040` makes and the
only way to check it: a vault that can only be opened where a compiled
dependency happens to be installed has traded one risk for another. Twenty
probes plus six CLI entry points run under the blocker, with zero leaks.

## Lane 13, measured at last

Held `UNAVAILABLE` for eleven rounds with the note that "avoids" is not
"verified". Round twelve ran it — bare environments with only pytest, on
CPython 3.8.20, 3.9.23, 3.10, 3.12 and 3.13, plus the full environment on
3.11 — and the first run found three defects that no amount of avoiding had
prevented:

- `CP-054`: the base tier imported `typing_extensions` at module scope, so
  `import scikitplot.cleanprompt` failed on a bare 3.12 or 3.13. The import
  probe blocks a *list* of packages and the backport was not on it; a new test
  asserts the positive property instead (only the standard library loads).
- `CP-055`: on 3.8, `df.loc[:, "col"]` parses as `ExtSlice`, and the column
  was not discovered.
- `CP-056`: seven tests exercised a live optional tier without a skip, and the
  corpus bridge let a 3.8 `TypeError` from corpus's own import escape.

PyYAML is tested at both ends of its declared range, 5.1 on 3.8 and 6.0.3 on
3.13, against the custom-pack, drift and CLI tests; the compiled catalogue is
identical under both. Platforms other than Linux are lane 27.

## Lane 30, and the filter that saw nothing

`redacting()` had been "defence in depth" since round one, and it scrubbed
nothing that mattered: a filter on a logger sees only that logger's own
records, and every module here logs through a child logger (`CP-059`). The
lane now measures the property — a value logged through a child logger inside
the block, and a value held by a live guard, never reach the stream — and the
test that asserts it was mutation-checked by restoring the old attachment.

## Lane 26, and the file that redacted nothing

`patient.json` — an MRN, a date of birth, a diagnosis code and an insurer id —
went through the round-eleven pipeline with zero redactions and a clean report.
Nothing in it had the *shape* of anything, and the key that said what each
value was had never been read. This lane measures the record formats by the
values left in the output, before and after, and holds the one-label-per-value
property across a prose note and a record: the same patient is `[MRN-1]` in
both.

## Lane 33, and the limit a history could exhaust

The lane compares a chunked encode with one pass over the same file, with the
limits raised for the one pass so both can run. Its first run failed for a
reason that was not about chunking: the engine counted every seeded entry
against `max_entries`, so the pieces of a large file — and every long
conversation — shared one lifetime budget (`CP-064`).

## Lane 30, measured with values held

The first shared log filter passed its timing test in a fresh process and
failed in the scale probe, which runs after cleaners holding 126 000 values
have been created: 2000 records took 10.9 s. A timing claim about the log
filter is only evidence when it is measured with values held.

## Lane 35, and a defect worse than a leak

Every earlier lane asked whether a value could reach the model. This one asks
whether a value can reach the *wrong person*. With threads sharing one guard,
two values got one label and a reply was decoded with the other one's value
(`CP-067`). The lane shrinks the interpreter's switch interval so threads
interleave inside an encode; with the default interval the race still
happens, only rarely — which is how it survived fifteen rounds.

## Reproducing

```sh
python -B -m pytest scikitplot/cleanprompt/tests -q -p no:cacheprovider
python -B maintenances/cleanprompt/_maintenance/evidence/probe_isolation.py
python -B maintenances/cleanprompt/_maintenance/evidence/probe_negative.py
python -B maintenances/cleanprompt/_maintenance/evidence/probe_engines.py
python -B maintenances/cleanprompt/_maintenance/evidence/probe_gallery.py
python -B maintenances/cleanprompt/_maintenance/evidence/probe_fuzz.py
python -B maintenances/cleanprompt/_maintenance/evidence/probe_scale.py
python -B -m scikitplot.cleanprompt packs --check
python -B maintenances/cleanprompt/_maintenance/tools/check_contract.py
python -B maintenances/cleanprompt/_maintenance/check_trackers.py --json
python -B maintenances/cleanprompt/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/cleanprompt/_maintenance/tests -q -p no:cacheprovider
```

`probe_engines.py` is the only one that needs an entity engine installed; it
reports which are usable and exits cleanly when none are.

`probe_gallery.py` must be run from the repository root, because it locates the
examples at `galleries/examples/cleanprompt`. It sandboxes `HOME` for each
script, so it deliberately runs the examples as a machine without NLTK data
would, and records the resulting skips.
