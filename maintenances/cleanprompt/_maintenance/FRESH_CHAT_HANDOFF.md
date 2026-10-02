# Fresh chat handoff

## Read in this order

1. `DESIGN.md` — the whole picture, and the reasoning. Everything else assumes it.
2. `SUBMODULE_STRUCTURE.md` — where things live and which rules are enforced.
3. `REVIEW.json` — the defect register; `CP-001` … `CP-078` are all closed,
   each with a named regression test and an executable probe.
4. `VERIFICATION.md` — which lanes are green and which are `UNAVAILABLE`.
5. `STATE.json` — current status per plane.

## Then run

```sh
python -B -m pytest scikitplot/cleanprompt/tests -q -p no:cacheprovider
python -B -m scikitplot.cleanprompt packs --check
python -B maintenances/cleanprompt/_maintenance/check_trackers.py --json
python -B maintenances/cleanprompt/_maintenance/evidence/probe_engines.py
python -B maintenances/cleanprompt/_maintenance/evidence/probe_gallery.py
python -B maintenances/cleanprompt/_maintenance/evidence/probe_fuzz.py
python -B maintenances/cleanprompt/_maintenance/evidence/probe_scale.py
```

`probe_engines.py` is the only one that needs an entity engine installed; it
reports which are usable and exits cleanly when none are. `probe_gallery.py`
must run from the repository root and takes a couple of minutes: it executes
all nine published examples in sandboxed homes.

## What is actually open

Nothing in the runtime: 2467 tests pass, 10 skipped, the maintenance plane's
tests pass, and both the maintenance and the runtime contract are `PASS`. One
thing is *not proven*, which is different:

- other platforms (lane 27). Lane 13, the Python matrix, is now measured —
  CPython 3.8 to 3.13 in bare environments and 3.11 with every tier — and its
  first run found `CP-054`, `CP-055` and `CP-056`. Every run was on Linux.

Two lanes that used to be open are now closed with evidence rather than
argument. Lane 12 runs both vault ciphers — the portable construction with
nothing installed, and Fernet under the `crypto` tier — and the
import-isolation probe runs a full encrypt/decrypt round trip with
`cryptography` blocked, 20 probes plus 6 CLI entry points, zero leaks. The
entity-engines lane runs real spaCy 3.8.16 with `en_core_web_sm` and real NLTK
3.10.3 with its corpora, over 120 fuzzed documents per engine mode.
`tests/test__ner.py` still uses a stub; the stub proves span translation and
nothing about recognition quality, so it is still not that lane.

What lane 12 does *not* claim is worth carrying into any change here: it
asserts that the portable construction fails closed on a wrong passphrase, a
flipped bit, a swapped label, a truncated token and a reused nonce. It does not
assert that the construction is secure, and no lane could.

## The documentation surface

`galleries/examples/cleanprompt/` holds the published examples: a `README.txt`
that Sphinx-Gallery requires (without it the folder is skipped and the build
succeeds having published nothing) and nine `plot_*.py` scripts.

They are **executed** during a documentation build, so treat them as code:

- every script that starts the CLI must point `CLEANPROMPT_VAULT` at a
  `TemporaryDirectory` in its first cell and clean it up in its last, because a
  vault is clear text and the default location is the builder's state
  directory;
- every printed value must come from a reserved range — `example.*` domains,
  `192.0.2.0/24`, `2001:db8::/32`, `+1 555 0100`–`0199`;
- an absent optional capability is a visible, specific `[SKIP]`; a failed round
  trip or a surviving value is a failure.

`tests/test_gallery.py` asserts the cheap half of that statically on every
commit. Lane 23 (`probe_gallery.py`) asserts the rest by running them with
`HOME` redirected into a sandbox and checking the sandbox is empty afterwards.
Guard **detection**, not just construction: redirecting `HOME` hides
`nltk_data`, so `build_detectors` succeeds and the failure appears only when a
sentence is tokenised. That is a real reader's machine, and it is how an
example that passes here crashes there.

## The artefact layer

Four modules turn `cleanprompt` from a prose tool into one that reads a
notebook or a module: `_documents.py` (regions and roles), `_code.py`
(syntactic schema discovery), `_schema.py` (role-preserving stand-ins) and
`_artifacts.py` (orchestration). `DESIGN.md` section 12 has the reasoning.

Four rules to hold onto before changing any of them:

- **offsets index the raw file.** Never re-serialise a notebook. `json.dumps`
  reformats it, drops cell ids and widget state, and turns the single rewrite
  pass every invariant is stated over into a pass across different text.
- **discovery is conservative; rewriting is complete.** `df.x` is not a
  discovery site — it cannot be told from a method call — and *is* a rewrite
  site once `x` is known from a subscript. Collapsing the two either misses
  columns or corrupts code.
- **hiding is deductive; role assignment is evidential.** A discovered column
  is hidden whether or not its role is known. Only the choice of stand-in
  depends on evidence, and an unestablished role is reported as `field_n`
  rather than guessed. `--infer-roles` is opt-in and labelled.
- **the parse unit is a cell, not a source line.** A notebook stores source as
  an array of lines; a statement spans them. Regions stay per-line for
  rewriting and are joined per cell for parsing.

Structure is not data: `output_type`, `name`, `execution_count` and cell ids
must come back byte-identical, or the file will not open. There is a test.

## Packs, formats and the fluent plan (round twelve)

`DESIGN.md` section 13 is the reasoning; the working rules are these.

- **Edit YAML, then compile.** Built-in packs and formats are authored in
  `scikitplot/cleanprompt/_config/{packs,formats}/*.yaml` and read from
  `_config/_compiled.json`. After any edit run
  `python -m scikitplot.cleanprompt packs --compile` (needs PyYAML); the suite
  and `packs --check` fail on drift (`I10`).
- **A pattern ships with examples that run.** `examples_yes` must each match
  in full and `examples_no` must not match, at load (`I11`). If a pattern needs
  a label to be sure, wrap the part to hide in `(?P<value>...)` so the vault
  keys on the value (`CP-051`).
- **Behaviour is named, never supplied.** A new checksum is a function in
  `_hooks.VALIDATORS`, reviewed like any code; a pack only names it.
- **One field, one meaning.** A field name may have one kind across all packs;
  `test__catalog` combines the whole catalogue to prove it.
- **A new round-trip format needs a sample** in `tests/test__runtime.SAMPLES`,
  or `test_every_round_trip_format_has_a_sample` fails.
- **JSON stays JSON.** Anything that changes span handling for `json`/`jsonl`
  must keep `_TokenBound` clipping and the sentinel pass; the output is parsed
  before it is returned.
- **Nothing unexamined is written.** Skipped and refused files never reach the
  output; keep it that way in any new container splitter.

## The gate and logging (round thirteen)

`DESIGN.md` section 14 is the reasoning. The rules:

- **Encode, then check, then call.** `Guard.outgoing` must run
  `Cleaner.leaks` after encoding and raise before any caller function runs.
  Never add a path that calls a model without passing through it.
- **The gate owns no network.** No vendor SDK, no socket, no key; `_bridge`
  runs the user's command with `shell=False` and kills it on any early exit.
- **Streaming equals whole.** Any change to `StreamDecoder` must keep
  `test__guard.TestStream.test_any_chunking_decodes_like_the_whole` green.
- **Log facts, never values or file names.** Use `audit()` for events; every
  logger comes from `get_logger` so active scrubbers reach it.
- **No MCP tool returns a value.** Tool results enter the model's context;
  `test__mcp.TestNothingReachesTheModel` must stay green for any new tool.
- **Two passes when remembering.** A new batch entry point must learn before
  it writes (`Cleaner._learn`), or file order decides what leaks.
- **A new field picks its prose `span` by its format**: `line` unless the
  format provably has no comma (`clause`) or no whitespace (`token`).

## Scale (round fifteen)

`DESIGN.md` section 16 is the reasoning. The rules:

- **Chunked equals whole.** A record file is cut only at `record_starts`,
  with regions from the whole file, the vault carried forward and known
  values frozen per file. `TestChunkedRecords` and `probe_scale.py` hold it;
  a format without independent records is never cut.
- **A limit bounds one call.** Nothing seeded counts against `max_entries`
  (`CP-064`); do not add a limit that a conversation's history can exhaust.
- **The log filter's cost follows the record, not the vault.** Adding or
  releasing a value is O(1); finding values in a record reads the record
  once. `CP-065` was fixed twice: the first fix removed the per-session cost
  and kept a per-held-value one, which only showed when the probe ran in a
  process that already held 126 000 values. Measure log cost with values
  held, never in a fresh process.
- **Suppress lint with `# noqa: CODE`.** `# ruff: ignore[...]` is not a
  suppression ruff 0.15 honours; the package carries 132 such comments and
  all are inert. Round 15 converted the ones that hid real `F401` findings
  (the type-checking re-exports in `__init__.py`) and removed eight unused
  test imports; the rest name rules the default configuration does not
  enable. Check a suppression with `ruff check --isolated` on a one-line file
  before relying on it.
- **The fingerprint hashes source, not caches.** `check_contract.py`
  skips `CACHE_DIRECTORIES`; a new tool that writes into the package needs
  its cache added there, or every run of it breaks the contract.
- **A dry run is the real walk.** `survey_tree` must stay `_walk_tree` with
  no output, on a scratch cleaner; never a separate estimate.

## Many callers (round sixteen)

`DESIGN.md` section 17 is the reasoning. The rules:

- **Anything that reads or writes a vault runs under the owner's lock.** Add
  `@_synchronized` to any new `Cleaner` or `Session` method that touches
  `_entries`, `_vault` or the sentinel counter, and never to a generator.
  `tests/_concurrency.py` is the harness; `CP-067` is what happens otherwise.
- **Async is the same gate.** A new async method must reuse the sync
  method's helpers (`_answer`, `_reply`, `_messages`), never re-implement a
  step.

## One vault file (round seventeen)

`DESIGN.md` section 18. Any new command that reads a vault and writes it back
must hold `_vault_lock(path, stderr)` from the read to the write, after
reading standard input; any new writer of a secret file uses
`_files.atomic_write`. Never delete the `.lock` file.

## The same value (round eighteen)

`DESIGN.md` section 19. "Is this the same value?" has one answer:
`_canonical.value_pattern` over `_canonical.canonical` text. Use it for any
new place that looks for held values; never compare raw strings. Add an
equivalence only if it is one (same string under a fixed rule) and keep
`canonical` length-preserving — `test__canonical` checks both.

## The way back (round nineteen)

`DESIGN.md` section 20. Restoration of literal stand-ins is exact matches
*plus* bounded, unambiguous equivalence matches, each reported as a repair.
If you change `RESTORE_MAX_GAP` or the equivalence, the stream decoder's reach
(`StreamDecoder._literal_prefixes`) must follow, and
`TestRewrittenStandInsStream` is the test that says whether it did. Matcher
caches are keyed on vault keys only; never put a value in one.

## A tool call is an exit (round twenty)

`DESIGN.md` section 21. Never add a path that restores values into a tool
call without an allow-list: `decode_tool_arguments` is the only one, and
`chat()` must keep returning tool calls encoded. A new pattern's negative
examples are asserted on the outgoing text, not on "the match is not the
whole example".

## A tool result is a document (round twenty-one)

`DESIGN.md` section 22. Guard what will be sent in the form it will be sent:
`encode_object` serialises and uses the `json` format; never walk a structure
and guard pieces. JSON string regions are decided on decoded text and mapped
through `_string_offsets`; keep that the only place escapes are interpreted.

## What is committed is read by scanners (round twenty-two)

`DESIGN.md` section 23. Invariant `I14`: no file under the package holds a
whole value a `secrets`-pack pattern accepts. A positive example for a key
pattern is written as fragments — `[['sk_live_', '0123…']]` — which the loader
joins; `_compiled.json` stores the fragments. A test fixture that needs a
key-shaped value builds it at run time. `packs --compile` refuses otherwise,
and names the file, line and kind, never the value.

Two things this round learned the hard way. Run the suite after any lint or
format pass: one such pass broke the base-tier import, the corpus bridge and
every session turn, and the suite already knew. And a commit that has been
refused must be rewritten, not followed by a fix — the scanner reads every
commit in the push, so the value has to leave history, not just the tip.

## Evidence discipline

A log's round label changes only when the command behind it has run again.
Round 17 nearly relabelled `packs-formats.log` without re-running it, and caught
it; a relabelled log is a claim with no evidence behind it.

## The things not to break

1. `import scikitplot.cleanprompt` must import no third-party package. If you
   need one, put the import inside a function and gate it on `require(tier)`
   *before* the import, never after.
2. Detectors receive the original text. If you find yourself wanting to run one
   stage over another's output, you are re-creating `CP-006`.
3. The vault stays structurally separate from the redacted text, and
   `_types.as_dict` is the only sanctioned serializer for results.
4. Entity labels are canonicalised at the detector boundary. spaCy says `ORG`
   where NLTK says `ORGANIZATION`; if a raw label escapes, a vault written
   under one engine stops restoring under the other.
5. No removed value ever reaches a log record. That holds by construction at
   every call site; `SecretFilter` is the second line, not the first.
6. Every command that takes text accepts all three sources — a positional
   `TEXT`, `--in PATH`, and standard input — through the one `_resolve_text`
   helper, and each of the three is asserted through *both* frontends. That is
   `CP-030`: a user pasting a paragraph after `inspect --ner` was told `Error:
   Got unexpected extra argument`, which names their own text as the fault, and
   running the command bare blocked on stdin with no output at all. If you add a
   text command, wire it to the same helper and add the parity cases; a source
   that works under argparse and not under click is `CP-021` again, in a place
   where the user is already confused.
7. Every new option is asserted in **both** frontends for two things: that an
   abbreviation of it is refused, and that `--` still ends the options in front
   of it. Those are `CP-032` and `CP-033`. argparse accepts an unambiguous
   prefix and click accepts none, so `--form` for `--format` worked or exited 2
   depending on which library was installed — and an abbreviation breaks on one
   machine too, as soon as a later option shares its prefix. argparse also
   classifies any token containing a space as a positional, so `inspect
   "--secret is a@b.co"` was redacted as if a mistyped flag were the user's
   prompt, with the option they meant never applied. `allow_abbrev=False` must
   stay on the root parser *and on every subparser* — subparsers do not inherit
   it — and a dash-leading positional stays refused unless a `--` authorised it.
   Nothing is checked past a `--`, and a lone `-` stays valid as standard input.
8. The default vault path must never resolve inside the working directory. A
   vault holds the removed values in clear text and this tool is used inside
   checkouts — the session that produced `CP-034` ran from
   `/work/.git_clones/learn/docs` on a checked-out branch — so a default beside
   the input file is one `git add .` from committing the exact values the user
   was redacting so as not to send them anywhere. `default_vault_path()`
   resolves `CLEANPROMPT_VAULT`, else the platform state directory
   (`$XDG_STATE_HOME/cleanprompt/vault.json`, falling back to
   `~/.local/state/cleanprompt/`, and `%LOCALAPPDATA%\cleanprompt\` on
   Windows), with the directory created `0700` and the file `0600` at creation
   rather than chmod-ed after. If you add a "vault next to the input" or
   "vault in the current directory" convenience, you are reintroducing that.
9. Append must never merge a vault that has no `index`. That is `CP-035`: a
   format-1 vault records no label-to-category mapping, so merging would
   renumber from `[KIND-1]` while the old vault already used that label for a
   different value, leaving one placeholder standing for two and restoration
   handing the user someone else's data. The refusal is the behaviour, not a
   missing feature — it costs one `--vault-mode overwrite`, and guessing costs
   a disclosure. Format-1 vaults must keep restoring normally; only appending
   is refused.
10. Any feature that makes this tool retain personal data by default ships the
    command that deletes it, in the same change. That is `CP-038`, and it is
    the one finding in the register this plane caused itself: `CP-034` gave the
    conversational command a vault at a fixed default path and `CP-035` made it
    append, both correct, and together they left the removed values piling up
    in one clear-text file at a location the user never chose, with nothing
    that cleared it. A tool that begins collecting personal data by default and
    offers no way to stop is worse than one that never collected it. `forget`
    (alias `clear-vault`) is that command: it reports and stops, `--force`
    deletes, the report names counts and categories but never a value, it
    deletes a vault it cannot parse — `ValueError`, `OSError` and
    `CleanPromptError` are all caught, and only the summary is lost — and it
    states that unlinking is not shredding, pointing at `--encrypt` with a
    user-held key for values that must not be recoverable. Do not strengthen
    that wording; the filesystem does not back it. When you add a default, ask
    what it now keeps, where, and what removes it.
11. A security feature must not depend on an optional tier. That is `CP-040`:
    `--encrypt` required the `crypto` tier and its compiled `cryptography`
    dependency, so the one protective feature here was the only part of the
    base workflow that needed an installed package, and a vault encrypted on
    one machine could not be opened on another. The base tier being pure
    standard library is this submodule's central promise, and making the
    mitigation the tool itself recommends the single exception inverted that
    promise where it mattered most — the user least able to install a package
    is not less entitled to encrypt. `_vaultcrypt.py` is base tier and stays
    base tier; `cryptography` buys `--cipher fernet` and nothing else. The same
    test applies to `doctor --new-key`, which generates its passphrase from the
    standard library because a key generator behind an extra is no use to the
    person being told to encrypt. If a new protective feature needs a package,
    that is a reason to reconsider the feature, not to add the tier.
12. `_vaultcrypt.py`'s construction may only change behind a **new versioned
    cipher name**. The vault records `cipher` as `cleanprompt-hmac-v1` and the
    full `kdf` parameters, and decryption dispatches on what the document says
    rather than on what this build happens to do. That is what lets today's
    vaults keep opening after tomorrow's change — a vault is written once and
    read much later, possibly by a newer build, and a silent change to the
    derivation, the key split, the keystream or the tag input makes every
    existing vault undecryptable with no way to tell why. Change the steps only
    under a new name, keep the old name readable, and leave the label inside
    the MAC input: without it someone with write access could swap two tokens
    and make `[EMAIL-1]` restore to a different person's address with every
    check still passing. Nothing written about this module may go further than
    what it is — a standard composition, not a reviewed implementation of AES,
    with Fernet the better primitive where it can be had.
13. A span produced by a third party is validated before it becomes a vault
    value. That is `CP-041`: on `Mustafa Kemal Atatürk[e] (c. 1881)` spaCy
    returns the person as `'Mustafa Kemal Atatürk[e'`, swallowing the opening
    bracket and the footnote letter and leaving the closing one behind. The
    visible symptom was a malformed `[PERSON-1]]` in the prompt; the silent one
    was that the vault recorded the person's name **as**
    `Mustafa Kemal Atatürk[e`, so restoring into any other text would have
    produced that string as somebody's name. Every structural pattern already
    carried a validator — Luhn, mod-97, octet ranges — and an entity span,
    which comes from a model nobody in this project controls, had none.
    `trim_entity_span()` in `_engines.py` is that check and both engines apply
    it: truncate at the first unmatched opener, start after the last unmatched
    closer, leave balanced brackets alone so `Acme (Europe) Ltd` is untouched.
    Keep it narrow. Only brackets are trimmed, because a trailing full stop is
    legitimate in `Inc.` and trimming it would be a guess. If you add a third
    detector, or a new source of spans, it is validated at the same boundary —
    do not assume a dependency's output is well-formed because it usually is.
14. Surrogate reversal stays a **single merged pass**. Under `--style
    surrogate` the reply is scanned once: grammar matches and literal
    surrogate matches, collected through `LiteralDetector` over the *original*
    reply text, are merged and rewritten in one pass. A second pass over the
    first pass's output would let an already-restored value be re-matched as
    another key — `CP-006` reappearing on the restoration side, where it
    corrupts the model's answer rather than the prompt. If you add another
    stand-in mechanism, collect it into the same pass; chaining a second one is
    the failure, however convenient it looks. The related invariant on the
    encode side is I6, and it is the same reason: stages compose over spans,
    never over each other's strings.

## If you add an option or a tier

A default may degrade quietly. A *request* may not. `CP-024` was an explicit
`--ner` that could not be met: it exited 0 having run no engine, telling the
user their text had been scanned for names when nothing had looked at it. Pass
`required=True` through `build_detectors` on any path that came from an
explicit flag, and let `CapabilityError` name each engine's own status and
remedy.

Compute remedies rather than writing them down. `CP-025` shipped an instruction
to download `en_core_web_lg` after the default had moved to `sm`, so following
it meant a 560 MB download that still left the tier misconfigured. Resolve the
model name through `resolve_model`, the same resolver the detector uses.

## If you add a pattern

If it recognises a credential, put it in a pack named in
`_catalog.AT_REST_PACKS` and write its positive examples as fragments
(`[['prefix_', 'rest']]`); a whole one fails `packs --compile` (`I14`).

State its intent in a sentence, give it positive **and** negative examples, and
prefer a small validator over a cleverer expression. If it needs a validator,
re-read the `RegexDetector` docstring first: `CP-015` and `CP-016` are both
about how validated patterns must be scanned, and getting that wrong leaks.

Your examples are run in prose as well as in isolation — a class-level test
puts every `examples_yes` through twelve ordinary sentence positions. That test
found `CP-028`, where `IPV4`'s trailing guard rejected an address at the end of
a sentence so it went to the model in the clear, and `CP-029`, where a URL
swallowed the sentence's full stop into the placeholder. Write a trailing guard
that says exactly what it means. One known cost stands: a URL genuinely ending
in `)` has that bracket left in the clear, which discloses nothing.
