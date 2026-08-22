# Maintenance Model — `scikitplot._cli`

> The terminal consumer: nothing depends on `_cli`. Read `DEPENDENCY_MAP.md` §3
> before touching the delegation registry.

---

## WHY

**Role: the command-line front door. It delegates to submodules by string, never
by import.**

A CLI's failure mode is not a wrong answer — it is **a promise the user can see
and cannot use**. `scikitplot mcp ...` appears in `--help` whether or not
`scikitplot.mcp` is installed, importable, or working, because the delegation
target is a string resolved only when the command runs.

Three shapes of that:

* **A dangling target.** `"scikitplot.mcp.__main__:main"` names a module and an
  attribute. If either is renamed, the command fails at *use time* with an
  import error rather than a message.
* **A silent exit code.** A CLI's contract is `(stdout, stderr, exit code)`. An
  exit code that does not match the outcome is undetectable by a human and
  catastrophic in a script.
* **Frontend divergence.** Two frontends (`argparse`, `click`) expose the same
  commands. When they disagree, one set of users gets different behaviour.

> A command that appears in `--help` must either work or fail with a message
> that says what to install.

---

## WHEN — triggers

| Trigger | Response |
|---|---|
| A delegation target is added or renamed | **The gate checks it resolves** — run it |
| A command is added | Both frontends, or an explicit note that it is single-frontend |
| An exit code changes | It is part of the contract; test it |
| A submodule `_cli` delegates to changes its `__main__` | Re-run the gate |
| Output format changes | The format-coverage tests exist for this |
| `check_trackers.py` fails | Drift, a dangling target, or a crossed tripwire |

**Not a trigger:** elapsed time.

---

## WHERE

```text
scikitplot/_cli/
├── MAINTAINING.md
├── app.py  cli.py  __main__.py       entry points
├── registry.py  _spec.py  loader.py  the delegation registry and resolver
├── context.py  output.py  logging.py  errors.py  exit_codes.py   the contract
├── _commands/                        built-in commands
├── _frontends/                       argparse + click
├── _backup/                          OLD implementation — see structure §3
├── tests/
└── _maintenance/                     this folder
```

---

## WHICH — what `_cli` owns

| Owns | Purpose |
|---|---|
| the delegation registry | `registry.py` — string targets, resolved at runtime |
| the output contract | `(stdout, stderr, exit code)` and the format matrix |
| the two frontends | `argparse` and `click`, expected to stay at parity |
| built-in commands | `doctor`, `info`, `sysinfo`, `show_config`, `show_versions`, `greet` |

**Out of scope:** anything a delegated submodule does. `_cli` owns *how* a
command is reached, never what it does.

---

## HOW MANY

```text
source   33 files /   7383 LOC
tests    14 files /    636 LOC
backup   11 files /   2781 LOC
delegation targets: 1
```

test : source LOC = **0.09**

| Metric | Now | Tripwire |
|---|---|---|
| delegation targets that resolve | see gate | **any that do not** |
| `_backup/` LOC | 2781 | should be 0 |
| `__pycache__` in the tree | **present** | any |
| frontends | 2 | a third without a parity test |
| test : source | 0.09 | < 0.08 |

---

## HOW MUCH

> **Match the effort to the blast radius, and the evidence to the claim.**

| Change | Required evidence |
|---|---|
| Help text | green suite |
| A new command | tests in **both** frontends + an exit-code test |
| A delegation target | the gate resolves it |
| An exit code | a test — scripts depend on these |
| An output format | the format-coverage matrix |
| Removing a command | a deprecation note; users have scripts |

The CLI's asymmetry: **its users are scripts.** A human notices a changed
message; a pipeline notices a changed exit code only by failing silently three
steps later.
