# Submodule Structure — `scikitplot._cli`

## 1. Role

**The command-line front door.** It owns *how* a command is reached, never what
it does.

## 2. Where a new thing goes

| Adding | Put it |
|---|---|
| A built-in command | `_commands/`, **and tests in both frontends** |
| A delegated command | `registry.py` — a string target; the gate must resolve it |
| A new output format | `output.py` + the format-coverage matrix |
| A new exit code | `exit_codes.py`, **and document it** — scripts depend on these |
| A third frontend | `_frontends/`, with a parity test, or do not add it |
| Logic belonging to a submodule | **not here** — `_cli` delegates |

## 3. Structural debt

### `_backup/` — 11 files, 2 781 LOC of unreachable code

The previous CLI: `optparse` and `click` option modules, `_misc.py`, its own
`cli.py` and commands. The three largest files in the submodule are all here.

| Verdict | What |
|---|---|
| **ARCHIVE** → `_maintenance/history/` or out of the tree | all of `_backup/` |

Same verdict as `corpus/_backup`, `annoy/_annoy/_backup` and
`_sphinx_ai_assistant/_static/_backup`. This project has a habit of keeping
history in directories; git already does that, and the directories ship.

### `__pycache__/` in the archive

**O-6**, recorded during the Corpus campaign and routed here. Still present.

### The delegation registry has one entry

`scikitplot.mcp.__main__:main`. The machinery is general; the usage is not yet.
That is fine — but it means the registry's edge cases (malformed target, missing
attribute, submodule that raises on import) are exercised by *tests* rather than
by real targets, which is worth knowing when reading them.

## 4. Review checklist

```text
[ ] Does it import a delegated submodule?          -> reject; that is the contract
[ ] Does it add a delegation target?               -> the gate must resolve it
[ ] Does it change an exit code?                   -> scripts depend on it; test it
[ ] Is the command present in BOTH frontends?      -> or explicitly noted
[ ] Does --help still work with nothing installed?
[ ] python _maintenance/check_trackers.py          -> exit 0
[ ] pytest scikitplot/_cli -q                      -> green
```

## 5. Directions, with prerequisites

| Direction | Needs first | Value |
|---|---|---|
| Archive `_backup/` | nothing | Removes 2 781 LOC of unreachable code |
| Remove `__pycache__` from the archive | a packaging fix | Closes O-6, open since the Corpus campaign |
| Publish an exit-code table | a decision | Scripts currently depend on undocumented values |
| Delegate to `corpus` and the annoy family | those campaigns closing | One front door for the whole package |
| A `doctor` check per delegated submodule | the capability models already built | `scikitplot doctor` could report `BROKEN` vs `ABSENT` using Corpus's `CapabilityStatus` |

The last one is the most interesting: **Corpus already built a seven-state
capability model, and `_cli` already has a `doctor` command.** Connecting them
gives a user a real answer to "why doesn't this work", and neither side needs new
design.
