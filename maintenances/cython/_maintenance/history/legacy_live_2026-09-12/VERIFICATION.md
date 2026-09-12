# Verification — `scikitplot.cython`

## Commands

```console
$ python scikitplot/cython/_maintenance/check_trackers.py
$ python -m pytest scikitplot/cython -q -p no:cacheprovider
```

The suite needs a **working toolchain** — Cython and a C compiler. A skipped
build test and a passing one look similar in `-q` output; read the skip count.

## What the gate checks

| Check | Fails when |
|---|---|
| DRIFT | inventory differs from the tree by more than 10% |
| INDEPENDENCE | a sibling `scikitplot` submodule is imported |
| STALE | `__pycache__` present in the source tree |
| TRIPWIRE | test:source below 0.80; a module over 2 500 LOC |

## What a green suite does *not* prove

| Failure mode | Why a green run can miss it |
|---|---|
| a lock probe destroying another process's lock | needs a second process |
| GC removing an artifact mid-load | needs concurrency |
| cache commit racing a writer | needs concurrency |
| a stale lock after abnormal exit | needs a killed process |
| irreproducibility from an unpinned flag | needs two machines, or two toolchains |

`test__interprocess_exclusivity.py` and `test__gc_coordination.py` cover the
first two. The rest are open.

## Evidence standard

- **"I tested it" is insufficient.** Paste the command and its output, including
  the skip count.
- A finding is resolved **with evidence**, never deleted.
- A test is never weakened to make a change pass.
- A security claim requires the strict suite, not the default one.
