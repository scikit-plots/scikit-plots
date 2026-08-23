# Verification — `scikitplot.memmap`

## 1. The commands

```console
$ python scikitplot/memmap/_maintenance/check_trackers.py
$ <build>                     # meson/ninja per the project's build docs
$ python -m pytest scikitplot/memmap -q -p no:cacheprovider
```

Expected from the gate:

```text
scikitplot.memmap: tracker matches the tree (4 source / 2 test files, ...)
```

## 2. What the gate checks

| Check | Fails when |
|---|---|
| DRIFT | recorded inventory differs from the tree by more than 10% |
| SHARED-SOURCE | a `cdef extern` names a header absent from `cexternals/_annoy/src/` |
| FAMILY | `cexternals` imports a Python layer built on top of it |

**This submodule requires a compiled build.** Unlike Corpus, a green gate does
not imply a working module — the extension must be built. A tracker check on an
unbuilt tree is necessary, not sufficient.

## 3. Evidence standard

Inherited unchanged from the Corpus and MCP campaigns:

- **"I tested it" is insufficient.** Paste the command and its output.
- A finding is marked resolved **with evidence**, never deleted.
- A test is never weakened to make a change pass.
- A capability claim requires a probe, not an assumption.

After a deliberate structural change:

```console
$ python scikitplot/memmap/_maintenance/check_trackers.py --update
```

then regenerate `TRACKER_PHYSICAL.md` to match.
