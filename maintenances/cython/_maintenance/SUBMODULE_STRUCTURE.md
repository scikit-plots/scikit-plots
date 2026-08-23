# Submodule Structure — `scikitplot.cython`

## 1. Role

**A runtime Cython/C build service.** Source in, loaded module out.

## 2. Where a new thing goes

| Adding | Put it |
|---|---|
| A build profile | `_profiles.py` — **and pin what it changes** |
| A toolchain flag | `_pins.py` first, then wherever it is used |
| A security rule | `_security.py`, **and the strict suite** |
| A template | `_templates/<family>/`, with its `.meta.json` and a containment test |
| A capability probe | `_api.py`; report `UNKNOWN`, never assume |
| Lock behaviour | `_lock.py` — **and an interprocess test** |
| Cache or staging logic | `_cache.py`/`_loader.py`, with rollback covered |
| A public symbol | `_public.py` **and** `__init__.pyi` — parity is tested |
| Anything importing a sibling submodule | **nowhere** — this submodule has no edges, and that is a feature |

## 3. Structural debt

### `__pycache__/` in the source tree

Observation **O-6** again — third submodule to carry it. A packaging fix, not a
code one.

### `_builder.py` at 2 097 lines

The largest module and a trust boundary: it constructs the arguments handed to a
compiler subprocess. Under the tripwire (2 500), but this is the file where size
and risk coincide, so it is the one to split first if it grows.

A natural seam if it does: **argument construction** (pure, testable) separate
from **subprocess invocation** (effectful).

### 306 template files

Not debt — an asset, and correctly treated as *test inputs* rather than
documentation. Worth stating explicitly because the instinct on seeing 306 files
is to prune them, and pruning a template removes a test case.

## 4. Review checklist

```text
[ ] Does it import a sibling submodule?           -> reject; this module has no edges
[ ] Does it touch _security.py?                   -> strict suite + what is now permitted
[ ] Does it touch _lock.py?                       -> an INTERPROCESS test, not threading
[ ] Does it touch cache/loader?                   -> staging, commit AND rollback
[ ] Does it add a template?                       -> .meta.json + containment test
[ ] Does it change the public surface?            -> .pyi parity
[ ] Does it add an unpinned toolchain flag?       -> pin it
[ ] python _maintenance/check_trackers.py         -> exit 0
[ ] pytest scikitplot/cython -q                   -> green
```

## 5. Directions, with prerequisites

| Direction | Needs first | Value |
|---|---|---|
| Remove `__pycache__` from the archive | a packaging fix | Closes O-6, open across three submodules |
| Split argument construction out of `_builder.py` | nothing | Makes the trust boundary pure and testable |
| Publish the security gate's permitted-source rules | a decision | Users currently discover the boundary by hitting it |
| A capability report in Corpus's vocabulary | Corpus's `CapabilityStatus` (built) | `BROKEN` vs `ABSENT` for a missing compiler is exactly that distinction |

The last one is the only place this submodule touches another campaign's work,
and it needs no edge: it can adopt the *vocabulary* without importing the module.

## 6. What *not* to do

| Tempting | Why not |
|---|---|
| Import this from `corpus` or `annoy` to compile a kernel on demand | It would create the project's first edge into a subprocess-invoking, filesystem-locking service. It needs its own review run, not a convenience import. |
| Prune templates to reduce file count | Each is a test input |
| Validate source after compiling | The gate must run first, or it is not a gate |
| Test lock behaviour with threads | The known defect was cross-**process** |
| Add a build flag without a pin | Silent irreproducibility |
