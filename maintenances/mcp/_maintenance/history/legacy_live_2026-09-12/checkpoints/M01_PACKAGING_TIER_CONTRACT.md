# Run M01 — Packaging and Python tier contract

```text
run_id            M01
date              2026-08-18
source_sha256     119855928dd052165c71efa61fa505aa206726265c7c375943084e64793c4018
scope             pyproject.toml [mcp] extra, Python tiering, scikitplot/mcp packaging surface
guide             MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md §6
production code   NOT MODIFIED
exit gate         PARTIAL — 6 of 9 checklist items closed; 3 deferred to M12 with cause
```

---

## 0. Correction to M00-R: M01 was not fully blocked

M00-R marked M01 `BLOCKED` pending `MCP_COMPATIBILITY_POLICY.md`. **That was too
strong, and this run corrects it.**

The guide states the packaging prohibition itself, in §6:

```text
No:
mcp; python_version < official SDK floor
```

That is an available source of truth. Combined with deterministic marker
semantics and the SDK's published metadata, it settles `MCP-M00-10` on evidence
without the missing document. What still needs a policy document is *ratifying*
the decision, not *making* it.

**M01 is downgraded from `BLOCKED` to `PARTIAL`.** The correction is recorded as
a lesson in §6.

---

## 1. Checklist status

| Guide §6 item | Status |
|---|---|
| base install | **VERIFIED** — §2 |
| `[mcp]` extra | **VERIFIED, DEFECTIVE** — §3, §4 |
| Python floor | **VERIFIED** — §2 |
| SDK Requires-Python interaction | **VERIFIED** — §3 |
| Pydantic declaration | **VERIFIED** — declared `pydantic>=2,<3`, correct, no marker |
| direct vs transitive dependencies | **DEFECTIVE** — `M01-01`, §4 |
| wheel metadata | **DEFERRED** → M12 (§5) |
| README install commands | **DEFECTIVE** — `M01-03`, §3d |
| CLI install hints | **VERIFIED** — `_server.py:282` names the interpreter floor and the exact range |

---

## 2. Tier contract — holds

```text
scikit-plots requires-python   >=3.8          (no upper bound)
MCP SDK (mcp 2.0.0)            >=3.10         (verified from PyPI metadata)
declared server tier           >=3.10         (intersection — correct)
```

Base dependencies contain **no** `mcp`, `pydantic` or `starlette`; the server
tier is confined to the `[mcp]` extra. Confirmed at runtime in M00-R: importing
`scikitplot.mcp` loads `numpy` and nothing else third-party.

The three-tier model in the guide (base retrieval / server = intersection with
SDK support / prerelease readiness) is **structurally correct in this tree**.
Only the third leg is unevidenced — see `M01-04`.

---

## 3. `MCP-M00-10` adjudicated

### 3a. The current line complies with the guide

```toml
'mcp>=2.0.0,<3; python_version >= "3.10"',
```

Marker evaluation, deterministic:

```text
py3.8  -> mcp requirement DROPPED
py3.9  -> mcp requirement DROPPED
py3.10 -> mcp requirement KEPT
py3.11+ -> KEPT
```

The guide forbids declaring `mcp` **below** the SDK floor. This marker declares
it only at and above the floor. **Guide §6 is satisfied.** My M00 characterisation
— "the extra violates policy" — was imprecise and is corrected here.

### 3b. But the failure mode the in-file comment describes is real

The pyproject comment states the stricter rule:

> *"NO python marker on the SDK line … A marker would instead silently drop the
> SDK and leave a partial (pydantic-only) extra, which does not mean 'server
> installed'."*

The marker evaluation above **confirms exactly that behaviour**: on 3.8/3.9 the
`mcp` requirement is dropped, `pydantic>=2,<3` survives, and
`pip install "scikit-plots[mcp]"` **succeeds** while installing a server extra
that cannot serve.

So the tree implements the position the comment argues against, and the test
that would have caught it is commented out (`MCP-M00-10`). Someone implemented
the opposite of the comment and disabled the assertion rather than reconciling
the two.

### 3c. The no-marker option strictly dominates — no policy document needed

Verified from PyPI: **`mcp` 2.0.0 declares `Requires-Python: >=3.10`.**

Therefore, with the marker removed:

| | marker present (today) | marker removed |
|---|---|---|
| py3.10+ | resolves, server works | resolves, server works |
| py3.8/3.9 | **succeeds**, installs pydantic only, server absent | **fails resolution** — SDK's own `Requires-Python` rejects |
| guide §6 (`no mcp below floor`) | satisfied | satisfied — metadata never declares support below the floor; the SDK rejects |
| pyproject comment | **violated** | satisfied |
| module's fail-closed philosophy | violated | satisfied |

Removing the marker satisfies **both** rules; keeping it satisfies only the
weaker one. There is no trade-off to adjudicate, so the missing policy document
is not required to choose — only to ratify.

This is consistent with the module's established stance everywhere else: the
private SDK seam fails closed (`_server.py:191`), capabilities must be probed not
assumed, wire validation is strict. A silently-degrading extra is the outlier.

**Recommendation:** remove the marker; re-enable the commented-out assertion in
`test_mcp_import_surface.py`; record an ADR. Implementation stays closed until
the runs finish.

### 3d. `M01-03` (P3) — README documents the behaviour but not the trap

`README.md:41`:

```bash
pip install "scikit-plots[mcp]"     # installs mcp>=2.0.0,<3 on Python >= 3.10
```

Accurate about 3.10+, silent about 3.8/3.9 — where the same command **succeeds**
and yields no server. A 3.9 user gets a success message and a broken install with
nothing pointing at the cause. If the marker is removed (§3c) this line becomes
correct as written; if it is kept, the README must state the partial-extra
outcome explicitly.

---

## 4. `M01-01` (P2) — `starlette` is an undeclared direct dependency

`starlette` is **imported by MCP's own code** and **not declared anywhere** in
`pyproject.toml`:

```text
_server.py:341,344              from starlette.requests import Request
                                from starlette.responses import JSONResponse   (call time)
tests/test_server.py:16         from starlette.requests import Request         (MODULE SCOPE)
```

The reliance is explicit in a source comment:

```python
except ImportError as exc:  # pragma: no cover - installed with MCP SDK
    raise RuntimeError("Starlette is required for the HTTP health endpoint") from exc
```

*"installed with MCP SDK"* is a statement that the project depends on a
**transitive** dependency of the SDK. That violates the standing dependency
policy (*"Do not rely on transitive dependencies"*) and the guide's §6
`direct vs transitive dependencies` item.

**Impact today is latent, not active** — the SDK does currently pull `starlette`
in, so nothing is broken. It becomes real the moment the SDK restructures its
dependencies, and it fails in two different ways:

- `create_server(..., health_path="/healthz")` raises `RuntimeError` at runtime;
- `tests/test_server.py` fails at **collection**, because its import is at module
  scope and unguarded — the same defect class as `MCP-M00-01`.

**Recommendation:** declare `starlette` in the `[mcp]` extra with a documented
range, or make the health endpoint's dependency explicit as its own extra. Note
this is an *addition* to the extra, so it does not narrow any compatibility range.

---

## 5. Deferred to M12 — with cause

The guide's M01 test list requires:

```text
clean wheel │ base import │ wheel[mcp] │ unsupported interpreter requested-extra
behavior │ minimum supported SDK │ latest supported SDK
```

`base import` is **VERIFIED** (M00-R §5a). The rest are `DEFERRED`, not skipped:

| Test | Why deferred |
|---|---|
| clean wheel / `wheel[mcp]` | scikit-plots builds via meson-python with Cython/C++ extensions; no build toolchain or wheel was produced in this environment. A wheel-metadata claim without a built wheel would be `INFERRED` presented as `VERIFIED`. |
| unsupported-interpreter extra behaviour | needs a real 3.8/3.9 interpreter and a resolver run. §3a establishes the marker semantics deterministically, which is the *cause*; the end-to-end `pip` behaviour remains unobserved. |
| minimum / latest supported SDK | no SDK installed here; this is M12's real-SDK closure matrix by design. |

### `M01-05` (P3) — the min/latest SDK matrix is currently degenerate

PyPI shows **`mcp` 2.0.0 is the only stable 2.x release** (the other 2.x entries
are `a1`–`a3`, `b1`, `b2` prereleases). So under `mcp>=2.0.0,<3`, *minimum
supported SDK == latest supported SDK == 2.0.0* today.

The standing policy (*"tested against both minimum and latest supported
versions"*) is therefore trivially satisfiable now and will silently stop being
so on the first 2.1 release. Worth pinning as an explicit M12 gate rather than
discovering it later.

---

## 6. `M01-04` (P3) — Python 3.14 is claimed but unevidenced

`pyproject.toml:169` declares:

```toml
"Programming Language :: Python :: 3.14",
```

`requires-python = ">=3.8"` has no upper bound, so 3.14 installs are permitted
and the classifier advertises support. Nothing in the tree shows the suite has
been run on 3.14.

This sits directly against the standing evidence rule in `MAINTENANCE_MODEL.md`
§"HOW MUCH": *"Claiming SDK compatibility → tested against that SDK, or the claim
is not made."* The same standard should apply to interpreter claims. The guide's
`next-Python prerelease readiness` item is the place to either produce the CI
evidence or drop the classifier.

Note the base-dependency block already carries 3.13-specific handling
(`numpy <= 2.4.6; python_version == '3.13'`) but nothing for 3.14 — so the
readiness question is live, not theoretical.

---

## 7. Lesson captured

```markdown
## 2026-08-18: A missing document is not automatically a blocker

**Context:** M00-R marked M01 BLOCKED pending MCP_COMPATIBILITY_POLICY.md.

**Issue:** M01 was substantially answerable. The review guide states the
packaging rule itself (§6), and marker semantics plus the SDK's published
Requires-Python settle MCP-M00-10 deterministically. Blocking the run deferred
work that evidence already supported.

**Root cause:** I treated "the document that would adjudicate is missing" as
equivalent to "the question is unanswerable", without first checking whether
another source of truth in the evidence precedence list covered it. The guide
sits at precedence rank 1-3 (source snapshot, deterministic reproducers, package
metadata); the missing policy is rank 5 (source-side maintenance documents).

**Prevention rule:**
- When: a run is about to be marked BLOCKED on a missing document.
- Then: walk the evidence precedence list top-down and record, per blocked item,
  which rank could settle it. Mark BLOCKED only for items no higher-ranked
  source can reach; mark the rest PARTIAL and proceed.
- Verified by: any BLOCKED item in STATE.json names the precedence ranks checked
  and why each failed.
```

---

## 8. Run record

```text
run_id                  M01
source_sha256           119855928dd052165c71efa61fa505aa206726265c7c375943084e64793c4018
scope                   packaging / Python tier contract
commands                marker evaluation via packaging.markers (deterministic)
                        PyPI metadata lookup: mcp 2.0.0, Requires-Python >=3.10
                        pyproject dependency + classifier inspection
                        starlette usage scan across scikitplot/mcp/**
confirmed               MCP-M00-10 (refined and adjudicated, §3)
new                     M01-01 (P2), M01-03 (P3), M01-04 (P3), M01-05 (P3)
corrected               M00-R's "M01 BLOCKED" -> PARTIAL; MCP-M00-10 wording
                        ("violates the guide" -> "satisfies the guide's letter
                        while producing the failure its own comment identifies")
deferred                wheel build/metadata, unsupported-interpreter resolver
                        behaviour, min/latest SDK matrix -> M12
production code changed NO
next exact action       M02 (import / optionality contract) is UNBLOCKED and is the
                        natural next run: its subject matter -- __all__, __getattr__,
                        __dir__, integration imports, star-import safety -- is already
                        evidenced by MCP-D04, MCP-D02 and MCP-M00-12, all of which
                        M02 must formally decide. The three missing KEEP documents
                        remain outstanding and still gate ratification of §3c, but
                        they do not gate M02.
```
