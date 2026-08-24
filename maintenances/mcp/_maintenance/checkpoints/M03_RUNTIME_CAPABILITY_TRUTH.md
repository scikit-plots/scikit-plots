# Run M03 — Runtime capability truth

```text
run_id            M03
date              2026-08-18
source_sha256     119855928dd052165c71efa61fa505aa206726265c7c375943084e64793c4018
scope             _capabilities.py, static vs effective surface, create_server registration
guide             MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md §8
production code   NOT MODIFIED
exit gate         MET — capability model assessed against all six states and five fields;
                  vocabulary decision taken on verified evidence
```

---

## 1. Required states — 2 of 6 expressible

The guide requires capability reporting to distinguish six states. Measured
against `server_runtime_status()`, whose entire vocabulary is
`sdk_present: bool` plus `reason ∈ {None, "python<3.10", "mcp-sdk-not-installed"}`:

| Guide state | Expressible? | How it presents today |
|---|---|---|
| `ABSENT` | **yes** | `sdk_present: false`, `reason: "mcp-sdk-not-installed"` |
| `PRESENT_COMPATIBLE` | **partially** | `sdk_present: true`, `reason: null` — but *never verified*, only assumed |
| `PRESENT_INCOMPATIBLE` | **no** | no version is read, so an out-of-range SDK reports as compatible |
| `BROKEN` | **no** | reports as `PRESENT_COMPATIBLE` — see §2 |
| `MISCONFIGURED` | **no** | no equivalent |
| `UNKNOWN` | **no** | collapsed into `ABSENT` — see §3 |

**`M03-01` (P2).** Four of six states cannot be represented, and the two that can
are not distinguished from the four that cannot — a client receiving
`server_available: true` cannot tell whether that was probed or assumed.

---

## 2. `M03-02` (P2) — a broken SDK reports as compatible

Constructed a present-but-broken install (`mcp/__init__.py` raising `ImportError`,
the shape of a corrupt or ABI-mismatched package):

```console
$ cd /tmp/m03 && python -c "…server_runtime_status()"
   retrieval_available    True
   server_available       True      <-- claims the server can run
   sdk_present            True
   reason                 None      <-- claims nothing is wrong

   actual import: FAILS -> ImportError: simulated BROKEN mcp install
```

The report says `PRESENT_COMPATIBLE`; the truth is `BROKEN`. This is the same
class as `MCP-D01` (§4) but a distinct trigger: D01's is a *false positive from
path shadowing*, this is a *false positive from an unhealthy install*.

`find_spec` answers "is a module of this name locatable", which is neither "is it
importable" nor "is it the right version".

---

## 3. `M03-03` (P2) — probe failure is silently reported as ABSENT

```python
# _capabilities.py:168-171
def _present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
```

An exception during probing means *the probe did not succeed* — which is
`UNKNOWN`. Returning `False` reports it as `ABSENT`, i.e. "not installed, go
install it", which may be false and sends the user down the wrong path.

This directly contradicts the standing rule in `MAINTENANCE_MODEL.md`:

> *"An unverified claim is worse than a narrow one. Prefer `UNKNOWN` or a
> declared degradation over a confident guess."*

The function converts an unverified condition into a confident — and possibly
wrong — claim.

---

## 4. `MCP-D01` — root cause located

M00-R proved D01 at runtime (any directory named `mcp/` on `sys.path` yields
`server_available: true` with no SDK). M03 locates *why* it is unfixable within
the current design:

**Required fields, per guide §8:**

| Field | Present? |
|---|---|
| `sdk_present` | yes |
| `sdk_version` | **MISSING** |
| `sdk_compatible` | **MISSING** |
| `server_available` | yes |
| `reason_code` | present as `reason` — values are machine-readable codes, so this is a *naming* mismatch, not a gap |

Without `sdk_version` there is nothing from which `sdk_compatible` could be
computed, so the declared range `mcp>=2.0.0,<3` is never enforced at runtime.
D01 is therefore not a bug in the probe — it is a **missing field in the model**.
`M03-01`, `M03-02` and `D01` all resolve together or not at all.

---

## 5. The vocabulary decision — adopt Corpus's `CapabilityStatus`

`TRACKER_LOGICAL.md` §5 records that MCP must not define its own capability enum.
Verified that the Corpus vocabulary exists and is sufficient:

```text
scikitplot/corpus/_capabilities.py:106
CapabilityStatus = AVAILABLE | ABSENT | BROKEN | INCOMPATIBLE |
                   MISCONFIGURED | UNREACHABLE | UNKNOWN        (7 states)
```

Mapped against the guide's six:

| Guide | Corpus |
|---|---|
| `ABSENT` | `ABSENT` |
| `PRESENT_COMPATIBLE` | `AVAILABLE` |
| `PRESENT_INCOMPATIBLE` | `INCOMPATIBLE` |
| `BROKEN` | `BROKEN` |
| `MISCONFIGURED` | `MISCONFIGURED` |
| `UNKNOWN` | `UNKNOWN` |

**All six are covered**, with `UNREACHABLE` spare. Adopting the Corpus enum
satisfies guide §8 exactly, with no parallel vocabulary and no new design work.

### Corpus already fixed this exact defect, and wrote down why

From the same file's docstring:

> *"A boolean was not enough. Finding F-R02-05 measured that a backend whose
> `is_available()` **raised** — an installed but corrupt, mis-linked or
> ABI-incompatible native library — reported exactly the same as one that was
> never installed."*

That is `M03-03`, described in Corpus's own words, about Corpus's own code,
already resolved there. MCP's `_present()` reproduces the defect Corpus retired.

This is the strongest available argument for the "consume, don't redefine" rule
in `SUBMODULE_STRUCTURE.md` §4: the cost of a parallel vocabulary is not just
duplication, it is **re-deriving the same bugs**. Recommend citing F-R02-05 in
the eventual ADR.

**Decision:** `server_runtime_status()` should return a `CapabilityStatus` value
plus `sdk_version`, consuming Corpus's enum by value (not by import at module
scope — the boundary still applies; the enum is a `str` enum, so the wire mapping
is trivial).

---

## 6. Static vs effective — the module's strongest area

Both required views exist and are properly separated:

```text
server_capabilities()            kind="static"     what the implementation could expose
effective_server_capabilities()  kind="effective"  what this configuration exposes
```

`effective_server_capabilities()` filters `resources` by their declared
`requires`, and strips the `requires` key from the effective view. Measured
across all eight configurations:

```text
document_reader_enabled=False -> resources: []
document_reader_enabled=True  -> resources: ['docs://chunk/{doc_id}']
```

This exactly matches `create_server`'s `if document_reader is not None:` gate at
`_server.py:366`. **The guide's rule — "Never advertise an unregistered
resource/tool" — is SATISFIED**, and the `requires`-driven derivation is a good
design: the effective view is computed from the static one rather than
maintained beside it.

Both are SDK-free, confirmed in M02.

### `M03-04` (P3) — health route: registered without transport awareness

The one divergence in the eight-configuration matrix:

| transport | health_path | effective reports | `create_server` registers | |
|---|---|---|---|---|
| `stdio` | `/healthz` | no | **yes** | **diverges** |
| `stdio` | `None` | no | no | ok |
| `streamable-http` | `/healthz` | yes | yes | ok |
| `streamable-http` | `None` | no | no | ok |

`create_server` gates only on `if health_path is not None:` (`_server.py:335`) and
**takes no `transport` parameter at all** — it cannot know the transport. The
transport coupling lives in the *caller*: `__main__.py:568` passes
`health_path=config.health_path if config.transport == "streamable-http" else None`.

So the CLI is correct, and `--list-capabilities` never lies. But the invariant
holds **by caller convention, not by construction** — the same pattern that
produced `MCP-M00-05` and `MCP-M00-07`. A library caller using the documented
default `create_server(retriever)` gets `health_path="/healthz"` and an HTTP route
registered on a stdio server, where it is unreachable.

**This compounds `M01-01`.** That default path executes
`from starlette.requests import …` (`_server.py:341`), so the default library call
requires an **undeclared** dependency in order to register a route that the
default transport cannot serve. If the SDK ever stops pulling `starlette` in,
`create_server(retriever)` raises `RuntimeError` for a feature the caller never
asked for.

*Evidence class:* `VERIFIED` by control-flow reading (`_server.py:335,341,347`);
not executed, since `create_server` needs pydantic and the SDK. Runtime
confirmation belongs to M12.

**Recommendation:** make `create_server` transport-aware, or default
`health_path=None` so the route is opt-in. Either removes the divergence *and*
the accidental `starlette` requirement. Sequence with M07 (transport lifecycle).

---

## 7. Status changes

| ID | Before | After |
|---|---|---|
| `MCP-D01` | OPEN (runtime-proven) | **root cause located** — missing `sdk_version`/`sdk_compatible` fields; resolves with `M03-01`/`M03-02` |
| `M03-01` | — | **NEW P2** — 4 of 6 required states inexpressible |
| `M03-02` | — | **NEW P2** — broken SDK reports as compatible |
| `M03-03` | — | **NEW P2** — probe failure reported as ABSENT, not UNKNOWN |
| `M03-04` | — | **NEW P3** — health route registered without transport awareness; compounds `M01-01` |
| guide "never advertise unregistered" | untested | **SATISFIED** |
| capability vocabulary | undecided | **DECIDED** — adopt Corpus `CapabilityStatus` (7 states, covers all 6) |

---

## 8. Run record

```text
run_id                  M03
source_sha256           119855928dd052165c71efa61fa505aa206726265c7c375943084e64793c4018
scope                   runtime capability truth
commands                server_runtime_status/capabilities/effective under blocked deps
                        constructed BROKEN mcp install -> reported PRESENT_COMPATIBLE
                        8-configuration effective-vs-registered matrix
                        Corpus CapabilityStatus verified: 7 states, covers all 6 guide states
confirmed               static/effective split correct; "never advertise unregistered" SATISFIED
new                     M03-01 (P2), M03-02 (P2), M03-03 (P2), M03-04 (P3)
decided                 adopt Corpus CapabilityStatus; add sdk_version + sdk_compatible
production code changed NO
next exact action       M04 (Corpus neutral-result integration). Guide §9 required it to
                        "wait for the Corpus campaign to settle" -- it HAS settled
                        (R00-R16 review, IMPL-01-18 implementation), and this run confirmed
                        the contracts are live in-tree by reading CapabilityStatus directly
                        from scikitplot/corpus/_capabilities.py. So M04 is unblocked and
                        concrete. Its required invariant, "FAILED retrieval != EMPTY
                        retrieval", is ALREADY DISPROVED in this tree by MCP-D03: the
                        DocsRetriever Protocol returns a bare list, and strict=False is the
                        default, so total backend failure returns [] -- exactly the
                        "No matching documentation" lie the guide forbids. M04 must decide
                        the Protocol's return shape, which makes it the highest-value
                        remaining run.
                        M05 remains the pivot for MCP-M00-07 / D02 / D08.
```
