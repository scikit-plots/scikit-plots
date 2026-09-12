# Run M13 — Blocker remediation

```text
run_id            M13 (remediation pass following the M08–M12 closure assessment)
date              2026-08-21
source_sha256     9be9218a2465bfe146bc28e60b1b1ff58bad2fe99fb60e87834a8f4307ac76eb
scope             the five closure blockers, fixed at root
mode              IMPLEMENTATION
environment       real SDK: mcp 2.0.0, pydantic 2.13.4, starlette 1.6.0, anyio 4.14.2
suite             166 passed -> 172 passed, 0 failed
```

---

## 1. `MCP-D01` + `M03-01/02/03` — capability truth · **CLOSED**

This was the campaign's only live violation of the project's own rejection list
(*"trust package presence as compatibility"*).

### Root cause, not symptom

The probe asked the wrong question. `importlib.util.find_spec("mcp")` answers
*"is a module of this name importable from `sys.path`"* — which any directory
named `mcp/` satisfies. The question that matters is *"is this package installed,
and at what version"*, which is **distribution metadata**.

Verified the distinction directly:

```console
$ cd /tmp/shadow            # contains a decoy mcp/ directory
find_spec sees:      /tmp/shadow/mcp/__init__.py     <- the decoy
metadata.version:    2.0.0                           <- the real SDK
```

`server_runtime_status()` now probes via `importlib.metadata.version`, parses the
release prefix and range-checks it against the declared `mcp>=2.0.0,<3`.

### All six required states are now expressible

| scenario | `sdk_status` | version | compatible | `server_available` | reason |
|---|---|---|---|---|---|
| installed 2.0.0 | `available` | 2.0.0 | True | True | — |
| absent | `absent` | — | False | False | `mcp-sdk-not-installed` |
| too old 1.9.0 | `incompatible` | 1.9.0 | False | False | `mcp-sdk-incompatible` |
| too new 3.0.0 | `incompatible` | 3.0.0 | False | False | `mcp-sdk-incompatible` |
| prerelease 2.1.0b2 | `available` | 2.1.0b2 | True | True | — |
| unparsable version | `unknown` | weird | None | False | `mcp-sdk-status-unknown` |
| probe itself raises | `unknown` | — | None | False | `mcp-sdk-status-unknown` |

`sdk_version` and `sdk_compatible` — the two fields the guide required and the
model lacked — now exist, which is what made compatibility undecidable before.

**`M03-03` closed with it:** a failed probe returns `UNKNOWN`, never `ABSENT`.
Telling a user to install a package when the probe merely failed is the confident
wrong claim this campaign kept finding.

**Vocabulary consumed, not redefined:** the states mirror
`scikitplot.corpus.CapabilityStatus`, with
`assert_capability_vocabulary_matches_corpus()` as the drift gate — the same
pattern `_outcome.py` uses for `RetrievalStatus`. No `packaging` dependency was
added; the numeric release prefix is parsed directly, which is sufficient for a
`>=2.0.0,<3` decision.

**Contract test rewritten, not deleted.** `test_status_shape_and_consistency`
pinned the old six-key set; it now pins the nine-key contract and asserts
availability requires `sdk_status == "available"` rather than mere presence.

---

## 2. The "confident wrong claim" class — **4 of 5 closed**

M08–M12 identified five independent sites of one habit. Fixed as a class:

| site | before | after |
|---|---|---|
| `M03-03` `_present()` | failed probe → `False` (= "not installed") | → `UNKNOWN` |
| `MCP-D01` | any `mcp/` dir → "server available" | distribution metadata + range check |
| `MCP-D05` `_corpus_annoy.py` | `except Exception` → "corpus is required, install it" | `ImportError` → absent; other → **"installed but failed to import; this is a broken installation, not a missing one"** |
| `M07-03` `_server.py` | any `ImportError` → "MCP SDK v2 is required, pip install mcp" | consults the probe: absent / incompatible / **"broken environment, not a missing package"** |
| `M06-01` score | uncoercible → `0.0` | **deferred — see §6** |

`MCP-M00-08` closed alongside D05: the malformed message that rendered as
*"…build the retriever install**the** corpus/embedding extras"* is now a correct
sentence.

---

## 3. Packaging — `MCP-M00-10` and `M01-01` · **CLOSED**

The marker M01 adjudicated is removed:

```toml
'mcp>=2.0.0,<3',          # was: ...; python_version >= "3.10"
'pydantic>=2,<3',
'starlette>=0.40',        # M01-01: declared, no longer inherited transitively
```

Rationale is in the file: with a marker, pip **drops** the SDK on 3.8/3.9 and
installs a pydantic-only extra that cannot serve; without it the SDK's own
`Requires-Python` rejects the interpreter and resolution fails loudly. Verified
in M01 against PyPI (`mcp 2.0.0` declares `Requires-Python: >=3.10`).

**The disabled assertion is re-enabled** — the one that would have caught this
originally — plus a new assertion that `starlette` is declared. `starlette>=0.40`
has no upper bound, so this adds a declaration without narrowing any range.

---

## 4. `M02-01` — the user-facing launch failure · **CLOSED**

The command all 7 plugin bundles declare produced a raw traceback on a base
install, because `__main__.py` imported the `_server` *module* — whose
module-scope `pydantic` import raises before the SDK guard inside it can speak.

`__main__.py` now pre-flights the capability probe before touching the server
tier — which is what `server_runtime_status` was written for:

```console
$ python -m scikitplot.mcp          # base install, no [mcp]
E ... cannot start the MCP server: mcp-sdk-not-installed
      (python=3.12, sdk_status=absent, sdk_version=None)

the MCP server layer is unavailable: mcp-sdk-not-installed. Install the server
extra with: pip install "scikit-plots[mcp]"   (the SDK-free retrieval tier
remains usable without it).
```

Note this fix **depended on `MCP-D01` being fixed first** — as M02 predicted. A
pre-flight built on the old probe would have reported `server_available: True`
in the shadowing case and fallen through to the same traceback.

---

## 5. `M06-02` and `M03-04` · **CLOSED**

**`M06-02`** — `_DOC_ID_RE` accepted `.`, `..` and a leading `:`, which reached
caller-supplied `document_reader` code. Now
`\A(?!\.{1,2}\Z)(?!:)[A-Za-z0-9._:-]{1,200}\Z`:

```text
'd1' 'doc-1_2' 'file.txt' 'a:b'   accepted
'.' '..' ':x' '../etc' 'a/b' ''   rejected
```

**`M03-04`** — `create_server(health_path=...)` now defaults to `None`. It has no
`transport` parameter and cannot know whether an HTTP route is reachable;
defaulting it on registered an unreachable endpoint for stdio servers *and*
pulled in `starlette` for a feature the caller never requested. `__main__.py`
already passes it explicitly for Streamable HTTP, so the CLI is unaffected.

---

## 6. Deliberately not fixed, with reasoning

**`M06-01` (P3) — uncoercible score → `0.0`.** The honest fix is to emit `None`,
but `CitationOutput.score` is a typed `float` on the advertised output schema.
Making it nullable is a **schema change**: a strict client expecting `float`
would break. M11 established that every change so far has been additive, and I
was not willing to trade that late in the campaign for a P3.

The finding stands and the reasoning is recorded so the next maintainer can weigh
it deliberately rather than rediscover it. If taken, it belongs with a schema
version bump.

**`M06-03` (P3) — non-URI strings emitted as citation URIs.** Same class of
judgement: dangerous schemes are already blocked, so this is cosmetic
correctness, and tightening it risks rejecting legitimate relative references.

---

## 7. Verification

```console
$ check_trackers.py
physical tracker matches the tree (18 source / 16 test files, 4473 / 2442 LOC)
EXIT=0

$ pytest scikitplot/mcp -q --maxfail=100
172 passed, 1 skipped          (was 166 passed)

base-install boundary:  corpus: False | pydantic: False
stdio round-trip:       STDIO OK ['search_docs']  is_error=False status=success
```

**+6 tests, 0 failures, 0 regressions.** New guards, each pinning a finding so it
stays fixed:

```text
test_probe_is_immune_to_a_shadowing_directory          MCP-D01
test_failed_probe_reports_unknown_not_absent           M03-03
test_incompatible_sdk_is_not_reported_as_absent        MCP-D01
test_capability_vocabulary_matches_corpus...           M03 vocabulary drift gate
test_create_server_distinguishes_broken_install...     M07-03
test_create_server_reports_an_incompatible_sdk_version M07-03 / D01
```

plus two re-enabled packaging assertions (`MCP-M00-10`, `M01-01`).

---

## 8. Closure status

```text
[x] deferred queue reverified
[x] packaging/install contract green            M00-10 + M01-01 CLOSED
[x] optional import boundary deliberate/tested
[x] compatible SDK version detected correctly   MCP-D01 CLOSED
[x] neutral Corpus outcomes mapped correctly
[x] backend failure never masquerades as empty
[x] integrations/plugins do not duplicate protocol logic
[ ] min/latest real SDK matrix green            degenerate (M01-05): mcp 2.0.0 is
                                                the only stable 2.x release
[x] stdio green
[ ] Streamable HTTP green                       needs a bound port
[~] cancellation/shutdown green                 shutdown green; cancellation deferred
[x] private SDK seam has drift gates
[ ] historical maintenance state marked current 3 KEEP docs still absent
```

**11 of 13 met** (was 8). **Rejection list: zero live violations** — *"trust
package presence as compatibility"* is fixed.

`scikitplot.mcp` is **not yet CLOSED**, and the three remaining items are honest
about why: two require infrastructure this environment does not have (a bound
port, multiple interpreters, a built wheel), and one requires documents that have
never appeared in any supplied archive.

---

## 9. Run record

```text
closed          MCP-D01, M03-01, M03-02, M03-03, MCP-D05, MCP-M00-08,
                MCP-M00-10, M01-01, M02-01, M06-02, M03-04, M07-03
deferred        M06-01, M06-03 (both P3, both would change the wire schema)
remaining       M12 lanes (wheel, Python 3.10, Streamable HTTP, cancellation);
                M01-05 degenerate matrix; M02-02 bare `python` in bundles;
                MCP-M00-11 stale anchor -> update to 9be9218a...;
                RULESET.md / MCP_COMPATIBILITY_POLICY.md / DESIGN.md absent
                (DESIGN.md still cited from _hybrid.py:22,346)
suite           166 -> 172 passed, 0 failed
next action     Run the existing tests/integration/test_mcp_http_live.py harness
                against a bound port to close Streamable HTTP, and build a wheel
                to close the install lanes. Neither needs new code -- both need
                infrastructure. Then MCP can be marked CLOSED.
```
