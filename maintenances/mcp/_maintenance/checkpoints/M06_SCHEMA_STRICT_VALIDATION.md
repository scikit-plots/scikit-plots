# Run M06 — Tool/resource schema and strict validation

```text
run_id            M06
date              2026-08-21
source_sha256     ee7593f81b35bd90a1dd2ba03691aadd2fa491599bce5a2b85dd70f88b14bb2c
scope             search_docs schema, output contract, resources, error mapping, seam
guide             MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md §11
mode              REVIEW ONLY — no production code changed
exit gate         MET for every item testable without the SDK; 2 items deferred to M12
```

---

## 1. Audit results

| Guide item | Verdict |
|---|---|
| `search_docs` input schema | **PASS** — §2 |
| closed / `extra="forbid"` semantics | **PARTIAL** — MCP's own models verified; SDK-side deferred (§6) |
| structured output | **PASS** — §3 |
| citations | **PASS** with one caveat (`M06-01`) |
| resource URIs | **PASS** with one caveat (`M06-02`) |
| health routes | covered in M03 (`M03-04` open) |
| error mapping | **DEFECT** — `M06-04`, §5 |
| unknown tool arguments | **DEFERRED** to M12 (needs a real SDK) |
| non-finite scores | **PASS** — §3 |
| oversized query | **PASS** — §2 |
| oversized `k` | **PASS** — §2 |
| unsafe URI / content | **PASS** with one caveat (`M06-03`) |
| private seam isolated to one module/function | **PASS** — §6 |

M05 paid off here: validation is now a single Tier-L entry point
(`SearchCoordinator.validate`), so this audit examined one implementation rather
than reconciling two tiers.

---

## 2. Input validation — complete and fail-closed

Every boundary rejected with an actionable `ValueError`:

```text
empty query        query must not be empty
whitespace query   query must not be empty
non-str query      query must be a string
oversized query    query must be at most 1024 characters
k = 0 / -1 / 21    k must be between 1 and 20
k = True           k must be an integer        <- bool rejected before int
k = 2.5 / "3"      k must be an integer
"transport", 3     OK
```

`bool` is rejected explicitly rather than passing as `int` — the classic Python
trap, handled.

### Declared schema matches enforcement exactly

```text
declared: query {type str, min_length 1, max_length 1024}
          k     {type int, minimum 1, maximum 20, default 5}

query.max_length == MAX_QUERY_CHARS   True
k.maximum        == MAX_RESULTS       True
k.minimum        == enforced minimum  True
query.min_length == enforced minimum  True
```

The capability report is derived from the same constants the coordinator
enforces, so `--list-capabilities` cannot advertise bounds the server will not
apply. Tool annotations are declared and correct: `read_only`, `idempotent`,
`open_world: False`.

---

## 3. Output contract

**Non-finite scores are neutralised.** `nan`, `inf`, `-inf`, a string and `None`
all emerge as JSON-safe values, so no unserialisable float can reach the wire.

**Unsafe URI schemes are stripped:**

```text
javascript:alert(1)      -> ''
data:text/html,<script>  -> ''
file:///etc/passwd       -> ''
vbscript:x               -> ''
https://ok.test/y        -> 'https://ok.test/y#a'
HtTpS://Ok.Test/z        -> 'https://Ok.Test/z#a'    (scheme case-normalised)
```

**Untrusted content is sanitised:** `NUL` and ANSI escape bytes are removed, and
every passage carries the untrusted-reference notice. A prompt-injection payload
embedded in retrieved text is delivered as inert data.

**Citation invariant holds:** `count == len(passages) == len(citations)`.

### `M06-01` (P3) — an invalid score is coerced to a *legitimate* value

Every rejected score becomes `0.0`:

```text
in=nan -> 0.0    in=inf -> 0.0    in='abc' -> 0.0    in=None -> 0.0
```

`0.0` is a valid similarity score, so "this score was unusable" is now
indistinguishable from "this hit genuinely scored zero". A client ranking or
thresholding on `score` cannot tell them apart.

This is the campaign's recurring principle again — `MAINTENANCE_MODEL.md`:
*"Prefer `UNKNOWN` or a declared degradation over a confident guess."* Emitting
`None` for an uncoercible score states the truth; `0.0` asserts something false.

Note this is the same defect shape as `M03-03` (`_present()` returning `False`
for a failed probe) and `MCP-D05` (`BROKEN` reported as `ABSENT`). Three
independent sites, one habit.

### `M06-03` (P3) — non-URI strings pass through as citations

```text
'not a uri'  ->  'not a uri#a'
```

Dangerous schemes are blocked, so this is not a security hole; but an arbitrary
string is emitted in `source_uri` and an anchor appended to it, which a client
may render as a link. A citation whose URI is not a URI is a broken citation.
Either reject it (empty, as with blocked schemes) or validate that it parses as
a relative reference.

---

## 4. Resources

`_read_resource` validates before doing anything: `_DOC_ID_RE` is
`\A[A-Za-z0-9._:-]{1,200}\Z`, applied with `fullmatch`, then the result is
length-capped to `_MAX_RESOURCE_CHARS` (20 000) and passed through the same
sanitising path as search results.

Traversal is blocked because `/` is not in the character class:

```text
'../../etc/passwd'  rejected      'a/b'   rejected
'a b'               rejected      ''      rejected
'd1#frag'           rejected      'd1?q=1' rejected
'x'*201             rejected      '%2e%2e' rejected
```

### `M06-02` (P3) — `.` and `..` are accepted

```text
'..'  accepted=True
```

Both characters are in the allowed class, so the bare strings `.` and `..` pass
validation and reach the **caller-supplied** `document_reader`. Full traversal
needs `/` and is blocked, so this is not exploitable through MCP itself — but
`document_reader` is third-party code, and a reader doing `Path(base) / doc_id`
receives a directory reference rather than a document identifier.

`:` is likewise accepted, which on Windows permits drive-relative (`C:`) or
alternate-data-stream forms in a reader that concatenates paths.

Defence in depth: MCP validates the identifier precisely so the reader does not
have to. Excluding `.`, `..` and a leading `:` is a one-line tightening that
narrows no legitimate identifier.

---

## 5. `M06-04` (P2) — the same failure maps to two different wire outcomes

Measured with one retriever whose backend is down, varying only `strict`:

```text
strict=False  ->  RESULT   isError=True   retrieval_status='failed'
strict=True   ->  RAISES   RuntimeError: documentation search is temporarily unavailable
```

Identical underlying condition; two different protocol outcomes. A client sees
either a successful tool call carrying an error-flagged result, or a thrown
tool error — decided by a retriever construction flag it cannot observe.

`strict` was designed as a *diagnostic* control (re-raise so failures surface in
development). After M04 it also silently selects the wire error channel, because
M04 gave the non-strict path a truthful `FAILED` status while the strict path
still raises.

This is not a regression introduced by M04 — the divergence existed before, but
was invisible because the non-strict path lied (`EMPTY`, `isError: False`). M04
made both paths truthful and thereby exposed that they are two paths.

**Recommendation:** decide which channel is canonical.
`structuredContent.retrieval_status` + `isError` is the richer one — it carries
per-leg detail, and `RuntimeError` collapses everything to one opaque sentence.
Keep `strict` for logging/`exc_info` only, and stop letting it change the
contract. This belongs with M07 (transport lifecycle), where error propagation
to the transport is decided.

---

## 6. Private SDK seam — isolated, as required

The guide requires the private hardening seam be confined to one
module/function. Verified by scanning all private-SDK attribute access:

```text
_server.py:192   getattr(server, "_tool_manager", None)
_server.py:201   getattr(metadata, "fn_metadata", None)
_server.py:202   getattr(metadata, "arg_model", None)
_server.py:210   dict(argument_model.model_config)
_server.py:212   argument_model.model_config = ConfigDict(..., extra="forbid")
_server.py:213   argument_model.model_rebuild(force=True)
```

All six sites are inside `_forbid_unknown_tool_arguments` in `_server.py`.
**Requirement satisfied.** The seventh hit, `_server.py:57`
(`model_config = ConfigDict(extra="forbid")`), is on MCP's own `_ClosedModel`
base — MCP's models are closed by construction, not SDK internals.

`M03`'s residual risk stands: line 212 mutates `model_config` on the
SDK-generated class and forces a rebuild. If a future SDK caches argument models
across servers, that is shared-state mutation. Runtime confirmation needs a real
SDK — M12.

### Deferred to M12, with cause

- **`extra="forbid"` runtime semantics** and **unknown-argument rejection**
  cannot be executed without the MCP SDK installed. Static reading shows the seam
  fails closed (raises `RuntimeError` when any expected private attribute is
  absent), which M00 verified; whether the *resulting schema* rejects unknown
  arguments end-to-end is exactly M12's real-SDK closure matrix.

---

## 7. Carried forward from M04

Corpus's `ErrorRecord.code` / `ErrorCategory` (`SOURCE`, `PARSE`, `VALIDATION`,
`CAPABILITY`) are still **not** mapped onto the wire — MCP's `LegRecord.error`
remains a plain string. M04 deferred this to M06; M06 defers it again, now with a
reason: `M06-04` must be settled first, because the category mapping's
destination depends on which error channel is canonical.

---

## 8. Run record

```text
run_id                  M06
scope                   tool/resource schema and strict validation
commands                input-boundary matrix (11 cases) against SearchCoordinator
                        declared-schema vs enforced-bounds comparison
                        non-finite score / unsafe URI / control-char matrices
                        _DOC_ID_RE acceptance matrix (12 identifiers)
                        error-mapping comparison across strict=False/True
                        private-SDK attribute-access scan
passed                  input validation; schema-enforcement agreement; structured
                        output; non-finite scores; unsafe schemes; content sanitisation;
                        traversal rejection; seam isolation
new                     M06-01 (P3) invalid score coerced to a legitimate 0.0
                        M06-02 (P3) '.' and '..' accepted as document identifiers
                        M06-03 (P3) non-URI strings emitted as citation URIs
                        M06-04 (P2) one failure, two wire outcomes, selected by `strict`
deferred to M12         extra=forbid runtime semantics; unknown-argument rejection
production code changed NO
next exact action       M07 (transport lifecycle). It must decide M06-04 -- which error
                        channel is canonical -- since that is a transport-facing contract,
                        and M03-04 (health route registered without transport awareness,
                        pulling in undeclared starlette) is already queued for the same run.
                        Once M06-04 is settled, map Corpus ErrorCategory onto the chosen
                        channel; that is the last piece of M04's deferred work.
```
