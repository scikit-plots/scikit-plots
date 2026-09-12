# Corpus Verification Guide

This file defines the minimum verification behavior for future changes.

## 1. Canonical rule

A historical green suite does not prove a changed source tree.

Always record:

```text
source hash
Python
platform
installed optional dependencies
pytest configuration
command
pass/skip/xfail/fail counts
```

## 2. Historical implementation baseline

IMPL-18 reported:

```text
3206 passed
27 skipped
4 xfailed
```

under the canonical project pytest configuration.

Use this only as a regression reference.

## 3. Canonical full-suite gate

Prefer the project configuration after installing the declared test
dependencies.

Conceptually:

```console
python -m pytest scikitplot/corpus
```

Do not silently bypass missing required pytest plugins and then call the result
the canonical suite.

## 4. Focused post-implementation smoke gate

For changes touching the new contracts, include the relevant focused suites:

```text
tests/test__diagnostics.py
tests/test__retrieval.py
tests/test__agentic.py
tests/test__graph.py
tests/test__artifact.py
tests/test__embedding_manifest.py
tests/test__plan.py
tests/test__catalog.py
tests/test__retrievers.py
_similarity/tests/test__capability_contract.py
_similarity/tests/test__score_provenance.py
```

Add subsystem-specific tests for any changed ingestion/storage/security path.

## 5. Environment caveat discovered during this maintenance refresh

In the current execution harness, a fresh Python subprocess already contains
`requests` in `sys.modules` before importing Corpus.

Therefore the focused test:

```text
TestValidation.test_configuration_performs_no_io
```

can fail with:

```text
configuration imported requests
```

even though the module was preloaded by the environment rather than Corpus.

A control subprocess confirmed:

```text
python -c "import sys; print('requests' in sys.modules)"
```

already reports `True` in this harness.

Do not weaken the source test because of this environment artifact.

Verify import hygiene in a genuinely clean interpreter/environment, or make the
test compare modules newly imported by the configuration action instead of
assuming an empty interpreter.

## 6. Mandatory semantic gates

### Retrieval outcomes

Test:

```text
successful hit
successful empty
single-leg failure
partial/hybrid degradation
all legs failed
cancelled
budget-exhausted graph traversal
```

No failed execution may masquerade as `EMPTY`/complete success.

### Filtering

Every requested filter must be:

```text
applied
equivalently emulated
or rejected
```

Never silently ignored.

### Embedding/index compatibility

Test:

```text
same dimension + different embedding manifest
wrong generation
corrupt manifest
wrong ordinal sidecar
wrong metric/threshold scale
```

All incompatible combinations must fail before query.

### Graph

Test:

```text
cycles
high-degree fanout
node/edge/evidence budgets
deadline exhaustion
deterministic ordering
edge provenance
generation consistency
```

### Agentic

Test:

```text
hard step limit
hard retrieval-call limit
deadline
degraded retrieval
no-progress termination
explicit stop reason
no durable memory write by default
```

### Fluent configuration

Test:

```text
nested == fluent canonical plan
independent fragment order commutes
same-domain duplicate errors
explicit replacement works
configuration does no network/model/backend initialization
plan fingerprint stable for equivalent plans
stage order explicit, not inferred from method-call order
```

## 7. Security positive controls

Keep existing guards green for:

```text
SSRF and redirect validation
DNS fail-closed behavior
archive byte/depth budgets
transactional archive publish
XML/DTD/entity hardening
safe resource download policy
atomic writes
safe artifact integrity verification
post-load document validation
```

Do not duplicate these primitives in new code.

## 8. Tracked xfails

Four HIGH-04 tests remain intentionally tracked.

Do not remove or convert them casually.

If one unexpectedly passes:

```text
inspect why
decide the intended API contract
then update the test/registry with evidence
```

## 9. Deferred environment gates

Before claiming full coverage for the corresponding capability, execute:

```text
U-5 network adversarial battery
U-6 optional media-parser adversarial battery
U-7 real I/O fault injection
U-8 multimodal cache-key verification
U-9 native ANN recall/performance
```

## 10. Closure record for a future change

A maintenance change is complete only when its record contains:

```text
source hash
issue/requirement
root cause
changed files
tests
result
compatibility impact
security/resource impact
registry update
state update
```
