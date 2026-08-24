# CORPUS G12 — Public dependency-free local helpers

Date: `2026-08-19`

Status: **VERIFIED / FOCUSED GREEN**

Source authority:

```text
scikit-plots(20260818-204251).zip
SHA-256 4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

## Scope

G12 promotes reusable, dependency-light functionality that previously existed
only as local teaching code or embedded example data:

```text
HAMLET_TEXT
HashEmbedder
SimpleEnricherSpec
SimpleFrequencyEnricher
```

The gallery examples are intentionally **not changed** in this increment.

## Public API

All four names are available directly from the Corpus facade:

```python
from scikitplot.corpus import (
    HAMLET_TEXT,
    HashEmbedder,
    SimpleEnricherSpec,
    SimpleFrequencyEnricher,
)
```

Implementation ownership:

```text
_samples.py                 HAMLET_TEXT
_embeddings/_hashing.py     HashEmbedder
_enrichers/_simple.py       SimpleEnricherSpec, SimpleFrequencyEnricher
```

## `HAMLET_TEXT`

Contract:

```text
bundled public-domain convenience excerpt
no I/O
no network
no optional dependency
not presented as an authoritative scholarly edition
```

It is intended for deterministic tests, examples, smoke checks, and quick local
experiments. `_samples.py` must stay small; it is not a general dataset
subsystem.

## `HashEmbedder`

Contract:

```text
input             iterable[str]
output            float32 ndarray (n_texts, dimension)
hash              BLAKE2b signed feature hashing
tokens            Unicode-aware words + internal apostrophes
case handling      token-first, then casefold
normalization      L2 per non-zero row
network/model I/O  none
```

Use cases:

```text
offline deterministic retrieval baseline
unit/integration tests
examples without model downloads
constrained environments
```

Non-goal:

```text
learned semantic similarity
```

`HashEmbedder` must not be described as equivalent to a transformer/API
embedding model.

## `SimpleEnricherSpec` / `SimpleFrequencyEnricher`

Contract:

```text
optional NLP resources      none
tokenization                Unicode-aware regex
keyword ranking             (-frequency, token) deterministic order
source text                 normalized_text when present, otherwise text
copy-on-write               yes
existing tokens             reused when overwrite=False
existing complete result    original document identity preserved
```

Direct convenience form:

```python
enricher = SimpleFrequencyEnricher(
    min_token_length=3,
    max_keywords=8,
)
```

Declarative Fluent form:

```python
fluent = FluentCorpus().enricher(
    SimpleEnricherSpec(
        min_token_length=3,
        max_keywords=8,
    )
)
```

`RuntimeCorpus.materialize()` resolves `SimpleEnricherSpec` directly to a
`SimpleFrequencyEnricher`; no NLTK/spaCy/model resource is consulted.

## Unicode regression found during G12

The first implementation lowercased text before regex matching. For Turkish
capital dotted I, Python case normalization yields an `i` plus combining dot;
matching after that transformation split the token.

G12 fixes the helper rule to:

```text
match complete token on original Unicode text
    -> casefold the token
    -> length filter
```

The same token-first rule is used by `HashEmbedder`.

## Verification

Focused helper/API/plan gate:

```text
56 passed
1 deselected (known harness: requests already loaded)
```

Broader regression gate covering all embedding tests plus pipeline/plan/API and
new helper tests:

```text
401 passed
1 deselected
```

The deselected test is the pre-recorded
`TestValidation.test_configuration_performs_no_io` harness condition where
`requests` is already loaded by the surrounding source-tree environment.

A broader optional-resource-heavy attempt reached:

```text
589 passed
1 deselected
51 environment failures
```

All 51 failures were existing NLTK `stopwords` resource requirements raising
`ResourceUnavailableError` with managed downloads disabled. G12 does not enable
network/resource downloads to make that suite green.

Public smoke:

```text
HAMLET_TEXT chars        3819
HashEmbedder shape       (1, 32)
SimpleEnricherSpec       Fluent validation PASS
```

Integrated public-helper runtime smoke:

```text
HAMLET_TEXT -> temp source
SimpleEnricherSpec -> RuntimeCorpus materialization
HashEmbedder(128) -> bruteforce hybrid retrieval
documents          12
embedding dimension 128
top query evidence  "To be, or not to be..."
result              PASS
```

Physical tracker:

```text
83 source files
81 test files
57,629 source LOC
32,217 test LOC
tracker PASS
```

## Invariants retained

```text
RuntimeCorpus closed/open/run/add semantics unchanged
NLPEnricher optional-resource behavior unchanged
EmbeddingEngine model/API/cache behavior unchanged
gallery source files unchanged
network policy unchanged
security policy unchanged
```
