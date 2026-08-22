# Corpus Review and Implementation History

This is a **compact lineage index**, not a live finding register.

Do not use historical counts or old symbol names as current source truth.

## Source and evidence bundles

Current source reviewed for this maintenance refresh:

```text
scikit-plots(7).zip
sha256: 099f9c414ffb6a80648d4a48b78670f243400a18b6362e1d735c44db89c15d13
```

Deep-review evidence:

```text
CORPUS_R.zip
sha256: 6177c7d69b5eccf5089f2b00b87af731fa9599dc08f3585c34e702d7de9483bf
```

Implementation logs:

```text
CORPUS_IMPL.zip
sha256: 42c0e721e8e1a6cfdd6678bb486857ffc8ea64ec34cc2a9c19a1e4ed74d75027
```

## Review campaign

The multi-run review completed R00 through R16.

Final review state recorded:

```text
55 confirmed findings
  4 P1
 39 P2
 12 P3

23 disproofs
56 proposals
159 decisions
5 environment-blocked unknowns
campaign status: COMPLETE
```

Important review result:

```text
plausible success after incomplete execution
```

was identified as the shared failure pattern behind:

```text
unsupported filter behavior
inert hierarchy semantics
silent hybrid-leg degradation
```

That drove the structured diagnostic/outcome contracts implemented later.

## Implementation campaign

Implementation was applied cumulatively through IMPL-18.

### I0 — hygiene / correctness

- deprecated shim cleanup;
- naming normalization;
- registry identity/overwrite/conformance hardening;
- packaging/test/docs/import gates.

### I1 — common contracts

- `ErrorRecord`;
- error-policy extension;
- `RetrievalResponse`;
- per-leg status;
- explicit filter capability handling;
- schema version/hierarchy integrity;
- capability status and `ComponentCatalog`;
- `EmbeddingManifest`;
- composable `CorpusPlan` / `FluentCorpus`;
- generation identity.

### I2 — local vector-index foundation

- backend capability declarations;
- metric/threshold-scale guards;
- `ANNIndexArtifact`;
- versioned ordinal-to-document-ID sidecar.

### I3 — filtering and score semantics

- backend-neutral filter AST;
- native score/metric/rank provenance;
- rank-fusion policy.

### I4 — graph

- G0 derived graph;
- typed/provenance-bearing edges;
- bounded traversal.

### I5 — retrieval legs

- lexical/dense/graph retriever objects;
- per-leg outcomes;
- hit contribution/fusion provenance.

### I6 — agentic retrieval

- `BudgetPolicy`;
- rule-based router;
- bounded retrieve/evaluate/refine loop;
- sufficiency signals;
- explicit stop reasons;
- caller-supplied token accounting where available.

Final IMPL-18 log reported:

```text
3206 passed
27 skipped
4 xfailed
waves I0-I6 COMPLETE
```

## Why old live maintenance files were retired

The previous live files described a pre-implementation architecture using names
such as:

```text
ANNBackend
SearchConfig
SimilarityIndex
SearchResult
```

and old campaign states such as:

```text
17 findings resolved
3 partial
rest open
```

Those descriptions are historically useful but wrong as current continuation
state after IMPL-18.

Therefore they should not remain the fresh-chat authority.

## Historical files retired from the live path

Retire/archive outside the live maintenance set:

```text
_maintenance/scikitplot_corpus_DEEP_SEMANTIC_REVIEW_GUIDE.md
_maintenance/SESSION_LOG.md
_maintenance/CORPUS_SEARCH_RESOLUTION_LOG.md
_maintenance/METHODOLOGY.md
_maintenance/STALE_FILES.md
_maintenance/stale_lifecycle.py
```

Version-control history plus `CORPUS_R.zip` and `CORPUS_IMPL.zip` preserve the
evidence.

## Rule for historical investigation

When a regression references an old finding/proposal/decision:

1. identify the exact historical ID;
2. open the external review/implementation bundle;
3. reproduce against the **current** source;
4. update the current registry rather than resurrecting the old live handbook.


## Post-Corpus project continuation

The completed Corpus campaign intentionally hands off to the remaining module
campaigns in this order:

```text
MCP
→ Annoy
→ CLI
→ cross-module verification
```

The purpose is not ownership dependency; it is review sequencing.

Corpus remains the neutral contract source while the other modules verify their
adapter/backend/presentation boundaries against it.
