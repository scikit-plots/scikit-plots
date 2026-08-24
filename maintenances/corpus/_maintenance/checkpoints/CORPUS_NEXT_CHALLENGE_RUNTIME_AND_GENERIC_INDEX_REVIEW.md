# Corpus Next Challenge Review
## First-Class Runtime Materialization + Generic Vector Index Configuration

Status: **review / implementation plan**
Scope: `scikitplot.corpus`
Primary goals:

1. Promote the gallery-local `RuntimeCorpus` / `materialize_plan()` concept into the Corpus submodule as a supported runtime boundary.
2. Make vector-index configuration backend-generic instead of growing backend-specific fields such as `annoy_*` indefinitely.
3. Preserve existing behavior and compatibility while establishing a cleaner long-term architecture.

---

# 1. Executive decision

The next Corpus increment should add a **first-class runtime compilation layer**:

```text
FluentCorpus
    ↓
CorpusPlan
    ↓
materialize_plan(...)
    ↓
RuntimeCorpus
    ↓
run / add / search / export / close
```

At the same time, retrieval configuration should evolve from:

```text
RetrievalConfig
├── search policy
├── backend selector
└── Annoy-specific constructor fields
```

toward:

```text
RetrievalConfig
├── search policy
├── backend selector
└── generic index/backend kwargs
          ↓
VectorIndexBackend
```

The change must be **additive first**.

Do not immediately change:

```python
FluentCorpus.build() -> CorpusPlan
```

into:

```python
FluentCorpus.build() -> RuntimeCorpus
```

because that would silently change an already-tested public contract.

Instead add:

```python
FluentCorpus.materialize(...)
materialize_plan(...)
RuntimeCorpus
```

and keep `build()` as the validated-plan boundary during the first compatibility cycle.

---

# 2. Verified current baseline

The latest reviewed Corpus source already establishes several important contracts.

## 2.1 `FluentCorpus` is declarative

Current `_plan.py` explicitly states:

```text
configuration calls select WHICH component
configuration-call order does not define WHEN it executes
configuration performs no network/model/backend initialization
```

`FluentCorpus.build()` currently:

```text
validate
→ raise on plan problems
→ return CorpusPlan
```

Its own developer note says that wiring the plan to concrete component construction is the next increment.

This makes a first-class materializer a natural continuation rather than a parallel architecture.

---

## 2.2 The current plan already has the correct domains

Current configuration domains are:

```text
source
reader
normalizer
chunker
enricher
embedder
storage
index
retrieval
export
```

This is already enough to describe a complete runtime.

The missing piece is a canonical translation from these fragments into concrete runtime components.

---

## 2.3 `BuilderConfig.index_kwargs` currently means `RetrievalConfig` kwargs

Current builder behavior is effectively:

```python
idx_cfg = RetrievalConfig(**cfg.index_kwargs)
index = RetrievalIndex(config=idx_cfg)
index.build(documents)
```

Therefore current `BuilderConfig.index_kwargs` does **not** mean:

```text
kwargs passed directly to a vector index backend
```

It means:

```text
kwargs passed to RetrievalConfig
```

This naming is now ambiguous after the `SimilarityIndex` → `RetrievalIndex` redesign.

---

## 2.4 `RetrievalConfig` mixes retrieval policy and Annoy construction

Current `RetrievalConfig` contains generic retrieval fields:

```text
top_k
match_mode
semantic_threshold
keyword_threshold
hybrid_alpha
rrf_k
use_normalized_text
case_sensitive
backend
```

but also backend-specific fields:

```text
annoy_n_trees
annoy_metric
annoy_search_k
annoy_impl
annoy_dtype
annoy_index_dtype
```

This does not scale.

Adding another backend with ten tunables should not require adding ten more fields to `RetrievalConfig`.

---

## 2.5 The backend selector is already close to generic

Current `select_backend(...)` already accepts:

```python
**kwargs
```

and forwards them to backend constructors.

The remaining specialization is that Annoy parameters are separately named in the selector signature and manually forwarded by `RetrievalIndex`.

So the implementation is already near a generic constructor-kwargs design.

---

# 3. Challenge A — promote RuntimeCorpus into the Corpus submodule

## 3.1 Current gap

The real-data FluentCorpus gallery currently needs a local helper conceptually equivalent to:

```python
plan = fluent.build()
runtime = materialize_plan(plan)
result = runtime.pipeline.run(...)
runtime.storage.save_batch(...)
runtime.index.build(...)
runtime.index.search(...)
```

That helper belongs in `scikitplot.corpus`, not in gallery code.

The gallery should eventually become:

```python
runtime = (
    FluentCorpus()
    .source(...)
    .reader(...)
    .normalizer(...)
    .chunker(...)
    .enricher(...)
    .embedder(...)
    .storage(...)
    .index(...)
    .retrieval(...)
    .export(...)
    .materialize()
)

result = runtime.run()
hits = runtime.search("...")
```

---

# 4. Proposed public runtime API

Add a new public module:

```text
scikitplot/corpus/
├── _runtime.py
```

with public exports:

```python
RuntimeCorpus
RuntimePolicy
materialize_plan
```

Recommended initial public surface:

```python
from scikitplot.corpus import (
    RuntimeCorpus,
    RuntimePolicy,
    materialize_plan,
)
```

---

# 5. `RuntimeCorpus`

Recommended initial shape:

```python
@dataclass
class RuntimeCorpus:
    plan: CorpusPlan
    pipeline: CorpusPipeline

    storage: StorageBase | None = None
    index: RetrievalIndex | None = None

    retrieval_config: RetrievalConfig | None = None
    export_config: Any = None

    capabilities: dict[str, Any] = field(default_factory=dict)

    _closed: bool = False
```

Do **not** make this frozen.

It owns runtime state.

The immutable object is `CorpusPlan`; `RuntimeCorpus` is intentionally operational.

---

# 6. RuntimeCorpus responsibilities

`RuntimeCorpus` should own orchestration of already-resolved runtime components.

Recommended methods:

```python
runtime.run(source=None)
runtime.add(source, ...)
runtime.search(query, ...)
runtime.query_storage(...)
runtime.export(...)
runtime.close()
```

Context-manager support:

```python
with fluent.materialize() as runtime:
    result = runtime.run()
    hits = runtime.search("...")
```

Must guarantee:

```text
close() is idempotent
__exit__ calls close()
owned temporary resources are released
external caller-owned resources are not closed unless ownership was transferred
```

---

# 7. RuntimeCorpus must not become another CorpusBuilder

This is a critical architecture rule.

Do not duplicate:

```text
reader dispatch
normalization implementation
chunking implementation
enrichment implementation
embedding implementation
storage implementation
retrieval math
backend distance conversion
export implementation
```

`RuntimeCorpus` should orchestrate existing components.

Correct dependency direction:

```text
RuntimeCorpus
    ↓
CorpusPipeline
StorageBase
RetrievalIndex
export_documents
```

Not:

```text
RuntimeCorpus
    ↓
new second implementation of each stage
```

---

# 8. `materialize_plan(plan)`

Recommended public function:

```python
def materialize_plan(
    plan: CorpusPlan,
    *,
    policy: RuntimePolicy | None = None,
    registry: ComponentRegistry | None = None,
) -> RuntimeCorpus:
    ...
```

Its job is:

```text
validate plan
→ resolve fragment specifications
→ validate cross-component compatibility
→ capability probe
→ construct runtime components
→ return RuntimeCorpus
```

It must **not** process the source.

Construction and execution remain separate.

---

# 9. Materialization phases

Recommended phases:

```text
Phase M0
validate CorpusPlan

Phase M1
resolve declarative fragments

Phase M2
validate resolved component contracts

Phase M3
probe required capabilities

Phase M4
construct runtime objects

Phase M5
return RuntimeCorpus
```

This separation is important for useful diagnostics.

Do not collapse every failure into:

```text
RuntimeError("could not build corpus")
```

---

# 10. Fragment resolution rules

Current `CorpusPlan` accepts arbitrary fragment values.

A runtime compiler therefore needs deterministic resolution rules.

Recommended precedence:

```text
1. already-constructed compatible instance
2. explicit config/spec dataclass
3. registered short name
4. compatible component class
5. explicit Python factory/callable
```

Do not auto-import arbitrary FQCN strings from untrusted serialized input.

Example:

```python
FluentCorpus().chunker(ParagraphChunker(...))
```

→ use existing instance.

Example:

```python
FluentCorpus().chunker(ParagraphChunkerConfig(...))
```

→ construct `ParagraphChunker(config)`.

Example:

```python
FluentCorpus().chunker("paragraph")
```

→ resolve through the Corpus component registry.

Example:

```python
FluentCorpus().chunker(MyChunker)
```

→ instantiate only after contract validation.

---

# 11. Add typed resolver helpers

Do not put one huge resolver inside `materialize_plan()`.

Recommended internal structure:

```text
_runtime.py
_runtime_resolvers.py
```

or:

```text
_runtime/
├── __init__.py
├── _runtime.py
├── _resolvers.py
└── _policy.py
```

Potential internal functions:

```python
_resolve_source(...)
_resolve_reader(...)
_resolve_normalizer(...)
_resolve_chunker(...)
_resolve_enricher(...)
_resolve_embedder(...)
_resolve_storage(...)
_resolve_index(...)
_resolve_retrieval(...)
_resolve_export(...)
```

Each resolver must have one contract and focused tests.

---

# 12. Runtime policy

Materialization is the first point where side effects may become possible.

Add an explicit policy object rather than scattering booleans.

Recommended:

```python
@dataclass(frozen=True)
class RuntimePolicy:
    allow_network: bool = False
    allow_model_download: bool = False
    allow_native_backends: bool = True
    allow_subprocess: bool = False
    capability_policy: str = "strict"
```

Exact fields may evolve, but the architectural requirement is important:

```text
configuration policy
≠
execution policy
```

Gallery/CI usage should be able to request:

```python
RuntimePolicy(
    allow_network=False,
    allow_model_download=False,
)
```

without monkeypatching unrelated components.

---

# 13. `FluentCorpus.materialize()`

Add:

```python
def materialize(
    self,
    *,
    policy: RuntimePolicy | None = None,
    registry: ComponentRegistry | None = None,
) -> RuntimeCorpus:
    return materialize_plan(
        self.build(),
        policy=policy,
        registry=registry,
    )
```

This gives a natural user API:

```python
runtime = (
    FluentCorpus()
    .source(...)
    .chunker(...)
    .embedder(...)
    .storage(...)
    .index(...)
    .retrieval(...)
    .materialize()
)
```

---

# 14. Keep `build()` backward compatible

First release:

```python
FluentCorpus.build() -> CorpusPlan
FluentCorpus.materialize() -> RuntimeCorpus
```

Do not overload:

```python
build(runtime=True)
```

That makes the meaning of `build()` context-dependent.

If a future major release wants:

```python
build() -> RuntimeCorpus
```

that should be a separately reviewed compatibility decision.

---

# 15. Runtime execution semantics

Recommended `run()`:

```python
result = runtime.run()
```

uses:

```text
explicit source argument
        ↓
or plan.source
        ↓
CorpusPipeline.run(...)
        ↓
optional storage save
        ↓
optional index build
        ↓
return BuildResult / RuntimeResult
```

Do not silently rebuild an index on every search unless documented.

Prefer:

```text
run
→ build corpus state once

search
→ query current generation
```

---

# 16. Runtime result type

Do not overload `PipelineResult` with every runtime concern.

Consider:

```python
@dataclass
class RuntimeResult:
    pipeline_result: PipelineResult
    documents: tuple[CorpusDocument, ...]
    storage_count: int | None
    index_generation: IndexGeneration | None
    plan_fingerprint: str
```

However this is optional for the first increment.

A minimal first release may return the existing `PipelineResult` while storing runtime state on `RuntimeCorpus`.

---

# 17. Challenge B — make index configuration backend-generic

## 17.1 Current problem

This:

```python
RetrievalConfig(
    backend="annoy",
    annoy_metric="angular",
    annoy_n_trees=20,
    annoy_search_k=-1,
    annoy_impl="auto",
)
```

is practical for one backend but not extensible.

A future backend should not require:

```text
hnsw_m
hnsw_ef_construction
hnsw_ef_search
qdrant_...
milvus_...
some_future_backend_...
```

to be added directly to `RetrievalConfig`.

---

# 18. Immediate additive API — `index_kwargs`

Add:

```python
index_kwargs: Mapping[str, Any] = field(default_factory=dict)
```

to `RetrievalConfig`.

Preferred new usage:

```python
RetrievalConfig(
    match_mode="hybrid",
    backend="annoy",
    index_kwargs={
        "metric": "angular",
        "n_trees": 20,
        "search_k": -1,
        "impl": "auto",
        "dtype": "float32",
        "index_dtype": "uint64",
    },
)
```

For another backend:

```python
RetrievalConfig(
    backend="my_backend",
    index_kwargs={
        "parameter_a": 10,
        "parameter_b": "value",
    },
)
```

This is the minimum useful genericization.

---

# 19. Preserve old Annoy fields

Keep existing fields initially:

```text
annoy_n_trees
annoy_metric
annoy_search_k
annoy_impl
annoy_dtype
annoy_index_dtype
```

Existing user code must continue to work.

New documentation should prefer:

```python
backend="annoy",
index_kwargs={...}
```

Old syntax remains supported:

```python
backend="annoy",
annoy_n_trees=20,
```

---

# 20. Legacy-to-generic normalization

Centralize compatibility conversion.

Example internal helper:

```python
def _resolved_index_kwargs(config: RetrievalConfig) -> dict[str, Any]:
    kwargs = dict(config.index_kwargs)

    if config.backend == "annoy":
        legacy = {
            "metric": config.annoy_metric,
            "n_trees": config.annoy_n_trees,
            "search_k": config.annoy_search_k,
            "impl": config.annoy_impl,
            "dtype": config.annoy_dtype,
            "index_dtype": config.annoy_index_dtype,
        }

        for key, value in legacy.items():
            if key not in kwargs and value is not None:
                kwargs[key] = value

    return kwargs
```

Do this once.

Do not reproduce compatibility mapping in:

```text
RetrievalIndex
CorpusBuilder
MCP
gallery examples
CLI
```

---

# 21. Conflict rule

Never silently accept contradictory old and new configuration.

Example:

```python
RetrievalConfig(
    backend="annoy",
    annoy_n_trees=10,
    index_kwargs={"n_trees": 50},
)
```

The compatibility layer needs a documented policy.

Preferred rule:

```text
if a legacy value is non-default
and the equivalent generic value is also present
and values differ
→ raise ValueError
```

Do not silently choose one.

During the transition, default-valued legacy fields may be treated as implicit when a generic key is present.

---

# 22. Stop hard-coding valid backend names in RetrievalConfig

Current validation contains a fixed tuple conceptually equivalent to:

```text
auto
annoy
faiss
voyager
bruteforce
brute
```

That prevents third-party/custom backend names.

After generic backend registration exists:

```python
RetrievalConfig(backend="my_backend")
```

should be schema-valid.

Availability/name resolution belongs at runtime/backend-selection time.

So validation layers should become:

```text
RetrievalConfig.__post_init__
→ validate type / non-empty backend identifier

backend resolver
→ validate registered name

runtime materialization/build
→ validate availability
```

---

# 23. Extend backend selection beyond strings

Recommended backend spec:

```python
VectorBackendSpec = (
    str
    | VectorIndexBackend
    | type[VectorIndexBackend]
    | Callable[..., VectorIndexBackend]
)
```

Preferred accepted forms:

```python
RetrievalConfig(
    backend="annoy",
    index_kwargs={...},
)
```

```python
RetrievalConfig(
    backend=MyBackend,
    index_kwargs={...},
)
```

```python
RetrievalConfig(
    backend=my_backend_factory,
    index_kwargs={...},
)
```

For an already-created instance:

```python
RetrievalConfig(
    backend=my_backend_instance,
)
```

If an instance is supplied together with constructor kwargs:

```text
raise
```

because constructor configuration can no longer be applied safely.

---

# 24. Backend registry

The current backend registry is a private `_BACKENDS` mapping.

To support generic index names cleanly, add a reviewed public registration seam.

Recommended direction: extend the existing `ComponentRegistry`.

Example:

```python
registry.register_vector_backend(
    "hnswlib",
    HnswlibBackend,
)
```

Then:

```python
RetrievalConfig(
    backend="hnswlib",
    index_kwargs={
        "m": 16,
        "ef_construction": 200,
    },
)
```

Alternative public helpers may wrap the registry:

```python
register_vector_backend(...)
get_vector_backend(...)
list_vector_backends(...)
```

But avoid maintaining a second unrelated global registry if the existing Corpus registry can own the category.

---

# 25. Capability snapshot must follow the registry

Current capability reporting reads the private backend mapping.

If vector backends become registerable:

```text
capability_snapshot()
```

must enumerate the registry, not a frozen built-in dict.

Preserve:

```text
AVAILABLE
ABSENT
BROKEN
```

and alias reporting.

Third-party backend registration must not break capability snapshot generation.

---

# 26. Recommended longer-term separation: `VectorIndexConfig`

`RetrievalConfig` currently mixes:

```text
retrieval/search policy
vector-index construction policy
```

The generic `index_kwargs` addition fixes the immediate extensibility problem.

The cleaner long-term design is:

```python
@dataclass(frozen=True)
class VectorIndexConfig:
    backend: VectorBackendSpec = "auto"
    kwargs: Mapping[str, Any] = field(default_factory=dict)
```

and:

```python
@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 10
    match_mode: str = "semantic"
    semantic_threshold: float = 0.0
    keyword_threshold: float = 0.0
    hybrid_alpha: float = 0.5
    rrf_k: int = 60
    use_normalized_text: bool = True
    case_sensitive: bool = False
```

Then the Fluent plan becomes semantically clean:

```python
(
    FluentCorpus()
    .index(
        VectorIndexConfig(
            backend="annoy",
            kwargs={
                "metric": "angular",
                "n_trees": 20,
            },
        )
    )
    .retrieval(
        RetrievalConfig(
            match_mode="hybrid",
            top_k=5,
            hybrid_alpha=0.5,
        )
    )
)
```

This matches the existing two domains:

```text
index
retrieval
```

instead of storing two different `RetrievalConfig` objects in both domains.

---

# 27. Recommended migration sequence for RetrievalConfig

Do not jump directly to removing backend fields.

Use three stages.

## Stage B1 — additive generic kwargs

Add:

```python
RetrievalConfig.index_kwargs
```

Keep:

```text
backend
annoy_*
```

Internally normalize both to backend constructor kwargs.

Documentation encourages generic kwargs.

---

## Stage B2 — introduce `VectorIndexConfig`

Add:

```python
VectorIndexConfig
```

Allow:

```python
RetrievalIndex(
    config=retrieval_config,
    index_config=vector_index_config,
)
```

Fluent `index` domain should prefer `VectorIndexConfig`.

Fluent `retrieval` domain should prefer `RetrievalConfig`.

Legacy `RetrievalConfig.backend` and `annoy_*` remain accepted.

---

## Stage B3 — compatibility retirement only after evidence

Only after:

```text
docs migrated
gallery migrated
MCP migrated
CLI migrated
external deprecation window elapsed
```

consider deprecating:

```text
RetrievalConfig.annoy_*
```

Do not remove them during the runtime-materializer increment.

---

# 28. Fix the `BuilderConfig.index_kwargs` naming ambiguity

Current:

```python
BuilderConfig(
    build_index=True,
    index_kwargs={
        "backend": "annoy",
        "annoy_n_trees": 20,
    },
)
```

means:

```text
construct RetrievalConfig from this dict
```

If `RetrievalConfig` gains its own `index_kwargs`, this becomes awkward:

```python
BuilderConfig(
    index_kwargs={
        "backend": "annoy",
        "index_kwargs": {
            "n_trees": 20,
        },
    },
)
```

That is legal but poor UX.

---

# 29. Add a canonical BuilderConfig retrieval field

Recommended additive API:

```python
@dataclass
class BuilderConfig:
    ...
    retrieval_config: RetrievalConfig | None = None

    # compatibility
    index_kwargs: dict[str, Any] = field(default_factory=dict)
```

Resolution:

```text
if retrieval_config is provided
    use it

elif legacy index_kwargs is provided
    RetrievalConfig(**index_kwargs)

else
    RetrievalConfig()
```

If both are provided:

```text
raise configuration conflict
```

Do not silently merge them.

New user code:

```python
BuilderConfig(
    embed=True,
    build_index=True,
    retrieval_config=RetrievalConfig(
        match_mode="hybrid",
        backend="annoy",
        index_kwargs={
            "n_trees": 20,
            "metric": "angular",
        },
    ),
)
```

Old code continues:

```python
BuilderConfig(
    build_index=True,
    index_kwargs={
        "backend": "annoy",
        "annoy_n_trees": 20,
    },
)
```

---

# 30. MCP cross-module impact

The current MCP Annoy integration uses:

```text
BuilderConfig.index_kwargs
```

and sets Annoy-specific fields there.

Migration order:

```text
Corpus compatibility support first
→ MCP tests remain green unchanged
→ migrate MCP to retrieval_config + index_kwargs
→ remove duplicated Annoy naming from MCP
```

MCP must still preserve:

```text
one Corpus-owned vector index
same embedding model for corpus/query
no duplicate distance→score conversion
```

Do not let generic index configuration reintroduce a second MCP-owned ANN index.

---

# 31. RetrievalIndex changes

Current dense build path manually does conceptually:

```python
select_backend(
    cfg.backend,
    annoy_metric=cfg.annoy_metric,
    annoy_n_trees=cfg.annoy_n_trees,
    ...
)
```

Replace with one generic seam:

```python
backend = select_backend(
    cfg.backend,
    **cfg.resolved_index_kwargs(),
)
```

or, after `VectorIndexConfig`:

```python
backend = select_backend(
    index_config.backend,
    **index_config.kwargs,
)
```

That should be the **only** constructor forwarding point.

---

# 32. `select_backend` changes

Current selector is almost generic but still has Annoy-specific named parameters.

Target:

```python
def select_backend(
    backend: VectorBackendSpec = "auto",
    **backend_kwargs: Any,
) -> VectorIndexBackend:
    ...
```

Rules:

```text
"auto"
→ resolve first available registered backend

registered string
→ resolve class and construct with kwargs

backend class
→ verify subclass and construct with kwargs

factory
→ call with kwargs and validate returned object

backend instance
→ require no constructor kwargs
→ use instance directly
```

Explicit unavailable registered backend:

```text
still fail fast
```

`auto`:

```text
still allowed to fall back
```

These current semantics must not regress.

---

# 33. Constructor-kwargs error quality

Generic forwarding can create poor Python errors such as:

```text
TypeError: __init__() got an unexpected keyword argument 'foo'
```

Wrap this with backend context:

```text
invalid index_kwargs for backend 'faiss':
unexpected option 'foo'
```

Preserve the original exception as `__cause__`.

Do not swallow backend constructor failures.

---

# 34. Security rule for generic backends

Generic extensibility must not become arbitrary code execution from untrusted configuration.

Safe configuration source:

```text
registered symbolic backend name
+ JSON-safe kwargs
```

Python-only trusted source may additionally use:

```text
class
factory
instance
```

Do not deserialize:

```text
"some.package.ClassName"
```

into an imported executable class merely because it appears in untrusted JSON.

If FQCN loading is ever supported, require an explicit trusted policy/allowlist.

---

# 35. Serialization and fingerprints

`CorpusPlan` fingerprints component fragments.

For generic backend configs, prefer frozen dataclasses:

```python
VectorIndexConfig(
    backend="annoy",
    kwargs={...},
)
```

But mutable `dict` fields are poor fingerprint material.

Recommended representation:

```python
@dataclass(frozen=True)
class VectorIndexConfig:
    backend: str = "auto"
    kwargs: Mapping[str, Any] = ...
```

Before hashing/serialization:

```text
sort mapping keys
normalize aliases
normalize backend name
reject non-deterministic values when persistence/fingerprint is requested
```

Do not let object memory addresses become plan identity.

---

# 36. Runtime materializer + generic index integration

The two challenges should meet at one clear seam.

Example future user flow:

```python
runtime = (
    FluentCorpus()
    .source("hamlet.txt")
    .reader("text")
    .normalizer(...)
    .chunker(...)
    .enricher(...)
    .embedder(...)
    .storage(...)
    .index(
        VectorIndexConfig(
            backend="annoy",
            kwargs={
                "metric": "angular",
                "n_trees": 20,
            },
        )
    )
    .retrieval(
        RetrievalConfig(
            match_mode="hybrid",
            top_k=5,
        )
    )
    .export(...)
    .materialize()
)

result = runtime.run()
hits = runtime.search("death and dreams")
```

That is the desired big picture.

---

# 37. Runtime ownership model

`RuntimeCorpus` must explicitly track ownership.

Example:

```python
RuntimeCorpus(
    storage=caller_owned_storage,
)
```

must not automatically close the caller's database unless ownership was transferred.

Recommended internal ownership record:

```python
_owned_components: set[str]
```

or typed lifecycle wrappers.

At minimum document:

```text
materializer-created component
→ RuntimeCorpus owns it

caller-supplied instance
→ caller owns it by default
```

---

# 38. Runtime generation identity

A runtime should expose:

```text
plan fingerprint
embedding manifest
index generation
storage generation/version where available
```

Recommended properties:

```python
runtime.plan_fingerprint
runtime.index_generation
runtime.embedding_manifest
```

This helps detect:

```text
stale search result
wrong index
wrong embedding generation
runtime built from a different plan
```

---

# 39. Empty and partial-state behavior

Runtime execution must preserve current fail-soft rules.

Examples:

```text
zero documents
→ do not build vector index
→ keep observable result status

missing optional OCR backend
→ source-specific failure
→ do not fabricate documents

invalid embeddings
→ preserve existing dense-disable / sparse-continuation policy

explicit missing vector backend
→ fail fast
```

Do not let RuntimeCorpus turn existing structured partial success into one generic exception.

---

# 40. WASM / JupyterLite

Runtime materialization must not assume:

```text
native threads
subprocess
memory mapping
native Annoy
filesystem persistence
unrestricted networking
```

Recommended browser-safe plan:

```python
VectorIndexConfig(
    backend="bruteforce",
)
```

The runtime compiler should allow capability-driven plans without changing search semantics.

No fake backend result should be generated merely because a native backend is unavailable.

---

# 41. Proposed module layout after implementation

Recommended:

```text
scikitplot/corpus/
├── _plan.py
├── _runtime.py
├── _runtime_resolvers.py
├── _runtime_policy.py
├── _corpus_builder.py
├── _pipeline.py
├── _similarity/
│   ├── _similarity.py
│   ├── _backends.py
│   └── ...
├── _registry/
│   └── ...
└── tests/
    ├── test__plan.py
    ├── test__runtime.py
    └── ...
```

If runtime code grows beyond one focused file:

```text
_runtime/
├── __init__.py
├── _runtime.py
├── _resolvers.py
└── _policy.py
```

is preferable.

Do not start with a package unless complexity actually requires it.

---

# 42. Proposed public exports

Add:

```python
RuntimeCorpus
RuntimePolicy
materialize_plan
```

Potential second-wave exports:

```python
VectorIndexConfig
register_vector_backend
list_vector_backends
```

Keep low-level:

```python
VectorIndexBackend
select_backend
```

public only if the project intends them as extension APIs.

---

# 43. Test plan — RuntimeCorpus

Add focused tests before gallery migration.

## RUNTIME-01 plan validation

```text
invalid plan
→ materialize_plan raises before component construction
```

## RUNTIME-02 no source execution during materialization

```text
materialize
→ no source read/download
```

## RUNTIME-03 correct component resolution

Cover:

```text
instance
config dataclass
registered name
class
factory
```

## RUNTIME-04 immutable plan preserved

```text
runtime.plan == input plan
runtime.plan_fingerprint == plan.fingerprint
```

## RUNTIME-05 real local flow

```text
text source
→ chunk
→ enrich
→ embed
→ storage
→ index
→ search
```

## RUNTIME-06 empty corpus

```text
no docs
→ no index build
→ explicit state
```

## RUNTIME-07 partial source failure

```text
some sources fail
→ successful evidence preserved
```

## RUNTIME-08 lifecycle

```text
close twice
→ safe

context manager
→ cleanup

caller-owned storage
→ not closed unexpectedly
```

## RUNTIME-09 offline policy

```text
network/model-download forbidden
→ no accidental external access
```

## RUNTIME-10 browser-safe configuration

```text
bruteforce backend
→ no native ANN requirement
```

---

# 44. Test plan — generic index configuration

## INDEXCFG-01 legacy Annoy compatibility

All current tests using:

```text
annoy_metric
annoy_n_trees
annoy_search_k
annoy_impl
annoy_dtype
annoy_index_dtype
```

remain green.

## INDEXCFG-02 generic Annoy kwargs

```python
RetrievalConfig(
    backend="annoy",
    index_kwargs={"n_trees": 5},
)
```

must reach the Annoy backend constructor.

## INDEXCFG-03 legacy/generic conflict

Contradictory explicit values:

```text
raise
```

## INDEXCFG-04 generic custom backend

Registered custom backend:

```text
construct
build
query
```

without modifying `RetrievalConfig`.

## INDEXCFG-05 custom class/factory

If supported:

```text
class
factory
instance
```

resolve correctly.

## INDEXCFG-06 unknown kwarg

Must produce an actionable backend-specific error.

## INDEXCFG-07 auto policy

Preserve:

```text
Annoy → FAISS → Voyager → brute-force
```

unless the registered policy is intentionally redesigned.

## INDEXCFG-08 explicit unavailable backend

Still:

```text
raise RuntimeError
```

No silent downgrade.

## INDEXCFG-09 capability snapshot

Registered backends appear once; aliases do not inflate counts.

## INDEXCFG-10 score contract

Every backend still returns:

```text
cosine similarity [-1, 1]
higher is better
deterministic descending order
```

---

# 45. Builder compatibility tests

Add:

```text
BuilderConfig.index_kwargs legacy path
→ still works
```

Add:

```text
BuilderConfig.retrieval_config new path
→ works
```

Both supplied:

```text
→ explicit conflict error
```

No silent precedence.

---

# 46. Cross-module verification

After Corpus tests pass, verify:

```text
scikitplot.mcp
scikitplot.annoy integration
CLI
Sphinx gallery
JupyterLite/WASM smoke
```

Especially MCP:

```text
must still consume the one Corpus-owned retrieval index
```

---

# 47. Documentation changes

Add a user-facing runtime section:

```text
FluentCorpus
→ plan
→ materialize
→ run
```

Explain clearly:

```text
FluentCorpus configuration is side-effect-free.
materialize() constructs runtime components.
run() processes sources.
```

Add vector backend examples:

```python
RetrievalConfig(
    backend="annoy",
    index_kwargs={"n_trees": 20},
)
```

and later:

```python
VectorIndexConfig(...)
```

Avoid teaching backend-specific top-level fields as the preferred modern style.

---

# 48. Gallery changes after implementation

The current real-data FluentCorpus gallery should be simplified.

Remove its local:

```python
RuntimeCorpus
materialize_plan
```

and replace with:

```python
runtime = fluent.materialize()
```

That gallery then becomes a regression test for the public runtime API.

Keep one explicit section showing:

```text
plan
→ runtime
```

so new users understand the boundary.

---

# 49. Implementation waves

## Wave R0 — freeze current contracts

Before code changes:

```bash
pytest -q scikitplot/corpus/tests/test__plan.py
pytest -q scikitplot/corpus/_similarity/tests/test__backends.py
pytest -q scikitplot/corpus/_similarity/tests/test__similarity.py
```

Record current public behavior.

---

## Wave R1 — runtime core

Implement:

```text
RuntimeCorpus
materialize_plan
RuntimePolicy minimal form
```

No FluentCorpus API change yet.

Focused tests only.

---

## Wave R2 — Fluent integration

Add:

```python
FluentCorpus.materialize()
```

Keep:

```python
build() -> CorpusPlan
```

Update docs and real-data gallery.

---

## Wave R3 — generic `RetrievalConfig.index_kwargs`

Implement generic forwarding while preserving all old Annoy fields.

Centralize legacy normalization.

No custom registration yet if that would make the patch too broad.

---

## Wave R4 — vector backend registration

Extend component registry or add a reviewed unified vector-backend registry seam.

Update:

```text
select_backend
capability_snapshot
aliases
auto-selection
```

---

## Wave R5 — BuilderConfig cleanup

Add canonical:

```python
retrieval_config
```

Keep legacy:

```python
index_kwargs
```

Migrate Corpus docs/examples.

---

## Wave R6 — cross-module migration

Update:

```text
MCP
Annoy integration
CLI
gallery
maintenance docs
```

Run full regressions.

---

# 50. Priority classification

## P1 — Runtime compiler boundary

Why P1:

```text
FluentCorpus currently describes a complete architecture
but cannot directly become an executable runtime.
```

This forces users/examples to write ad-hoc materializers, risking multiple incompatible execution paths.

---

## P1 — generic vector-index kwargs

Why P1:

```text
RetrievalConfig currently hard-codes Annoy implementation details.
```

This blocks clean extension to future backends and conflates retrieval policy with backend construction.

---

## P2 — backend registration

Important for ecosystem extension, but can follow generic kwargs if needed.

---

## P2 — BuilderConfig naming migration

Necessary for API clarity but should not block the initial runtime materializer.

---

# 51. Compatibility matrix

| Existing behavior | Required result |
|---|---|
| `FluentCorpus.build()` returns `CorpusPlan` | Preserve |
| Fluent calls perform no I/O | Preserve |
| duplicate domain defaults to error | Preserve |
| plan order independence | Preserve |
| `RetrievalConfig(backend="annoy", annoy_n_trees=...)` | Preserve |
| `BuilderConfig.index_kwargs` | Preserve as compatibility path |
| explicit unavailable backend raises | Preserve |
| `auto` backend may fall back | Preserve |
| unified cosine score contract | Preserve |
| invalid dense embeddings may disable dense leg observably | Preserve |
| optional backend capability discovery is lazy | Preserve |

---

# 52. Recommended final user API

Short form:

```python
runtime = (
    FluentCorpus()
    .source("docs/")
    .reader("auto")
    .chunker("paragraph")
    .embedder("all-MiniLM-L6-v2")
    .storage("memory")
    .index(
        VectorIndexConfig(
            backend="annoy",
            kwargs={"n_trees": 20},
        )
    )
    .retrieval(
        RetrievalConfig(
            match_mode="hybrid",
            top_k=5,
        )
    )
    .materialize()
)

runtime.run()
results = runtime.search("How does this work?")
```

Transitional form before `VectorIndexConfig`:

```python
runtime = (
    FluentCorpus()
    .source("docs/")
    .embedder("all-MiniLM-L6-v2")
    .index(
        RetrievalConfig(
            backend="annoy",
            index_kwargs={
                "n_trees": 20,
                "metric": "angular",
            },
        )
    )
    .retrieval(
        RetrievalConfig(
            match_mode="hybrid",
            top_k=5,
        )
    )
    .materialize()
)
```

Legacy form still accepted:

```python
RetrievalConfig(
    backend="annoy",
    annoy_n_trees=20,
)
```

but no longer the preferred documentation style.

---

# 53. Architectural end state

The desired end state is:

```text
                    ┌──────────────────────┐
                    │     FluentCorpus     │
                    │ declarative / frozen │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      CorpusPlan      │
                    │ canonical identity   │
                    └──────────┬───────────┘
                               │
                      materialize_plan()
                               │
                               ▼
                    ┌──────────────────────┐
                    │    RuntimeCorpus     │
                    │ lifecycle + state    │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                     ▼
   CorpusPipeline          StorageBase         RetrievalIndex
                                                      │
                                                      ▼
                                            VectorIndexConfig
                                                      │
                                                      ▼
                                           VectorIndexBackend
                                                      │
                            ┌──────────┬──────────────┼───────────┐
                            ▼          ▼              ▼           ▼
                          Annoy      FAISS         Voyager     custom
```

This preserves the most important Corpus principle:

> Configuration describes intent; runtime materialization resolves capabilities; execution produces evidence.

And for vector retrieval:

> Retrieval policy should not need to know every constructor option of every vector-index implementation.

---

# 54. Acceptance gate

Do not mark this challenge complete until all are true:

```text
[x] RuntimeCorpus is public and documented in the module/checkpoint.
[x] materialize_plan is public and tested.
[x] FluentCorpus.materialize exists.
[x] FluentCorpus.build remains backward compatible.
[x] materialization does not process sources implicitly.
[x] runtime lifecycle/ownership is explicit.
[x] RetrievalConfig supports generic index/backend kwargs.
[x] old annoy_* fields remain functional.
[x] legacy/generic conflicts are observable.
[x] backend constructor forwarding exists in one place.
[ ] arbitrary registered backend can be used without editing RetrievalConfig.
[x] explicit unavailable backend still fails fast.
[x] auto fallback semantics remain intact.
[ ] capability snapshot handles registered backends correctly.
[x] BuilderConfig legacy index_kwargs remains functional.
[x] canonical BuilderConfig retrieval configuration path exists.
[ ] MCP uses the one Corpus-owned index.
[x] no score conversion logic is duplicated.
[ ] real FluentCorpus Hamlet gallery uses public materialization API.
[x] Corpus focused suite passes.
[ ] full Corpus suite passes.
[ ] cross-module MCP/Annoy/CLI checks pass.
[ ] offline Sphinx-Gallery build passes.
[ ] JupyterLite/WASM-safe path remains available.
```

---

# 55. Recommended next implementation step

Start with **Runtime R0/R1 only**:

```text
1. freeze current plan tests
2. add `_runtime.py`
3. implement `RuntimeCorpus`
4. implement `materialize_plan`
5. support the real-data Hamlet plan with public components
6. add lifecycle + no-I/O-during-materialization tests
7. stop
8. review before touching RetrievalConfig
```

Then implement generic retrieval/index configuration as a separate focused wave.

This keeps failures attributable and avoids mixing two architectural changes into one unreviewable patch.

---

# Implementation checkpoint — 2026-08-18

## Status

```text
IMPLEMENTED / FOCUSED VERIFIED
canonical full-suite closure: PENDING declared project test environment
```

## Source anchor

```text
archive: scikit-plots(10).zip
sha256: ee7593f81b35bd90a1dd2ba03691aadd2fa491599bce5a2b85dd70f88b14bb2c
```

## Requirement executed

1. Move the gallery-local `RuntimeCorpus` / `materialize_plan()` seam into the
   Corpus submodule as a public, additive runtime boundary.
2. Add backend-generic index constructor kwargs without removing existing
   Annoy configuration or changing unrelated retrieval behavior.
3. Keep the change inside related Corpus plan/runtime/retrieval/builder sections
   and maintenance records.

## Root cause

`FluentCorpus` could describe a complete plan but had no supported transition to
operational runtime objects, forcing examples/consumers to create ad-hoc
materializers. Separately, `RetrievalConfig` embedded Annoy-specific constructor
fields and `RetrievalIndex` forwarded those fields manually, so each future
backend risked expanding the retrieval policy surface.

## Production files changed

```text
scikitplot/corpus/__init__.py
scikitplot/corpus/_plan.py
scikitplot/corpus/_runtime.py                         NEW
scikitplot/corpus/_corpus_builder.py
scikitplot/corpus/_similarity/_similarity.py
scikitplot/corpus/_similarity/_backends.py
```

## Tests changed

```text
scikitplot/corpus/tests/test__plan.py
scikitplot/corpus/tests/test_orchestration.py
scikitplot/corpus/_similarity/tests/test__similarity.py
scikitplot/corpus/_similarity/tests/test__backends.py
```

## Runtime contract delivered

```text
FluentCorpus.build()
        -> CorpusPlan                         unchanged

FluentCorpus.materialize()
        -> materialize_plan(CorpusPlan)
        -> RuntimeCorpus                     added

RuntimeCorpus
        -> CorpusPipeline
        -> optional StorageBase
        -> optional RetrievalIndex
        -> export_documents
```

Materialization validates/resolves/constructs components but does not read the
configured source. Source I/O starts only in `RuntimeCorpus.run()` / `add()`.
The runtime delegates all actual processing/storage/retrieval/export mechanics
to existing Corpus components.

## Generic index contract delivered

Preferred additive form:

```python
RetrievalConfig(
    backend="annoy",
    index_kwargs={
        "metric": "angular",
        "n_trees": 20,
    },
)
```

Legacy remains valid:

```python
RetrievalConfig(
    backend="annoy",
    annoy_metric="angular",
    annoy_n_trees=20,
)
```

A `VectorIndexBackend` subclass may also be selected directly with
`index_kwargs`; no fifth registry was introduced. Contradictory explicit legacy
and generic values raise instead of choosing silently.

## Builder compatibility

```text
BuilderConfig.retrieval_config  canonical new path
BuilderConfig.index_kwargs      legacy compatibility path retained
both supplied                   explicit configuration error
```

## Verification

Core focused gate:

```text
159 passed, 1 deselected
```

Broader related gate:

```text
277 passed, 8 skipped, 3 deselected
```

Additional API/capability/catalog/deprecation gate:

```text
36 passed
```

Environment observations reproduced on the pristine uploaded source and not
changed by this increment:

```text
4 import-hygiene failures: PIL / requests already imported in harness
1 no-I/O configuration test: requests already present in sys.modules
2 NLP enricher tests: NLTK stopwords data absent; downloads disabled by policy
```

The canonical pytest invocation is not claimable in this harness because the
project configuration requests `sphinx.testing.fixtures` and Sphinx is not
installed here.

## Compatibility impact

```text
FluentCorpus.build return type                  unchanged
existing FluentCorpus conflict/order semantics unchanged
existing RetrievalConfig annoy_* calls         unchanged
existing BuilderConfig.index_kwargs calls      unchanged
explicit missing backend behavior              unchanged
auto fallback order                            unchanged
score/metric/generation semantics               unchanged
```

## Security/resource impact

- default `RuntimePolicy` rejects network sources at execution time unless
  explicitly allowed; existing URL/SSRF validation remains authoritative;
- materialization itself performs no source read/download;
- no automatic model/resource download policy was weakened;
- runtime closes only resources it creates/owns;
- no new registry, subprocess, serialization loader, or arbitrary FQCN import
  path was added;
- generic backend constructor errors propagate with backend context.

## Maintenance updates

Updated in this checkpoint:

```text
MAINTAINING.md
_maintenance/REGISTRY.md
_maintenance/STATE.json
_maintenance/VERIFICATION.md
_maintenance/HISTORY.md
_maintenance/TRACKER_LOGICAL.md
_maintenance/TRACKER.json
_maintenance/TRACKER_PHYSICAL.md
_maintenance/SUBMODULE_STRUCTURE.md
```

## Remaining closure gates

```text
[ ] canonical full Corpus suite in declared test environment
[ ] migrate real FluentCorpus gallery to public .materialize() API
[ ] focused MCP / Annoy / CLI compatibility checks
[ ] offline Sphinx-Gallery build
[ ] JupyterLite/WASM smoke for brute-force runtime path
```

## Exact next action

Run the canonical full Corpus suite in the declared test environment. If green,
mark this maintenance increment closed, then migrate the real-data FluentCorpus
gallery from its local materializer to `FluentCorpus.materialize()` and execute
the cross-module compatibility gates.
