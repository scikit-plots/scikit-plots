# `_sphinx_ai_assistant` Logical Contract Tracker

The initial statuses are deliberately conservative. `VIOLATED_OR_UNPROVED` means existing review evidence suggests a gap or the desired invariant cannot yet be proved from the supplied source without a dedicated checkpoint. It prevents aspirational documentation from becoming false source truth.

| ID | Contract | Owner | Status | Invariant | Proof |
|---|---|---|---|---|---|
| `AIA-C01` | `BuildConfiguration` | `__init__.py` | **PARTIAL** | Sphinx config is validated, serializes only intended client-safe fields, and has explicit ownership. | config inventory + serialization tests |
| `AIA-C02` | `ClientSafeSerialization` | `__init__.py -> window globals` | **VIOLATED_OR_UNPROVED** | Production secrets never enter generated HTML/browser globals. | secret leakage positive-control tests |
| `AIA-C03` | `BrowserRuntime` | `_static/ai-assistant.js` | **PARTIAL** | Browser owns presentation/state only; security decisions remain server-enforced. | browser/service bypass tests |
| `AIA-C04` | `RepresentationConsumer` | `_sphinx_llm facade/static artifacts` | **PLANNED** | Assistant consumes canonical static representation instead of owning canonical conversion. | static-first integration tests |
| `AIA-C05` | `PromptAuthority` | `model/proxy service` | **VIOLATED_OR_UNPROVED** | Server owns system policy; client/page content cannot set authoritative system role. | direct API prompt-role tests |
| `AIA-C06` | `EndpointDiscovery` | `proxy GET / + browser client` | **PARTIAL** | Discovery is non-secret, schema-versioned, and client/server drift is tested. | discovery contract tests |
| `AIA-C07` | `ProxyRouting` | `_hf_spaces_proxy` | **VIOLATED_OR_UNPROVED** | Server credentials are destination-bound to approved upstreams. | malicious BACKEND_URL test |
| `AIA-C08` | `CorsOriginPolicy` | `proxy + worker` | **VIOLATED_OR_UNPROVED** | Production CORS is explicit least privilege; parallel relays match policy. | origin allow/deny tests |
| `AIA-C09` | `ShareAuthorization` | `proxy persistence/routes` | **VIOLATED_OR_UNPROVED** | Read ID/capability does not imply edit/delete authority. | share read/edit capability tests |
| `AIA-C10` | `ClientIdentity` | `proxy/worker` | **VIOLATED_OR_UNPROVED** | Forwarded identity is accepted only from trusted proxy boundary. | spoofed forwarding tests |
| `AIA-C11` | `RequestLimits` | `proxy/model/worker` | **VIOLATED_OR_UNPROVED** | Body/resource limits prevent pre-validation memory exhaustion. | oversize streaming/buffering tests |
| `AIA-C12` | `ModelServicePolicy` | `_hf_spaces_model` | **VIOLATED_OR_UNPROVED** | Direct model endpoint enforces role/policy invariants independently of UI. | direct completion tests |
| `AIA-C13` | `FeedbackProvenance` | `proxy dataset logic` | **VIOLATED_OR_UNPROVED** | Persisted feedback/training data has consent/provenance and is never implicitly trusted. | dataset schema/provenance tests |
| `AIA-C14` | `SettingsRegistry` | `browser settings` | **PARTIAL** | Client settings have one validated schema/write path with explicit persistence and sensitivity. | Node registry tests + integration guard |
| `AIA-C15` | `CorpusBoundary` | `cross-submodule` | **PLANNED** | Corpus owns retrieval/evidence semantics; assistant consumes published retrieval contracts. | import/boundary tests |
| `AIA-C16` | `MCPBoundary` | `cross-submodule` | **PLANNED** | MCP owns protocol transport; assistant does not invent parallel MCP semantics. | integration boundary tests |
| `AIA-C17` | `RuntimeHtmlFallback` | `browser/representation compatibility` | **CURRENT_LEGACY** | DOM conversion remains observable fallback during staged static migration, not canonical target. | fallback-selection tests |
| `AIA-C18` | `ServiceParity` | `proxy/model/worker` | **VIOLATED_OR_UNPROVED** | Alternative service paths cannot bypass auth/CORS/routing/model-policy controls. | cross-path security matrix |

## Status vocabulary

- `HOLDS` — current source and a regression gate prove it.
- `PARTIAL` — useful implementation exists but the entire invariant is not proved.
- `VIOLATED_OR_UNPROVED` — evidence indicates risk or source truth has not yet been adjudicated; do not claim safe.
- `CURRENT_LEGACY` — intentionally still live during migration, not desired end-state.
- `PLANNED` — new architecture not yet live.
- `DEFERRED` / `SUPERSEDED` — explicit lifecycle states.
