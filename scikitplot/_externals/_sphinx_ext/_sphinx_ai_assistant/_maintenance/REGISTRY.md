# `_sphinx_ai_assistant` Active Registry

| ID | Severity | Status | Finding / goal | Evidence | Exact next action | Closure gate |
|---|---|---|---|---|---|---|
| `AIA-001` | **P0** | **OPEN** | Current maintenance target claims must be reconciled with current source | Existing docs contain desired server-only secret policy while source has client token/config escape hatches. | B00/B02: annotate current vs target truth with source evidence. | no desired invariant mislabeled HOLDS |
| `AIA-002` | **P0** | **OPEN** | Canonical representation ownership must move to `_sphinx_llm` | Current `__init__.py` still owns post-build HTML->Markdown/llms generation. | Block migration until producer A11; then B08/B09 staged cutover. | static producer primary + no duplicate canonical owner |
| `AIA-003` | **P0** | **OPEN** | Prompt authority must be server-owned | Browser constructs a system message containing page/context in the anchored source path; direct service policy requires revalidation. | B04 direct endpoint/browser request review. | client cannot select authoritative system role |
| `AIA-004` | **P0** | **OPEN** | Destination-bound credential policy | Existing security review identified configurable backend + server token risk. | B02 reproduce; B05 remediate with allow/bind policy. | malicious destination receives no credential |
| `AIA-005` | **P0** | **OPEN** | CORS parity across proxy/worker | Existing review identified permissive origins in more than one relay. | B02 reproduce; B05 define shared least-privilege contract. | origin allow/deny matrix |
| `AIA-006` | **P0** | **OPEN** | Share read/edit authorization separation | Existing review identified share UUID possession as edit authority risk. | B02 reproduce; B06 capability split. | read locator cannot PATCH/DELETE |
| `AIA-007` | **P0** | **OPEN** | Direct model/relay bypass policy | Multiple service paths can undermine browser-only controls. | B02 map all paths; B04/B05 enforce server parity. | cross-path security matrix |
| `AIA-008` | **P0** | **OPEN** | Feedback/training poisoning/provenance boundary | User feedback may feed persistence/training paths. | B07 consent/provenance/authenticity schema. | untrusted contribution stays labeled and auditable |
| `AIA-009` | **P1** | **OPEN** | Trusted forwarded identity boundary | Forwarded headers need trusted proxy semantics. | B06 revalidate and gate spoofing. | spoofed forwarding test |
| `AIA-010` | **P1** | **OPEN** | Pre-buffer resource limits | Post-allocation limits do not prevent memory exhaustion. | B06 inspect request paths and enforce early limits. | oversize request bounded before full allocation |
| `AIA-011` | **P1** | **OPEN** | Settings/config schema consolidation | 90 Sphinx config calls plus browser registries create drift/sensitivity risk. | B10 evaluate/promote schema-driven registry after security/representation migration. | single ownership record; secret class nonserializable |
| `AIA-012` | **P1** | **OPEN** | Known monolith decomposition | 22k JS / 14k CSS / 5.5k Python contain multiple responsibilities. | B11 extract only after tests pin behavior. | no new responsibility mixing + green gates |
| `AIA-013` | **P1** | **BLOCKED** | Static Markdown consumer migration | Requires `_sphinx_llm` stable facade and manifest. | Wait for producer A11 then activate B08. | static canonical primary |
| `AIA-014` | **P2** | **OPEN** | Corpus/MCP boundary integration | Future retrieval/MCP features should consume existing contracts, not duplicate them. | B12 after core security/representation stabilization. | boundary/import tests |

## Priority order

Security authority and producer/consumer boundaries precede monolith refactoring. Do not restructure large files first and hope the security/representation architecture becomes clearer afterward.
