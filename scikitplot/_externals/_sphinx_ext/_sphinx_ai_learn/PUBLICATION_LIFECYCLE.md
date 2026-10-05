# AI Learn publication lifecycle

## Authority model

AI Learn intentionally separates private generation, durable source data, and
rendered documentation:

```text
browser/local AI draft
        ↓
review + revision-bound publication transaction
        ↓
canonical JSON proposal
        ↓
GitHub pull request
        ↓
human/CI review + merge
        ↓
ReadTheDocs build
        ↓
_sphinx_ai_learn JSON → RST materialization
        ↓
Sphinx RST → HTML
        ↓
live documentation
```

The repository authority is JSON. RST and HTML are derived. There is no
`catalog.json` publication step, no publication-time RST exporter, and no model
or repository write inside the Sphinx build.

## Filesystem model

A Topic has one immutable route and granular same-stem JSON/RST artifacts:

```text
learn-ai/topics/<timestamp>-<digest>/
├── index.json / index.rst
├── topic.json / topic.rst
├── summary.json / summary.rst
├── video.json / video.rst
├── audio.json / audio.rst
├── document.json / document.rst
├── whiteboard.json / whiteboard.rst
├── topic-prompts/
│   ├── index.json / index.rst
│   ├── eli14.json / eli14.rst
│   └── knowledge-gaps.json / knowledge-gaps.rst
├── skills/
│   ├── index.json / index.rst
│   └── skill-check-reference.json / skill-check-reference.rst
├── open-problems.json / open-problems.rst
├── continue-learning.json / continue-learning.rst
├── tweets.json / tweets.rst
└── hackernews.json / hackernews.rst
```

Empty result JSON is intentional. It gives every generation target a stable
repository address before anyone generates content.

Reusable definitions live outside a specific Topic:

```text
learn-ai/topic-prompts/<prompt-id>/index.json / index.rst
learn-ai/skills/<skill-id>/index.json / index.rst
```

A definition answers *what should the interaction do?* A Topic-owned result
answers *what did this interaction produce for this subject?* Those are never
the same artifact.

## Include and toctree modes

Every `learn.record.v2` can choose `add_toctree`.

Default `false` means content composition:

```text
index.rst
  ├─ include summary.rst
  ├─ include topic-prompts/eli14.rst
  └─ include skills/skill-check-reference.rst
```

Children begin with `:orphan:` and `:no-search:` and expose a compiler sentinel.
The parent owns headings/public anchors and includes only content after that
sentinel. Child files contain no toctree.

Explicit `true` means navigation composition: the parent emits `.. toctree::`,
children omit `:orphan:`, and each child owns its standalone title/anchor.

This option changes only document composition. Canonical JSON contracts and
result locations do not change.

Every renderable canonical artifact also owns `hide_secondary_sidebar` (default
`true`). Publication projection carries the current record and section values forward
rather than resetting them, while newly created pages/results receive the explicit
hidden default. This layout policy is independent of include/toctree composition.

## Reusable interaction registries

Topic Prompts and Skills share these invariants:

- stable machine ID; display title lives in JSON;
- explicit deterministic order;
- bounded plain-text instruction/description;
- no duplicate ID or order within a registry;
- Prompt and Skill IDs are mutually disjoint;
- neither registry may reuse fixed Topic structural IDs;
- Python does not duplicate semantic definition content.

Skills may carry `domains` and graph `related` references. They are exposed as
virtual `kind="skill"` subjects in the normalized in-memory catalog so normal
related-record/navigation code can address them. That projection is not a
second repository source model.

## Publication transaction

`_publication.py` validates a transaction against the exact current semantic
tree revision. Supported logical operations include:

- `create-record` for ordinary durable records;
- `upsert-section` for a record-owned result section;
- `attach-source` for reviewed source evidence;
- `create-topic-prompt` for a reusable Prompt definition;
- `create-skill` for a reusable Skill definition.

The planner applies the transaction to normalized state and uses the same
canonical JSON projectors as `_materialize.py` to calculate the future tree.
It returns exact changed/added JSON bytes, deleted JSON paths, routes, affected
IDs, base revision, and future revision. It never emits RST.

A stale base revision is a hard failure. The caller must review/replan against
the new tree rather than silently rebasing generated content.

### Definition fan-out

Adding a Prompt or Skill is atomic across all Topics. One operation proposes:

1. the top-level reusable definition;
2. every existing Topic's updated `index.json` references; and
3. every existing Topic's empty result JSON for that interaction.

Future Topics are automatically projected with all current Prompt/Skill
definitions. Cross-registry and structural-ID collisions are rejected before a
review bundle can be produced.

## Review bundle and provider boundary

`_publication_cli.py` writes only canonical `.json` proposal files plus a
`publication-plan.json` manifest containing SHA-256 hashes, byte lengths,
routes, affected IDs, deletions, and before/after revisions.

The GitHub/Hugging Face service can reuse authenticated contribution
infrastructure, but browser clients must never choose arbitrary repository paths
or obtain repository credentials. The service receives a validated reviewed
transaction/bundle and maps it onto the fixed canonical subtree.

### Reviewed GitHub transport

The implemented publication adapter keeps repository authority out of the browser.
The browser sends a reviewed ``learn.publication-request.v1`` envelope to the
active ``publication`` endpoint only after an explicit **Open pull request**
action. ``Generate`` itself never performs a repository write.

The proxy has three fail-closed modes: ``disabled``, ``stub`` (full validation
with zero GitHub writes), and ``github`` (the Scikit-plots deployment default). Repository identity, default
branch, canonical subtree, workflow file, and GitHub credential are server-owned
policy and cannot be overridden by a browser request. ``POST /v1/learn``
with ``action=test`` verifies this policy without dispatching a publication.

In ``github`` mode the proxy may dispatch only the configured repository workflow.
It does **not** calculate canonical repository paths and does not hold direct
content/pull-request mutation logic. The checked-out repository workflow is the
second trust boundary and must:

1. reconstruct and hash-bind the exact reviewed request;
2. reject a stale semantic revision for content publication; append-only feedback
   may cross an older page revision only when its immutable generation target still
   exists in the current tree;
3. rerun ``_publication.py`` against the current canonical JSON tree;
4. apply only planner-owned ``docs/source/learn-ai/**/*.json`` paths;
5. reject every non-JSON repository mutation;
6. validate the proposed future tree with JSON→RST materialization twice in an
   isolated clean room;
7. use a deterministic request-bound branch and replay an existing open review
   instead of creating duplicate PRs; and
8. open a human-review pull request without bypassing branch protection.

The proxy GitHub token therefore needs only the permission required to dispatch
the fixed workflow. Repository ``contents`` and ``pull-requests`` write authority
belongs to the repository workflow's short-lived ``GITHUB_TOKEN``. Secrets are
never serialized into Sphinx HTML, endpoint profiles, browser storage, publication
receipts, or review bundles.

### Publication state, evidence review, and contributor credit

These are intentionally separate trust domains:

- **Publication** is complete when reviewed canonical JSON is merged and therefore
  present in the repository revision used for the site build. A rendered canonical
  section reports ``Published``; it does not infer GitHub PR state from browser
  storage.
- ``section.review`` is **evidence/citation review metadata**, not repository PR
  review. Its UI state is independently ``reviewed``, ``pending``, ``stale``, or
  ``not-applicable``.
- ``learn.publication-request.v1`` may contain a bounded public ``contributor``
  object with only ``display_name``. This is self-declared credit, never verified
  identity. Empty credit normalizes to ``Anonymous``; email/account identifiers or
  other fields are not accepted by the contract.
- Contributor credit is publication metadata only. It is not included in model
  generation context. Section edits accumulate unique bounded display names, new
  records project credit into ``authors``, and Prompt/Skill definitions use their
  existing ``author`` field.

This separation prevents a merged PR from appearing "review pending" merely
because no evidence assessment exists, while also preventing evidence review from
being misrepresented as repository approval.

### Authorship, accepted generations, and community feedback

Authorship and participation are intentionally different metadata layers:

- ``record.authors`` is the ordered bounded public credit list for stewardship of
  the overall Topic/Source/Open Problem/Media record. It is not duplicated into
  every section file, which avoids drift when record stewardship changes.
- Topic Prompt and Skill definitions expose plural ``authors`` while retaining the
  historical singular ``author`` field as a compatibility projection.
- A section generation owns its own ``contributors`` list. Different accepted
  generations can therefore credit different participants without rewriting the
  record-level authorship history.

``learn.section.v1`` remains a valid legacy contract. A reviewed *content update*
projects its previously accepted non-empty content into a deterministic legacy
generation and writes ``learn.section.v2`` with an explicit ``active_generation_id``
plus a bounded append-only ``generations`` ledger. Empty placeholders are scaffolding
and are never fabricated into generation history. New accepted generations are
appended; older accepted generations are not overwritten. Promotion of the
published/active generation remains a reviewed repository decision and is never
performed automatically by ratings.

New reviewed feedback is stored as an immutable metadata sidecar rather than by
rewriting the section file::

    <record>/feedback/<section-id>/<generation-id>/<feedback-id>.json

The sidecar contract is ``learn.generation-feedback.v1``. V62 embedded generation
feedback remains readable for compatibility, but new events are written only as
sidecars. A feedback identifier is globally unique across both storage forms, so an
event cannot be represented twice and accidentally double-counted. Metadata
sidecars participate in canonical validation and the full tree/event digest but
intentionally do **not** change the authored-content catalog revision and own no RST
page. This lets independent feedback PRs touch independent files, avoids a popular
section becoming a permanent Git hot file, and prevents a community vote from
invalidating unrelated authored drafts or evidence-review freshness.

Quick feedback submits only ``-1`` or ``+1``. Detailed feedback uses the same
eleven-value scale as the Assistant, from ``-5`` through ``+5`` inclusive, with
optional bounded plain-text details and self-declared public credit. The displayed
community score is derived from merged canonical events; browsers never increment it
optimistically. A neutral ``0`` still counts as one reviewed rating event. Generation
history can be viewed with published first, highest-score, or newest-generated
ordering without changing canonical order.

Feedback is append-only and generation-bound, so it has a narrower concurrency
contract than authored content. A static page may submit feedback using an older
catalog revision if the exact ``generation_id`` still exists in the current tree.
If the generation does not exist, the repository boundary fails closed. Browser
retries reuse the same ``feedback_id`` after an ambiguous transport failure; the
browser keeps that pending envelope in session-scoped storage so a same-tab reload can
continue the same idempotent request. The repository treats that ID as an idempotency
key, preventing a dispatch whose response was lost from being counted twice.
Browser-authored timestamps are accepted only for legacy transport compatibility and
are not persisted as canonical review time; Git history/PR review remains the
acceptance chronology.

V71 feedback identifiers are 192-bit browser-CSPRNG event nonces encoded as
``feedback-`` plus 48 hexadecimal characters. They contain no timestamp, account,
device, browser, network address, page-history value, or stable participant identifier.
V70 128-bit/32-hex nonces remain accepted so an ambiguous V70 retry can finish. New
reviewed writes require one of those opaque nonce forms independently at the proxy,
workflow-request parser, publication planner, and feedback-sidecar validator, so a
direct workflow invocation cannot replace the event nonce with a semantic/user-shaped
identifier. Existing embedded legacy events remain readable. The publication
``request_id`` is a SHA-256 commitment to canonical request JSON for
idempotent GitHub dispatch; because every feedback request contains a fresh high-entropy
feedback nonce, that digest is not a participant fingerprint and is not derived from
client metadata. The full digest remains workflow authority but is no longer repeated in
pull-request prose; the public review shows only the existing short review reference.

The browser does not add IP address, user agent, language, timezone, screen/device data,
account identity, cookies, or telemetry fields to a feedback request. The proxy still
needs an abuse-control identity for rate limiting. V71 derives a publication-and-scope
specific HMAC pseudonym before either limiter backend sees the identity. Local mode uses
an ephemeral process secret; Redis mode uses the deployment rate-limit secret so replicas
agree without making publication limiter identities linkable to chat/media limiter
identities. The local limiter retains that pseudonym only as bounded rate-window control
state and discards stale entries; it is not a participant identifier. That pseudonym is
rate-limit control state only: it is never
part of request hashing, receipts, workflow inputs, branches, commits, pull requests, or
canonical feedback JSON. Optional comment text and public credit are different: they are
user-authored publication content and become public if the reviewed PR is merged.

Contributor display names are self-declared public credit, not authenticated voter
identity. Therefore ``score`` and ``rating count`` describe merged/reviewed feedback
events, not a one-person-one-vote ballot. Duplicate/spam review remains a repository
moderation responsibility, with transport rate limits as an additional abuse bound.
Unlike the private Assistant feedback store, AI Learn does not claim a secure
client-side retract/supersede identity for public repository events: a correction is
another reviewed immutable event, and maintainers decide whether contradictory or
duplicate proposals should merge. This avoids letting a browser forge a retraction of
somebody else's public event merely by naming its visible ``feedback_id``.

A generation may accumulate at most 4096 reviewed feedback events across legacy embedded
metadata and immutable sidecars. This is a deterministic materializer/repository safety
bound, not a voting identity rule; it prevents one hot generation from creating
unbounded filesystem/build fan-out. Empty optional comments canonicalize to omission,
and unsupported control characters fail at the browser/proxy/repository boundaries.

Feedback-only metadata also has a narrower Sphinx invalidation boundary. Authored
JSON remains a global Learn dependency and defines the catalog/content revision.
Feedback sidecars are fingerprinted per record; only the owning record's pages (and
external documents that explicitly consumed that record's feedback) are reparsed when
its feedback digest changes. The V63 environment-schema revision forces one upgrade
reparse so incremental-build caches register those record-scoped consumers safely.

## Sphinx build boundary

At `config-inited`, `_sphinx_ai_learn` may:

- read and validate canonical JSON;
- validate record/section/interaction graph integrity;
- deterministically derive sibling RST;
- prune only extension-owned stale RST;
- preserve unchanged output mtimes; and
- expose the normalized graph to directives.

It must not:

- call text/image/audio/video models;
- fetch sources/evidence from the public network;
- create branches/PRs;
- write datasets or contribution services;
- mutate canonical JSON; or
- silently repair invalid canonical content.

Invalid canonical state fails visibly.

## Strict preview

The browser can keep a fast immediate projection for editing, but the final
"what will this page look like?" preview should exercise the real compiler:

```text
draft/proposed JSON
        ↓
temporary canonical tree
        ↓
_sphinx_ai_learn materializer
        ↓
temporary RST
        ↓
isolated Sphinx build
        ↓
HTML preview in sandboxed iframe
```

That worker should have no repository credentials and no network, and should be
bounded by input size, CPU, memory, and wall-clock limits. Cache previews by
semantic artifact/tree hash. This gives strict output parity without waiting for
GitHub review + ReadTheDocs deployment.

## Security and determinism

The compiler rejects unsafe/ambiguous inputs including duplicate JSON keys,
non-finite numbers, path traversal, symlinks, unknown contracts, invalid
ownership, orphan sections, registry collisions, bad graph references, and
handwritten RST output collisions.

Generated body text is data inside trusted directives, not arbitrary RST. This
prevents a model-generated string from introducing `raw`, `include`, `toctree`,
or other structural directives.

Graph validation is strict and diagnostic. Missing `related` subjects, missing
citation targets, and citations that resolve to a non-Source subject are aggregated
with their canonical JSON owner paths before materialization stops. Publication
validation uses the same graph rules, so an invalid future tree must be rejected
before a review bundle is returned.

Writes are atomic per file. A failed materialization never changes canonical
JSON; rerunning against the same JSON converges to the same RST. Semantic tree
hashes ignore JSON whitespace while repository serialization remains stable and
human-reviewable.

## CI contract

CI should independently require:

1. canonical JSON validation;
2. materialization with no unexpected checked-in RST diff;
3. a second materialization with zero changes;
4. Python and JavaScript static checks;
5. unit/security/publication tests;
6. a real warning-strict Sphinx build in the docs environment; and
7. preview/publication tests confirming planned revisions and JSON-only diffs.

This keeps checked-in generated RST useful for review while ensuring JSON remains
the sole durable content authority.
