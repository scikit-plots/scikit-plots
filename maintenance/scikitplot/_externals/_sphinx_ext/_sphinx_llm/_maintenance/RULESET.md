# `_sphinx_llm` Durable Ruleset

## Upstream and provenance

1. Pin every NVIDIA import to an exact commit and record its archive/hash.
2. Preserve upstream licensing/notices and mark modified upstream-derived files.
3. Classify every implementation file as `UPSTREAM_PRESERVED`,
   `UPSTREAM_PATCHED`, or `DOWNSTREAM_ONLY`.
4. Prefer wrappers/adapters over invasive edits to upstream-preserved files.
5. Never create/import a top-level local `sphinx_llm` package that shadows the
   external NVIDIA package.

## Canonical representation

6. The resolved Sphinx semantic document is canonical; HTML is a human output,
   not the source of canonical LLM Markdown.
7. Per-page Markdown, `llms.txt`, manifest, compatibility report, and provenance
   are build-time static artifacts.
8. A post-build HTML converter is lower-fidelity compatibility output and must
   be labeled as such.
9. Runtime DOM conversion is last-resort fallback only and must never be marked
   canonical.
10. `html`, `dirhtml`, suffix-mode, version, and locale routing are explicit
    contracts with tests.

## Directives and nodes

11. Presentation state must not erase semantics: dropdown/toggle/tab content is
    fully represented unless explicitly excluded by author policy.
12. Unknown semantic leaf nodes may not disappear silently.
13. Unknown container nodes with meaningful children may use a transparent
    fallback, but must be reported until classified.
14. Media nodes preserve semantic metadata (title/alt/caption/provider/link/
    source) rather than browser iframe/layout markup.
15. Raw executable HTML/script/event-handler/`javascript:` content is never
    copied blindly into agent Markdown.
16. New scikit-plots directives should emit standard Docutils/Sphinx nodes or
    register an explicit LLM semantic adapter.

## Curation and artifacts

17. `llms.txt` is a small standards-facing navigation/index artifact with
    semantic sections, links, and descriptions.
18. Author-provided page metadata wins over generated summary text.
19. `llms-full.txt` is optional compatibility/convenience output and has explicit
    size policy.
20. Never silently truncate a full corpus and still call it complete.
21. Page/block exclusions, source-code inclusion, ordering, and URI templates
    are explicit curation policy, not ad hoc string processing.
22. All generated artifacts have deterministic ordering and provenance.
23. The manifest is the stable machine inventory; consumers do not independently
    rescan output directories when the manifest is available.

## Build-time LLM augmentation

24. Build-time LLM generation is opt-in and never required for deterministic
    representation.
25. Credentials are read from named environment variables, not stored as Sphinx
    config values or generated artifact metadata.
26. Authenticated provider calls over non-loopback plain HTTP are rejected by
    default.
27. Generated summaries are untrusted generated data with recorded origin,
    provider/model/prompt/cache fingerprint, and deterministic fallback.

## Boundaries

28. `_sphinx_llm` may not import `_sphinx_ai_assistant`.
29. `_sphinx_ai_assistant` runtime/browser consumers use published static artifacts. Any optional Python build-time integration uses only a stable public/internal facade, never private upstream implementation details.
30. `scikitplot.corpus` owns retrieval/evidence semantics; `_sphinx_llm` owns
    documentation representation. Do not duplicate retrieval architecture here.
31. `scikitplot.mcp` owns MCP wire/protocol behavior. Representation artifacts may
    be exposed through MCP later, but `_sphinx_llm` does not own MCP transport.

## Maintenance truth

32. Desired invariants must be labeled `PLANNED` until source/tests prove them.
33. A tracker contract naming a missing module or adapter is drift and must be
    reconciled.
34. Deleted/weakened regression gates require explicit rationale.
35. Never declare a checkpoint complete while its **selected closure proof mode**
    is `NOT_RUN` or `ENVIRONMENT_BLOCKED`, or when the claim is based only on a
    previous chat transcript. A separately defined alternate proof mode may close
    a checkpoint only when the rules/checkpoint name its prerequisites explicitly,
    those prerequisites are machine-checked, and any blocked local proof remains
    recorded as blocked rather than being relabeled `GREEN`.
36. A path-dependent or checkout-location-dependent vendor tree hash may be
    retained as historical vendoring evidence, but it may not be the sole
    portable integrity proof; verified vendor baselines require relative-path
    file digests or an equivalently relocation-stable manifest.
37. Preserved upstream tests that depend on repository-relative fixtures must be
    executed in an ephemeral upstream-shaped staging tree; do not edit preserved
    tests or misclassify a vendored-layout path mismatch as upstream behavior.

38. A01 **local** pinned-baseline behavior proof uses the exact preserved
    `uv.lock` resolution for the selected Python baseline (currently Python 3.13).
    Tests against compatible-but-different dependency versions are A02
    compatibility evidence, not a substitute for A01 provenance/baseline proof.
39. A01 may alternatively use `PINNED_UPSTREAM_CI_EQUIVALENT` only when all of
    the following are true for the same exact NVIDIA commit: local vendored
    source/tests/license files are byte-verified; the pinned `docs/source`,
    `pyproject.toml`, `uv.lock`, and test workflow are byte-verified; the staged
    upstream layout is GREEN; the official workflow/job for the selected Python
    and Sphinx baseline reports SUCCESS; and the workflow executes the preserved
    upstream test path using the preserved project/lock semantics. This proof mode
    never changes a blocked local exact-lock run into local GREEN.
40. `GREEN_EXACT_LOCK` means every distribution in `UPSTREAM_TEST_LOCKSET.json`
    matches exactly and required import probes are GREEN; matching only selected
    anchor packages is insufficient.
41. A Markdown sub-build may be called semantically faithful only when every
    primary-build input that can change resolved document content crosses the
    boundary. At minimum this includes the configuration directory, Sphinx tags,
    and explicit config overrides (`-D` / `app.config.overrides`). A GREEN tag
    test does not imply GREEN config-override parity.

42. A downstream compatibility fix for upstream behavior belongs outside
    `sphinx_llm/**` whenever the behavior can be intercepted safely without
    modifying `UPSTREAM_PRESERVED` bytes; such a shim must have its own regression
    gate and provenance/license classification.
43. Primary-build configuration values are potentially sensitive build data.
    Arbitrary effective values may not be copied to process arguments or logs;
    only explicitly allow-listed early Sphinx core settings may use `-D` transport.
44. The full effective-config handoff to a child Sphinx build must be private,
    integrity-checked before deserialization, consumed only within the same
    build-host trust boundary, and deleted promptly after child read.
45. A02 cannot close on source inspection, helper-unit tests, or a single GREEN
    Sphinx environment. The programmatic config-override/`ifconfig` regression must
    prove identical semantic selection in HTML and canonical Markdown across all 10
    required downstream Python/Sphinx compatibility cells (or an explicitly reviewed
    equivalent matrix), with per-cell evidence.

46. A02 CircleCI aggregate output is not trusted as a standalone closure claim.
    Closure review must independently recompute the aggregate from all ten canonical
    per-cell records against the current plan, upstream commit, builder pin, and
    implementation fingerprint; duplicate, stale, unsafe, or inconsistent evidence
    fails closed.
47. Post-CI A02 evidence verification is read-only. A GREEN artifact decision makes
    A02 eligible for a separate human-reviewed reconciliation; tooling must not
    automatically rewrite `STATE.json`, tracker/baseline status, history, or A03
    eligibility.

48. A02 CircleCI closure eligibility requires coherent built-in execution provenance across all ten cell records: one pipeline ID, one workflow ID, one project ID, one revision, the expected `scikit-plots/scikit-plots` project identity, and ten unique job IDs. These fields are provenance consistency metadata, not cryptographic authentication; downloaded artifacts remain subject to human review. All-GREEN local/manual records are diagnostic unless a separate equivalent proof mode is explicitly reviewed and machine-defined.

49. A02 reconciliation preparation remains read-only. A closure-eligible artifact may produce a review receipt only after the maintenance checker is GREEN; that receipt must bind the exact evidence set and SHA-256 of every planned closure-target file. Receipt generation may not update baseline cells, checkpoint status, history, registry, or A03 eligibility, and any target drift requires regenerating the receipt before human review.

50. A02 repository-CI drift must be rebased semantically, not by overwriting a newer `.circleci/config.yml` with a historical full-file snapshot. Rebase tooling must read the maintainer's current config, require unambiguous top-level `parameters`/`jobs`/`workflows` anchors, render a separate candidate/diff, refuse partial or in-place mutation, and pass YAML-structure candidate verification before human application. The historical integrated digest remains the strict gate only for the checked-in reviewed snapshot.

51. A checkpoint whose selected closure proof mode is permanently abandoned is
    recorded `DEFERRED_PERMANENTLY`, never `COMPLETE`. The status requires a
    written `deferral_rationale` and `residual_risk` in `STATE.json`, keeps its
    gap entry open rather than closed, and leaves `checkpoint_complete` false.
    Every later checkpoint that reaches `COMPLETE` while a permanent deferral is
    in force must record `depends_on_unproved` naming it. Rule 35 is unaffected:
    deferral is not completion, and no deferral may make `COMPLETE` reachable
    without the original evidence or an explicitly reviewed equivalent matrix.
