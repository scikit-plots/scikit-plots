# AI Assistant

## Scope

Primary package:

`docs/source/scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/`

The module owns the documentation-side AI Assistant experience: generated Markdown companions, provider links, the floating Assistant panel, browser-side conversation/runtime state, multimodal generation UI, resource/file handling, sharing/contribution workflows, and the static assets copied by Sphinx.

Deployable sibling services live inside the same package:

- `_hf_spaces_proxy/` — public browser-facing proxy/service boundary.
- `_hf_spaces_model/` — self-hosted model endpoint used by supported proxy routes.
- `_cf_worker/` — optional worker/deployment helper.

## Main ownership boundaries

### Sphinx extension

`_sphinx_ai_assistant/__init__.py` owns build-time configuration and output integration. Important responsibilities include:

- generating canonical `page.md` companions from final HTML;
- generating `llms.txt` from those Markdown files;
- exposing configuration to the browser runtime;
- registering/copying Assistant CSS, JavaScript, icons, and optional isolation assets;
- keeping build-time helpers usable without forcing provider/network execution.

### Browser runtime

`_static/ai-assistant.js` owns the interactive panel and browser state. `ai-assistant.css` owns presentation. Browser code must treat build configuration as public data: never place API keys, bearer credentials, repository write tokens, or other server authority in `conf.py` or generated HTML.

The icon registry in `ai-assistant.js`, Python fallbacks in `_static/__init__.py`, and standalone SVG files are synchronized representations. New action-specific icons should be additive rather than replacing a generic icon used elsewhere.

Current feedback icon ownership:

- `commentDiscussion` remains the generic Feedback workspace/discussion icon.
- `feedbackDetail` is the default icon for the Assistant answer action that opens the detailed feedback form.
- The detailed-feedback action falls back to `commentDiscussion`, then `chat`, if a downstream/custom runtime lacks the new icon.

### Proxy/service

`_hf_spaces_proxy/app.py` owns browser-facing service routes and server-side provider authority. Current route families include:

- chat: `/v1/chat/completions`;
- image generation: `/v1/image` and `/v1/image-generations`;
- audio generation: `/v1/audio` and `/v1/audio-generations`;
- document generation: `/v1/document` and `/v1/document-generations`;
- video generation: `/v1/video` and `/v1/video-generations`;
- generated artifacts and provider-output lifecycle;
- resource capability discovery;
- AI Learn publication bridge routes;
- contribution/review lifecycle under `/v1/contribute`;
- share lifecycle under `/v1/share`;
- explicit Assistant review feedback under `/v1/feedback/review`;
- generic page feedback under `/v1/feedback`.

Provider credentials stay server-side. The browser sends bounded requests to this service; the proxy selects and executes supported provider adapters.

## Resources and multimodal requests

Resource handling is capability-driven. Do not infer that every model accepts image/audio/video/document bytes merely because the UI supports those resource types.

The browser and proxy distinguish:

- a requested Assistant/model provenance;
- the active runtime/service that actually owns generation/rendering/publishing;
- safe public/transcript state;
- private management or artifact capabilities.

Bearer-like management capabilities must not be copied into transcripts, shared conversations, exports, or public build artifacts.

## Activity and generated-file previews

Activity UI exposes public operational summaries, not hidden model reasoning. Streaming may report bounded progress/activity and complete generated files. The browser should preserve latest-revision semantics for generated files and provide immediate previews without treating an incomplete stream fragment as a final artifact.

Cancellation and broken connections must leave the UI in a recoverable state. Persist only state that is safe for the chosen browser storage boundary.

## Feedback boundaries

Keep these concepts separate:

- Assistant quick/local rating UI;
- explicit content-bearing Assistant review submission;
- generic page feedback owned by `_sphinx_feedback`;
- dataset/contribution submission and maintainer review.

A local rating must not silently become a training contribution or repository review. Content-bearing review/contribution is an explicit reader action with its own server-side lifecycle.

## Isolation and browser authority

Optional separate-origin isolation is implemented by the isolation host/frame assets in `_static/`. When enabled:

- parent origins are explicit;
- browser cookies/ambient credentials remain disabled unless a deployment deliberately enables them;
- microphone/device permission remains opt-in;
- frame messaging is versioned and bounded;
- storage/context crossing the frame boundary is explicit;
- production CSP/frame-ancestor headers are deployment responsibilities.

Treat a hostile or compromised documentation parent as part of the threat model. Isolation reduces ambient authority; it does not make public page content secret.

## Dataset, contribution, and review storage

The proxy supports provider-neutral record storage with one Primary and optional Mirrors. Configuration topology belongs in non-secret variables; provider credentials belong in secret storage.

Core invariants:

- the browser never receives repository/storage write credentials;
- contribution identity/deduplication uses explicit record/request identifiers, not client IP identity;
- provider-native review is an explicit quarantine/review flow;
- a successful Primary write is authoritative; mirror failure must be reported rather than silently redefining authority;
- retraction/edit/retry operations are receipt/capability scoped;
- shared receipt/state backends are required when continuity must survive process/container replacement.

The detailed operator material remains next to `_hf_spaces_proxy/`; this root file is only the current architecture summary.

## Files to inspect first

For most Assistant work:

- `__init__.py` — Sphinx integration and build-time contracts.
- `_static/ai-assistant.js` — panel behavior and browser state.
- `_static/ai-assistant.css` — presentation/responsive behavior.
- `_static/__init__.py` — packaged static/icon fallbacks.
- `_example_conf.py` — supported configuration examples.
- `_hf_spaces_proxy/app.py` — public service routes.
- `_hf_spaces_proxy/_providers/` — provider adapters/policies.
- `_hf_spaces_proxy/_utils/` — shared contracts, storage, resource, and artifact logic.
- `_hf_spaces_model/app.py` — self-hosted model service.
- `tests/` — browser, integration, proxy, architecture, and static contract tests.

## Safe editing rules

1. Preserve separation between public browser state and server/private authority.
2. Keep provider-specific behavior behind provider adapters instead of branching throughout UI code.
3. Add icons and UI primitives without repurposing an existing semantic asset used elsewhere.
4. Keep static-file/Python fallback representations in sync.
5. Keep Sphinx import-time behavior free of unnecessary heavy/provider dependencies.
6. Test both static/build-time contracts and browser runtime behavior after UI changes.
7. Do not document implementation chronology here; document only the current contract.
