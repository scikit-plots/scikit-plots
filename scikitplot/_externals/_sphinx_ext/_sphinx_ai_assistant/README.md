# ✨ Sphinx AI Assistant

A Sphinx extension that adds AI-powered features to documentation pages, making it easier to use your documentation with AI tools.

## Start here

Choose the path that matches what you are trying to do:

| Goal | Start with | Backend required? |
|---|---|---:|
| Copy/view documentation as Markdown | [Basic setup](#basic-setup) | No |
| Open ChatGPT/Claude/Gemini with documentation context | [AI provider configuration](#configuration) | No |
| Run the in-page assistant with stub models | [AI Assistant Panel](#ai-assistant-panel) | No |
| Run real models securely | [Endpoint profiles](#endpoint-profiles--one-service-flexible-routes) + proxy README | Yes |
| Let readers share one-Q&A feedback with maintainers | [Feedback and maintainer review](#feedback-and-maintainer-review) | Yes |
| Let readers submit reviewed dataset content | [Dataset contribution and review](#dataset-contribution-and-review) | Yes |
| Configure HF/GitHub/GitLab/Bitbucket storage | [`_hf_spaces_proxy/DATASET_CONTRIBUTION_GUIDE.md`](_hf_spaces_proxy/DATASET_CONTRIBUTION_GUIDE.md) | Yes |
| Deploy the bundled Hugging Face Space proxy | [`_hf_spaces_proxy/README.md`](./_hf_spaces_proxy/README.md) | Yes |
| Run a separate-origin assistant frame | [`ISOLATION_DEPLOYMENT.md`](./ISOLATION_DEPLOYMENT.md) | Optional |

### Deployment levels

The extension can be adopted incrementally:

```text
Level 0  Static docs only
         Markdown export + AI deep-links

Level 1  Local/stub assistant
         Full panel UX, no model credential or network backend required

Level 2  Live model proxy
         Browser -> your proxy -> model provider
         provider credentials remain server-side

Level 3  Reviewed feedback + model improvement
         one Q&A -> review-sharing permission -> provider PR/MR
         merge -> eligible Q&A + normalized quality signal

Level 4  Dataset contribution
         explicit content consent -> quarantine/review -> approved canonical dataset
```

Do not put model, storage, review, or repository-write tokens in `conf.py`. Sphinx
configuration is rendered into public documentation output. Credentials belong in
the server-side proxy's secret store.

## Features

### Markdown Export
- **Copy as Markdown**: Convert any documentation page to Markdown format with a single click
- **View as Markdown**: Open the markdown version of the current page in a new browser tab
- Perfect for pasting into ChatGPT, Claude, or other AI tools
- Preserves code blocks, headings, links, and formatting
- Clean conversion that removes navigation, headers, and other non-content elements

### Integration with AI tools
- **Direct AI Chat Links**: Open ChatGPT or Claude with pre-filled documentation context
- **Smart Content Strategy**: Uses pre-generated markdown files for clean, unlimited-length context
- **Customizable AI Providers**: Built-in support for Claude, ChatGPT, and custom AI services
- **No Backend Required**: Pure static files, works on any hosting
- **MCP (Model Context Protocol) integration**: Connect VS Code and Claude to your MCP

### Export as PDF
- **"Export as PDF" button** added to the bottom of the dropdown menu (after MCP tools)
- Default behaviour: calls the browser's built-in `window.print()` → user saves as PDF
- Optional: set `ai_assistant_pdf_export_url` to a server-side endpoint
  (e.g. a WeasyPrint URL, GitBook-style `~gitbook/pdf?page=…`, or any static `.pdf` URL)
  and the button will open that URL in a new tab instead
- Icon mirrors the Font Awesome `file-pdf` style used by sphinx-book-theme and GitBook

### AI Assistant Panel
- **Floating chat panel** anchored to the bottom-right viewport corner
- Opens via the last dropdown entry ("AI Assistant" or your custom label)
- Slide-in / slide-out animation; fully keyboard-accessible (Enter submits, Escape closes)
- **Stub mode** (default, `ai_assistant_panel_api_enabled = False`): renders the full UI
  with a polite placeholder response — zero network calls, works on any static site
- **API mode** (`ai_assistant_panel_api_enabled = True`): sends the bounded
  `scikitplot-chat-v1` request contract to a configured proxy endpoint and streams
  the answer. Provider credentials stay server-side; the browser must not embed them.
- Compatible with PyData Sphinx Theme, Furo, sphinx-book-theme, and Read the Docs
- Dark-mode aware via the same three-layer CSS variable chain as the rest of the widget

### AI Assistant Panel — v0.3 additions

- **Mouse-resizable**: drag the top-left grip to resize (clamped to the
  viewport, size persisted per tab)
- **Conversation persistence**: when `ai_assistant_panel_persist = True`, the
  **Remember conversation in this tab** switch starts from the configurable
  `ai_assistant_panel_remember_conversation` site default (**True** by default).
  The reader can override it for the current tab; that explicit ON/OFF choice
  survives same-tab page navigation but disappears when the tab closes. The
  transcript stays in `sessionStorage` only, and invalid/oversized stored state
  fails closed and is cleared.
- **Start a new chat**: refresh-icon button clears the conversation without a
  page reload
- **Export as txt**: download the whole conversation as a plain-text file
- **Copy this answer**: per-answer copy button under each assistant reply
- **Feedback**: configurable quick + detailed local rating UI with synchronized
  controls and an optional note. Anonymous rating telemetry is a separate
  browser preference whose built-in initial value is **False** and which never
  contains Q&A, note, model, page URL, or stable conversation identity.
  **Maintainer feedback review** (**Share with maintainers**) has an independently
  configurable initial value (**True** by default) and can place exactly one Q&A
  into an updatable provider-native feedback review. Explicit reader ON/OFF choices
  override both site defaults. The Feedback tab owns both feedback permissions and
  exposes separate keyboard-accessible **JSON** request and readable **JSONL** repository views before review;
  Endpoint Configuration does not duplicate those consent switches. Originating model attribution
  is required so the reviewed Q&A remains useful and auditable. A maintainer merge
  makes that Q&A training-eligible together with a normalized quality score/percentage.
  Canonical schema-v5 saved rows also retain a bounded rating lineage
  (`feedbackChainId`, scalar `prevFeedbackId`, ordered `prevFeedbackIds[]`, and
  `editCount`) so repeated rating changes can be resolved to the terminal valid
  revision without relying on network arrival order.
  Host-page lifecycle
  events remain independently permissioned.
- **Keyboard shortcut**: toggle the panel with a configurable chord
  (`ai_assistant_panel_shortcut`, default `Alt+Shift+A`; a modifier is
  required, a bare key is rejected)
- **Privacy & Responsibility sheet**: a built-in, fully customizable in-panel
  explainer that clearly separates the extension's responsibilities from the
  integrated model's
- **First-message privacy status**: when chat starts, onboarding is replaced by
  a compact shield notice at the top of the transcript. Its **More information**
  action opens the same Privacy & Responsibility sheet. Site owners can supply
  a plain-text stronger guarantee only when their deployed endpoint actually
  provides it; the default copy never invents anonymity, zero-retention, or
  no-training claims.
- **Quick model switching in answer actions**: **More → Change model** expands a
  compact current-model list in place, with provider/model metadata and active
  checkmark; **Model configuration…** remains the full search/edit escape hatch.
- **Observable activity + generated-file previews**: each live turn can show a
  collapsible work-status timeline with reader-visible request/tool/verification
  summaries and a **Stop** control. It never requests or renders hidden
  chain-of-thought. Complete fenced files can opt into a latest-revision preview
  ledger, and a deduplicated **Changed files** section is appended after the
  answer. See [`ACTIVITY_AND_FILE_PREVIEW_GUIDE.md`](ACTIVITY_AND_FILE_PREVIEW_GUIDE.md).
- **Standalone AI search-bar** (opt-in, default off): an additive search input
  that forwards text into the panel; never touches the theme's own search
- **API mode now uses a configurable proxy** (`ai_assistant_panel_api_url`).
  A browser cannot call Anthropic directly (no CORS, key would leak), so API
  mode must point at your own proxy that injects the key server-side. With no
  proxy set, API mode shows a clear, actionable message instead of failing
  silently.

See [`_example_conf.py`](_example_conf.py) for every new option, its type,
default, and rationale.

### Optional separate-origin isolation

For deployments that do not want documentation-origin scripts to have ambient
access to assistant DOM, transcript, preferences, model state, or management
receipts, configure `ai_assistant_isolation_origin` to a **distinct HTTPS
origin**. Isolation is fail-closed: when requested, the full same-origin runtime
is suppressed even if the host bridge or frame handshake fails.

The parent page exposes only a small versioned capability bridge for bounded
page context, canonical Markdown reads, print, UI sizing, and separately
consented public integration events. B42 protocol 2.0.0 uses a build-generated
exact parent-origin policy and a **frame-generated WebCrypto nonce** that never
appears in `iframe.src`; the parent consumes the valid HELLO before later page
listeners can observe it, then transfers one `MessageChannel`. Runtime messages
are bounded, exactly sequenced, and capability-allowlisted. Configuration and
endpoint descriptors are snapshotted/sanitized at host startup, and isolated Web
Storage is namespaced by parent origin + docs-root path.

The isolated frame cannot self-navigate HTTP(S) onto the docs origin, popup
sandbox escape is not granted, and cross-origin microphone permission is an
independent site-owner opt-in. Assistant-service fetches omit ambient cookies by
default; a separate compatibility flag can permit only same-origin credentials.

This reduces the same-origin confidentiality surface; it does **not** make a
fully compromised parent page trustworthy. Production deployments must also
serve the isolated origin with restrictive response headers. See
[`ISOLATION_DEPLOYMENT.md`](ISOLATION_DEPLOYMENT.md) for the deployment contract
and residual threat boundary.

### Endpoint profiles — one service, flexible routes

Endpoint profiles use one absolute `base` service URL. Each feature endpoint can
then be configured in any of three forms:

- **absolute** — `https://proxy.example.com/v1/share`
- **relative** — `v1/share` or `/v1/share` (joined beneath `base`)
- **inherited** — `""`, `None`, or omitted (uses `base` + the built-in default route)

Surrounding whitespace is trimmed. Endpoint values are bounded and canonicalised
before use. The browser rejects embedded URL credentials, fragments, protocol-
relative authorities, private/reserved runtime hosts, control/bidi characters,
ambiguous backslashes, traversal (including encoded forms), invalid percent-
encoding, overlong paths/queries, and non-HTTP(S) schemes. Relative routes cannot
switch authority/scheme and are always resolved beneath `base`. Old custom
profiles restored from browser storage are re-sanitised before use. Build-time
`conf.py` private/local hosts remain available for trusted local-development
workflows but emit a privacy-safe Sphinx warning.

`datasetRepo` is optional metadata and is normally auto-discovered from
`GET {base}/` via `training.dataset_repo`. URL validation is defense-in-depth;
production proxies should still enforce their own destination allowlist/network
policy because client-side lexical validation cannot prove DNS/redirect safety.

```python
ai_assistant_endpoint_profiles = {
    "hf": {
        "label": "Scikit-plots HF",
        "base": "https://scikit-plots-ai.hf.space",
        "chat": "v1/chat/completions",  # relative
        "share": "/v1/share",  # relative with leading slash
        "feedback": "",  # inherit default
        "training": None,  # inherit default
        # "datasetRepo": "scikit-plots/ai-assistant-contributions",
    },
}
ai_assistant_endpoint_default_profile = "hf"
```

Absolute provider-specific endpoints are also supported and are used verbatim,
so heterogeneous deployments can override only the routes that need a different
host or path. Legacy host-only feature values remain compatible.

## Feedback and maintainer review

Feedback is intentionally separate from anonymous telemetry and dataset contribution.
The **Feedback & contribution** workspace exposes three tabs:

```text
[ Feedback ] [ Dataset contribution ] [ Activity ]
```

The **Activity** tab is deliberately a management ledger, not an audit archive. It
keeps only bounded tab-local private receipts that may still be useful. Each tracked
review can be **Forgotten** locally, and each Feedback/Dataset section provides a
two-click **Forget all** action. Forgetting never changes remote provider or dataset
state. Terminal or missing reviews are removed automatically when their status is
checked; the UI does not poll providers in the background.

A quick or detailed rating always updates local state first. With **Maintainer feedback
review / Share with maintainers** Off, no Q&A is uploaded for repository review or
training use. After the reader explicitly enables that permission, the same logical
feedback item keeps one native provider review:

```text
first rating              -> feedback PR/MR #27 · revision 1
same rating/note again    -> no-op
changed quick rating      -> feedback PR/MR #27 · revision 2
saved detailed note       -> feedback PR/MR #27 · revision 3
withdraw                  -> close/remove according to lifecycle state
```

Feedback lives under the Primary target's `feedback/` path. The open PR/MR is
not training-eligible, but it carries future canonical `_source=feedback`,
`trainingStatus=eligible` bytes. On maintainer merge, the Q&A becomes eligible and
retains both the raw rating and server-derived `qualityScore` (`0..1`) /
`qualityPercent` (`0..100`). Quick and detailed buttons are synchronized, and
withdrawal removes the active canonical feedback view and clears the local rating
state so old button selections do not remain visible.

The existing **Send anonymous rating telemetry** switch still controls only
privacy-minimal `/v1/feedback` metadata. `FEEDBACK_PERSIST_ENABLED=false` can make
that telemetry intentionally non-persistent even while the browser permission is On.
It does not disable `/v1/feedback/review`, and telemetry consent never authorizes
content-bearing review.

For provider setup, reviewer behavior, update/no-op rules, withdrawal, persistence,
and troubleshooting, read [`FEEDBACK_REVIEW_GUIDE.md`](_hf_spaces_proxy/FEEDBACK_REVIEW_GUIDE.md).

## Dataset contribution and review

The dataset workflow is deliberately separate from Share and rating feedback. A
reader must open **Contribute to dataset**, choose the exact scope, inspect the
JSON, pass the privacy preflight, and explicitly consent before any conversation
content is sent for dataset review.

For human-maintained repositories, the recommended server-side mode is:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
```

The Primary storage provider then becomes the review authority:

```text
Reader -> Submit for review
       -> PRIMARY provider PR/MR
       -> IN REVIEW / trainingEligible=false
              |
              +-- merge --------> ELIGIBLE on canonical branch
              |
              +-- close/decline -> NOT ACCEPTED
```

Provider mapping:

| Primary provider | Native review object | Maintainer accepts by | Rejects by |
|---|---|---|---|
| Hugging Face | Pull Request | Merge | Close |
| GitHub | Pull Request | Merge | Close |
| GitLab | Merge Request | Merge | Close |
| Bitbucket Cloud | Pull Request | Merge | Decline |

Only the **Primary** decides eligibility. Mirrors are not independent review
authorities. If Hugging Face is Primary and GitHub is a Mirror, a submission
creates an HF review, not a second GitHub review.

The reader receives a private management capability with multiple recovery paths:

- **Save private receipt** — download the private capability as JSON;
- **Copy private withdrawal code** — copy the same authority as compact text for a password manager/private note;
- **Recover withdrawal access** — import/paste either form after reopening the panel or returning later;
- **Check status** — observe open/closed/merged provider-review state;
- **Copy support reference** — copy a **non-secret** receipt/review/`ct_….jsonl` locator for maintainer support;
- **Delete pending / withdraw training use** — close a pending review or record a post-approval training withdrawal.

Closing the panel no longer hides an active capability: reopening the contribution
sheet restores its management actions while the tab still has the receipt. Private
receipt files/codes are the portable path when browser state is gone. Never place the
private receipt/code in an issue, PR, log, URL, or repository file; use the support
reference for maintainer contact instead.

The provider PR/MR and the receipt lifecycle are separate durability concerns. A
provider review can survive a proxy restart while a `memory` receipt ledger cannot.
For production use, configure restart-durable SQLite for one persistent instance or
a shared Redis receipt authority for multiple replicas.

Read [`_hf_spaces_proxy/DATASET_CONTRIBUTION_GUIDE.md`](_hf_spaces_proxy/DATASET_CONTRIBUTION_GUIDE.md) for the
complete user + maintainer workflow, provider-specific review locations, Variables
vs Secrets, topology examples, approval/rejection/withdrawal scenarios, and
troubleshooting. For deep storage/deduplication operations, continue with
[`_hf_spaces_proxy/DATASET_COLLECTION_GUIDANCE.md`](./_hf_spaces_proxy/DATASET_COLLECTION_GUIDANCE.md).


### Re-submission without reviewer queue spam

A pending contribution is now one evolving review, not a sequence of unrelated
reviews. Re-submission from the same management receipt updates the same review. While the browser still holds the management receipt for that logical
conversation scope:

```text
first submit         -> PR/MR #42, revision 1
same content again   -> no-op; PR/MR #42 unchanged
conversation changed -> PR/MR #42, revision 2
changed again        -> PR/MR #42, revision 3
withdraw             -> close/decline PR/MR #42
```

The sheet says **Update existing review** while this continuity is active and
shows the revision number. Maintainers should review the latest commit; older
commits provide an audit trail. Provider review IDs are persisted with the
receipt so normal status/update operations do not scan a 100- or 1000-item
review queue.

This does not globally coalesce identical submissions from different readers.
Independent receipts remain independent because their delete/withdraw authority
must not be merged.

### Minimal provider-native setup

Proxy Variable:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
```

Provider-neutral storage topology is supplied through `RECORD_STORAGE_TARGETS`:

```json
[
  {
    "id": "hf-primary",
    "label": "Hugging Face Dataset",
    "provider": "huggingface",
    "role": "primary",
    "repo": "example-org/assistant-contributions",
    "branch": "main",
    "paths": {"feedback": "feedback", "contributions": "contributions"},
    "token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
    "token_type": "fine-grained",
    "expose_links": true
  }
]
```

The JSON contains only the **name** of the secret environment variable. The actual
provider token belongs in the proxy's secret store.

### Variables versus Secrets

Typical public/non-sensitive Variables:

```text
CONTRIBUTION_REVIEW_MODE
RECORD_STORAGE_TARGETS
ALLOWED_MODELS
HF_SPACES_MODEL_NAMESPACES
ALLOWED_ORIGINS
ALLOWED_ORIGINS_MODE
TRAINING_DATASET_REPO          # legacy repository ID, not normally a secret
```

Typical private Secrets:

```text
HF_TOKEN
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY
AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR
AI_RECORD_STORAGE_TOKEN_GITLAB_*
AI_RECORD_STORAGE_TOKEN_BITBUCKET_*
CONTRIBUTION_REVIEW_TOKEN      # optional API-driven promotion
```

Never put token values inside `RECORD_STORAGE_TARGETS`; use `token_env` names.

## Installation

This directory is the **scikit-plots adapted/bundled extension**, not merely the
original standalone upstream package. Choose the import path that matches how you
ship it.

### Bundled with scikit-plots

```python
extensions = [
    # ...
    "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
]
```

### Vendored into a documentation project

If the extension is copied into a local Sphinx `_sphinx_ext/` package:

```python
extensions = [
    # ...
    "_sphinx_ext._sphinx_ai_assistant",
]
```

Keep the complete extension directory together, including `_static/`, proxy
packages, and any files used by the deployment mode you enable.

The original `sphinx-ai-assistant` project is credited in the source headers, but
this adapted copy contains additional scikit-plots security, privacy, proxy,
contribution, isolation, and provider-storage behavior. Do not assume an upstream
standalone release has the same configuration surface.

## Usage

### Basic Setup

1. Add the extension to your `conf.py` using the import path from the
   [Installation](#installation) section. For the scikit-plots bundled copy:

```python
extensions = [
    # ... your other extensions
    "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
]
```

2. Build your documentation:

```bash
sphinx-build -b html docs/ docs/_build/html
```

That's it! The AI Assistant button will now appear on every page:
- Main button: Copy page as Markdown
- Dropdown:
  - Copy or view page as Markdown
  - Ask Claude and ChatGPT
  - Connect to MCP server in VS Code and Claude Desktop

### Configuration

For details, see [`_example_conf.py`](_example_conf.py)

You can customize the extension in your `conf.py`:

```python
# Enable or disable the extension (default: True)
ai_assistant_enabled = True

# Button position: 'sidebar' or 'title' (default: 'sidebar')
# 'sidebar': Places button in the right sidebar (above TOC in Furo)
# 'title': Places button near the page title
ai_assistant_position = "sidebar"

# CSS selector for content to convert (default: 'article')
# For Furo theme, you might want: 'article'
# For other themes, adjust as needed
ai_assistant_content_selector = "article"

# Enable/disable specific features (default: as shown)
# CRITICAL: Always supply ALL keys explicitly.  If any key is absent the JS
# widget falls back to its FEATURE_DEFAULTS where ai_panel = false — this
# silently hides the AI-panel button even if you expect it to appear.
ai_assistant_features = {
    "markdown_export": True,  # Copy to clipboard
    "view_markdown": True,  # View as Markdown in new tab
    "ai_chat": True,  # AI chat links
    "mcp_integration": False,  # MCP tool connect buttons (opt-in)
    "theme_toggle": True,  # Dark/light/system color-scheme toggle
    "pdf_export": True,  # "Export as PDF" button (window.print or custom URL)
    "ai_panel": True,  # Floating AI assistant chat panel
}

# PDF export button
# ─ None / "" → browser print dialog (window.print)
# ─ Non-empty string → opened in a new tab as the PDF download URL
#   Examples:
#     ai_assistant_pdf_export_url = "/_pdf/{pagename}.pdf"
#     ai_assistant_pdf_export_url = "https://docs.example.com/~gitbook/pdf?page=…"
ai_assistant_pdf_export_url = None  # default: browser print dialog

# Show the URL/Print mode toggle below the PDF button (default True).
# Set False to hide the toggle and lock to the mode implied by pdf_export_url.
ai_assistant_pdf_url_mode_toggle = True

# AI assistant panel (floating chat drawer)
ai_assistant_panel_title = "AI Assistant"  # header label in the panel
# Whether readers may permanently show or hide the floating "Ask AI" pill
# themselves, via a switch on the "AI Assistant" dropdown row (default True).
# The switch starts from ai_assistant_panel_start_minimized and stores the
# reader's choice in the browser. Set False to hide the switch and pin the
# pill to the build value. Ignored when features['ai_panel'] is False.
# While a minimized conversation is waiting the pill is pinned visible and the
# switch is locked to match, so the two can never disagree on screen.
ai_assistant_panel_trigger_toggle = True
ai_assistant_panel_placeholder = "Ask a question about this page…"
# False → stub mode (safe for any static build, no API calls)
# True  → live mode through a configured server-side proxy
ai_assistant_panel_api_enabled = False

# Built-in deterministic diagnostic models. Enabled by default both in Sphinx
# and in the standalone JS fallback; set False to remove all six from the UI.
# The separately deployed proxy also has an independent STUB_ENABLED switch.
ai_assistant_panel_stub_models = True
# Available modes: echo, mirror, error, hostile, qa, slow.
# Mirror is the advanced client→server request inspector: it decomposes user
# text, one-turn file text, page context and controls, while redacting known
# secret shapes from the displayed diagnostic answer.

# Conversation persistence capability and default. The transcript remains in
# sessionStorage only; readers may change the switch for their current tab.
ai_assistant_panel_persist = True
ai_assistant_panel_remember_conversation = True

# Reader-facing privacy/runtime initial values. A stored reader choice wins.
ai_assistant_panel_feedback_telemetry_default = False  # privacy-first
ai_assistant_panel_feedback_review_default = True  # set False for local-only/dev checks
ai_assistant_panel_page_integration_default = False  # private event bus by default
ai_assistant_panel_streaming_default = True  # reader preference
# Hard SSE capability ceiling; False disables streaming regardless of preference.
ai_assistant_panel_api_streaming = True

# Build-time markdown generation from topics
ai_assistant_generate_markdown = True

# Patterns to exclude from markdown generation
ai_assistant_markdown_exclude_patterns = [
    "genindex",
    "search",
    "py-modindex",
    "_sources",  # Exclude source files
]

# llms.txt generation
ai_assistant_generate_llms_txt = True
ai_assistant_base_url = "https://docs.example.com"  # Or use html_baseurl

# AI provider configuration
ai_assistant_providers = {
    "claude": {
        "enabled": True,
        "label": "Ask Claude",
        "description": "Ask Claude about this topic.",
        "icon": "anthropic-logo.svg",
        "url_template": "https://claude.ai/new?q={prompt}",
        "prompt_template": "Get familiar with the documentation content at {url} so that I can ask questions about it.",
    },
    "chatgpt": {
        "enabled": True,
        "label": "Ask ChatGPT",
        "description": "Ask ChatGPT about this topic.",
        "icon": "chatgpt-logo.svg",
        "url_template": "https://chatgpt.com/?q={prompt}",
        "prompt_template": "Get familiar with the documentation content at {url} so that I can ask questions about it.",
    },
    # Example: Custom AI provider
    "custom": {
        "enabled": True,
        "label": "Ask Perplexity",
        "url_template": "https://www.perplexity.ai/?q={prompt}",
        "prompt_template": "Analyze this documentation: {url}",
    },
}
```

## How It Works

### Markdown Conversion

When you click "Copy content":

1. The extension identifies the main content area of the page
2. Removes non-content elements (navigation, headers, footers, etc.)
3. Converts the HTML to clean Markdown using [Turndown.js](https://github.com/mixmark-io/turndown)
4. Copies the result to your clipboard
5. Shows a confirmation notification

The converted Markdown includes:
- All text content
- Headings (with proper ATX-style formatting)
- Code blocks (with language syntax highlighting preserved)
- Links and images
- Lists and tables
- Block quotes

### AI Chat Integration

When you click "Ask Claude" or "Ask ChatGPT":

**With build-time markdown generation (recommended):**
1. Extension checks if `.md` file exists for current page
2. Opens AI chat with clean URL to markdown file
3. AI can fetch unlimited content directly from your server

**Without markdown generation (fallback):**
1. Converts page to markdown using JavaScript
2. Embeds markdown in URL query parameter
3. Truncates if needed (URL length limits)

## Examples

### Using with AI Tools

After copying a page as Markdown, you can paste it into:

**ChatGPT/Claude:**
```
Here's the documentation for [feature]:

[paste markdown here]

Can you help me understand how to use this?
```

**Cursor/VS Code:**
```
# Context from docs

[paste markdown here]

# Question
How do I implement this in my project?
```

## Development

### Project Structure

```text
_sphinx_ai_assistant/
├── __init__.py                        # Sphinx config registration + HTML integration
├── _static/                           # Browser runtime, CSS, icons, isolation frame
├── _hf_spaces_proxy/                  # FastAPI proxy + provider-neutral storage
│   ├── app.py
│   ├── README.md                      # Proxy deployment / Variables / Secrets
│   ├── DATASET_COLLECTION_GUIDANCE.md # Deep storage + dedup operations
│   └── _utils/
├── _cf_worker/                        # Cloudflare Worker proxy alternative
├── tests/                             # Python + registered Node/browser harnesses
├── ISOLATION_DEPLOYMENT.md            # Separate-origin deployment contract
├── _example_conf.py                   # Complete Sphinx configuration example
└── README.md                           # Start here
```

### Building Documentation

```bash
cd docs/
sphinx-build -b html . _build/html
```

Creates:
```
docs/_build/html/
├── index.html
├── index.md          # Generated markdown
├── tutorial.html
├── tutorial.md       # Generated markdown
└── llms.txt          # List of all markdown pages
```

## Theme Compatibility

Currently optimized for:
- **Furo** - Full support with sidebar integration
- **Alabaster** - Supported
- **Read the Docs** - Supported
- **Book Theme** - Supported

The extension should work with most themes but may require CSS adjustments.

## Troubleshooting

### Markdown files not generated

```bash
# Install dependencies
pip install beautifulsoup4 markdownify

# Check configuration
grep ai_assistant_generate_markdown conf.py

# Rebuild
sphinx-build -b html docs/ docs/_build/html
```

### AI chat has no content

1. Check if `.md` file exists:
   ```bash
   curl -I https://your-docs.com/page.md
   ```

2. Check browser console for errors

### Markdown features not working

This happens when `.md` file doesn't exist.

Solution: Generate `.md` files with `ai_assistant_generate_markdown = True`

### Dataset submission says QUARANTINED but no repository review appears

Open the proxy status page and inspect:

```json
"contribution_review_mode": "ledger"
```

`ledger` is the compatibility workflow and does not automatically create a
provider PR/MR. For native repository review set the server Variable:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
```

and restart/redeploy the proxy. If it already reports `provider-pr`, verify the
Primary repository token and branch permissions. See
[`_hf_spaces_proxy/DATASET_CONTRIBUTION_GUIDE.md`](_hf_spaces_proxy/DATASET_CONTRIBUTION_GUIDE.md).

### A provider review exists but Check status returns 404 after restart

The provider review and the contribution receipt ledger have separate durability.
If startup reports `backend=memory durability=process_local`, a restart can lose
the receipt-management authority. Use a persistent SQLite ledger for one instance
or Redis for a shared multi-replica deployment.

## Performance

### Build Time
- Adds few seconds per 100 pages for markdown generation

### Runtime
- **With .md files:** Instant (just opens URL)
- **Without .md files:** 100-500ms for first conversion
- Cached for subsequent uses

### File Size
- Markdown files are 40-60% smaller than HTML
- Example: 45 KB HTML → 18 KB Markdown

## License

This adapted copy contains MIT-origin upstream code and BSD-3-Clause scikit-plots adaptations. Follow the repository-level license files and per-file SPDX headers.

## Questions or Issues?

- Check [`_example_conf.py`](_example_conf.py)
- Read [`_hf_spaces_proxy/DATASET_CONTRIBUTION_GUIDE.md`](_hf_spaces_proxy/DATASET_CONTRIBUTION_GUIDE.md) for dataset review
- Open a scikit-plots issue: https://github.com/scikit-plots/scikit-plots/issues
- For the original upstream project, see https://github.com/mlazag/sphinx-ai-assistant

## Acknowledgments

- Built with [Turndown.js](https://github.com/mixmark-io/turndown) for HTML to Markdown conversion
- Uses [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) and [markdownify](https://github.com/matthewwithanm/python-markdownify) for build-time conversion
- Designed to work seamlessly with the [Furo](https://github.com/pradyunsg/furo) Sphinx theme
- Inspired by the need to make documentation more AI-friendly


### Microphone input selection and push-to-talk

Microphone rows are a real radio group. The dynamically refreshed device list uses one stable delegated interaction handler (`closest(.ai-assistant-mic-device-item)`) so clicks on nested labels/icons remain selectable across enumeration refreshes, and the non-interactive capability status cannot intercept the last row. Choosing a row commits the routing preference immediately so the checkmark and keyboard state follow the user's click without waiting on a temporary capture stream. The selected source is acquired and verified with exact `getUserMedia` constraints only when recording starts; failed or disconnected devices do not silently fall back to another physical microphone. Opening the microphone picker only enumerates devices and never prompts for capture. Permission is requested only from an explicit user action. `ai_assistant_panel_mic_space_shortcut=True` enables hold-Space push-to-talk while interaction is inside the assistant panel; text entry, IME composition, menus/sheets, and page scrolling retain normal Space behavior. Where supported, the selected live `MediaStreamTrack` is passed directly to `SpeechRecognition.start(track)`; older engines are labeled as potentially using the browser system-default speech input.

### Composer file attachments

The footer `+` menu, panel-scoped `Alt+U`, file picker, folder picker, and OS drag/drop all enter one metadata-first resource pipeline. Up to **256 staged files** may be represented without eagerly reading their bodies. Large batches render a bounded card shelf plus Resource Manager; folder/ZIP inventories are separately bounded, paged, cancellable, and require explicit selection before extracted entries become staged resources. Choosing a top-level ZIP now preserves the **original archive as one first-class resource** by default; **Inspect contents** opens the existing traversal/CRC/ZIP-bomb-resistant inventory workflow without replacing the original archive identity.

Preview capability, transport modality, and model capability are intentionally separate. Text/source/notebook files use bounded inert local preview and default to the shared **48,000-character context plane**. JPEG/PNG/WebP/GIF images, SVG/vector documents, PDF, audio, video (for example MP3/WAV/MP4/WebM), spreadsheets/data, ZIP archives, and arbitrary binaries are represented in the **raw resource plane** with a modality hint (`image`, `animated_image`, `vector_image`, `document`, `audio`, `video`, `data`, `archive`, or `binary`). PDF preview remains signature-gated and never embeds a blob-backed PDF iframe; audio/video preview uses user-invoked browser controls with `preload=metadata`; raw SVG is never embedded directly because it is active-capable content. Local preview never implies that the selected AI model supports that modality.

When raw resources are sent, the panel uses `scikitplot-chat-v1` over `multipart/form-data`: one JSON `request` control part plus byte-for-byte `resource:<id>` parts. The browser never base64-encodes large files or authors the multipart boundary. The trusted proxy re-measures every upload, computes SHA-256, performs bounded signature/MIME classification, and routes only through a provider/model adapter that explicitly declares a compatible native/tool/extract/context path. Browser-declared MIME/modality and the model name are diagnostic, not provider authority.

Run 127 adds capability-driven provider planning. The panel discovers the selected model's exact resource routes from the proxy (`/health`, with on-demand `/v1/resource-capabilities?model=...` for models outside the bounded health map) and enforces server byte/count ceilings before building multipart. A `plan-only` model is rejected locally before any raw bytes leave the browser; `stub/mirror` remains an enabled all-modality diagnostic route. Provider identity follows the actual configured upstream path, so an `openai/...` model served through Hugging Face is still governed by the Hugging Face resource adapter. Merely choosing or previewing a resource performs no network upload; Send is the transport boundary.


Run 128 adds a credential-neutral provider execution lifecycle behind that routing
plane. Route planning and provider execution use separate registries; provider-side
opaque file IDs are request-private and never enter route plans, browser state,
exports, or public receipts. Provider resources are prepared sequentially and
released LIFO, with cleanup shielded through cancellation and partial failures.
This common ownership state machine is shared by the OpenAI, Anthropic, Gemini, and
Hugging Face executors and remains the required base for later self-hosted/task-specific
executors.
Historical/export state persists metadata and provenance, not raw local bytes. Same-page previews use a bounded, evictable runtime capability cache, and UI actions derive from live byte authority so an evicted/reloaded PDF/image/media resource cannot keep stale Open/Download controls. Retry/Edit can reuse bounded text context, while raw-file reuse requires bytes still owned by the current runtime or an explicit restage. The legacy `ai-assistant-attach` page-integration event remains a separate **Page attachment hook** action with its independent permission boundary.


### Panel Escape shortcut

`Escape` is panel-scoped and mirrors the header close button when no lighter transient surface owns the key. Active microphone capture or model generation is stopped first; open popups/menus/sheets close before the panel itself. The close button advertises `aria-keyshortcuts="Escape"`.

Run 129 implements the first provider-specific preparation backend: OpenAI resources
can be streamed to the official Files API with short-lived provider ownership and
request-scoped cleanup, while provider file IDs remain server-private.

Run 130 completes the first end-to-end real-provider executor. OpenAI resource
execution becomes `enabled` only when **all** deployment authorities agree: Path 1 is
exactly `https://api.openai.com/v1/chat/completions`,
`BACKEND_RESOURCE_ADAPTER=openai`, and a dedicated `BACKEND_AUTH_TOKEN` is present.
The proxy then uploads verified resources, consumes the private `file-*` handles through
the Responses API, bridges buffered or SSE output back to the panel's provider-neutral
chat vocabulary, and deletes temporary provider files after completion, failure, or
stream cancellation. Native raster images use `input_image`; raw text/PDF documents
use bounded direct `input_file` delivery; archive/data/binary tool routes use Code
Interpreter without client-side archive destruction. OpenAI audio/video remain
unsupported by this adapter unless a later model-specific route is explicitly added.

Run 131 adds the second real-provider executor for the exact official Anthropic
`https://api.anthropic.com/v1/messages` Path-1 route. Verified files are streamed
byte-for-byte to the Anthropic Files API with a one-hour expiration safety net and
request-final deletion; provider file IDs remain private opaque capabilities whose
format is never assumed. Plain text/PDF resources use Messages `document` blocks,
JPEG/PNG/GIF/WebP use `image` blocks, and archive/data/vector/binary resources use
`container_upload` only when the selected Claude family is explicitly documented for
Code Execution. Unknown/future Claude families fail closed for tool routes while their
native image/text/PDF routes remain independently discoverable. An enabled Anthropic
executor also owns text-only requests so the proxy speaks native `/v1/messages` rather
than incorrectly forwarding an OpenAI-Chat-shaped body. Audio/video remain unsupported
until an explicit Anthropic model route exists.

Run 132 makes execution capability **route-level**, not merely provider-level. Every
enabled executor declares the exact modality/route pairs it can perform for the selected
model. Capability discovery intersects provider planning with that executor authority,
and the proxy repeats the check before provider preparation. This prevents an enabled
provider from accidentally advertising an unimplemented `extract`/`tool` route and
ensures a client that bypasses browser preflight still fails before provider I/O.

Run 133 adds the first native **Gemini Files + Interactions** executor. It is enabled
only when Path 1 is exactly `https://generativelanguage.googleapis.com/v1beta/interactions`,
`BACKEND_RESOURCE_ADAPTER=gemini`, and a dedicated `BACKEND_AUTH_TOKEN` is present.
Reviewed general multimodal Gemini models (`gemini-3.8-flash`, `3.7-flash`, `3.6-flash`,
`3.5-flash`, `3.5-flash-lite`, and `gemini-3-flash-preview`) can receive original
image/GIF, audio, video, and PDF resources through the resumable Files API. The proxy
pins the provider-issued upload URL to the official Google origin, streams original
bytes without base64, verifies provider size/SHA-256 metadata when supplied, waits
boundedly for `PROCESSING -> ACTIVE`, and rolls back the provider file on validation,
processing, timeout, or cancellation failure. Interactions output is bridged back to
the provider-neutral Chat shape; thought/signature deltas stay private and streaming
succeeds only after `interaction.completed`.

Run 134 adds reviewed **Gemini File Search** tool routing without weakening the native
media path. File Search-compatible Gemini models may send original text/source files,
ZIP archives, and XLS/XLSX workbooks into one request-scoped File Search store. The
store is created once per chat request, all selected retrieval resources are indexed
with bounded long-running-operation polling, Interactions receives one `file_search`
tool reference, and the store is force-deleted on success, error, or cancellation.
The browser/proxy capability contract advances to resource-transport v3 so each
modality/route can publish a separate `max_file_bytes` and MIME allowlist. This lets a
large native video remain eligible while a ZIP over Gemini's 100 MiB File Search
document ceiling, or an unsupported data MIME such as Parquet, fails locally before
FormData/upload. `/health` sanitizes route constraints to those two public fields only;
provider IDs, URLs, arbitrary metadata, and credentials cannot enter the capability
document. Current File Search support is intentionally narrower than generic `data`: ZIP
and XLS/XLSX are enabled, while ODS/Parquet/Arrow/Feather remain unsupported until the
provider documents and the adapter reviews those MIME types.


Run 135 adds a task-aware **Hugging Face Chat/VLM executor** for Path 3. When no Path-1
backend wins, `HF_TOKEN` is present, and `HF_BASE` is exactly
`https://router.huggingface.co`, ordinary HF text chat is executed through the official
OpenAI-compatible Chat Completion endpoint. Raw files remain fail-closed by default:
only exact reviewed conversational VLMs (plus exact deployment opt-ins from
`HF_RESOURCE_VLM_MODELS`) receive native JPEG/PNG/WebP routes. Browser→proxy bytes stay
raw multipart; only the server-private HF adapter converts a bounded image to the
`image_url` data-URL representation required by Chat Completion. GIF/audio/video/PDF/ZIP/
data/binary are **not** inferred as chat-capable merely because Hugging Face exposes other
task APIs for those modalities. Provider-qualified variants of a reviewed VLM retain the
reviewed base-model authority, while deployment additions are exact model strings.

Run 136 adds a separate **Hugging Face Automatic Speech Recognition task route** rather
than pretending Chat Completion understands audio. The reviewed built-in model
`openai/whisper-large-v3` (plus exact `HF_RESOURCE_ASR_MODELS` deployment opt-ins) gets
`audio/native`; its text resource route is disabled because ASR is not a conversational
LLM. One verified audio file (WAV/MP3/FLAC/OGG/M4A/AAC/WebM vocabulary, bounded locally to
25 MiB) is sent byte-for-byte to the pinned HF Inference task endpoint and the returned
transcript is bridged into the normal assistant-text response. Route constraints now also
carry `max_files`; the panel enforces that count before multipart and the proxy repeats it
before provider preparation, so a single-input task cannot become a multi-file upload
storm when browser preflight is bypassed.

Run 137 adds a provider-neutral **tree-preserving ZIP edit workspace** on the trusted
server side. This is intentionally separate from provider ZIP pass-through: the original
archive remains the tree authority, and a future edit/artifact route may authorize only
replacement of exact existing regular-file paths. The workspace never calls filesystem
extraction APIs and does not materialize an extracted project directory. It rejects adds,
deletes, renames, directory replacement, symlinks/special files, traversal/drive/backslash
paths, Unicode/case/trailing-dot aliases, file/descendant collisions, unsupported
compression, malformed extra fields, compression bombs, and bounded-size overflows.

The reference limits are 4,096 entries, 64 MiB per uncompressed entry, 512 MiB total
uncompressed content, 256 MiB aggregate replacement bytes, and a 500x compression ratio.
Rewrites stream in 1 MiB chunks and spool output after 8 MiB. The implementation preserves
entry order, archive/entry comments, timestamps, permissions/external attributes, internal
attributes, creator system, and well-formed portable extra metadata. The public receipt
contains only source/output hashes, entry count, changed paths, unchanged count, and
preservation booleans—never provider IDs, tokens, or raw provider diagnostics.

Run 138 now performs the higher-fidelity rewrite that Run 137 intentionally deferred.
Untouched entries are copied as exact local-record bytes (local header + already-compressed
payload + optional data descriptor), changed entries alone are regenerated, and the central
directory is rebuilt with safe new offsets. This means a 205-file archive with two edits can
preserve all 203 untouched compressed records exactly instead of decompressing and
recompressing them during the write path. Central ZIP64 transport metadata is normalized
away because the workspace is far below ZIP64 limits.

The stronger fidelity does not weaken validation: the completed archive is reopened and
unchanged payloads still go through decompression/CRC verification, so damaged DEFLATE data
continues to fail closed. After the caller-visible source generation is SHA-256 bound, Run
138 copies it into a bounded server-owned archive spool and performs inspection/rewrite from
that stable snapshot; the caller source is hashed again before and after rewriting. This
closes record-by-record races without creating an extracted project directory.

Run 138 also rejects directory entries carrying hidden file payloads, unsupported
general-purpose flags, multi-disk or archive-level ZIP64 layouts, trailing bytes after EOCD,
malformed local extras, unsupported central-directory adjunct records, and local/central
flag inconsistencies. Self-extracting preambles are treated as wrapper data and are not
copied into the returned project ZIP.

Run 139 connects that primitive to a provider-neutral artifact API at
`POST /v1/artifacts/zip-edit`. The multipart request carries one strict
`scikitplot-zip-edit-v1` manifest, one complete source archive, and one raw replacement part
for each proposed edit. The manifest deliberately separates `authorization.paths` from
`proposal.replacements`: proposal paths must be a subset of user-authorized paths, every
authorized path must resolve to an exact existing regular file in the bound source archive,
and the model/provider never receives authority to add, delete, rename, or choose archive
structure. Source and replacement generations are independently size/SHA-256 bound before
rewrite. Replacement uploads remain spooled/streamed rather than being collected into one
large in-memory byte mapping.

The returned response is the complete verified ZIP (`application/zip`) with a server-derived
filename and path-free bounded receipt headers containing source/output hashes, counts, and
preservation booleans. No provider IDs, model metadata, credentials, URLs, file bodies, or
changed-path list are copied into public provenance. A browser UI must build
`authorization.paths` only from an explicit user selection/approval gesture; it must never
copy a model-produced authorization field into the server manifest.

Run 140 implements that browser boundary as **Edit ZIP safely**. ZIP inventory remains
metadata-only until the reader checks exact existing files. The AI proposal lane is
intentionally narrower than the server's binary-capable artifact primitive: at most eight
reader-selected UTF-8 text files, each at most 32 KiB, are CRC-verified and decoded locally
before model egress. The configured proxy must independently advertise
`scikitplot-zip-edit-v1` on `/health`; otherwise the edit action stays fail-closed. The
browser consumes the server-advertised source/entry/replacement byte ceilings instead of
inventing a second artifact-limit contract.

The model is not asked to return archive paths. For each selected file the browser assigns
an ephemeral opaque proposal id (`f1`, `f2`, …) and sends the selected relative path only as
a reader-visible reasoning label. The strict `scikitplot-zip-text-proposal-v1` response is
`{id, content}` only. Unknown/duplicate ids, extra fields, NUL/binary content, invalid UTF-16,
unauthorized files, oversized replacements, and proposals whose **complete diff** cannot fit
the bounded review surface are rejected. The browser maps accepted opaque ids back to its
own selected paths and separately generates server multipart ids (`r1`, `r2`, …), so model
output never populates `authorization.paths` or a server proposal path.

Before selected text leaves the browser, the existing local privacy preflight warns about
high-confidence credential-like data, conservative personal-information signals, and
invisible/bidi controls. The reader can keep the files local, send unchanged, or send a
redacted model-context copy; any returned replacement is still diffed against the actual
original file. Mixed-newline source files fail closed in this v1 text lane to avoid an
invisible whole-file line-ending rewrite. UTF-8 BOM and a uniform original newline style are
preserved when replacement bytes are constructed.

Application is a second explicit gesture: each proposed file has its own acceptance
checkbox and the reader must confirm that the accepted complete diffs were reviewed. Only
then is `authorization.paths` reconstructed from current UI selection and posted to Run 139.
The source ZIP and replacements are SHA-256 hashed in bounded chunks, the server receipt is
checked for source/output hashes, authorization/applied/tree counts and preservation
booleans, and the original attachment is never silently replaced. Browsers with the File
System Access API stream the verified result to the chosen file; the fallback path itself is
stream-read through a hard 128 MiB ceiling rather than buffering an unknown-length response.


Run 141 extends that client boundary with **typed ZIP edit inputs** without widening server
mutation authority. UTF-8 text remains editable and SVG gains a separate editable source
lane (64 KiB) that must parse as SVG and satisfy a conservative passive-content policy:
DOCTYPE/entities, scripts, event handlers, `foreignObject`, embedded HTML containers,
external/relative URL references, CSS imports, and non-local CSS `url(...)` references fail
closed. Local fragment references and bounded embedded raster data are allowed. SVG is
reviewed as complete source diff; the panel never renders model-produced SVG as trusted
markup during approval.

Raster/animated images, audio, and video are different lanes in Run 141: they are
**read-only model references**, not replacement targets. A media checkbox is enabled only
when the exact active model's discovered resource capability advertises executable raw
routing for that modality, subject to independent 8 MiB image, 16 MiB audio, and 32 MiB
video client ceilings plus the proxy's own resource constraints. The browser assigns `mN`
resource ids, sends bytes through the existing multipart `scikitplot-chat-v1` resource
transport, and the proxy independently measures, hashes, signature-classifies, and routes
them. Model output can still contain only editable `fN` ids; `mN` ids never enter
`authorization.paths` or `proposal.replacements`.

This split is deliberate. “The reader permits the model to inspect this media” is not the
same capability as “the server may mutate this archive path.” Run 141 therefore does not
base64 binary replacements into chat output and does not pretend that a vision/video input
route is a binary-generation route. Binary/media replacement can be added later only behind
a separately discovered output capability or an explicit reader-supplied local replacement,
while Run 139/138 remain the final path/tree authority. The consent copy also states that
raw media bytes (including metadata such as EXIF) are not text-redacted by the local privacy
preflight.

Run 142 implements the first binary-write lane as an **explicit reader-supplied local
replacement**, not as model output. Supported targets are deliberately conservative: PNG,
JPEG, GIF, WebP, WAV, FLAC, Ogg/Opus-family audio, MP3, M4A, MP4/MOV, WebM, MPEG, and
AVI. The inventory exposes `Replace locally…` independently from the model-input checkbox.
A replacement path can enter the server authorization set only from this reader-owned staged
state; `mN` model-reference ids remain read-only and cannot be promoted into write authority.

Before staging, the browser extracts and validates the original archive entry and validates
the chosen local replacement separately. It requires a byte-level container/image signature
matching the archive target, applies 16 MiB image / 32 MiB audio / 64 MiB video client
replacement ceilings (also clamped by the server-advertised per-entry limit), bounds decoded
images/video frames to 16,384 pixels per dimension and 40 million pixels total, and requires
finite media metadata with a maximum one-hour audio or 30-minute video duration. Image
header dimensions are cross-checked against browser decode dimensions. Unsupported formats
such as AVIF remain fail-closed until an equally explicit parser/review lane exists.

Binary review is not a text diff. The panel shows local original/replacement image or media
previews with byte size, dimensions, and/or duration, then requires the same final aggregate
review checkbox before application. Changing any accepted-set checkbox invalidates a previous
review attestation so a newly accepted replacement cannot ride on an earlier approval. The
replacement is whole-file: its metadata is preserved as supplied, while original EXIF/media
metadata is not copied or merged. Preview object URLs are short-lived runtime capabilities
and are revoked when replacements are changed, removed, or the workflow closes.

Run 142 still does **not** infer provider binary output from provider media input. Chat text,
JSON, or base64 returned by a model cannot become a PNG/audio/video replacement. A future
provider-generated binary lane must advertise a separate output-artifact contract and return
server-verifiable binary artifact bytes/provenance; until then only reader-owned local binary
files can use this lane. Run 139 continues to re-hash every replacement and Run 138 continues
to own the complete ZIP tree/rewrite verification.


Run 143 implements that provider-generated branch as a separate, first-class
`scikitplot-provider-artifact-output-v1` contract. It is deliberately not a chat response
format and it is not inferred from `resource_transport`. `/health` advertises a bounded list
of exact output generators plus `chat_text_is_output_authority: false` and
`resource_input_is_output_authority: false`; the browser fails closed if either separation
claim, the receipt contract, limits, or relative output endpoint is missing/malformed.
Diagnostic stub generators remain discoverable for regression but are never shown as normal
user generation actions.

The output request contains only generator id, kind, prompt, target MIME, and bounded options:
there is no archive path, ZIP generation, authorization set, or replacement id. The proxy
invokes a separately registered output executor, validates the returned byte signature,
binds provider/model/generator + prompt SHA-256 + output SHA-256/size/MIME into the bounded
`scikitplot-provider-artifact-output-receipt-v1` receipt, and streams only the verified bytes.
The client independently caps the response, hashes it before accepting the receipt, rejects
receipts containing path authority, then runs the same Run 142 signature/decode/dimension or
duration checks against both the original archive entry and generated candidate.

A generated candidate has a distinct `provider-binary` / `binary-provider` provenance lane.
It becomes a ZIP replacement only after local original/replacement preview, explicit acceptance,
and the aggregate review attestation. Immediately before Run 139 upload the browser hashes the
candidate again and re-checks its provider receipt; only then does reader-owned UI state map that
exact blob to the existing path. The generation receipt itself never grants path authority and
is not copied into the ZIP authorization manifest.

Production binary output is disabled by default. Even when the proxy is pinned to the official
OpenAI backend, Run 143 registers production generators only when the operator separately
includes `openai` in `PROVIDER_ARTIFACT_OUTPUT_ADAPTERS` **and** supplies the distinct
server-only `PROVIDER_ARTIFACT_OPENAI_TOKEN`. The output plane never silently reuses
`BACKEND_AUTH_TOKEN`; operators may deliberately place the same secret in both variables, but
that is an explicit deployment decision. These separate gates prevent chat/resource-input
configuration from silently creating a paid output surface. With that authority, the dedicated
image-generation API supplies PNG/JPEG/WebP
candidates and the dedicated speech API supplies MP3/WAV candidates. Other providers remain
unsupported until they implement the exact output executor contract. Chat/base64 text is never
decoded into a file write.

Run 144 adds a bounded **ephemeral lifecycle** around each provider-generated binary
candidate without turning that lifecycle into archive authority or persistent storage. The
`provider_artifact_output` health capability advances to version 2 and advertises exact
`scikitplot-provider-artifact-lifecycle-v1` and `scikitplot-provider-artifact-cancel-v1`
contracts, a relative cancellation endpoint, a 15-minute candidate TTL, and a short duplicate
suppression window. Version-1 output capability remains readable for compatibility but does not
receive lifecycle authority.

The server lifecycle registry is process-local and metadata-only. It stores generated lifecycle
ids, SHA-256 generation fingerprints, SHA-256 cancellation capabilities, provider/model/generator
identity, prompt SHA-256, output SHA-256/size, state, and bounded timestamps. It never stores the
raw prompt, generated bytes, ZIP paths, provider request/response bodies, or provider secrets.
Candidate bytes remain browser-resident. Registry records are bounded and expire automatically;
a stalled generation is cancelled when its record expires rather than reserving duplicate
authority indefinitely.

Generation now has explicit lifecycle semantics across the server/browser boundary. The server
uses `generating -> ready -> delivered -> reserved -> applied`, while `accepted` / `downloaded`
remain browser-only review events. The transient `reserved` state is acquired atomically for the
full set of correlated provider candidates before ZIP construction; a competing ZIP request cannot
reserve the same lifecycle, and a failed/interrupted ZIP build releases the reservation back to its
exact prior ready/delivered state. The browser owns reader-only review events; the server owns
generation, delivery, expiry, cancellation, reservation, and final ZIP provenance. A cryptographic
cancel token is held only by the browser and only its SHA-256 is retained server-side. Closing the
ZIP editor or pressing **Cancel generation** aborts the local request and calls the dedicated
cancellation endpoint, which cancels the attached provider task when still in flight. Equivalent
requests are suppressed for the bounded duplicate window. A new candidate after an existing one
requires explicit regeneration and carries only its predecessor lifecycle id; regeneration never
revives the old candidate's write eligibility.

Expiry actively releases generated browser blobs/previews and removes server correlation
authority. Reader acceptance is still required independently, and changing the accepted set still
invalidates the aggregate review attestation. At final apply the browser re-hashes the candidate
again and sends only the 32-hex lifecycle id beside the already size/SHA-bound replacement. Run
139 independently validates that lifecycle against the exact bytes. The returned ZIP receipt may
contain only the bounded `provider_artifact_ids` correlation list—never provider prompts, file
paths, generated content, cancellation capabilities, or secrets. One lifecycle id may correlate
to only one replacement in a ZIP request, and after the correlated ZIP body finishes streaming
the lifecycle becomes terminal/non-replayable. A later generation must create a new lifecycle.

Run 145 makes that lifecycle authority **multi-worker safe without making it durable content storage**.
`PROVIDER_ARTIFACT_LIFECYCLE_BACKEND=memory` preserves the Run 144 process-local default;
`redis` moves only lifecycle metadata into one Redis consistency domain. Every duplicate check,
regeneration check, cancellation transition, multi-candidate ZIP reservation, release, and apply
transition is atomic across replicas. Redis keys share one cluster hash tag so the transaction
scripts remain single-slot. Worker shutdown never clears shared records.

Cross-worker cancellation is explicit: the worker receiving the cancellation capability performs
the shared cancellation CAS, while the worker that owns the provider task polls the tiny shared
status record and cancels its local task. A cancellation that lands between `begin()` and task
attachment is re-checked before attachment and fails closed. A bounded shared expiry sweeper runs
while any replica is alive, so idle records are removed at their TTL rather than waiting for a later
user operation; startup also performs one cleanup pass. Redis stores no task object, raw
prompt, generated bytes, ZIP path, provider body, URL, or credential. `PROVIDER_ARTIFACT_LIFECYCLE_REQUIRE_SHARED=true`
lets horizontally scaled deployments reject provider-output/provenance operations unless the
shared authority initialized successfully. Health exposes only coarse backend/shared/ready facts.
Strict Redis transport continues to use the common verified-TLS policy. Production release evidence
now includes a dedicated `providerArtifactLifecycle` Redis plane with TLS, non-default identity,
least-privilege, and replication evidence; persistence/backup is intentionally not required because
loss of ephemeral lifecycle state invalidates candidates rather than recreating write authority.

Run 146 adds the Redis reconnect/chaos boundary. Exact lifecycle Lua mutations are idempotent for a
transport replay of the same operation, preventing ambiguous "committed but reply lost" outcomes from
creating duplicate lifecycle authority. Cancellation watchers tolerate a short Redis failover but
cancel local provider work after bounded consecutive authority failures. Redis topology is explicit:
`PROVIDER_ARTIFACT_LIFECYCLE_REDIS_TOPOLOGY=cluster` uses the Redis Cluster client, requires database
`0`, and keeps all lifecycle script keys in one cluster slot; `standalone` remains the default. A
real-Redis test gate can be made mandatory in Redis-enabled CI with
`RUN146_REDIS_CHAOS_REQUIRED=1`; without a local `redis-server`, only those live cases skip while the
portable reconnect/cluster/security doctor still runs.

Run 147 makes that live authority gate reproducible in parent CI without adding a new runtime
contract. `python _hf_spaces_proxy/ci/run_redis_chaos.py --mode all` requires Redis 7/8 and the
pinned redis-py client, executes the Run 146 standalone real-Lua/restart doctor, then builds a real
six-node Redis Cluster in loopback. The cluster doctor races two OS processes for one ZIP provenance
reservation, kills the primary that owns the `{provider-artifact}` slot, waits for replica
promotion, and proves terminal non-replay plus new lifecycle work after failover. Reference GitHub
Actions and CircleCI fragments run the same harness in isolated Redis 7.4.11 and 8.2.9 container images.


### Run 148 — signed Redis chaos release evidence

Release promotion can now bind Redis 7/8 standalone + cluster chaos results to the
exact extension source-tree SHA-256 and source revision. Chaos images are pinned by
version tag **and** immutable index digest; schema-v2 production evidence rejects
missing/stale/wrong-image attestations and requires a separately verified signer
identity record for every chaos payload. No production Redis authority, prompt, media,
ZIP path, provider token, or generated bytes are included in the attestation.

### Run 149 — transactional release promotion

Production release evidence now feeds a one-shot promotion transaction. The release
layer snapshots the verified source tree, proves the baseline→current patch recreates
that tree, builds and verifies the final ZIP, binds ZIP/patch/SBOM hashes into a
canonical release statement, and requires externally verified signing before final
promotion. `finalize` repeats the source/evidence/patch/ZIP/SBOM checks and atomically
emits a path-safe publish allowlist in `promotion-receipt.json`, preventing a verified
tree from being replaced by later rebuilt bytes.

### Run 150 — receipt-authorized release publication

`security/publish_release.py` consumes the Run 149 `promotion-receipt.json` as the
only publication allowlist. It rebinds the receipt to the included signed release
statement, snapshots every authorized object, and calls a publisher adapter using
create-only semantics followed by independent remote read-back. Interrupted releases
are resumable only when an existing remote object has the exact receipt SHA-256 and
size; overwrite/clobber behavior fails closed. Successful publication emits bounded
`publication-transparency.json` and `publication-receipt.json` evidence without local
paths, access URLs, credentials, prompts, user content, or provider secrets. See
`_hf_spaces_proxy/security/RELEASE_PUBLICATION_GUIDE.md`.

### Run 151 — signed post-publication transparency

`security/finalize_publication.py` now re-validates the complete Run 150 publication
evidence, then requires a separately identified read-only verifier to re-read every
remote release object without reusing publisher credentials. The resulting canonical
in-toto statement makes the exact `publication-transparency.json` SHA-256 its subject.
After an external release trust root signs/verifies that statement and its signer
identity, finalization re-reads the remote objects again and binds one deterministic
`release-publication-record.json` to the same release target using create-only + remote
read-back semantics. Retries reproduce the same final-record bytes, while fresh
verifier timestamps remain local audit evidence. See
`_hf_spaces_proxy/security/RELEASE_TRANSPARENCY_GUIDE.md`.

### Run 152 — append-only transparency and threshold witnessing

`security/witness_publication.py` now treats Run 151's deterministic
`release-publication-record.json` as an immutable external transparency subject. The
release pipeline must advance from a pinned previous checkpoint, use a distinct
read-only verifier to verify checkpoint signature + exact inclusion + consistency, and
obtain agreement from at least two witness identities across at least two distinct
operators. Only then is the deterministic
`release-transparency-witness-record.json` create-only attached to the original release
target with mandatory remote read-back. See
`_hf_spaces_proxy/security/RELEASE_WITNESS_GUIDE.md`.

### Run 153 — durable release history and cross-release equivocation detection

`_hf_spaces_proxy/security/preserve_release_history.py` now requires every witnessed
release to advance from the exact previously trusted history state and offline bundle.
At least three read-only gossip replicas are configured by default, with an N-of-M
quorum spanning independent operators; an available conflicting history/checkpoint/key
view fails closed rather than being outvoted. Log-key rotation and compromise recovery
use explicit policy-authorized transition records, retired keys cannot return, and
release/publication/witness subjects cannot be replayed at a later sequence.

The deterministic `release-history-bundle.json` is offline-verifiable and is
create-only replicated with exact remote read-back to at least two independent archive
operators so provenance remains auditable if the original transparency service later
disappears. See `_hf_spaces_proxy/security/RELEASE_HISTORY_GUIDE.md`.

### Run 154 — threshold-governed trust roots and disaster recovery

`_hf_spaces_proxy/security/govern_release_history.py` adds a governance plane above the
Run 153 durable release history. Governance genesis is accepted only by an explicit
external SHA-256 pin. Changes to policy authorities, emergency authorities, history
replicas, history archives, or their thresholds are canonical epoch transitions bound to
the exact Run 153 history head and require an M-of-N approval set from the currently
trusted authority. Compromise recovery uses a distinct emergency council and permanently
revokes compromised authority keys.

The same boundary can reconstruct the exact Run 153 trusted history state from an N-of-M
quorum of independent immutable archives. Available disagreement fails closed. Successful
governance produces an offline-verifiable governance bundle plus a self-contained
recovery snapshot that is create-only replicated and remotely read back from at least
two independent archive operators.

### Run 155 — cryptographic root-of-trust boundary

`_hf_spaces_proxy/security/seal_release_governance.py` makes the Run 154 governance
output a candidate until real Ed25519 threshold authorization succeeds. Bootstrap is
bound to an out-of-band root SHA-256 pin; root/governance/emergency roles are versioned,
expiring, operator-diverse, and root-role separated. Governance signatures are verified
in-process over the exact candidate subject rather than trusting an external
`signatureVerified: true` assertion.

Root rotation follows an old-root + new-root threshold rule and cannot skip versions.
The accepted `release-root-bundle.json` is offline-verifiable, while
`trusted-release-root-state.json` prevents governance/root rollback and binds the current
root chain to the exact accepted governance artifacts.

### Run 156 — delegated freshness and multi-channel root recovery

`_hf_spaces_proxy/security/maintain_release_trust.py` adds short-lived, threshold-signed
snapshot/timestamp metadata above the Run 155 cryptographic seal. A root-authorized
`release-delegation-root.json` defines disjoint snapshot and timestamp roles; the snapshot
binds every Run 155 seal artifact and the timestamp binds the exact snapshot. Monotonic
versions, root-chain-prefix rebinding, bounded expiry, and live freeze checks prevent
rollback to an otherwise valid but stale release view.

Break-glass root recovery uses a **separately pinned recovery root**, not the potentially
compromised release-root threshold. Recovery requires fresh Ed25519 M-of-N signatures
across independently pinned recovery channels and operators. The replacement root must
self-satisfy its new-root threshold, permanently exclude declared compromised root keys,
and preserve governance/emergency policy authority exactly. See
`_hf_spaces_proxy/security/RELEASE_DELEGATED_TRUST_GUIDE.md`.

### Run 157 — recovered-root continuity and X.509 key attestation

Run 157 makes an accepted Run 156 replacement root the only valid predecessor for later
cryptographic governance/root rotation. It adds provider-neutral X.509 attestation of
every recovered/new root key against independently SHA-256-pinned attestation trust roots
and a self-contained offline-verifiable continuity bundle. Details are in
`_hf_spaces_proxy/security/RELEASE_ROOT_CONTINUITY_GUIDE.md`.

### Run 158 — hardware attestation lifecycle and revocation

Run 158 adds root-threshold-governed attestation CA rotation, independent threshold-signed
certificate-status snapshots, sticky revocation with historical-time semantics, and
certificate-bound vendor/device claim verification on top of Run 157. See
`_hf_spaces_proxy/security/RELEASE_ATTESTATION_LIFECYCLE_GUIDE.md`.

### Run 159 — native OCSP/CRL status provenance and anti-equivocation

`_hf_spaces_proxy/security/verify_native_status_provenance.py` independently verifies
raw DER CRLs and OCSP responses for every Run 158 attestation leaf, preserves the exact
native bytes and responder certificate evidence, requires both source kinds by default,
and fails closed on any CRL↔OCSP or native↔Run-158 disagreement. CRL Number and native
update-time continuity prevent status rollback across refreshes. Optional vendor-native
evidence profiles require an explicitly configured executable verifier and otherwise
fail closed. See `_hf_spaces_proxy/security/RELEASE_NATIVE_STATUS_GUIDE.md`.

### Run 160 — immutable native-status evidence archival and recovery

`_hf_spaces_proxy/security/archive_native_status_evidence.py` turns the complete Run 159
native-status output into one deterministic archival subject, replicates it create-only to
independently operated immutable archives, and requires a separate read-only verifier for
each remote copy. Recovery requires byte-identical agreement from multiple independent
archive readers plus out-of-band pins for both the archive SHA-256 and Run 159 chain head;
any available conflicting copy fails closed. See
`_hf_spaces_proxy/security/RELEASE_NATIVE_ARCHIVE_GUIDE.md`.

### Run 161 — cryptographic archive retention and health continuity

`_hf_spaces_proxy/security/audit_archive_retention.py` adds an append-only durability layer
above Run 160. An out-of-band-pinned Ed25519 threshold root authorizes exact archive
membership and migration. Each storage provider signs its immutable version ID,
retention/legal-hold state, and challenge-bound read-back of the exact Run 160 artifact;
a separate read-only auditor signs an independent read-back of that same version.
Migration cannot authorize retirement until every member of the new set passes the live
audit and the minimum independent durable-copy count remains satisfied. See
`_hf_spaces_proxy/security/RELEASE_ARCHIVE_HEALTH_GUIDE.md`.

### Run 162 release-security note

Archive-retention health can be independently witnessed and a compromised retention root
can be recovered without granting retirement authority. See
`_hf_spaces_proxy/security/RELEASE_ARCHIVE_WITNESS_GUIDE.md`.

### Run 163 release-security note

External archive-health anchoring and witness-root recovery are documented in `_hf_spaces_proxy/security/RELEASE_ARCHIVE_ANCHOR_GUIDE.md`. Run 163 hash-links each Run 162 witness head across independent channels with separate observers and fails closed on any split view; recovery cannot rewrite already anchored epochs.

### Run 164 release-security note

Run 164 upgrades Run 163's signed hash-linked external channels to independently verified
RFC6962-style Merkle transparency. `verify_archive_merkle_transparency.py` verifies the
exact Run 163 leaf inclusion, cryptographic consistency from each log's previously
accepted tree root, and an independent cross-channel gossip view. See
`_hf_spaces_proxy/security/RELEASE_ARCHIVE_MERKLE_GUIDE.md`.

### Run 165 release-security note

Run 165 adds threshold-governed Merkle log/gossip key lifecycle above Run 164. Scheduled
rotation requires an explicit old + new checkpoint handoff; compromise recovery uses a
separately pinned recovery quorum and permanently revokes replaced log-key fingerprints.
See `_hf_spaces_proxy/security/RELEASE_ARCHIVE_LOG_AUTHORITY_GUIDE.md`.

### Run 166 — post-recovery Merkle authority continuity

Release-security tooling now includes Run 166, which continues RFC6962 archive-health Merkle trees using the rotated/recovered active log authority accepted by Run 165. The first post-handoff checkpoint proves consistency from Run 165's preserved Run 164 checkpoint; retired or compromised keys are not required to append future epochs. See `_hf_spaces_proxy/security/RELEASE_ARCHIVE_MERKLE_CONTINUITY_GUIDE.md`.

### Run 167 — recursive Merkle authority re-bridge
Run 167 re-bridges a later scheduled rotation or compromise recovery from the newest accepted Run 166/Run 167 RFC6962 checkpoint, then appends the next leaf under the replacement authority. Append-only epochs remain supported between rotations; see `_hf_spaces_proxy/security/RELEASE_ARCHIVE_MERKLE_REBRIDGE_GUIDE.md`.

### Run 168 release-authority cold recovery

The release-security chain can preserve a complete Run 167 recursive Merkle-authority
checkpoint across independent immutable archives and recover the active log/gossip
authority, permanent revocations, and latest RFC6962 checkpoints using out-of-band
rollback pins. See `_hf_spaces_proxy/security/RELEASE_ARCHIVE_MERKLE_RECOVERY_GUIDE.md`.

### Run 169 — hermetic release-security replay clock


Run 169 removes the wall-clock dependency from the synthetic Run 159 native-status fixture
that feeds Runs 160–168. Release/provenance replay is now deterministic across calendar
time, while production freshness and expiry checks remain live and fail closed. See
`_hf_spaces_proxy/security/RELEASE_TEST_CLOCK_HERMETICITY_GUIDE.md`.

### Run 170 — hermetic external-tool release gates

Run 170 isolates both synthetic and production release subprocess authority. Run 149's
Git patch proof is independent of executor `GIT_*` state, while production `git apply` uses
a system-default or explicitly pinned absolute Git executable with isolated configuration
and no ambient secrets. All publisher/verifier/witness/archive/gossip/recovery subprocesses
receive an explicit minimal environment rather than inheriting the proxy/release executor's
credentials. See `_hf_spaces_proxy/security/RELEASE_PROCESS_HERMETICITY_GUIDE.md`.
