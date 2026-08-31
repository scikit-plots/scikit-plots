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
| Let readers submit reviewed dataset content | [Dataset contribution and review](#dataset-contribution-and-review) | Yes |
| Configure HF/GitHub/GitLab/Bitbucket storage | [`DATASET_CONTRIBUTION_GUIDE.md`](./DATASET_CONTRIBUTION_GUIDE.md) | Yes |
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

Level 3  Dataset contribution
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
- **Feedback**: configurable local rating UI plus an optional note. Network
  rating telemetry is a separate explicit permission and never contains Q&A,
  note, model, page URL, or stable conversation identity. Host-page lifecycle
  events use another explicit permission; internal assistant coordination stays
  on a private bus and public events are bounded projections only
- **Keyboard shortcut**: toggle the panel with a configurable chord
  (`ai_assistant_panel_shortcut`, default `Alt+Shift+A`; a modifier is
  required, a bare key is rejected)
- **Privacy & Responsibility sheet**: a built-in, fully customizable in-panel
  explainer that clearly separates the extension's responsibilities from the
  integrated model's
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
        "chat": "v1/chat/completions",     # relative
        "share": "/v1/share",             # relative with leading slash
        "feedback": "",                   # inherit default
        "training": None,                  # inherit default
        # "datasetRepo": "scikit-plots/ai-assistant-contributions",
    },
}
ai_assistant_endpoint_default_profile = "hf"
```

Absolute provider-specific endpoints are also supported and are used verbatim,
so heterogeneous deployments can override only the routes that need a different
host or path. Legacy host-only feature values remain compatible.

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

Read [`DATASET_CONTRIBUTION_GUIDE.md`](./DATASET_CONTRIBUTION_GUIDE.md) for the
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
ai_assistant_position = 'sidebar'

# CSS selector for content to convert (default: 'article')
# For Furo theme, you might want: 'article'
# For other themes, adjust as needed
ai_assistant_content_selector = 'article'

# Enable/disable specific features (default: as shown)
# CRITICAL: Always supply ALL keys explicitly.  If any key is absent the JS
# widget falls back to its FEATURE_DEFAULTS where ai_panel = false — this
# silently hides the AI-panel button even if you expect it to appear.
ai_assistant_features = {
    'markdown_export': True,  # Copy to clipboard
    'view_markdown': True,    # View as Markdown in new tab
    'ai_chat': True,          # AI chat links
    'mcp_integration': False, # MCP tool connect buttons (opt-in)
    'theme_toggle': True,     # Dark/light/system color-scheme toggle
    'pdf_export': True,       # "Export as PDF" button (window.print or custom URL)
    'ai_panel': True,         # Floating AI assistant chat panel
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
ai_assistant_panel_title = "AI Assistant"          # header label in the panel
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

# Conversation persistence capability and default. The transcript remains in
# sessionStorage only; readers may change the switch for their current tab.
ai_assistant_panel_persist = True
ai_assistant_panel_remember_conversation = True

# Build-time markdown generation from topics
ai_assistant_generate_markdown = True

# Patterns to exclude from markdown generation
ai_assistant_markdown_exclude_patterns = [
    'genindex',
    'search',
    'py-modindex',
    '_sources',  # Exclude source files
]

# llms.txt generation
ai_assistant_generate_llms_txt = True
ai_assistant_base_url = 'https://docs.example.com'  # Or use html_baseurl

# AI provider configuration
ai_assistant_providers = {
    'claude': {
        'enabled': True,
        'label': 'Ask Claude',
        'description': 'Ask Claude about this topic.',
        'icon': 'anthropic-logo.svg',
        'url_template': 'https://claude.ai/new?q={prompt}',
        'prompt_template': 'Get familiar with the documentation content at {url} so that I can ask questions about it.',
    },
    'chatgpt': {
        'enabled': True,
        'label': 'Ask ChatGPT',
        'description': 'Ask ChatGPT about this topic.',
        'icon': 'chatgpt-logo.svg',
        'url_template': 'https://chatgpt.com/?q={prompt}',
        'prompt_template': 'Get familiar with the documentation content at {url} so that I can ask questions about it.',
    },
    # Example: Custom AI provider
    'custom': {
        'enabled': True,
        'label': 'Ask Perplexity',
        'url_template': 'https://www.perplexity.ai/?q={prompt}',
        'prompt_template': 'Analyze this documentation: {url}',
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
├── DATASET_CONTRIBUTION_GUIDE.md      # Reader + maintainer contribution workflow
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
[`DATASET_CONTRIBUTION_GUIDE.md`](./DATASET_CONTRIBUTION_GUIDE.md).

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
- Read [`DATASET_CONTRIBUTION_GUIDE.md`](./DATASET_CONTRIBUTION_GUIDE.md) for dataset review
- Open a scikit-plots issue: https://github.com/scikit-plots/scikit-plots/issues
- For the original upstream project, see https://github.com/mlazag/sphinx-ai-assistant

## Acknowledgments

- Built with [Turndown.js](https://github.com/mixmark-io/turndown) for HTML to Markdown conversion
- Uses [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) and [markdownify](https://github.com/matthewwithanm/python-markdownify) for build-time conversion
- Designed to work seamlessly with the [Furo](https://github.com/pradyunsg/furo) Sphinx theme
- Inspired by the need to make documentation more AI-friendly
