# scikitplot/_externals/_sphinx_ext/_sphinx_feedback/_example_conf.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Example Sphinx ``conf.py`` settings for the page-feedback extension.

Every ``feedback_*`` value the extension registers is assigned below, with
its default, its allowed values and what it is for. The live assignments
describe one complete, generic site that uses this extension **on its own**
(no AI assistant, any theme). The sections after it show what changes for a
site that also loads the AI assistant, for the two Scikit-Plots sites, and
for the service each site posts to.

Usage
-----
Copy the settings you need into your project's ``conf.py``. The extension is
named by where it is imported from:

* installed library: ``"scikitplot._externals._sphinx_ext._sphinx_feedback"``
* vendored stack in a documentation source tree:
  ``"_sphinx_ext._sphinx_feedback"``

Use one form for every ``_sphinx_ext`` member of a build; mixing the two is
refused at setup.

Quick start
-----------
The smallest configuration that renders working controls::

    extensions = ["scikitplot._externals._sphinx_ext._sphinx_feedback"]
    feedback_page_enabled = True
    feedback_site_id = "my-docs"
    feedback_endpoint = "https://feedback.example.org/v1/feedback"

All three are required: an enabled page with quick or detailed controls and
no ``feedback_endpoint`` stops the build with a configuration error. No
counters are shown until ``feedback_aggregate_file`` names a reviewed
snapshot for this site.

Notes
-----
**What every site must get right.** Three values must agree for submissions
to be accepted, and two for the build to succeed:

1. ``feedback_site_id`` (here) must be listed in the service's
   ``FEEDBACK_ALLOWED_SITE_IDS`` whenever that allowlist is set; otherwise the
   service answers ``422 site_not_allowed``.
2. The site's origin (scheme + host, no path: ``https://user.github.io``,
   not ``https://user.github.io/dev/``) must be in the service's CORS
   allowlist; otherwise the browser blocks the response.
3. The snapshot named by ``feedback_aggregate_file`` must carry the same
   ``site_id``; otherwise the build stops with a configuration error.

**Privacy.** No request is made on page view, and the browser keeps no
cookies or visitor IDs. ``conf.py`` values are published in every page's HTML,
so never put a token or secret here; service credentials live only in the
service's environment.

**Developer note.** This module is documentation that happens to be valid
Python. ``tests/test_sphinx_build.py`` builds a site from its ``feedback_*``
values, so an example that drifts from what the extension accepts fails the
test suite.

See Also
--------
README.md : Design, contracts, service deployment and threat model.
scikitplot._externals._sphinx_ext._sphinx_ai_assistant._example_conf :
    The AI-assistant settings; independent of everything here.
"""

# ---------------------------------------------------------------------------
# 1. Extension
# ---------------------------------------------------------------------------
# The page-feedback extension imports nothing from the AI assistant or AI
# Learn, so it can be listed alone. Listing the AI assistant as well is
# supported; see section 7.
extensions = [
    # ... your other extensions ...
    "scikitplot._externals._sphinx_ext._sphinx_feedback",
]

# ---------------------------------------------------------------------------
# 2. Switch and identity
# ---------------------------------------------------------------------------
# bool, default False. Nothing is rendered and no asset is added while False.
feedback_page_enabled = True

# str, default "docs". A stable ASCII identifier for this logical site. It
# is event data and the key the service's FEEDBACK_ALLOWED_SITE_IDS and the
# snapshot's "site_id" are checked against. Give every public site its own
# value; two sites that share a value share counters.
feedback_site_id = "my-docs"

# str, default "". When set, the snapshot must be pinned to the same
# revision, so feedback on an older revision is not shown as current.
feedback_page_revision = ""

# ---------------------------------------------------------------------------
# 3. Feedback service
# ---------------------------------------------------------------------------
# str, default "". Required while feedback_page_enabled is True (the build
# stops without it). The explicit /v1/feedback URL. HTTPS on the standard port,
# or http://localhost / 127.0.0.1 / [::1] for development. No credentials,
# query or fragment. It is never inherited from AI-assistant endpoint
# profiles, so a chat-only service cannot become feedback authority by
# accident.
feedback_endpoint = "https://feedback.example.org/v1/feedback"

# ---------------------------------------------------------------------------
# 4. Placement (theme-tolerant)
# ---------------------------------------------------------------------------
# "auto" | "sidebar" | "main-bottom" | "floating" | "none"; default "sidebar".
feedback_position = "sidebar"

# bool, default True. Also mirror the controls at the end of the main
# content. Both views share one controller, one request and one state.
feedback_page_main = True

# "main-bottom" | "none"; default "main-bottom". Used when no sidebar
# selector matches, so an unknown theme still shows the controls.
feedback_position_fallback = "main-bottom"

# list[str] or None, default None (built-in list). Tried in order. The
# built-in lists are:
#   sidebar: aside[role="complementary"], .bd-sidebar-secondary, .bd-toc,
#            .toc-sidebar, .toc-drawer, .sphinxsidebar
#   main:    article[role="main"], article.bd-article, div.rst-content,
#            [role="main"], main, div.document, div.body, article
# Set a list only for a theme none of these match; a non-empty list of at
# most 32 selectors replaces the built-in list.
feedback_sidebar_selectors = None
feedback_main_selectors = None

# Escape hatch: put ``.. feedback::`` (option ``:layout: compact|full``) in a
# page to mount the controls exactly there.

# ---------------------------------------------------------------------------
# 5. Which pages
# ---------------------------------------------------------------------------
# fnmatch patterns over Sphinx page names. Exclude wins.
feedback_include = ["**"]
feedback_exclude = ["search", "genindex", "py-modindex", "404"]
# A site that also enables AI Learn adds its pages here, for example
# "learn/**", because those pages own their section/generation feedback.

# ---------------------------------------------------------------------------
# 6. Controls and reviewed counters
# ---------------------------------------------------------------------------
feedback_quick_enabled = True  # thumbs up/down
feedback_detailed_enabled = True  # -5..+5 rating panel
feedback_comment_enabled = True  # optional comment in the panel
feedback_contributor_enabled = True  # optional credit in the panel

# Per-button count side: "left" | "right" for each key.
feedback_buttons_ratings = {
    "left_button_rating": "left",
    "right_button_rating": "right",
}

# bool, default True; "embedded" | "none", default "embedded". Counters are
# build-time data from a reviewed snapshot; nothing is fetched on page view.
feedback_counter_enabled = True
feedback_counter_source = "embedded"

# str, default "". Where the reviewed snapshot (contract
# "page.feedback-aggregate.v3") is read from at build time:
#
#   ""                      no snapshot: counters stay hidden (unknown).
#   "_feedback/agg.json"    no leading slash: a file inside this conf.py's
#                           directory. Use this for your own site.
#   "/page-feedback-aggregate.json"
#                           leading slash: a file shipped inside the
#                           extension's _static directory. That one file is
#                           shared by every site that installs the extension
#                           and holds the scikit-plots-learn snapshot, so
#                           other sites must not point at it.
#
# Neither form may leave its directory, and neither is a browser URL. The
# snapshot's "site_id" must equal feedback_site_id. Mark it "complete": true
# only when it covers every reviewed event for this site; then a page without
# a row shows 0 / 0, otherwise it shows nothing.
feedback_aggregate_file = ""

# ---------------------------------------------------------------------------
# 7. With the AI assistant
# ---------------------------------------------------------------------------
# Nothing above changes. List both extensions; they share no state and no
# configuration. When one public proxy hosts both, derive both URLs from one
# base so they move together, and still set feedback_endpoint explicitly:
#
#   import os
#
#   extensions += ["scikitplot._externals._sphinx_ext._sphinx_ai_assistant"]
#   _PROXY_BASE = (os.environ.get("AI_PROXY_BASE") or "https://proxy.example.org").rstrip("/")
#   ai_assistant_endpoint_profiles = {"default": {"label": "Default", "base": _PROXY_BASE}}
#   feedback_endpoint = (os.environ.get("FEEDBACK_PROXY_BASE") or _PROXY_BASE).rstrip("/") + "/v1/feedback"
#
# The AI assistant's own "Was this helpful?" panel feedback is a separate
# feature (ai_assistant_panel_feedback) and does not replace page feedback.

# ---------------------------------------------------------------------------
# 8. The two Scikit-Plots sites (one shared proxy)
# ---------------------------------------------------------------------------
# Both post to https://scikit-plots-ai.hf.space/v1/feedback. The proxy's
# default FEEDBACK_ALLOWED_SITE_IDS is "scikit-plots-learn,scikit-plots" and
# both origins are in its default CORS list.
#
#   https://scikit-plots-learn.readthedocs.io/en/latest/
#       feedback_site_id = "scikit-plots-learn"
#       feedback_aggregate_file = "/page-feedback-aggregate.json"   # packaged
#
#   https://scikit-plots.github.io/dev/
#       feedback_site_id = "scikit-plots"
#       feedback_aggregate_file = "_page_feedback/aggregate.json"   # docs/source

# ---------------------------------------------------------------------------
# 9. Your own service (environment of the service, never conf.py)
# ---------------------------------------------------------------------------
# Standalone ASGI adapter shipped with the extension:
#
#   uvicorn scikitplot._externals._sphinx_ext._sphinx_feedback._service.app:app
#
#   FEEDBACK_REVIEW_MODE=sqlite
#   FEEDBACK_SQLITE_PATH=/srv/feedback/feedback.sqlite3
#   FEEDBACK_ALLOWED_SITE_IDS=my-docs,my-other-docs     # blank = any site_id
#   FEEDBACK_ALLOWED_ORIGINS=https://docs.example.org,https://user.github.io
#   FEEDBACK_PAGE_AUTHORITY_FILE=/srv/feedback/page-authority.json  # optional
#   FEEDBACK_TRUSTED_PROXY_CIDRS=10.0.0.0/8             # only behind your proxy
#
# On the Scikit-Plots Hugging Face proxy the same FEEDBACK_ALLOWED_SITE_IDS
# variable overrides its default, and ALLOWED_ORIGINS adds browser origins.
