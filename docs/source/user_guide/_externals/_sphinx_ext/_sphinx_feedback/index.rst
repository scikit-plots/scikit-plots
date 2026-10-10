.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_feedback

.. _externals-sphinx-ext-sphinx-feedback-index:

======================================================================
Sphinx Feedback
======================================================================

``_sphinx_feedback`` provides privacy-minimal page feedback for Sphinx HTML
sites.  The package separates a dependency-light request/event contract from
the Sphinx UI and optional standalone service.

Page feedback is disabled by default.  Enabling the extension alone does not
make page-view requests.

Enable page feedback
----------------------------------------------------------------------

::

   extensions += [
       "scikitplot._externals._sphinx_ext._sphinx_feedback",
   ]

   feedback_page_enabled = True
   feedback_site_id = "docs"
   feedback_endpoint = "https://feedback.example.org/v1/feedback"

The endpoint may be left empty for a UI-only/static deployment, but a site must
provide an appropriate service endpoint before feedback can be submitted.

::

   # Generic page feedback is an independent reviewed-feedback subsystem. Its
   # endpoint is explicit rather than inherited from the active Assistant profile,
   # so a chat/share-only Cloudflare Worker cannot accidentally become feedback
   # authority. Override FEEDBACK_PROXY_BASE independently in deployments that
   # separate these services. AI Learn pages are excluded because they already own
   # section/generation feedback and should not render a competing page controller.
   _FEEDBACK_PROXY_BASE: str = (
       os.environ.get("FEEDBACK_PROXY_BASE") or _AI_PROXY_BASE
   ).rstrip("/")
   feedback_page_enabled = True
   feedback_position = "sidebar"
   feedback_page_main = True
   feedback_position_fallback = "main-bottom"
   feedback_site_id = "scikit-plots"
   feedback_endpoint = _FEEDBACK_PROXY_BASE + "/v1/feedback"
   feedback_counter_enabled = True
   feedback_counter_source = "embedded"
   # Quick reviewed-count placement is independently configurable per button. The
   # balanced default keeps the counts on the outside edges: [0 | 👎] [👍 | 0].
   feedback_buttons_ratings = {
       "left_button_rating": "left",
       "right_button_rating": "right",
   }
   # Generic page-feedback counters are build-time reviewed data. This complete V3
   # snapshot currently certifies that there are no reviewed generic page-feedback
   # events, so eligible pages may render authoritative 0 / 0 quick counts. Replace
   # or regenerate this packaged extension asset from the full reviewed event set
   # as feedback is merged; never mark a partial export complete merely to make
   # zero counters visible. The leading slash is a logical root-relative asset key;
   # the extension resolves it only inside _sphinx_feedback/_static at build time.
   feedback_aggregate_file = "/page-feedback-aggregate.json"
   feedback_include = ["**"]
   feedback_exclude = ["search", "genindex", "py-modindex", "404"]

Automatic placement and explicit mounts
----------------------------------------------------------------------

The extension can place feedback through theme-aware page/sidebar positions, or
a page can request an explicit mount::

   .. feedback::
      :layout: compact

``:layout:`` accepts only ``compact`` or ``full``.  Explicit mounts still
require ``feedback_page_enabled = True`` because that switch owns the page
assets and controller.

Core configuration
----------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Setting
     - Default
     - Meaning
   * - ``feedback_page_enabled``
     - ``False``
     - Master page-feedback switch.
   * - ``feedback_position``
     - ``"sidebar"``
     - Preferred automatic placement.
   * - ``feedback_page_main``
     - ``True``
     - Permit the main-page placement surface.
   * - ``feedback_quick_enabled``
     - ``True``
     - Show quick feedback actions.
   * - ``feedback_detailed_enabled``
     - ``True``
     - Enable detailed rating UI.
   * - ``feedback_comment_enabled``
     - ``True``
     - Enable the optional comment field.
   * - ``feedback_contributor_enabled``
     - ``True``
     - Enable contributor-related UI supported by the contract.
   * - ``feedback_counter_enabled``
     - ``True``
     - Allow reviewed aggregate counts to be shown.
   * - ``feedback_counter_source``
     - ``"embedded"``
     - Counter source policy.
   * - ``feedback_endpoint``
     - ``""``
     - Submission service endpoint.
   * - ``feedback_site_id``
     - ``"docs"``
     - Logical site authority.
   * - ``feedback_include`` / ``feedback_exclude``
     - all / common utility pages excluded
     - Page-selection patterns.

Privacy and authority boundary
----------------------------------------------------------------------

A feedback event represents a reaction to a page, not a durable person
identity.  The extension does not need a page-view request in order to render
the feedback surface.  Credentials for storage/repository providers belong in
the service deployment, not in public Sphinx configuration.

Do not turn missing aggregate data into a verified zero.  The implementation
only treats an absent page row as zero when the aggregate snapshot explicitly
identifies itself as complete; sparse/unknown aggregate input keeps the counter
unknown.

Standalone service
----------------------------------------------------------------------

The package also exposes service helpers for deployments that want to run the
feedback endpoint separately from Sphinx.  Storage/review providers and trusted
proxy/origin configuration are server concerns.  Keep browser configuration
bounded to the public endpoint and site/page contract.

Relationship to AI Assistant feedback
----------------------------------------------------------------------

This package owns generic page feedback.  Assistant-answer ratings and reviewed
Q&A contribution are separate workflows owned by
:doc:`../_sphinx_ai_assistant/index`; do not merge their consent or storage
semantics merely because both surfaces use feedback language.
