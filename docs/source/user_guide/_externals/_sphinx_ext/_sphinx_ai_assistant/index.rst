.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_ai_assistant

.. _externals-sphinx-ext-sphinx-ai-assistant-index:
.. _externals_-sphinx_ext_-sphinx_ai_assistant_-index:

======================================================================
Sphinx AI Assistant
======================================================================

``_sphinx_ai_assistant`` adds machine-friendly documentation artifacts and an
optional in-page assistant experience to a Sphinx HTML site.  Its build-time
features include per-page Markdown and ``llms.txt`` generation; its browser
features include Markdown/PDF actions, AI-provider deep links, MCP integration,
and an optional assistant panel.

The extension is designed so the static-documentation features can be used
without deploying a model backend.

Start with the smallest deployment
----------------------------------------------------------------------

::

   extensions += [
       "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
   ]

   ai_assistant_enabled = True
   ai_assistant_generate_markdown = True
   ai_assistant_generate_llms_txt = True

The defaults already enable Markdown and ``llms.txt`` generation.  The
assistant panel's network/API mode is disabled by default, so enabling the
extension alone does not require a model credential.

Deployment levels
----------------------------------------------------------------------

A useful way to deploy the extension is incrementally:

``static``
   Generate Markdown/``llms.txt`` and expose local browser actions.  No model
   backend is required.

``panel stub``
   Keep ``ai_assistant_panel_api_enabled = False``.  The panel UI can be tested
   without sending model requests.

``live panel``
   Set ``ai_assistant_panel_api_enabled = True`` and configure a server-side
   proxy URL.  The browser sends the bounded request contract to that proxy;
   provider credentials remain server-side.

``review/contribution``
   Add the separately permissioned feedback/review and dataset-contribution
   services only when the deployment has an explicit storage/review policy.

Core configuration
----------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 18 47

   * - Setting
     - Default
     - Purpose
   * - ``ai_assistant_enabled``
     - ``True``
     - Master switch.
   * - ``ai_assistant_position``
     - ``"sidebar"``
     - Places the launcher in ``sidebar``, ``title``, ``floating`` or ``none``.
   * - ``ai_assistant_content_selector``
     - ``"article"``
     - Browser-side main-content selector.
   * - ``ai_assistant_theme_preset``
     - ``None``
     - Adds theme-specific server-side content selectors.
   * - ``ai_assistant_generate_markdown``
     - ``True``
     - Generate canonical page Markdown after the HTML build.
   * - ``ai_assistant_generate_llms_txt``
     - ``True``
     - Generate a documentation ``llms.txt`` index.
   * - ``ai_assistant_llms_txt_full_content``
     - ``False``
     - Keep ``llms.txt`` as an index rather than embedding every page body.
   * - ``ai_assistant_panel_api_enabled``
     - ``False``
     - Enable live proxy-backed assistant requests.
   * - ``ai_assistant_panel_api_url``
     - ``""``
     - Proxy endpoint used by the live panel.
   * - ``ai_assistant_panel_persist``
     - ``True``
     - Permit same-tab conversation persistence.
   * - ``ai_assistant_panel_remember_conversation``
     - ``True``
     - Initial state of the per-tab remember switch.
   * - ``ai_assistant_isolation_origin``
     - ``""``
     - Optional distinct HTTPS origin for the isolated assistant frame.

The extension has additional provider, endpoint-profile, UI, feedback,
contribution and presentation settings.  Configure only the capability being
deployed; do not copy a large configuration block without understanding its
trust boundary.

Canonical Markdown versus browser copy
----------------------------------------------------------------------

The extension intentionally has two Markdown representations:

* generated ``page.md`` is a published build artifact and can be fetched by
  tools outside the browser;
* clipboard conversion is produced from the live DOM and is a convenience
  representation for the current tab.

Use the generated file as the canonical external retrieval target when both are
available.  A browser ``blob:`` or clipboard result is not a stable published
URL.

Live model requests: keep authority server-side
----------------------------------------------------------------------

Do not put model-provider keys, repository-write tokens, storage credentials or
review credentials in ``conf.py``.  Sphinx configuration can be serialized into
public output, and browser JavaScript cannot protect a provider secret.

A live deployment should therefore follow this authority boundary::

   documentation browser
          |
          | bounded public request
          v
   deployment-controlled proxy
          |
          | provider credential stays here
          v
   model / storage / review service

If separate-origin isolation is enabled, use a distinct HTTPS origin.  The
isolation mode is designed to fail closed rather than silently fall back to the
same-origin assistant runtime.

Conversation persistence
----------------------------------------------------------------------

When panel persistence is enabled, remembered conversation state is scoped to
``sessionStorage`` and therefore to the current browser tab.  The reader can
override the site's initial remember setting.  Invalid or oversized stored
state is cleared rather than trusted.

Feedback ownership
----------------------------------------------------------------------

Do not confuse the assistant's reviewed Q&A workflow with generic page
feedback:

* generic documentation-page feedback is owned by
  :doc:`../_sphinx_feedback/index`;
* assistant Q&A review/contribution uses the assistant proxy's explicit
  review/contribution routes and separate permissions.

The browser should never imply that a local rating has been shared unless the
reader has entered the explicit sharing workflow.

Troubleshooting
----------------------------------------------------------------------

**The static extension works but live chat does not**
   Confirm ``ai_assistant_panel_api_enabled`` and the proxy URL.  A browser
   should not be pointed directly at a provider API that requires a secret.

**The wrong page content is exported**
   Adjust the theme preset or content selectors.  Client-side and server-side
   selectors are distinct settings because they operate on different
   representations.

**You are also enabling ``_sphinx_llm``**
   Choose one owner for overlapping Markdown/``llms.txt`` publication until
   the deployment has explicitly reconciled the two artifact pipelines.  Do
   not rely on extension ordering to resolve two writers targeting the same
   output name.
