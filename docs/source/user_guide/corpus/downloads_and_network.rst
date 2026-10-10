.. currentmodule:: scikitplot.corpus

.. _corpus-downloads-network:

Downloads and network boundaries
================================

:class:`AnyDownloader` is the central URL transfer router. It delegates to
specialists for web/direct downloads, Google Drive, GitHub and YouTube while
preserving the common :class:`DownloadResult` contract.

DownloadPolicy: one transport contract
--------------------------------------

:class:`DownloadPolicy` centralizes the settings that describe transport
security and resource budgets. It contains no credentials, headers or output
paths, so it can be serialized in configuration/evidence without accidentally
serializing authentication material.

Built-in presets keep security controls enabled:

``secure``
    General-purpose defaults: TLS verification, private-IP blocking, bounded
    redirects, retries and 100 MiB maximum transfer.

``constrained`` / ``ci`` / ``docs``
    Smaller timeout/byte/retry budgets for deterministic CI and documentation.
    It does **not** disable TLS or SSRF protection.

``large`` / ``large-files``
    Larger byte/time budget while preserving the same security boundaries.

.. code-block:: python

   from scikitplot.corpus import AnyDownloader, DownloadPolicy

   policy = DownloadPolicy.from_config({
       "preset": "constrained",
       "max_bytes": 8 * 1024 * 1024,
   })
   downloader = AnyDownloader.from_policy(
       "https://example.com/data.csv",
       policy,
   )

Unknown configuration keys and non-boolean security flags fail early. This
prevents values such as ``verify_ssl="false"`` from being interpreted as a
truthy value. Explicit keyword arguments may still override a policy when the
caller intentionally needs a per-transfer exception.


Plan dispatch without touching the network
-----------------------------------------

Use :meth:`AnyDownloader.plan` when code, CI or a user interface needs to show
what **would** happen before any transfer starts.  The returned
:class:`DownloadPlan` uses the same URL-classification/per-URL parameter
resolver as the real specialist construction.  Planning intentionally does not
resolve DNS, issue HEAD/GET requests, allocate a temporary directory or import a
remote SDK.

.. code-block:: python

   from scikitplot.corpus import AnyDownloader, DownloadPolicy

   downloader = AnyDownloader.from_policy(
       "https://github.com/org/repo/blob/main/data.csv",
       DownloadPolicy.constrained(),
       github_token="...",
   )
   plan = downloader.plan()
   print(plan.downloader)              # GitHubDownloader
   print(plan.max_bytes)               # constrained transfer budget
   print(plan.github_token_configured) # True; token value is not serialized

``DownloadPlan`` reports whether credentials/custom headers are configured but
never includes their values.  It is a **dispatch plan**, not a security verdict:
SSRF DNS resolution and remote content-type probing belong to execution-time
network boundaries and are intentionally not disguised as side-effect-free
preflight.  For batch inputs, :meth:`AnyDownloader.plan_all` always returns one
plan per URL in input order.

Builder integration
-------------------

:class:`CorpusBuilder` no longer owns a second direct-download implementation.
Downloadable URLs flow through ``_make_downloader()`` and then through the same
``_make_reader()`` seam used by local files.

This separation matters because transfer policy and parsing policy are different:

.. code-block:: text

   URL -> downloader -> DownloadResult(output_path, suffix, MIME, filename)
                                    |
                                    v
                              DocumentReader

A temporary local filename is implementation detail; provenance can continue to
point at the original URL.

Advanced built-in downloader tuning does not require a custom factory. Put
explicit overrides in :class:`BuilderConfig` ``downloader_kwargs``:

.. code-block:: python

   from scikitplot.corpus import BuilderConfig, CorpusBuilder

   builder = CorpusBuilder(BuilderConfig(
       download_policy="constrained",
       downloader_kwargs={
           "max_redirects": 2,
           "youtube_language": "tr",
       },
   ))

Call-site downloader kwargs take precedence over this mapping; the mapping takes
precedence over ordinary builder defaults. Security-reducing values such as
``verify_ssl=False`` or ``block_private_ips=False`` are therefore explicit user
choices rather than hidden fallbacks.

Security defaults
-----------------

The built-in downloader layer keeps SSRF, redirect, scheme, TLS and size
controls. :class:`CustomDownloader` is intentionally a **trusted-code escape
hatch**: its handler can execute arbitrary Python and could make unrelated
network requests, so the wrapper cannot sandbox it. Instead the wrapper applies
what it can verify independently:

* the original input URL still receives the normal SSRF pre-check by default;
* timeout, size, TLS, redirect and user-agent policy are passed to the handler;
* the returned path must exist and resolve to a regular file;
* it must remain inside the configured output directory unless
  ``allow_external_output=True`` is explicit;
* returned file size is checked against ``max_bytes`` by default even if the
  handler ignored the hint.

A handler remains responsible for applying those transfer-policy arguments to
its own redirects/subrequests. Do not accept an untrusted downloaded Python
callable as a ``CustomDownloader`` handler. If a private network or custom
transport is intentionally allowed, make that decision in the custom
downloader/policy rather than weakening the global default.

Runtime plans
-------------

:class:`RuntimePolicy` defaults to ``allow_network=False``. A Fluent plan with a
URL source therefore requires an explicit network permission at execution time.
This is independent of backend/model download policy: allowing an input URL does
not automatically mean an OCR/ASR/model backend may fetch assets.


Policy/legacy configuration ambiguity
-------------------------------------

``downloader_kwargs`` remains the deliberate per-build override layer.

Do not mix an explicit ``download_policy`` with customized legacy scalar
fields such as ``download_timeout`` or ``max_download_bytes``. That is
ambiguous and :class:`BuilderConfig` rejects it instead of silently ignoring
one side. Put the value in :class:`DownloadPolicy`, or use
``downloader_kwargs`` when an intentional one-build override is required.
