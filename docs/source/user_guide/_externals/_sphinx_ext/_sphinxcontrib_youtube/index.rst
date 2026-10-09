.. currentmodule:: scikitplot._externals._sphinx_ext._sphinxcontrib_youtube

.. _externals-sphinx-ext-sphinxcontrib-youtube-index:

======================================================================
Sphinx video directives
======================================================================

``_sphinxcontrib_youtube`` is Scikit-Plots' vendored leaf-player layer.  It
registers individual ``youtube``, ``vimeo`` and ``peertube`` directives.  It
does not own gallery search, filtering, sorting or pagination; those behaviors
belong to the collection/gallery layers.

Enable it
----------------------------------------------------------------------

::

   extensions += [
       "scikitplot._externals._sphinx_ext._sphinxcontrib_youtube",
   ]

Do not also register another extension that owns the same ``youtube``, ``vimeo``
or ``peertube`` directive names in the same Sphinx application.

YouTube example
----------------------------------------------------------------------

::

   .. youtube:: https://www.youtube.com/watch?v=JXtISpdDPNY
      :title: Principal Component Analysis in Python
      :aspect: 16:9
      :width: 100%
      :privacy_mode:

The YouTube directive accepts supported watch/short/embed references in addition
to a bare video ID.  A start offset present in a recognized URL is carried into
the embed query instead of being discarded.

Shared player options
----------------------------------------------------------------------

The leaf option contract is shared with the YouTube gallery and includes
``width``, ``height``, ``aspect``, ``align``, ``title``, ``privacy_mode`` and
``url_parameters``.  Query strings are normalized and bounded by the shared
provider validator.

For YouTube, an enabled privacy mode uses the ``youtube-nocookie.com`` embed
host.  The generated iframe also receives an accessible title; when no explicit
``:title:`` is provided the player receives a platform-based fallback title.

Thumbnail downloads are opt-in
----------------------------------------------------------------------

``video_download_thumbnails`` defaults to ``False``.  Normal builds therefore
do not need remote thumbnail I/O.  When compatibility downloading is enabled,
additional limits bound request count, per-file bytes and total bytes.

This separation keeps an ordinary HTML build deterministic/offline with respect
to thumbnails unless the project explicitly chooses otherwise.

Other builders
----------------------------------------------------------------------

The extension provides builder-specific visitors rather than assuming every
builder can embed an iframe.  Validate HTML, LaTeX and EPUB output in a project
that relies on non-HTML publication, especially when changing player options or
thumbnail policy.
