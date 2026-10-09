.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_youtube_gallery

.. _externals-sphinx-ext-sphinx-youtube-gallery-index:

======================================================================
Sphinx YouTube Gallery
======================================================================

``_sphinx_youtube_gallery`` provides the ``youtube-gallery`` directive for
reviewed local YouTube video/channel catalogs.  It is a typed adapter: YouTube
records are normalized and queried by this package, then card layout and reader
controls are delegated to the generic gallery/collection stack.

Enable it
----------------------------------------------------------------------

::

   extensions += [
       "scikitplot._externals._sphinx_ext._sphinx_youtube_gallery",
   ]

Required sibling extensions are loaded by ``setup()``.  Use one namespace
(``scikitplot...`` or standalone ``_sphinx_ext...``) consistently within a
Sphinx application.

Inline video catalog
----------------------------------------------------------------------

::

   .. youtube-gallery::
      :grid-columns: 1 1 2 2

      videos:
        - id: JXtISpdDPNY
          title: Principal Component Analysis in Python
          channel: Statistics Globe
          tags: [pca, dimensionality-reduction]

A bare YAML list of videos is also accepted.  A directive invocation is
homogeneous: use video records or channel records, not both.

File-backed catalog
----------------------------------------------------------------------

::

   .. youtube-gallery:: ./_data/youtube.yaml
      :group-by: channel
      :sort: -published
      :grid-columns: 1 1 2 2

Source precedence is explicit: inline content, then the positional path, then
``:catalog:``, then ``youtube_catalog_path``.  Exactly one source is used; the
directive does not silently merge multiple catalogs.

The package uses the same bounded YAML and source-tree path confinement as the
generic gallery stack.

Channel views
----------------------------------------------------------------------

A native ``channels:`` catalog renders channel cards.  A video catalog can also
be projected offline into a deduplicated channel view::

   .. youtube-gallery:: ./_data/youtube.yaml
      :view: channels
      :interactive:
      :sort-fields: title,video_count
      :collection-id: learning-channel-index

The projection is deterministic and does not query YouTube during the Sphinx
build.

Build-time query versus browser controls
----------------------------------------------------------------------

Build-time options include channel/playlist/tag/match/date predicates, sorting,
grouping, offset and limit.  ``:interactive:``/``:searchable:`` add the shared
local browser controls **after** the build-time record set has been selected.

The browser UI is progressive enhancement: a complete static gallery remains
readable when JavaScript is unavailable.

Player customization
----------------------------------------------------------------------

``video-*`` options use the validators from
:doc:`../_sphinx_youtube_core/index`.  For example, gallery-wide player width,
aspect, alignment, privacy mode, title and query parameters are translated to
the leaf video directive without maintaining a second option grammar.

``youtube_catalog_max_embeds`` bounds how many video players may be emitted;
``youtube_catalog_path`` supplies a default catalog when a directive does not
provide one explicitly.

Network boundary
----------------------------------------------------------------------

Rendering an existing local catalog does not require a YouTube API request.
Catalog synchronization/fetch helpers are separate operations.  Keep provider
credentials out of RST/YAML and out of generated browser configuration.
