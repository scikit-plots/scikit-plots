.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_youtube_core

.. _externals-sphinx-ext-sphinx-youtube-core-index:

======================================================================
Sphinx YouTube core
======================================================================

``_sphinx_youtube_core`` is a dependency-free provider library shared by the
YouTube gallery and standalone player layers.  It owns YouTube reference
grammar and leaf-player option validation only.

It is **not** a Sphinx extension and should not be listed in ``extensions``.

Reference parsing
----------------------------------------------------------------------

The public reference layer recognizes supported YouTube reference forms and
normalizes them into a typed ``YouTubeReference``.  Public helpers include
``parse_reference``, ``parse_video_reference``, ``is_reference_url`` and
validators for handles, channel IDs and playlist IDs.

Keep provider parsing here rather than adding a second URL parser to the gallery
or player directive.  The purpose of this layer is to make pasted references
behave consistently across consumers.

Player option contract
----------------------------------------------------------------------

The shared leaf option validators cover:

* positive ``width`` and ``height`` values with optional ``px``/``%`` units;
* positive ``width:height`` aspect ratios;
* ``left``, ``center`` or ``right`` alignment;
* explicit/common boolean forms for privacy mode;
* one-line titles; and
* normalized query parameters with at most 128 key/value pairs.

Gallery-facing options are a namespaced ``video-*`` view of this same contract.
That keeps ``youtube-gallery`` generated players and standalone ``youtube``
players from drifting on validation semantics.

What this layer deliberately does not own
----------------------------------------------------------------------

It performs no network access and registers no browser/search UI.  Filtering,
pagination, result counts and collection controls belong to
:doc:`../_sphinx_collection/index`; gallery layout belongs to
:doc:`../_sphinx_gallery_grid/index`.
