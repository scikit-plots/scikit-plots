# scikitplot/_externals/_sphinx_ext/_sphinxcontrib_youtube/__init__.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Authors: Dr David Ham, Chris Pickel and others
# SPDX-License-Identifier: BSD-3-Clause

"""
Sphinx "youtube" extension.

..seealso::
  * https://github.com/sphinx-contrib/youtube
  * https://github.com/sphinx-contrib/youtube/commit/5238c057730f953ed7c38316aad692a5231294f1
"""

from . import peertube, utils, vimeo, youtube

# https://github.com/sphinx-contrib/youtube/blob/master/pyproject.toml
# authors = [{name = "Chris Pickel", email = "sfiera@gmail.com"}]
# maintainers = [{name = "David A. Ham", email = "david.ham@imperial.ac.uk"}]
__version__ = "1.5.0"

# https://github.com/sphinx-contrib/youtube
__hash__ = "5238c057730f953ed7c38316aad692a5231294f1"

def _backfill_epub_handlers(app, *_args):
    """
    Give the epub translator the html handlers other extensions omitted.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application.
    *_args
        Ignored; the signature accommodates any Sphinx event this is
        connected to.

    Notes
    -----
    scikit-plots local patch for a pre-existing conflict between this
    extension and any extension that registers nodes for ``html`` only.

    Sphinx's epub builder has format ``html``, so an extension that calls
    ``add_node(MyNode, html=...)`` normally works there by inheritance.
    But the moment *any* extension registers an explicit ``epub=`` handler
    -- as this one does, because an epub reader cannot run an ``<iframe>``
    and needs a link instead -- Sphinx creates a distinct ``epub`` handler
    set, and the html-only nodes are no longer found.

    The result is that ``sphinx_design`` + ``sphinxcontrib-youtube`` in the
    same project aborts ``sphinx-build -b epub`` with::

        NotImplementedError: <HTML5Translator> departing unknown node type:
        PassthroughTextElement

    This predates the scikit-plots changes and reproduces with neither of
    them loaded. Rather than patch ``sphinx_design``, this copies every
    ``html`` handler that has no ``epub`` counterpart into the ``epub``
    set, restoring the inheritance those extensions were relying on. Our
    own explicit ``epub`` handlers are already registered and are left
    alone, so the epub-specific video rendering is preserved.
    """
    handlers = app.registry.translation_handlers
    html_handlers = handlers.get("html", {})
    epub_handlers = handlers.setdefault("epub", {})
    for node_name, pair in html_handlers.items():
        epub_handlers.setdefault(node_name, pair)


def setup(app):
    """Set up Sphinx application."""
    from .._extension_setup import check_namespace

    check_namespace(app, __package__.rsplit(".", 1)[0])

    app.add_node(youtube.youtube, **youtube._NODE_VISITORS)
    app.add_directive("youtube", youtube.YouTube)
    app.add_node(vimeo.vimeo, **utils._NODE_VISITORS)
    app.add_directive("vimeo", vimeo.Vimeo)
    app.add_node(peertube.peertube, **peertube._NODE_VISITORS)
    app.add_directive("peertube", peertube.PeerTube)
    # scikit-plots local patch: see `_backfill_epub_handlers`. Connected to
    # `builder-inited` so it runs after every extension's `setup()` has
    # registered its nodes, whatever order they are listed in.
    # scikit-plots local patch: aggregate cap on latex thumbnail fetches.
    app.add_config_value(
        "video_download_limit", utils.DEFAULT_DOWNLOAD_LIMIT, "env", types=[int]
    )
    app.add_config_value(
        "video_download_max_bytes",
        utils.DEFAULT_DOWNLOAD_MAX_BYTES,
        "env",
        types=[int],
    )
    app.connect("builder-inited", _backfill_epub_handlers)
    app.connect("builder-inited", utils.configure_image_download)
    app.connect("env-merge-info", utils.merge_download_images)
    app.connect("env-updated", utils.download_images)
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
