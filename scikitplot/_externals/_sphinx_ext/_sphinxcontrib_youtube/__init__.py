# scikitplot/_externals/_sphinx_ext/_sphinx_contrib/__init__.py
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

"""Sphinx "youtube" extension."""

from . import peertube, utils, vimeo, youtube

# https://github.com/sphinx-contrib/youtube/blob/master/pyproject.toml
# authors = [{name = "Chris Pickel", email = "sfiera@gmail.com"}]
# maintainers = [{name = "David A. Ham", email = "david.ham@imperial.ac.uk"}]
__version__ = "1.5.0"


def setup(app):
    """Set up Sphinx application."""
    app.add_node(youtube.youtube, **youtube._NODE_VISITORS)
    app.add_directive("youtube", youtube.YouTube)
    app.add_node(vimeo.vimeo, **utils._NODE_VISITORS)
    app.add_directive("vimeo", vimeo.Vimeo)
    app.add_node(peertube.peertube, **peertube._NODE_VISITORS)
    app.add_directive("peertube", peertube.PeerTube)
    app.connect("builder-inited", utils.configure_image_download)
    app.connect("env-merge-info", utils.merge_download_images)
    app.connect("env-updated", utils.download_images)
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
