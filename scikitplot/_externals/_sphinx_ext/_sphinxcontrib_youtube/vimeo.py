# scikitplot/_externals/_sphinx_ext/_sphinx_contrib/vimeo.py
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

"""Directive dedicated to the vimeo platform."""

from . import utils


class vimeo(utils.video):
    """Empty video node class."""

    pass


class Vimeo(utils.Video):
    """Custom version of the Video Directive."""

    _node = vimeo
    _thumbnail_url = "https://vumbnail.com/{}.jpg"
    _platform = "vimeo"
    _platform_url = "https://player.vimeo.com/video/"
