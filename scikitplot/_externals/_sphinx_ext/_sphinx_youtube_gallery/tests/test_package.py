"""
Tests for the package surface of ``_sphinx_youtube_gallery``.

Covers the lazy package ``setup`` entry point and the two compatibility
facades (``reference`` and ``_video_options``), which must re-export the
canonical ``_sphinx_youtube_core`` objects with their identity intact.
"""

from __future__ import annotations

import importlib

import pytest

from .. import _video_options, reference
from ..model import CatalogError, parse_video_id

PACKAGE = __package__.rsplit(".", 1)[0]
ROOT = PACKAGE.rsplit(".", 1)[0]


class TestPackage:
    def test_only_setup_is_public(self):
        package = importlib.import_module(PACKAGE)
        assert package.__all__ == ["setup"]
        assert callable(package.setup)

    def test_the_package_documents_itself(self):
        package = importlib.import_module(PACKAGE)
        assert "youtube-gallery" in package.__doc__


class TestReferenceFacade:
    @pytest.mark.parametrize(
        "name",
        ["parse_reference", "parse_video_reference", "YouTubeReference",
         "ReferenceError", "validate_handle", "validate_channel_id",
         "validate_playlist_id", "is_reference_url", "VIDEO", "PLAYLIST", "CHANNEL"],
    )
    def test_names_keep_the_identity_of_the_canonical_grammar(self, name):
        core = importlib.import_module(ROOT + "._sphinx_youtube_core.reference")
        assert getattr(reference, name) is getattr(core, name)

    def test_the_facade_and_the_model_agree_on_what_a_video_is(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxxxxxxxxxxxxxxx"
        parsed = reference.parse_video_reference(url)
        assert parsed.video_id == parse_video_id(url) == "dQw4w9WgXcQ"
        assert parsed.watch_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_a_reference_error_surfaces_as_a_catalog_error_in_the_model(self):
        with pytest.raises(reference.ReferenceError):
            reference.parse_video_reference("https://evil.example/watch?v=dQw4w9WgXcQ")
        with pytest.raises(CatalogError):
            parse_video_id("https://evil.example/watch?v=dQw4w9WgXcQ")


class TestVideoOptionsFacade:
    @pytest.mark.parametrize("name", ["VIDEO_SPEC", "LEAF_VIDEO_SPEC", "player_options"])
    def test_names_keep_the_identity_of_the_canonical_validators(self, name):
        core = importlib.import_module(ROOT + "._sphinx_youtube_core.video_options")
        assert getattr(_video_options, name) is getattr(core, name)

    def test_gallery_options_are_the_namespaced_leaf_options(self):
        assert sorted(_video_options.VIDEO_SPEC) == [
            "video-align",
            "video-aspect",
            "video-height",
            "video-privacy-mode",
            "video-title",
            "video-url-parameters",
            "video-width",
        ]

    def test_player_options_default_the_title_and_drop_a_false_privacy_flag(self):
        options = _video_options.player_options(
            {"video-width": "100%", "video-privacy-mode": "off", "limit": 3},
            "  A   title\n",
        )
        assert options == {"title": "A title", "width": "100%"}
