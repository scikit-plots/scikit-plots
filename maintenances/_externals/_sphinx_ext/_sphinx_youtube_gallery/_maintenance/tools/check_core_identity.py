"""Dependency-free proof that compatibility facades point at canonical objects."""
from __future__ import annotations
from pathlib import Path
import sys
MAINT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAINT))
from bootstrap import activate
activate(__file__)
from _sphinx_ext._sphinx_youtube_core import reference as core_ref
from _sphinx_ext._sphinx_youtube_core import video_options as core_opts
from _sphinx_ext._sphinx_youtube_gallery import reference as facade_ref
from _sphinx_ext._sphinx_youtube_gallery import _video_options as facade_opts
assert facade_ref.parse_reference is core_ref.parse_reference
assert facade_ref.YouTubeReference is core_ref.YouTubeReference
assert facade_opts.LEAF_VIDEO_SPEC is core_opts.LEAF_VIDEO_SPEC
assert facade_opts.player_options is core_opts.player_options
print("YOUTUBE_CORE_IDENTITY_PASS")
