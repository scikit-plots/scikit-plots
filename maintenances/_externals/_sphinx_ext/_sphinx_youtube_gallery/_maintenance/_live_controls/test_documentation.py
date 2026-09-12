"""Ensure package documentation covers and imports the supported public surface."""
from pathlib import Path
import sys
import inspect

_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)
PACKAGE = ROOT / "_sphinx_youtube_gallery"
readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
required_sections = [
    "# YouTube Gallery for Sphinx",
    "## Enable the extension",
    "## Small inline catalog",
    "## File-backed catalog",
    "## Custom gallery metadata without losing schema safety",
    "## Build-time catalog query",
    "## Rendering modes and scale",
    "## Live reader controls",
    "## Grid, card, and player customization",
    "## Generic authored gallery",
    "## Catalog schema",
    "## Standalone player",
    "## Build and maintenance behavior",
    "## Migration",
]
for heading in required_sections:
    assert heading in readme, heading
for snippet in [
    ".. youtube-gallery::", "videos:", "channels:", ":interactive:",
    ":filter-fields:", ":sort-fields:", ":grid-columns:",
    ":grid-gutter:", ":card-shadow:", ":video-width:",
    ":video-aspect:", ".. gallery-grid::", ".. youtube::",
    "_sphinx_ext._sphinx_youtube_gallery",
    "scikitplot._externals._sphinx_ext._sphinx_youtube_gallery",
    ":view: channels", "_sphinx_gallery_grid", "fields:", "audience.level",
]:
    assert snippet in readme, snippet

from _sphinx_ext._sphinx_youtube_gallery.model import (
    VideoRecord, ChannelRecord, normalize_catalog, normalize_record,
    normalize_channel_record, normalize_gallery_catalog, derive_channel_records, parse_duration, parse_timestamp,
)
from _sphinx_ext._sphinx_youtube_gallery.query import Query, apply_query, group_records
from _sphinx_ext._sphinx_youtube_gallery.reference import parse_reference, parse_video_reference

# Data/model documentation is intentionally dependency-free and must always be
# inspectable. Directive/browser presentation helpers require Sphinx/docutils;
# inspect them only when those optional build dependencies are installed.
public = [VideoRecord, ChannelRecord, normalize_catalog, normalize_record,
          normalize_channel_record, normalize_gallery_catalog, derive_channel_records,
          parse_duration, parse_timestamp, Query, apply_query,
          group_records, parse_reference, parse_video_reference]
for item in public:
    doc = inspect.getdoc(item)
    assert doc and len(doc) >= 40, getattr(item, "__name__", repr(item))

try:
    from _sphinx_ext._sphinx_youtube_gallery import setup
    from _sphinx_ext._sphinx_youtube_gallery.directive import YouTubeGalleryDirective
    from _sphinx_ext._sphinx_youtube_gallery._video_options import player_options
    from _sphinx_ext._sphinx_collection._browser import collection_id, field_names, metadata_node, record_for_browser
    from _sphinx_ext._sphinx_collection._presentation import forwarded
except ModuleNotFoundError as exc:
    if exc.name not in {"docutils", "sphinx", "sphinx_design"}:
        raise
    print(
        f"README sections/examples and {len(public)} dependency-free public docstrings passed; "
        f"Sphinx-dependent docstrings skipped ({exc.name} not installed)"
    )
else:
    sphinx_public = [setup, YouTubeGalleryDirective, player_options, collection_id,
                     field_names, metadata_node, record_for_browser, forwarded]
    for item in sphinx_public:
        doc = inspect.getdoc(item)
        assert doc and len(doc) >= 40, getattr(item, "__name__", repr(item))
    print(f"README sections/examples and {len(public) + len(sphinx_public)} public docstrings passed")
