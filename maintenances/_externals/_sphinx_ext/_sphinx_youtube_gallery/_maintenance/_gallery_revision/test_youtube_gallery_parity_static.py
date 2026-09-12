"""Dependency-free architectural checks for youtube-gallery/gallery-grid parity."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)
DIRECTIVE = ROOT / "_sphinx_youtube_gallery" / "directive.py"
README = ROOT / "_sphinx_youtube_gallery" / "README.md"
GRID = ROOT / "_sphinx_gallery_grid" / "directive.py"

source = DIRECTIVE.read_text(encoding="utf-8")
tree = ast.parse(source)
classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
klass = classes["YouTubeGalleryDirective"]
methods = {n.name: n for n in klass.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

# The adapter must not recreate gallery-grid's collection root or browser data.
run_text = ast.get_source_segment(source, methods["run"]) or ""
assert "metadata_node" not in source
assert "SEARCHABLE_CLASS" not in source
assert "ensure_assets" not in source
assert "nodes.container(classes=" not in run_text
assert "_parse_gallery_grid(source)" in run_text

# Reader-facing options are forwarded to the one delegated gallery-grid block.
render_text = ast.get_source_segment(source, methods["_render_grid"]) or ""
for token in (
    '"searchable"', '"interactive"', '"filter-fields"', '"sort-fields"',
    '"search-fields"', '"search-label"', '"collection-id"', '"group-by"',
    '"limit"', '"offset"', '"show-count"', '"section-style"',
):
    assert token in render_text, token

# The old youtube-gallery-only visible body prose must never return.
card_text = ast.get_source_segment(source, methods["_card"]) or ""
assert "sd-card-text" in card_text  # negative explanatory comment is intentional
assert "Watch on YouTube" not in card_text
assert 'item["description"]' not in card_text

readme = README.read_text(encoding="utf-8")
assert "one `sk-collection` root" in readme
assert "channels:" in readme
assert ":view: channels" in readme
assert "derive_channel_records" in source
assert 'custom_fields = getattr(record, "fields", {})' in source
assert "_sphinx_gallery_grid" in source
grid_source = GRID.read_text(encoding="utf-8")
assert "_sk_collection_metadata_only" in source
assert "_sk_collection_metadata_only" in grid_source
assert "_source_format(self.env)" in grid_source
assert "_source_format(self.env)" in source
assert 'self.options.get("mode") == "list"' in run_text
assert "static lightweight renderer and cannot" in run_text
print("YOUTUBE_GALLERY_PARITY_STATIC_PASS")

# Live field options share one existence validator in gallery-grid, so a typo
# cannot silently create a useless control in either directive.
assert "has_field(item, field_name)" in grid_source
assert "does not exist in any gallery record" in grid_source

# Card-mode empty selections must still delegate to gallery-grid rather than
# bypassing the shared collection root with youtube-gallery-only prose.
assert 'return [nodes.paragraph(text="No items matched these filters.")]' not in source
assert "Delegating an empty typed selection to gallery-grid" in source

# Source escaping must not leak into local search/sort metadata.
assert '"_sk_collection_browser_title"' in card_text
assert '_BROWSER_TITLE_KEY' in grid_source
assert 'browser_item["title"] = browser_title' in grid_source

# Valueless Sphinx Design flags stay valueless through the forwarding layer.
presentation_source = (ROOT / "_sphinx_collection" / "_presentation.py").read_text(encoding="utf-8")
assert "argument is None and converted is None" in presentation_source
assert 'f":{key}:" if value is None' in grid_source
assert '"show-count": directives.flag' in source
assert 'if "show-count" in self.options:' in render_text
assert 'show_count=query.limit' not in source

# The leaf youtube directive consumes the same option validators as the gallery
# wrapper, including explicit false privacy semantics.
utils_source = (ROOT / "_sphinxcontrib_youtube" / "utils.py").read_text(encoding="utf-8")
youtube_source = (ROOT / "_sphinxcontrib_youtube" / "youtube.py").read_text(encoding="utf-8")
assert "dict(LEAF_VIDEO_SPEC)" in utils_source
assert "_privacy_enabled" in utils_source
assert "utils._privacy_enabled" in youtube_source

# Accessibility-only YouTube link prose is not part of typed-gallery search.
browser_source = (ROOT / "_sphinx_collection" / "_browser.py").read_text(encoding="utf-8")
assert 'item["_sk_collection_search_base"] = [record.title]' in card_text
assert "item.get('_sk_collection_search_base')" in browser_source

# Generated privacy mode is canonical valueless directive syntax.
assert 'if key == "privacy_mode" and value == ""' in card_text
