"""Focused checks for bounded YAML, regexes, responsive players and downloads."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import tempfile


_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)

import _sphinx_ext._sphinx_collection._yaml as bounded
import _sphinx_ext._sphinx_youtube_gallery.model as model
import _sphinx_ext._sphinx_gallery_grid.directive as generic_gallery
from _sphinx_ext._sphinx_collection._yaml import (
    BoundedYAMLError, MAX_YAML_ALIASES, load_bounded_yaml,
)
from _sphinx_ext._sphinx_youtube_gallery.model import CatalogError, normalize_catalog
from _sphinx_ext._sphinx_youtube_gallery.query import Query, apply_query
from _sphinx_ext._sphinxcontrib_youtube import utils


assert load_bounded_yaml("videos:\n  - id: JXtISpdDPNY\n", "inline")["videos"]
with patch.object(bounded, "MAX_YAML_BYTES", 4):
    try:
        load_bounded_yaml("value: long", "byte fixture")
    except BoundedYAMLError as exc:
        assert "bytes" in str(exc)
    else:
        raise AssertionError("oversized YAML accepted")
with patch.object(bounded, "MAX_YAML_DEPTH", 2):
    try:
        load_bounded_yaml("[[[x]]]", "depth fixture")
    except BoundedYAMLError as exc:
        assert "nesting" in str(exc)
    else:
        raise AssertionError("deep YAML accepted")
with patch.object(bounded, "MAX_YAML_NODES", 4):
    try:
        load_bounded_yaml("[a, b, c, d]", "node fixture")
    except BoundedYAMLError as exc:
        assert "values" in str(exc)
    else:
        raise AssertionError("large YAML graph accepted")
with patch.object(bounded, "MAX_YAML_SCALAR_CHARS", 3):
    try:
        load_bounded_yaml("[abcd]", "scalar fixture")
    except BoundedYAMLError as exc:
        assert "scalar text" in str(exc)
    else:
        raise AssertionError("large scalar payload accepted")
aliases = "base: &base x\nitems: [" + ",".join("*base" for _ in range(MAX_YAML_ALIASES + 1)) + "]\n"
try:
    load_bounded_yaml(aliases, "alias fixture")
except BoundedYAMLError as exc:
    assert "aliases" in str(exc)
else:
    raise AssertionError("excessive aliases accepted")

try:
    load_bounded_yaml("value: &self [*self]\n", "recursive fixture")
except BoundedYAMLError as exc:
    assert "recursive" in str(exc)
else:
    raise AssertionError("recursive alias accepted")

with patch.object(model, "MAX_COLLECTION_ITEMS", 1):
    try:
        normalize_catalog(["JXtISpdDPNY", "UNzCG3lw6O0"])
    except CatalogError as exc:
        assert "Split" in str(exc)
    else:
        raise AssertionError("oversized video collection accepted")
with patch.object(generic_gallery, "MAX_COLLECTION_ITEMS", 1):
    try:
        generic_gallery._coerce_items([{}, {}], "generic fixture")
    except ValueError as exc:
        assert "Split" in str(exc)
    else:
        raise AssertionError("oversized generic collection accepted")

catalog = normalize_catalog([
    {"id": "JXtISpdDPNY", "title": "PCA tutorial"},
    {"id": "UNzCG3lw6O0", "title": "Agent skills"},
])
selected, _ = apply_query(catalog, Query(match_regex=r"^PCA.{0,20}$"))
assert [record.id for record in selected] == ["JXtISpdDPNY"]
for unsafe in ("(a+)+$", "a*", "a|b", r"(a)\1", "a{1,}"):
    try:
        Query(match_regex=unsafe)
    except CatalogError:
        pass
    else:
        raise AssertionError(f"unsafe regex accepted: {unsafe!r}")


class Response:
    headers = {"Content-Type": "image/jpeg"}
    def raise_for_status(self): pass
    def iter_content(self, chunk_size):
        yield b"x" * 6
        yield b"y" * 6
    def close(self): self.closed = True


with tempfile.TemporaryDirectory() as directory:
    response = Response()
    app = SimpleNamespace(
        builder=SimpleNamespace(name="latex", status_iterator=lambda values, *args: values),
        outdir=directory,
        config=SimpleNamespace(video_download_limit=1, video_download_max_bytes=10),
    )
    env = SimpleNamespace(video_remote_images={
        "https://i.ytimg.com/example.jpg": Path("_video_thumbnail/example.jpg")
    })
    with patch.object(utils.requests, "get", return_value=response):
        utils.download_images(app, env)
    assert response.closed
    assert not (Path(directory) / "_video_thumbnail/example.jpg").exists()
    assert not (Path(directory) / "_video_thumbnail/example.jpg.part").exists()

print("Bounded YAML, safe regex, and streaming download checks passed")
