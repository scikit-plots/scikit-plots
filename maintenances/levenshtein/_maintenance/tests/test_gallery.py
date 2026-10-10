from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
GALLERY = ROOT / "galleries" / "examples" / "levenshtein"


def test_gallery_scripts_parse() -> None:
    scripts = sorted(GALLERY.glob("plot_levenshtein_*_script.py"))
    assert len(scripts) >= 5
    for script in scripts:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def test_gallery_has_no_network_calls() -> None:
    forbidden = ("requests.", "urllib.request", "httpx.", "socket.", "urlopen(")
    for script in GALLERY.glob("plot_levenshtein_*_script.py"):
        source = script.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), script.name
