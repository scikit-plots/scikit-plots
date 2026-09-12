"""Integration checks for RST/MyST metadata, nesting, links and empty results.

Exit code 2 means the optional Sphinx integration stack is unavailable.
"""
from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate

ROOT = activate(__file__)


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.frames = []
        self.links = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "iframe":
            self.frames.append(dict(attrs))
        if tag == "a":
            self.links.append(dict(attrs))

    def handle_data(self, data):
        self.text.append(data)


def main() -> int:
    missing = [
        name
        for name in ("sphinx", "docutils", "sphinx_design", "myst_parser")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        print("SPHINX_RENDERING_UNAVAILABLE: " + ", ".join(missing))
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "source"
        out = Path(tmp) / "html"
        src.mkdir()
        (src / "conf.py").write_text(
            'extensions=["_sphinx_ext._sphinx_youtube_gallery","myst_parser"]\n'
            'source_suffix={".rst":"restructuredtext",".md":"markdown"}\n'
            'myst_enable_extensions=["colon_fence"]\nmaster_doc="index"\nproject="Gallery regression"\n',
            encoding="utf-8",
        )
        (src / "index.rst").write_text(
            "Gallery tests\n=============\n\n.. toctree::\n\n   rst\n   myst\n",
            encoding="utf-8",
        )
        (src / "secret.txt").write_text("SHOULD_NEVER_BE_INCLUDED", encoding="utf-8")
        data = [{
            "id": "JXtISpdDPNY",
            "title": "A `code` [title] <tag>\n.. raw:: html",
            "description": ".. include:: secret.txt\n\n<script>METADATA</script>",
        }]
        (src / "videos.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        (src / "rst.rst").write_text(
            "RST\n===\n\n"
            + "".join(f".. youtube-gallery:: videos.yaml\n   :mode: {mode}\n\n" for mode in ["embed", "thumbnail", "list"])
            + """.. gallery-grid::

   - title: Nested video
     content: |
       .. admonition:: Video
           :class: dropdown

           .. youtube:: JXtISpdDPNY
""",
            encoding="utf-8",
        )
        (src / "myst.md").write_text(
            "# MyST\n\n"
            + "".join(f"```{{youtube-gallery}} videos.yaml\n:mode: {mode}\n```\n\n" for mode in ["embed", "thumbnail", "list"])
            + """~~~~~~~{gallery-grid}
- title: Nested video
  content: |
    ````{admonition} Video
    :class: dropdown

    ```{youtube} JXtISpdDPNY
    ```
    ````
~~~~~~~
""",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(ROOT.parent) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-m", "sphinx", "-b", "html", "-W", "--keep-going", str(src), str(out)]
        run = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if run.returncode:
            print(run.stdout, run.stderr)
            return 1
        for page in ["rst", "myst"]:
            html = (out / (page + ".html")).read_text(encoding="utf-8")
            tags = Tags()
            tags.feed(html)
            assert len(tags.frames) == 2, (page, len(tags.frames))
            assert "SHOULD_NEVER_BE_INCLUDED" not in html
            assert "<script>METADATA</script>" not in html
            assert ".. include:: secret.txt" not in "".join(tags.text)
            assert all(f["src"] == "https://www.youtube.com/embed/JXtISpdDPNY" for f in tags.frames)
            assert "Watch on YouTube" not in "".join(tags.text)
            assert sum(a.get("href") == "https://www.youtube.com/watch?v=JXtISpdDPNY" for a in tags.links) >= 2

        # Empty result remains visible; its existing build warning is expected.
        (src / "rst.rst").write_text(
            "Empty\n=====\n\n.. youtube-gallery:: videos.yaml\n   :match: no-such-title\n",
            encoding="utf-8",
        )
        (src / "myst.md").write_text(
            "# Empty\n\n```{youtube-gallery} videos.yaml\n:match: no-such-title\n```\n",
            encoding="utf-8",
        )
        run = subprocess.run([x for x in cmd if x != "-W"], env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert run.stderr.count("no items matched") == 2, run.stderr
        for page in ["rst", "myst"]:
            assert "No items matched these filters." in (out / (page + ".html")).read_text(encoding="utf-8")

    print("RST/MyST metadata, links, nesting, and empty results passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
