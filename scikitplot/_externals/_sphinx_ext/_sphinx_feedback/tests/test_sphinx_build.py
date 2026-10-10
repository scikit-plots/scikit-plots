"""
Real Sphinx builds: any site, with or without the AI assistant.

Notes
-----
The other tests check the adapter's pieces; these run ``sphinx.application``
end to end, because the failure they guard only appears in a build: a site
whose ``feedback_site_id`` differs from the packaged snapshot stopped at
``config-inited`` ("feedback aggregate site_id does not match
feedback_site_id"), so one Scikit-Plots site built and the other did not.
"""

from __future__ import annotations

import io
import json
import re
import runpy
from pathlib import Path

import pytest

sphinx_application = pytest.importorskip("sphinx.application")
from sphinx.errors import ConfigError  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
EXAMPLE = PACKAGE / "_example_conf.py"
EXTENSION = "_sphinx_ext._sphinx_feedback"
CONFIG_SCRIPT = re.compile(
    r'<script type="application/json" id="sphinx-feedback-config">(.*?)</script>',
    re.DOTALL,
)


def _example_values() -> dict:
    """Return the ``feedback_*`` values ``_example_conf.py`` assigns."""
    namespace = runpy.run_path(str(EXAMPLE))
    return {k: v for k, v in namespace.items() if k.startswith("feedback_")}


ENDPOINT = "https://feedback.example.org/v1/feedback"


def _build(tmp_path: Path, conf: dict, *, extensions=(EXTENSION,), files=None):
    """Build a one-page HTML site and return ``(page_config, html)``."""
    conf = {"feedback_endpoint": ENDPOINT, **conf}
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    lines = [
        f"extensions = {list(extensions)!r}",
        'html_theme = "basic"',
        *(f"{key} = {value!r}" for key, value in conf.items()),
    ]
    (src / "conf.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (src / "index.rst").write_text("Title\n=====\n\nBody.\n", encoding="utf-8")
    for relative, payload in (files or {}).items():
        target = src / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
    app = sphinx_application.Sphinx(
        str(src),
        str(src),
        str(out),
        str(tmp_path / "doctrees"),
        "html",
        status=None,
        warning=io.StringIO(),
        freshenv=True,
    )
    app.build()
    html = (out / "index.html").read_text(encoding="utf-8")
    match = CONFIG_SCRIPT.search(html)
    return (json.loads(match.group(1)) if match else None), html


def _snapshot(site_id: str, *, complete: bool = True) -> dict:
    return {
        "contract": "page.feedback-aggregate.v3",
        "site_id": site_id,
        "complete": complete,
        "pages": {},
    }


def test_example_conf_builds_a_standalone_site(tmp_path):
    values = _example_values()
    config, html = _build(tmp_path, values)
    assert config["site_id"] == values["feedback_site_id"]
    assert config["endpoint"] == values["feedback_endpoint"]
    assert config["page_id"] == "index"
    assert config["counter"] is None  # no snapshot: unknown, not zero
    assert "sphinx-feedback.js" in html
    assert "ai-assistant" not in html


@pytest.mark.parametrize("site_id", ["scikit-plots", "my-docs"])
def test_any_site_id_builds_without_a_snapshot(tmp_path, site_id):
    config, _ = _build(
        tmp_path,
        {"feedback_page_enabled": True, "feedback_site_id": site_id},
    )
    assert config["site_id"] == site_id
    assert config["counter"] is None


def test_site_snapshot_beside_conf_py_gives_known_zero(tmp_path):
    config, _ = _build(
        tmp_path,
        {
            "feedback_page_enabled": True,
            "feedback_site_id": "scikit-plots",
            "feedback_aggregate_file": "_page_feedback/aggregate.json",
        },
        files={"_page_feedback/aggregate.json": _snapshot("scikit-plots")},
    )
    assert config["counter"] == {
        "count": 0,
        "score": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
    }


def test_packaged_snapshot_still_serves_its_own_site(tmp_path):
    packaged = json.loads(
        (PACKAGE / "_static" / "page-feedback-aggregate.json").read_text("utf-8")
    )
    config, _ = _build(
        tmp_path,
        {
            "feedback_page_enabled": True,
            "feedback_site_id": packaged["site_id"],
            "feedback_aggregate_file": "/page-feedback-aggregate.json",
        },
    )
    assert config["site_id"] == packaged["site_id"]
    assert config["counter"]["count"] == 0


def test_packaged_snapshot_of_another_site_fails_closed(tmp_path):
    with pytest.raises(ConfigError, match="site_id does not match"):
        _build(
            tmp_path,
            {
                "feedback_page_enabled": True,
                "feedback_site_id": "scikit-plots",
                "feedback_aggregate_file": "/page-feedback-aggregate.json",
            },
        )


@pytest.mark.parametrize(
    "selector",
    ["../outside.json", "a/../../outside.json", "a\\b.json", "C:/x.json", "a//b.json"],
)
def test_site_snapshot_cannot_leave_the_source_directory(tmp_path, selector):
    (tmp_path / "outside.json").write_text(
        json.dumps(_snapshot("my-docs")), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="feedback_aggregate_file"):
        _build(
            tmp_path,
            {
                "feedback_page_enabled": True,
                "feedback_site_id": "my-docs",
                "feedback_aggregate_file": selector,
            },
        )


def test_site_snapshot_symlink_out_of_the_source_directory_is_refused(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_snapshot("my-docs")), encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    try:
        (src / "link.json").symlink_to(outside)
    except OSError:
        pytest.skip("this platform cannot create symlinks")
    # No conf.py: Sphinx then uses the source directory as confdir (-C).
    with pytest.raises(ConfigError, match="stay inside the documentation source"):
        sphinx_application.Sphinx(
            str(src),
            None,
            str(tmp_path / "out"),
            str(tmp_path / "doctrees"),
            "html",
            confoverrides={
                "extensions": EXTENSION,
                "feedback_page_enabled": True,
                "feedback_site_id": "my-docs",
                "feedback_aggregate_file": "link.json",
                "feedback_endpoint": ENDPOINT,
            },
            status=None,
            warning=io.StringIO(),
            freshenv=True,
        )


def test_builds_alongside_the_ai_assistant(tmp_path):
    pytest.importorskip("_sphinx_ext._sphinx_ai_assistant")
    config, html = _build(
        tmp_path,
        {
            "feedback_page_enabled": True,
            "feedback_site_id": "scikit-plots",
            "feedback_endpoint": "https://feedback.example.org/v1/feedback",
            "ai_assistant_enabled": True,
            "html_baseurl": "https://docs.example.org/",
        },
        extensions=(EXTENSION, "_sphinx_ext._sphinx_ai_assistant"),
    )
    assert config["site_id"] == "scikit-plots"
    assert config["endpoint"] == "https://feedback.example.org/v1/feedback"
    assert html.count('id="sphinx-feedback-config"') == 1
