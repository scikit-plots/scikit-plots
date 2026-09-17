"""
Catalog-membership regressions for ``llms.txt`` (slice S-32).

``llms.txt`` was built from ``outdir.rglob("*.md")``, so its membership was
whatever happened to be on disk: a page deleted from the source tree stays in
the output directory until a clean build, and any markdown copied in by
``html_extra_path`` or shipped under ``_static`` looked like documentation the
project publishes. The catalog now lists what this build produced, recorded by
the generator as it produces it.

See Also
--------
scikitplot._externals._sphinx_ext._sphinx_ai_assistant.generate_markdown_files
scikitplot._externals._sphinx_ext._sphinx_ai_assistant.generate_llms_txt
"""

import types
from pathlib import Path

import pytest
from sphinx.builders.html import StandaloneHTMLBuilder

from scikitplot._externals._sphinx_ext import _sphinx_ai_assistant as aia


class _Config(types.SimpleNamespace):
    """Stand-in for ``app.config``; an unset option reads as ``None``."""

    def __getattr__(self, name):
        return None


def _app(outdir, **options):
    """Return a stub application whose builder reports ``outdir``."""
    builder = object.__new__(StandaloneHTMLBuilder)
    builder.outdir = str(outdir)
    defaults = {
        "ai_assistant_generate_markdown": True,
        "ai_assistant_generate_llms_txt": True,
        "ai_assistant_base_url": "https://example.invalid/docs",
        "html_baseurl": "",
        "project": "Probe",
        "ai_assistant_llms_txt_format": "flat",
        "ai_assistant_llms_txt_full_content": False,
        "ai_assistant_llms_txt_max_entries": None,
        "ai_assistant_markdown_exclude_patterns": [],
        "ai_assistant_content_selectors": [],
        "ai_assistant_max_workers": 1,
        # Sphinx supplies defaults for these; the stub must too, or the hook
        # fails on the stub rather than on anything it is being tested for.
        "ai_assistant_strip_tags": ["script", "style"],
    }
    defaults.update(options)
    return types.SimpleNamespace(builder=builder, config=_Config(**defaults))


def _seed(outdir, generated=(), foreign=()):
    """Write generated and foreign markdown, returning the app with a registry."""
    for name in list(generated) + list(foreign):
        target = Path(outdir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {name}\n\nbody\n", encoding="utf-8")
    app = _app(outdir)
    aia.set_generated_markdown(app, list(generated))
    return app


def _entries(text):
    """Return the markdown paths a catalog lists."""
    return [line.strip() for line in (text or "").splitlines()
            if line.strip().endswith(".md")]


def _catalog(outdir):
    """Return the written catalog text, or ``None``."""
    path = Path(outdir) / "llms.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def test_only_this_builds_pages_are_listed(tmp_path):
    """Stale, vendored and source-copy markdown are not members."""
    app = _seed(
        tmp_path,
        generated=["index.md", "guide/install.md"],
        foreign=["stale/removed-page.md", "_static/vendor/CHANGELOG.md",
                 "_sources/raw.md"],
    )
    aia.generate_llms_txt(app, None)
    entries = _entries(_catalog(tmp_path))
    assert len(entries) == 2
    assert not any("removed-page" in e for e in entries)
    assert not any("CHANGELOG" in e for e in entries)
    assert not any("_sources" in e for e in entries)


def test_a_registered_page_missing_on_disk_is_not_listed(tmp_path):
    """The registry is a claim about what was produced, still checked against disk."""
    app = _seed(tmp_path, generated=["index.md"])
    aia.set_generated_markdown(app, ["index.md", "never-written.md"])
    aia.generate_llms_txt(app, None)
    entries = _entries(_catalog(tmp_path))
    assert len(entries) == 1
    assert "never-written" not in "".join(entries)


def test_no_registry_means_no_catalog(tmp_path):
    """Without a registry the hook declines rather than falling back to a scan."""
    (tmp_path / "leftover.md").write_text("# leftover\n", encoding="utf-8")
    app = _app(tmp_path)
    aia.generate_llms_txt(app, None)
    assert _catalog(tmp_path) is None


def test_an_empty_registry_produces_no_catalog(tmp_path):
    """A build that generated nothing advertises nothing."""
    (tmp_path / "leftover.md").write_text("# leftover\n", encoding="utf-8")
    app = _seed(tmp_path, generated=[], foreign=["leftover.md"])
    aia.generate_llms_txt(app, None)
    assert _catalog(tmp_path) is None


def test_the_markdown_hook_records_what_it_produced(tmp_path):
    """The generator writes the registry, so the two hooks share one truth."""
    app = _app(tmp_path)
    aia.generate_markdown_files(app, None)
    assert aia.get_generated_markdown(app) == []


def test_entry_point_ordering_is_preserved(tmp_path):
    """Registered pages keep the entry-point-first ordering."""
    app = _seed(tmp_path, generated=["zzz.md", "index.md", "aaa.md"])
    aia.generate_llms_txt(app, None)
    entries = _entries(_catalog(tmp_path))
    assert entries[0].endswith("index.md")


@pytest.mark.parametrize("failed_build", [RuntimeError("build failed")])
def test_a_failed_build_publishes_no_catalog(tmp_path, failed_build):
    """A build that raised does not get a catalog written for it."""
    app = _seed(tmp_path, generated=["index.md"])
    aia.generate_llms_txt(app, failed_build)
    assert _catalog(tmp_path) is None


# -- S-33, S-34, S-36: cap, bound and failure outcome ----------------------


def _entry_lines(text):
    """Return the markdown entries a catalog lists."""
    return [line.strip() for line in (text or "").splitlines()
            if line.strip().endswith(".md")]


def test_the_cap_is_applied_after_ordering(tmp_path):
    """
    An entry point survives a small cap whatever unrelated pages are named.

    Notes
    -----
    The cap previously sliced a path-sorted list before the entry-point
    reordering ran, so whether ``index.md`` appeared depended on the
    alphabetical position of pages that have nothing to do with it.
    """
    app = _seed(tmp_path, generated=["aaa.md", "bbb.md", "ccc.md", "ddd.md",
                                     "index.md"])
    app.config.ai_assistant_llms_txt_max_entries = 2
    aia.generate_llms_txt(app, None)
    entries = _entry_lines(_catalog(tmp_path))
    assert len(entries) == 2
    assert any(entry.endswith("index.md") for entry in entries)


def test_a_capped_catalog_declares_itself_partial(tmp_path):
    """A consumer can tell a capped catalog from a complete one."""
    app = _seed(tmp_path, generated=["a.md", "b.md", "c.md"])
    app.config.ai_assistant_llms_txt_max_entries = 1
    aia.generate_llms_txt(app, None)
    text = _catalog(tmp_path)
    assert "1" in text and "3" in text
    assert "truncate" in text.lower() or "partial" in text.lower()


def test_an_uncapped_catalog_does_not_claim_truncation(tmp_path):
    """The marker states a fact, so it must be absent when nothing was cut."""
    app = _seed(tmp_path, generated=["a.md", "b.md"])
    aia.generate_llms_txt(app, None)
    text = _catalog(tmp_path)
    assert "truncate" not in text.lower()


def test_an_inlined_catalog_is_bounded_in_bytes(tmp_path):
    """Entry count alone does not bound a file that inlines whole pages."""
    pages = [f"page{n:02d}.md" for n in range(30)]
    for name in pages:
        (tmp_path / name).write_text("# x\n\n" + "y" * 20000, encoding="utf-8")
    app = _app(tmp_path)
    aia.set_generated_markdown(app, pages)
    app.config.ai_assistant_llms_txt_full_content = True
    app.config.ai_assistant_llms_txt_max_bytes = 50_000
    aia.generate_llms_txt(app, None)
    text = _catalog(tmp_path)
    assert len(text.encode("utf-8")) <= 60_000
    assert "truncate" in text.lower()


# -- S-35, S-36: substitution reported, failures escalated -----------------


def test_an_undecodable_page_is_reported_not_silently_substituted(tmp_path):
    """Published text that differs from the source says so."""
    app = _seed(tmp_path, generated=["good.md"])
    (tmp_path / "broken.md").write_bytes(b"# broken\n\n\xff\xfe raw\n")
    aia.set_generated_markdown(app, ["good.md", "broken.md"])
    app.config.ai_assistant_llms_txt_full_content = True
    aia.generate_llms_txt(app, None)
    text = _catalog(tmp_path)
    assert "broken.md" in text
    assert "not valid UTF-8" in text
    assert "differs from the source" in text


def test_a_clean_catalog_makes_no_encoding_claim(tmp_path):
    """The note states a fact, so it is absent when nothing was substituted."""
    app = _seed(tmp_path, generated=["a.md"])
    app.config.ai_assistant_llms_txt_full_content = True
    aia.generate_llms_txt(app, None)
    assert "not valid UTF-8" not in _catalog(tmp_path)


def test_conversion_failures_are_recorded_as_an_outcome(tmp_path):
    """A page that failed to convert is visible, not only logged."""
    app = _app(tmp_path)
    aia.set_conversion_failures(app, [("guide/x.html", "no main content")])
    failures = aia.conversion_failures(app)
    assert failures == [("guide/x.html", "no main content")]


def test_a_build_with_no_failures_reports_none(tmp_path):
    """The ordinary build reports an empty list, not an absent attribute."""
    app = _app(tmp_path)
    aia.generate_markdown_files(app, None)
    assert aia.conversion_failures(app) == []


def test_strict_mode_escalates_a_conversion_failure(tmp_path, monkeypatch):
    """Strict already escalates a missing dependency; a 404 page is the same class."""
    from sphinx.errors import ExtensionError

    app = _app(tmp_path)
    app.config.ai_assistant_strict = True
    monkeypatch.setattr(aia, "_process_html_file_worker",
                        lambda *a, **k: ("error", "guide/x.html", "injected"))
    (tmp_path / "guide").mkdir()
    (tmp_path / "guide" / "x.html").write_text("<html><body>x</body></html>",
                                               encoding="utf-8")
    with pytest.raises(ExtensionError) as excinfo:
        aia.generate_markdown_files(app, None)
    assert "404" in str(excinfo.value)


# -- S-44: membership follows Sphinx, not the output directory -------------


def test_only_documents_sphinx_holds_are_converted(tmp_path):
    """
    A stale HTML file left in the output directory is not a member.

    Notes
    -----
    Sphinx does not purge ``outdir``, so a page deleted from the source tree
    keeps its rendered HTML. Walking the directory converted it again and
    entered it in the registry as though this build had produced it; a real
    incremental build reproduces that, and it is what the Run 07 reasoning
    missed.
    """
    for name in ("alpha.html", "stale.html"):
        (tmp_path / name).write_text(
            "<html><body><main><h1>x</h1><p>body</p></main></body></html>",
            encoding="utf-8",
        )
    app = _app(tmp_path)
    app.env = types.SimpleNamespace(found_docs={"alpha"})
    aia.generate_markdown_files(app, None)
    assert aia.get_generated_markdown(app) == ["alpha.md"]


def test_an_application_without_an_environment_falls_back_to_the_scan(tmp_path):
    """A stub or an unusual builder must not silently produce nothing."""
    (tmp_path / "alpha.html").write_text(
        "<html><body><main><h1>x</h1><p>body</p></main></body></html>",
        encoding="utf-8",
    )
    app = _app(tmp_path)
    aia.generate_markdown_files(app, None)
    assert aia.get_generated_markdown(app) == ["alpha.md"]


def test_nested_documents_are_matched_by_docname(tmp_path):
    """A docname is a path without its suffix, not a bare file name."""
    nested = tmp_path / "guide"
    nested.mkdir()
    (nested / "install.html").write_text(
        "<html><body><main><h1>x</h1><p>body</p></main></body></html>",
        encoding="utf-8",
    )
    app = _app(tmp_path)
    app.env = types.SimpleNamespace(found_docs={"guide/install"})
    aia.generate_markdown_files(app, None)
    assert aia.get_generated_markdown(app) == ["guide/install.md"]
