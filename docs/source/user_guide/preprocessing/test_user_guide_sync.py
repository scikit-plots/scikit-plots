from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for parent in HERE.parents:
        if (parent / "scikitplot").is_dir() and (parent / "docs").is_dir():
            return parent
    raise AssertionError("repository root not found")


def _all_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in HERE.glob("*.rst"))


def test_required_pages_exist():
    for name in ['getting_started.rst', 'get_dummies.rst', 'dummy_code_encoder.rst', 'sklearn_integration.rst', 'python_api.rst', 'known_limits_and_troubleshooting.rst']:
        assert (HERE / name).is_file(), name


def test_index_links_required_pages_and_anchor():
    index = (HERE / "index.rst").read_text(encoding="utf-8")
    assert ".. _preprocessing-index:" in index
    for name in ['getting_started.rst', 'get_dummies.rst', 'dummy_code_encoder.rst', 'sklearn_integration.rst', 'python_api.rst', 'known_limits_and_troubleshooting.rst']:
        assert Path(name).stem in index, name


def test_key_public_names_are_documented_and_grounded_in_source():
    repo = _repo_root()
    docs = _all_text()
    source = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (repo / "scikitplot" / "preprocessing").rglob("*")
        if path.is_file() and path.suffix in {".py", ".pyi", ".pyx", ".pxd"}
    )
    for name in ['GetDummies', 'DummyCodeEncoder']:
        assert name in docs, name
        assert name in source, name


def test_user_guide_keeps_limits_visible():
    text = _all_text().lower()
    assert "limit" in text or "troubleshoot" in text


def test_root_user_guide_keeps_module_discoverable():
    repo = _repo_root()
    root_index = (repo / "docs/source/user_guide/index.rst").read_text(encoding="utf-8")
    assert "./preprocessing/index.rst" in root_index or "preprocessing/index" in root_index


def test_gallery_anchor_exists_and_is_linked():
    repo = _repo_root()
    gallery_readme = repo / "galleries/examples/preprocessing/README.txt"
    assert gallery_readme.is_file()
    assert ".. _preprocessing_examples:" in gallery_readme.read_text(encoding="utf-8")
    assert ":ref:`preprocessing_examples`" in _all_text()
