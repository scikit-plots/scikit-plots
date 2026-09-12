from pathlib import Path
import sys
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from paths import discover_repository_root, discover_runtime_sphinx_ext

def _runtime(root: Path):
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "_extension_setup.py").write_text("")
    (root / "_sphinx_collection").mkdir()
    return root

def test_discovers_wide_repository(tmp_path):
    runtime = _runtime(tmp_path / "scikitplot/_externals/_sphinx_ext")
    (tmp_path / "maintenances/_externals/_sphinx_ext").mkdir(parents=True)
    probe = tmp_path / "maintenances/_externals/_sphinx_ext/x/y.py"
    probe.parent.mkdir(parents=True); probe.touch()
    assert discover_repository_root(probe) == tmp_path
    assert discover_runtime_sphinx_ext(probe) == runtime

def test_discovers_standalone_runtime(tmp_path):
    runtime = _runtime(tmp_path / "bundle/_sphinx_ext")
    probe = runtime / "_sphinx_collection/probe.py"; probe.touch()
    assert discover_runtime_sphinx_ext(probe) == runtime

def test_rejects_unrelated_named_directory(tmp_path):
    fake = tmp_path / "_sphinx_ext"; fake.mkdir(); (fake / "__init__.py").touch()
    with pytest.raises(FileNotFoundError):
        discover_runtime_sphinx_ext(fake)
