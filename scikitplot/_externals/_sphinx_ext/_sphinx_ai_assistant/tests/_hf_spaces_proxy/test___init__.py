# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Package/deployment layout owned by :mod:`_hf_spaces_proxy.__init__`."""
from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import importlib
from pathlib import Path
import shutil
import subprocess
import sys

PROXY = RUNTIME_ROOT / "_hf_spaces_proxy"
UTILS = PROXY / "_utils"
PROVIDERS = PROXY / "_providers"
EXPECTED_PACKAGE_ROOT_PY = {"__init__.py", "app.py", "deduplicate_dataset.py"}
EXPECTED_DEPLOY_ENTRYPOINTS = {"app.py", "deduplicate_dataset.py"}


def test_package_init_is_importable_without_eager_proxy_startup() -> None:
    mod = importlib.import_module(
        "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy"
    )
    assert mod.__name__.endswith("._hf_spaces_proxy")
    assert not getattr(mod, "__all__", ())


def test_proxy_root_contains_only_package_marker_and_supported_entrypoints() -> None:
    assert {p.name for p in PROXY.glob("*.py")} == EXPECTED_PACKAGE_ROOT_PY


def test_docker_copies_private_packages_and_only_deploy_entrypoints() -> None:
    docker = (PROXY / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=1000:1000 _utils ./_utils" in docker
    assert "COPY --chown=1000:1000 _providers ./_providers" in docker
    assert "COPY --chown=1000:1000 app.py deduplicate_dataset.py ./" in docker
    assert "__init__.py" not in EXPECTED_DEPLOY_ENTRYPOINTS
    assert UTILS.is_dir() and PROVIDERS.is_dir()


def test_dockerignore_allows_every_local_docker_copy_source() -> None:
    """A deny-by-default context must re-include every local COPY source."""
    docker = (PROXY / "Dockerfile").read_text(encoding="utf-8")
    active = [
        line.strip()
        for line in (PROXY / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert active[0] == "*"

    local_sources: set[str] = set()
    for raw in docker.splitlines():
        line = raw.strip()
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        parts = line.split()
        sources = [part for part in parts[1:] if not part.startswith("--")][:-1]
        local_sources.update(sources)

    assert local_sources == {
        "requirements.lock",
        "_utils",
        "_providers",
        "app.py",
        "deduplicate_dataset.py",
    }

    for source in sorted(local_sources):
        path = PROXY / source
        assert path.exists(), source
        if path.is_dir():
            assert f"!{source}/" in active, source
            assert f"!{source}/**" in active, source
        else:
            assert f"!{source}" in active, source


def test_top_level_hf_space_import_resolves_private_utils() -> None:
    code = (
        "import app; from _utils import _shared_logic; "
        "print(app.PROXY_VERSION, _shared_logic.PROXY_VERSION)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROXY,
        check=True,
        capture_output=True,
        text=True,
    )
    left, right = proc.stdout.strip().split()
    assert left == right
    assert tuple(int(part) for part in left.split(".")) >= (6, 8, 0)


def test_direct_deduplicator_resolves_private_utils() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import deduplicate_dataset as d; print(d._SCHEMA_AVAILABLE)"],
        cwd=PROXY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "True"


def test_docker_runtime_copy_set_is_standalone_importable(tmp_path: Path) -> None:
    """The Dockerfile copy set must contain every package imported by app.py."""
    for name in EXPECTED_DEPLOY_ENTRYPOINTS:
        shutil.copy2(PROXY / name, tmp_path / name)
    shutil.copytree(UTILS, tmp_path / "_utils")
    shutil.copytree(PROVIDERS, tmp_path / "_providers")
    proc = subprocess.run(
        [sys.executable, "-c", "import app; print(app.PROXY_VERSION)"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tuple(int(part) for part in proc.stdout.strip().split(".")) >= (7, 4, 0)
