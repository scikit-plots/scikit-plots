import pytest
from threadpoolctl import threadpool_info

from ..._testing import ignore_warnings
from ...utils._show_versions import (
    _get_system_info,
    _get_dep_info,
    show_versions,
)


def test_get_system_info():
    """Test that system information contains expected keys."""
    sys_info = _get_system_info()

    assert "python" in sys_info
    assert "executable" in sys_info
    assert "OS" in sys_info
    assert "cores" in sys_info


def test_get_dep_info():
    """Test that dependency info includes core libraries."""
    with ignore_warnings():
        deps_info = _get_dep_info()

    # Core dependencies
    assert "pip" in deps_info
    assert "setuptools" in deps_info
    assert "numpy" in deps_info
    assert "scipy" in deps_info
    assert "pandas" in deps_info
    assert "matplotlib" in deps_info
    assert "joblib" in deps_info
    assert "threadpoolctl" in deps_info

    # Project version
    assert "scikitplot" in deps_info


def test_show_versions_stdout(capsys):
    """Test default stdout output."""
    with ignore_warnings():
        show_versions()
        out, err = capsys.readouterr()

    # Basic checks
    assert "System Information" in out
    assert "Python Dependencies" in out
    assert "python" in out
    assert "numpy" in out

    # Check threadpoolctl info if available
    info = threadpool_info()
    if info:
        assert "Threadpoolctl Information" in out


def test_show_versions_dict():
    """Test returning version info as a dictionary."""
    with ignore_warnings():
        data = show_versions(mode="dict")

    assert isinstance(data, dict)
    assert "system" in data
    assert "dependencies" in data
    assert "environment" in data
    assert "gpu" in data


def test_show_versions_yaml():
    """Test returning version info in YAML format (if PyYAML installed)."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        pytest.skip("PyYAML not installed")

    with ignore_warnings():
        data_yaml = show_versions(mode="yaml")

    assert isinstance(data_yaml, str)
    assert "system:" in data_yaml
    assert "dependencies:" in data_yaml
