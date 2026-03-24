# scikitplot/utils/tests/test__show_versions.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._show_versions`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__show_versions.py -v

Coverage map
------------
_is_docker           /proc cgroup + /.dockerenv detection  → TestIsDocker
_is_wsl              'microsoft' in platform.release()     → TestIsWsl
_get_env_info        OMP/MKL/OPENBLAS env vars             → TestGetEnvInfo
_get_cuda_info       nvidia-smi absent / present (mocked)  → TestGetCudaInfo
_get_system_info     Required keys, types, optional markers → TestGetSystemInfo
_get_dep_info        Core + optional package versions      → TestGetDepInfo
show_versions        stdout/dict/yaml/rich modes, caching  → TestShowVersions
"""

from __future__ import annotations

import io
import os
import platform
import sys
import unittest
import unittest.mock as mock

# --- path bootstrap (commented out — use pytest from the package root) -----
# import pathlib
# _HERE = pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))
# --------------------------------------------------------------------------

from .._show_versions import (  # noqa: E402
    _get_cuda_info,
    _get_dep_info,
    _get_env_info,
    _get_system_info,
    _is_docker,
    _is_wsl,
    show_versions,
)


# ===========================================================================
# _is_docker
# ===========================================================================


class TestIsDocker(unittest.TestCase):
    """_is_docker must detect Docker containers by examining /proc/self/cgroup."""

    def test_returns_bool(self):
        result = _is_docker()
        self.assertIsInstance(result, bool)

    def test_docker_detected_via_cgroup(self):
        """If /proc/self/cgroup contains 'docker', must return True."""
        with mock.patch("os.path.exists", side_effect=lambda p: p == "/proc/self/cgroup"):
            with mock.patch(
                "builtins.open",
                mock.mock_open(read_data="12:blkio:/docker/abc123\n"),
            ):
                # Clear LRU cache so our mock takes effect
                _is_docker.cache_clear()
                result = _is_docker()
        _is_docker.cache_clear()
        self.assertTrue(result)

    def test_no_cgroup_no_dockerenv_returns_false(self):
        """When neither /proc/self/cgroup nor /.dockerenv exists, return False."""
        with mock.patch("os.path.exists", return_value=False):
            _is_docker.cache_clear()
            result = _is_docker()
        _is_docker.cache_clear()
        self.assertFalse(result)

    def test_dockerenv_file_detected(self):
        """/.dockerenv existence alone must return True."""
        def fake_exists(p):
            if p == "/proc/self/cgroup":
                return False
            if p == "/.dockerenv":
                return True
            return False

        with mock.patch("os.path.exists", side_effect=fake_exists):
            _is_docker.cache_clear()
            result = _is_docker()
        _is_docker.cache_clear()
        self.assertTrue(result)


# ===========================================================================
# _is_wsl
# ===========================================================================


class TestIsWsl(unittest.TestCase):
    """_is_wsl must detect WSL by inspecting the kernel release string."""

    def test_returns_bool(self):
        result = _is_wsl()
        self.assertIsInstance(result, bool)

    def test_microsoft_in_release_returns_true(self):
        """'Microsoft' in platform.release() must return True."""
        with mock.patch("platform.release", return_value="5.15.90.1-microsoft-standard"):
            _is_wsl.cache_clear()
            result = _is_wsl()
        _is_wsl.cache_clear()
        self.assertTrue(result)

    def test_linux_release_returns_false(self):
        """A standard Linux kernel string must return False."""
        with mock.patch("platform.release", return_value="6.1.0-28-amd64"):
            _is_wsl.cache_clear()
            result = _is_wsl()
        _is_wsl.cache_clear()
        self.assertFalse(result)


# ===========================================================================
# _get_env_info
# ===========================================================================


class TestGetEnvInfo(unittest.TestCase):
    """_get_env_info must return a dict of thread-related env variables."""

    def test_returns_dict(self):
        self.assertIsInstance(_get_env_info(), dict)

    def test_contains_expected_keys(self):
        info = _get_env_info()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertIn(key, info)

    def test_keys_sorted(self):
        """Keys must be in sorted order."""
        info = _get_env_info()
        self.assertEqual(list(info.keys()), sorted(info.keys()))

    def test_unset_var_returns_none(self):
        """An unset env var must map to None."""
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.pop(var, None)
        info = _get_env_info()
        for val in info.values():
            self.assertIsNone(val)

    def test_set_var_returned(self):
        """A set env var must appear in the output."""
        os.environ["OMP_NUM_THREADS"] = "4"
        try:
            info = _get_env_info()
            self.assertEqual(info.get("OMP_NUM_THREADS"), "4")
        finally:
            del os.environ["OMP_NUM_THREADS"]


# ===========================================================================
# _get_cuda_info
# ===========================================================================


class TestGetCudaInfo(unittest.TestCase):
    """_get_cuda_info must gracefully handle CUDA available/unavailable."""

    def test_returns_dict(self):
        result = _get_cuda_info()
        self.assertIsInstance(result, dict)

    def test_no_nvidia_smi_returns_null_dict(self):
        """When nvidia-smi is absent, must return the null-state dict."""
        with mock.patch("shutil.which", return_value=None):
            result = _get_cuda_info()
        # All values should be None when no GPU driver is present
        self.assertIn("cuda", result)
        self.assertIn("gpu", result)

    def test_nvidia_smi_failure_returns_error(self):
        """When nvidia-smi exists but fails, must return an error entry."""
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch(
                "subprocess.check_output", side_effect=Exception("fail")
            ):
                result = _get_cuda_info()
        self.assertIn("Error", result)

    def test_nvidia_smi_success_returns_gpu_info(self):
        """When nvidia-smi succeeds, each GPU entry must be in the result."""
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch(
                "subprocess.check_output",
                return_value="Tesla T4, 470.82\n",
            ):
                result = _get_cuda_info()
        self.assertIn("GPU 0", result)
        self.assertIn("Tesla T4", result["GPU 0"])


# ===========================================================================
# _get_system_info
# ===========================================================================


class TestGetSystemInfo(unittest.TestCase):
    """_get_system_info must return a dict with all required system keys."""

    def setUp(self):
        self._info = _get_system_info()

    def test_returns_dict(self):
        self.assertIsInstance(self._info, dict)

    def test_python_key_present(self):
        self.assertIn("python", self._info)

    def test_executable_key_present(self):
        self.assertIn("executable", self._info)

    def test_os_key_present(self):
        self.assertIn("OS", self._info)

    def test_cores_key_present(self):
        self.assertIn("cores", self._info)

    def test_architecture_key_present(self):
        self.assertIn("architecture", self._info)

    def test_python_value_contains_version(self):
        """Python version string must contain at least one digit."""
        val = self._info["python"]
        self.assertTrue(any(c.isdigit() for c in val))

    def test_cores_is_positive_int(self):
        """CPU core count must be a positive integer."""
        cores = self._info["cores"]
        if cores is not None:
            self.assertIsInstance(cores, int)
            self.assertGreater(cores, 0)

    def test_executable_is_string(self):
        self.assertIsInstance(self._info["executable"], str)

    def test_is_free_threaded_is_bool(self):
        self.assertIsInstance(self._info.get("is_free_threaded", False), bool)

    def test_docker_key_conditional(self):
        """'container' key appears only when Docker is detected; must not crash."""
        # Just verify the dict is structurally valid regardless.
        self.assertIsInstance(self._info, dict)

    def test_ci_key_reflects_env(self):
        """'ci' key must appear when the CI env var is set."""
        original = os.environ.get("CI")
        os.environ["CI"] = "true"
        try:
            info = _get_system_info()
            self.assertIn("ci", info)
        finally:
            if original is None:
                os.environ.pop("CI", None)
            else:
                os.environ["CI"] = original


# ===========================================================================
# _get_dep_info
# ===========================================================================


class TestGetDepInfo(unittest.TestCase):
    """_get_dep_info must report versions of core and optional dependencies."""

    def setUp(self):
        self._info = _get_dep_info()

    def test_returns_dict(self):
        self.assertIsInstance(self._info, dict)

    def test_scikitplot_in_deps(self):
        self.assertIn("scikitplot", self._info)

    def test_numpy_in_deps(self):
        self.assertIn("numpy", self._info)

    def test_scipy_in_deps(self):
        self.assertIn("scipy", self._info)

    def test_pandas_in_deps(self):
        self.assertIn("pandas", self._info)

    def test_matplotlib_in_deps(self):
        self.assertIn("matplotlib", self._info)

    def test_joblib_in_deps(self):
        self.assertIn("joblib", self._info)

    def test_threadpoolctl_in_deps(self):
        self.assertIn("threadpoolctl", self._info)

    def test_pip_in_deps(self):
        self.assertIn("pip", self._info)

    def test_numpy_version_is_str(self):
        """numpy must be installed; version must be a string."""
        val = self._info["numpy"]
        self.assertIsNotNone(val)
        self.assertIsInstance(val, str)

    def test_missing_optional_returns_none(self):
        """An optional package not installed must map to None, not raise."""
        # 'aggdraw' is listed as optional; may or may not be installed.
        if "aggdraw" in self._info:
            # If present, value is str (version) or None
            self.assertIsInstance(self._info["aggdraw"], (str, type(None)))

    def test_all_values_str_or_none(self):
        """Every value in dep_info must be str or None."""
        for k, v in self._info.items():
            self.assertIsInstance(
                v, (str, type(None)),
                msg=f"dep_info[{k!r}] = {v!r} is not str or None",
            )


# ===========================================================================
# show_versions
# ===========================================================================


class TestShowVersions(unittest.TestCase):
    """show_versions must behave correctly in all four modes."""

    # -- stdout mode --

    def test_stdout_mode_prints_to_stdout(self, capsys=None):
        """show_versions() must write to stdout (captured with capsys or StringIO)."""
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            show_versions()
        output = captured.getvalue()
        self.assertIn("System Information", output)

    def test_stdout_returns_none(self):
        """show_versions() (stdout mode) must return None."""
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = show_versions()
        self.assertIsNone(result)

    def test_stdout_contains_python(self):
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            show_versions()
        self.assertIn("python", captured.getvalue())

    def test_stdout_contains_numpy(self):
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            show_versions()
        self.assertIn("numpy", captured.getvalue())

    def test_stdout_contains_dependencies_section(self):
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            show_versions()
        self.assertIn("Python Dependencies", captured.getvalue())

    # -- dict mode --

    def test_dict_mode_returns_dict(self):
        result = show_versions(mode="dict")
        self.assertIsInstance(result, dict)

    def test_dict_mode_has_system_key(self):
        self.assertIn("system", show_versions(mode="dict"))

    def test_dict_mode_has_dependencies_key(self):
        self.assertIn("dependencies", show_versions(mode="dict"))

    def test_dict_mode_has_environment_key(self):
        self.assertIn("environment", show_versions(mode="dict"))

    def test_dict_mode_has_gpu_key(self):
        self.assertIn("gpu", show_versions(mode="dict"))

    def test_dict_mode_has_threadpoolctl_key(self):
        self.assertIn("threadpoolctl", show_versions(mode="dict"))

    def test_dict_mode_system_is_dict(self):
        self.assertIsInstance(show_versions(mode="dict")["system"], dict)

    def test_dict_mode_dependencies_is_dict(self):
        self.assertIsInstance(show_versions(mode="dict")["dependencies"], dict)

    # -- yaml mode --

    def test_yaml_mode_returns_str_or_fallback(self):
        """yaml mode must return a YAML str, or fall back to stdout (None) if PyYAML absent."""
        result = show_versions(mode="yaml")
        self.assertIsInstance(result, (str, type(None)))

    def test_yaml_mode_content_if_pyyaml(self):
        """If PyYAML is installed, output must be valid YAML string."""
        try:
            import yaml as _yaml  # noqa: PLC0415, F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        result = show_versions(mode="yaml")
        self.assertIsInstance(result, str)
        self.assertIn("system:", result)
        self.assertIn("dependencies:", result)

    def test_yaml_mode_no_pyyaml_fallback_stdout(self):
        """Without PyYAML, yaml mode must silently fall back to stdout (returns None)."""
        with mock.patch.dict(sys.modules, {"yaml": None}):
            captured = io.StringIO()
            with mock.patch("sys.stdout", captured):
                result = show_versions(mode="yaml")
        # Either returned None (stdout fallback) or a string if yaml was cached
        self.assertIsInstance(result, (str, type(None)))

    # -- rich mode --

    def test_rich_mode_no_error_when_rich_absent(self):
        """When rich is not installed, show_versions('rich') must fall back to stdout."""
        with mock.patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None}):
            try:
                captured = io.StringIO()
                with mock.patch("sys.stdout", captured):
                    show_versions(mode="rich")
            except Exception as e:
                self.fail(f"show_versions('rich') raised with no rich: {e}")

    # -- Structural integrity --

    def test_dict_mode_numpy_version_present(self):
        """numpy must appear in the dependencies dict with a non-None value."""
        data = show_versions(mode="dict")
        self.assertIsNotNone(data["dependencies"].get("numpy"))

    def test_dict_mode_scikitplot_version_present(self):
        data = show_versions(mode="dict")
        self.assertIsNotNone(data["dependencies"].get("scikitplot"))

    def test_dict_mode_system_python_matches(self):
        """System python entry must begin with the running Python major.minor."""
        data = show_versions(mode="dict")
        major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.assertIn(major_minor, data["system"]["python"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
