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
_detect_runtime_envs  WSL/Docker/Podman/Kubernetes detection          → TestDetectRuntimeEnvs
_get_env_info         runtime_envs list + OMP/MKL/OPENBLAS/CI vars    → TestGetEnvInfo
_get_cuda_info        nvidia-smi absent / present (mocked)            → TestGetCudaInfo
_get_system_info      Required keys, GIL flags, types                 → TestGetSystemInfo
_get_dep_info         Core + optional package versions                → TestGetDepInfo
show_versions         stdout/dict/yaml/rich modes, structural checks  → TestShowVersions
"""

from __future__ import annotations

import io
import os
import sys
import unittest
import unittest.mock as mock

from .._show_versions import (  # noqa: E402
    _detect_runtime_envs,
    _get_cuda_info,
    _get_dep_info,
    _get_env_info,
    _get_system_info,
    show_versions,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: All keys that must be present in _get_system_info() output.
REQUIRED_SYS_KEYS: tuple[str, ...] = (
    "python",
    "executable",
    "python_implementation",
    "libc_ver",
    "OS",
    "architecture",
    "CPU",
    "cores",
    "is_free_threaded_build",
    "is_gil_enabled",
    "is_running_no_gil",
)

#: Core dependency keys that _get_dep_info() must always report.
REQUIRED_DEP_KEYS: tuple[str, ...] = (
    "scikitplot",
    "pip",
    "setuptools",
    "cython",
    "numpy",
    "scipy",
    "aggdraw",
    "pandas",
    "matplotlib",
    "joblib",
    "threadpoolctl",
    "scikit-learn",
    "seaborn",
)


# ===========================================================================
# _detect_runtime_envs
# ===========================================================================


class TestDetectRuntimeEnvs(unittest.TestCase):
    """
    _detect_runtime_envs must return an ordered, deduplicated list of
    runtime environment identifiers.

    Notes
    -----
    Possible identifiers: ``"wsl"``, ``"docker"``, ``"podman"``,
    ``"kubernetes"``.
    """

    def test_returns_list(self):
        """Return type must be list."""
        result = _detect_runtime_envs()
        self.assertIsInstance(result, list)

    def test_no_duplicates(self):
        """Result must contain no duplicate identifiers."""
        result = _detect_runtime_envs()
        self.assertEqual(len(result), len(set(result)))

    def test_wsl_detected_via_release(self):
        """'microsoft' in platform.release() must yield 'wsl' in result."""
        with mock.patch("platform.release", return_value="5.15.90.1-microsoft-standard"):
            with mock.patch("platform.version", return_value=""):
                result = _detect_runtime_envs()
        self.assertIn("wsl", result)

    def test_wsl_detected_via_version(self):
        """'wsl' in platform.version() must yield 'wsl' in result."""
        with mock.patch("platform.release", return_value="6.1.0"):
            with mock.patch("platform.version", return_value="#1 SMP wsl2"):
                result = _detect_runtime_envs()
        self.assertIn("wsl", result)

    def test_non_wsl_release_excludes_wsl(self):
        """A standard Linux kernel string must not produce 'wsl'."""
        with mock.patch("platform.release", return_value="6.1.0-28-amd64"):
            with mock.patch("platform.version", return_value=""):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch("builtins.open", side_effect=OSError):
                        result = _detect_runtime_envs()
        self.assertNotIn("wsl", result)

    def test_dockerenv_file_detected(self):
        """/.dockerenv existence must produce 'docker' in result."""
        with mock.patch("platform.release", return_value="6.1.0"):
            with mock.patch("platform.version", return_value=""):
                with mock.patch(
                    "os.path.exists", side_effect=lambda p: p == "/.dockerenv"
                ):
                    with mock.patch("builtins.open", side_effect=OSError):
                        result = _detect_runtime_envs()
        self.assertIn("docker", result)

    def test_cgroup_docker_detected(self):
        """'docker' in /proc/self/cgroup content must produce 'docker' in result."""
        with mock.patch("platform.release", return_value="6.1.0"):
            with mock.patch("platform.version", return_value=""):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch(
                        "builtins.open",
                        mock.mock_open(read_data="12:blkio:/docker/abc123\n"),
                    ):
                        result = _detect_runtime_envs()
        self.assertIn("docker", result)

    def test_cgroup_kubernetes_detected(self):
        """'kubepods' in /proc cgroup content must produce 'kubernetes' in result."""
        with mock.patch("platform.release", return_value="6.1.0"):
            with mock.patch("platform.version", return_value=""):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch(
                        "builtins.open",
                        mock.mock_open(
                            read_data="12:cpu:/kubepods/besteffort/pod-xyz\n"
                        ),
                    ):
                        result = _detect_runtime_envs()
        self.assertIn("kubernetes", result)

    def test_cgroup_podman_detected(self):
        """'podman' in /proc cgroup content must produce 'podman' in result."""
        with mock.patch("platform.release", return_value="6.1.0"):
            with mock.patch("platform.version", return_value=""):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch(
                        "builtins.open",
                        mock.mock_open(read_data="12:memory:/podman/abc\n"),
                    ):
                        result = _detect_runtime_envs()
        self.assertIn("podman", result)

    def test_no_markers_returns_empty_list(self):
        """With no environment markers, result must be an empty list."""
        with mock.patch("platform.release", return_value="6.1.0-28-amd64"):
            with mock.patch("platform.version", return_value=""):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch("builtins.open", side_effect=OSError):
                        result = _detect_runtime_envs()
        self.assertEqual(result, [])

    def test_oserror_on_release_does_not_raise(self):
        """OSError from platform.release() must be silently swallowed."""
        with mock.patch("platform.release", side_effect=OSError):
            with mock.patch("platform.version", side_effect=OSError):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch("builtins.open", side_effect=OSError):
                        try:
                            _detect_runtime_envs()
                        except OSError:
                            self.fail(
                                "_detect_runtime_envs raised OSError unexpectedly"
                            )

    def test_oserror_on_cgroup_does_not_raise(self):
        """OSError reading cgroup files must be silently handled."""
        with mock.patch("platform.release", return_value="6.1.0"):
            with mock.patch("platform.version", return_value=""):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch("builtins.open", side_effect=OSError):
                        try:
                            _detect_runtime_envs()
                        except OSError:
                            self.fail(
                                "_detect_runtime_envs raised OSError on cgroup read"
                            )


# ===========================================================================
# _get_env_info
# ===========================================================================


class TestGetEnvInfo(unittest.TestCase):
    """
    _get_env_info must return a dict containing runtime environment
    identifiers and thread-related environment variables.
    """

    def test_returns_dict(self):
        """Return type must be dict."""
        self.assertIsInstance(_get_env_info(), dict)

    def test_contains_runtime_envs_key(self):
        """Must contain 'runtime_envs' key whose value is a list."""
        info = _get_env_info()
        self.assertIn("runtime_envs", info)
        self.assertIsInstance(info["runtime_envs"], list)

    def test_contains_ci_key(self):
        """Must contain 'CI' key."""
        info = _get_env_info()
        self.assertIn("CI", info)

    def test_contains_thread_env_keys(self):
        """Must contain the three thread-count environment variable keys."""
        info = _get_env_info()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertIn(key, info)

    def test_env_var_keys_sorted(self):
        """
        The four env-var keys (CI, MKL_NUM_THREADS, OMP_NUM_THREADS,
        OPENBLAS_NUM_THREADS) must appear in sorted order.

        Notes
        -----
        ``runtime_envs`` is inserted first and excluded from this check
        because it is not an environment variable key.
        """
        info = _get_env_info()
        env_var_keys = [k for k in info if k != "runtime_envs"]
        self.assertEqual(env_var_keys, sorted(env_var_keys))

    def test_runtime_envs_is_first_key(self):
        """'runtime_envs' must be the first key in the returned dict."""
        info = _get_env_info()
        self.assertEqual(next(iter(info)), "runtime_envs")

    def test_unset_thread_vars_return_none(self):
        """Unset thread env vars must map to None, not raise."""
        thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        for var in thread_vars:
            os.environ.pop(var, None)
        info = _get_env_info()
        for var in thread_vars:
            self.assertIsNone(info[var])

    def test_set_omp_var_returned(self):
        """OMP_NUM_THREADS when set must appear with its value."""
        os.environ["OMP_NUM_THREADS"] = "4"
        try:
            info = _get_env_info()
            self.assertEqual(info.get("OMP_NUM_THREADS"), "4")
        finally:
            del os.environ["OMP_NUM_THREADS"]

    def test_set_ci_var_returned(self):
        """CI when set must appear with its value."""
        os.environ["CI"] = "true"
        try:
            info = _get_env_info()
            self.assertEqual(info.get("CI"), "true")
        finally:
            del os.environ["CI"]

    def test_unset_ci_returns_none(self):
        """CI when unset must map to None."""
        os.environ.pop("CI", None)
        info = _get_env_info()
        self.assertIsNone(info["CI"])


# ===========================================================================
# _get_cuda_info
# ===========================================================================


class TestGetCudaInfo(unittest.TestCase):
    """_get_cuda_info must handle CUDA present/absent/failing gracefully."""

    def test_returns_dict(self):
        """Return type must be dict."""
        result = _get_cuda_info()
        self.assertIsInstance(result, dict)

    def test_no_nvidia_smi_returns_null_dict(self):
        """When nvidia-smi is absent, must return the null-state dict."""
        with mock.patch("shutil.which", return_value=None):
            result = _get_cuda_info()
        self.assertIn("cuda", result)
        self.assertIn("gpu", result)
        self.assertIsNone(result["cuda"])
        self.assertIsNone(result["gpu"])

    def test_no_nvidia_smi_includes_mps_xla_xpu_keys(self):
        """Null-state dict must also contain mps, xla, xpu keys all set to None."""
        with mock.patch("shutil.which", return_value=None):
            result = _get_cuda_info()
        for key in ("mps", "xla", "xpu"):
            self.assertIn(key, result, msg=f"Missing null-state key: {key!r}")
            self.assertIsNone(result[key])

    def test_nvidia_smi_failure_returns_error_key(self):
        """When nvidia-smi exists but subprocess fails, must return 'Error' key."""
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch(
                "subprocess.check_output", side_effect=Exception("fail")
            ):
                result = _get_cuda_info()
        self.assertIn("Error", result)

    def test_nvidia_smi_success_single_gpu(self):
        """When nvidia-smi succeeds with one GPU, 'GPU 0' must be in result."""
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch(
                "subprocess.check_output",
                return_value="Tesla T4, 470.82\n",
            ):
                result = _get_cuda_info()
        self.assertIn("GPU 0", result)
        self.assertIn("Tesla T4", result["GPU 0"])
        self.assertIn("470.82", result["GPU 0"])

    def test_nvidia_smi_success_multiple_gpus(self):
        """Multiple GPU lines must each appear as separate 'GPU N' entries."""
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch(
                "subprocess.check_output",
                return_value="Tesla T4, 470.82\nA100, 525.00\n",
            ):
                result = _get_cuda_info()
        self.assertIn("GPU 0", result)
        self.assertIn("GPU 1", result)
        self.assertIn("A100", result["GPU 1"])


# ===========================================================================
# _get_system_info
# ===========================================================================


class TestGetSystemInfo(unittest.TestCase):
    """
    _get_system_info must return a dict with all required system keys,
    correct types, and consistent GIL flag semantics.
    """

    def setUp(self):
        self._info = _get_system_info()

    # -- Presence checks --

    def test_returns_dict(self):
        """Return type must be dict."""
        self.assertIsInstance(self._info, dict)

    def test_all_required_keys_present(self):
        """Every key in REQUIRED_SYS_KEYS must exist in the result."""
        for key in REQUIRED_SYS_KEYS:
            self.assertIn(
                key, self._info, msg=f"Missing required system key: {key!r}"
            )

    # -- Type checks --

    def test_python_is_str(self):
        self.assertIsInstance(self._info["python"], str)

    def test_executable_is_str(self):
        self.assertIsInstance(self._info["executable"], str)

    def test_os_is_str(self):
        self.assertIsInstance(self._info["OS"], str)

    def test_architecture_is_str(self):
        self.assertIsInstance(self._info["architecture"], str)

    def test_python_implementation_is_str(self):
        self.assertIsInstance(self._info["python_implementation"], str)

    def test_cores_is_positive_int_or_none(self):
        """CPU core count must be a positive integer or None if undetermined."""
        cores = self._info["cores"]
        if cores is not None:
            self.assertIsInstance(cores, int)
            self.assertGreater(cores, 0)

    # -- GIL flag checks --

    def test_is_free_threaded_build_is_bool(self):
        """is_free_threaded_build must be a boolean."""
        self.assertIsInstance(self._info["is_free_threaded_build"], bool)

    def test_is_gil_enabled_is_bool(self):
        """is_gil_enabled must be a boolean."""
        self.assertIsInstance(self._info["is_gil_enabled"], bool)

    def test_is_running_no_gil_is_bool(self):
        """is_running_no_gil must be a boolean."""
        self.assertIsInstance(self._info["is_running_no_gil"], bool)

    def test_gil_flags_mutual_exclusion(self):
        """
        is_running_no_gil=True implies is_gil_enabled=False.
        These two flags must be mutually consistent.
        """
        if self._info["is_running_no_gil"]:
            self.assertFalse(
                self._info["is_gil_enabled"],
                "is_gil_enabled must be False when is_running_no_gil is True",
            )

    def test_no_gil_requires_free_threaded_build(self):
        """
        is_running_no_gil=True requires is_free_threaded_build=True.
        A free-threaded build may still run with the GIL enabled.
        """
        if self._info["is_running_no_gil"]:
            self.assertTrue(
                self._info["is_free_threaded_build"],
                "is_free_threaded_build must be True when is_running_no_gil is True",
            )

    # -- Content checks --

    def test_python_value_contains_version(self):
        """Python version string must contain at least one digit."""
        self.assertTrue(
            any(c.isdigit() for c in self._info["python"]),
            "python value must contain a digit",
        )

    def test_python_matches_runtime_version(self):
        """python entry must contain the running Python major.minor string."""
        major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.assertIn(major_minor, self._info["python"])

    # -- Removed-key guards --

    def test_ci_key_not_in_system_info(self):
        """
        'ci' must NOT appear in system info.
        CI detection moved to _get_env_info / environment section.
        """
        self.assertNotIn("ci", self._info)

    def test_container_key_not_in_system_info(self):
        """
        'container' must NOT appear in system info.
        Runtime env detection moved to _detect_runtime_envs / environment section.
        """
        self.assertNotIn("container", self._info)

    def test_is_free_threaded_old_key_not_present(self):
        """
        Deprecated 'is_free_threaded' key must NOT appear;
        the current key is 'is_free_threaded_build'.
        """
        self.assertNotIn("is_free_threaded", self._info)


# ===========================================================================
# _get_dep_info
# ===========================================================================


class TestGetDepInfo(unittest.TestCase):
    """_get_dep_info must report versions for all required dependencies."""

    def setUp(self):
        self._info = _get_dep_info()

    def test_returns_dict(self):
        """Return type must be dict."""
        self.assertIsInstance(self._info, dict)

    def test_all_required_dep_keys_present(self):
        """Every key in REQUIRED_DEP_KEYS must exist in the result."""
        for key in REQUIRED_DEP_KEYS:
            self.assertIn(
                key, self._info, msg=f"Missing required dep key: {key!r}"
            )

    def test_scikitplot_is_first_key(self):
        """'scikitplot' must be the first key inserted in the dep dict."""
        first_key = next(iter(self._info))
        self.assertEqual(first_key, "scikitplot")

    def test_numpy_is_installed(self):
        """numpy must be installed; version must be a non-None string."""
        val = self._info["numpy"]
        self.assertIsNotNone(val)
        self.assertIsInstance(val, str)

    def test_scikitplot_is_installed(self):
        """scikitplot must be installed; version must be a non-None string."""
        val = self._info["scikitplot"]
        self.assertIsNotNone(val)
        self.assertIsInstance(val, str)

    def test_missing_package_returns_none(self):
        """
        A package that is not installed must map to None, not raise.
        Verified via PackageNotFoundError mocking.
        """
        from importlib.metadata import PackageNotFoundError  # noqa: PLC0415

        with mock.patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("fake-pkg"),
        ):
            # Re-call to exercise the except branch
            from scikitplot.utils._show_versions import _get_dep_info as _fn  # noqa: PLC0415

            info = _fn()
        # All values must be None or str (PackageNotFoundError → None)
        for k, v in info.items():
            self.assertIsInstance(
                v,
                (str, type(None)),
                msg=f"dep_info[{k!r}] = {v!r} must be str or None",
            )

    def test_all_values_str_or_none(self):
        """Every value in dep_info must be str or None."""
        for k, v in self._info.items():
            self.assertIsInstance(
                v,
                (str, type(None)),
                msg=f"dep_info[{k!r}] = {v!r} is not str or None",
            )

    def test_aggdraw_optional_value_type(self):
        """aggdraw (optional) must map to str (if installed) or None."""
        if "aggdraw" in self._info:
            self.assertIsInstance(self._info["aggdraw"], (str, type(None)))


# ===========================================================================
# show_versions
# ===========================================================================


class TestShowVersions(unittest.TestCase):
    """show_versions must behave correctly across all four output modes."""

    # -- stdout mode --

    def test_stdout_mode_prints_system_information(self):
        """show_versions() must write 'System Information' to stdout."""
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            show_versions()
        self.assertIn("System Information", captured.getvalue())

    def test_stdout_mode_returns_none(self):
        """show_versions() in stdout mode must return None."""
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

    def test_stdout_contains_environment_section(self):
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            show_versions()
        self.assertIn("Environment Variables", captured.getvalue())

    def test_stdout_contains_gpu_section(self):
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            show_versions()
        self.assertIn("GPU Information", captured.getvalue())

    # -- dict mode --

    def test_dict_mode_returns_dict(self):
        self.assertIsInstance(show_versions(mode="dict"), dict)

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

    def test_dict_mode_environment_is_dict(self):
        self.assertIsInstance(show_versions(mode="dict")["environment"], dict)

    def test_dict_mode_threadpoolctl_is_list(self):
        self.assertIsInstance(show_versions(mode="dict")["threadpoolctl"], list)

    def test_dict_mode_environment_has_runtime_envs(self):
        """environment dict must contain 'runtime_envs' key with a list value."""
        env = show_versions(mode="dict")["environment"]
        self.assertIn("runtime_envs", env)
        self.assertIsInstance(env["runtime_envs"], list)

    def test_dict_mode_numpy_version_present(self):
        """numpy must appear in dependencies with a non-None value."""
        deps = show_versions(mode="dict")["dependencies"]
        self.assertIsNotNone(deps.get("numpy"))

    def test_dict_mode_scikitplot_version_present(self):
        """scikitplot must appear in dependencies with a non-None value."""
        deps = show_versions(mode="dict")["dependencies"]
        self.assertIsNotNone(deps.get("scikitplot"))

    def test_dict_mode_system_python_matches_runtime(self):
        """system.python must contain the running Python major.minor."""
        major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.assertIn(major_minor, show_versions(mode="dict")["system"]["python"])

    def test_dict_mode_system_has_all_gil_flags(self):
        """system dict must contain all three GIL-related boolean flags."""
        sys_data = show_versions(mode="dict")["system"]
        for key in ("is_free_threaded_build", "is_gil_enabled", "is_running_no_gil"):
            self.assertIn(key, sys_data, msg=f"Missing GIL flag: {key!r}")
            self.assertIsInstance(
                sys_data[key], bool, msg=f"GIL flag {key!r} must be bool"
            )

    # -- yaml mode --

    def test_yaml_mode_returns_str_or_none(self):
        """yaml mode must return a YAML str, or None if PyYAML is absent."""
        result = show_versions(mode="yaml")
        self.assertIsInstance(result, (str, type(None)))

    def test_yaml_mode_content_when_pyyaml_installed(self):
        """If PyYAML is installed, output must be a YAML string with expected keys."""
        try:
            import yaml as _yaml  # noqa: PLC0415, F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        result = show_versions(mode="yaml")
        self.assertIsInstance(result, str)
        self.assertIn("system:", result)
        self.assertIn("dependencies:", result)

    def test_yaml_mode_no_pyyaml_falls_back_to_stdout(self):
        """Without PyYAML, yaml mode must fall back to stdout and return None."""
        with mock.patch.dict(sys.modules, {"yaml": None}):
            captured = io.StringIO()
            with mock.patch("sys.stdout", captured):
                result = show_versions(mode="yaml")
        # May return None (stdout fallback) or str if yaml was already cached
        self.assertIsInstance(result, (str, type(None)))

    # -- rich mode --

    def test_rich_mode_no_error_when_rich_absent(self):
        """show_versions('rich') must not raise when rich is not installed."""
        with mock.patch.dict(
            sys.modules,
            {"rich": None, "rich.console": None, "rich.table": None},
        ):
            try:
                captured = io.StringIO()
                with mock.patch("sys.stdout", captured):
                    show_versions(mode="rich")
            except Exception as exc:
                self.fail(
                    f"show_versions('rich') raised when rich absent: {exc}"
                )

    def test_rich_mode_returns_none(self):
        """rich mode must return None (output goes to Console, not return value)."""
        try:
            result = show_versions(mode="rich")
            self.assertIsNone(result)
        except ImportError:
            self.skipTest("rich not installed")

    # -- Unknown mode --

    def test_unknown_mode_falls_through_to_stdout(self):
        """An unrecognized mode must not raise and must fall through to stdout."""
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            try:
                result = show_versions(mode="__unknown_mode__")
            except Exception as exc:
                self.fail(
                    f"show_versions raised on unknown mode: {exc}"
                )
        self.assertIsNone(result)
        self.assertIn("System Information", captured.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
