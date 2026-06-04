# flake8: noqa: D213

# ruff: noqa
# ruff: noqa: PGH004

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility methods to display system and dependency information for debugging.

Adapted and expanded from :py:func:`pandas.show_versions`.
"""

from __future__ import annotations

import logging
import json  # noqa: F401
import platform
import sys
import os
import shutil
import sysconfig
import warnings  # noqa: F401
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError

# Only imports when type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Heavy import, only for type checking
    # Only imports when type checking, not at runtime
    from typing import Dict, List, Optional, Any, Final  # noqa: F401

from threadpoolctl import threadpool_info

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("scikitplot.utils._show_versions")

## Define __all__ to specify the public interface of the module,
## not required default all belove func
# _all_ignore = ["platform", "sys", "threadpool_info"]
__all__ = [
    "show_versions",
]


_CONTAINER_MARKERS: Final[tuple[str, ...]] = (
    "docker",
    "containerd",
    "kubepods",
    "podman",
)


def _detect_runtime_envs() -> list[str]:
    """
    Detect runtime environment markers.

    Returns
    -------
    list[str]
        Ordered unique environment identifiers.

        Possible values include:
        - docker
        - podman
        - kubernetes
        - wsl
    """
    envs: list[str] = []
    # ---- WSL ----
    try:
        release = platform.release().lower()
        version = platform.version().lower()
        if (
            "microsoft" in release
            or "microsoft" in version
            or "wsl" in release
            or "wsl" in version
        ):
            envs.append("wsl")
    except OSError:
        pass
    # ---- Container ----
    if os.path.exists("/.dockerenv"):
        envs.append("docker")
    cgroup_paths = (
        "/proc/self/cgroup",
        "/proc/1/cgroup",
    )
    for path in cgroup_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "kubepods" in content:
                envs.append("kubernetes")
            if "podman" in content:
                envs.append("podman")
            if any(marker in content for marker in ("docker", "containerd")):
                envs.append("docker")
        except OSError:
            continue
    # Stable dedup preserving order
    return list(dict.fromkeys(envs))


def _get_env_info() -> dict[str, Optional[str]]:
    """
    Retrieve relevant environment variable info when present.

    Returns
    -------
    env_info : dict
        A dictionary containing thread-related environment settings, if set.
    """
    # Environmental markers
    env_vars = sorted(
        ["CI", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
    )
    runtime_envs = {"runtime_envs": _detect_runtime_envs()}
    runtime_envs.update({var: os.getenv(var, None) for var in env_vars})
    return runtime_envs


def _get_cuda_info() -> Optional[dict[str, str]]:
    """
    Attempt to get GPU/CUDA information if available.

    MPS (Metal Performance Shaders) backend in Python. Metal is Apple's API.
    XPU backend, specifically tailored for Intel GPU optimization.

    ..seealso::
        * https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-device
    """
    if shutil.which("nvidia-smi"):
        try:
            import subprocess

            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                encoding="utf-8",
            )
            gpu_info = {}
            for idx, line in enumerate(result.strip().split("\n")):
                gpu_name, driver_version = [v.strip() for v in line.split(",")]
                gpu_info[f"GPU {idx}"] = f"{gpu_name} (Driver: {driver_version})"
            return gpu_info
        except Exception:
            return {"Error": "Unable to retrieve GPU info"}
    return {"cuda": None, "gpu": None, "mps": None, "xla": None, "xpu": None}


def _detect_linux_wheel_type():
    # print(platform.system())      # Same as platform_system
    # print(platform.platform())    # Shows musl / glibc details
    # print(platform.libc_ver())    # Helps detect musllinux (musl vs glibc)
    # print(platform.python_implementation())
    system = platform.system()
    machine = platform.machine().lower()
    libc, _ = platform.libc_ver()
    is_musl = libc == "musl"
    is_wasm = (
        sys.platform in ("emscripten", "wasi") or "emscripten" in sys.executable.lower()
    )

    if is_wasm:
        return "wasm"
    elif system == "Linux" and is_musl:
        return "musllinux"
    elif system == "Linux":
        return "manylinux"
    else:
        return system.lower()


def _get_system_info() -> dict[str, str]:
    """
    Gather detailed system and Python interpreter information.

    | Python                 | `is_free_threaded_build` | `is_gil_enabled` | `is_running_no_gil` |
    | ---------------------- | -----------------------: | ---------------: | ------------------: |
    | 3.8-3.12               |                    False |             True |               False |
    | 3.13-3.14 regular      |                    False |             True |               False |
    | 3.13-3.14 FT + GIL ON  |                     True |             True |               False |
    | 3.13-3.14 FT + GIL OFF |                     True |            False |                True |
    | 3.15+ FT + GIL OFF     |                     True |            False |                True |

    Returns
    -------
    sys_info : dict
        A dictionary containing OS, Python, CPU, execution, and environment info.
    """
    sys_info = {
        "python": sys.version.split("\n")[0],
        "executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "libc_ver": platform.libc_ver(),
        "OS": platform.platform(),
        "architecture": platform.machine(),  # platform.architecture()
        "CPU": platform.processor() or "Unknown",
        "cores": os.cpu_count(),
    }
    # is_free_threaded_build -> capability / ABI
    # is_gil_enabled         -> runtime state
    # is_running_no_gil      -> effective execution mode
    # Python 3.13+: runtime GIL state checker
    _is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    RUNTIME_INFO = {
        # Interpreter capability / ABI:
        # Was Python built with free-threading (no-GIL) support?
        "is_free_threaded_build": (
            # Python 3.15+
            getattr(sys, "is_free_threaded", lambda: False)()
            # Python 3.15+ ABI metadata
            or bool(
                getattr(
                    getattr(sys, "abi_info", None),
                    "free_threaded",
                    False,
                )
            )
            # Python 3.13–3.14 build flag
            or bool(
                getattr(
                    sysconfig,
                    "get_config_var",
                    lambda _name: None,
                )("Py_GIL_DISABLED")
            )
        ),
        # Effective execution mode:
        # Is Python currently running without the GIL?
        "is_running_no_gil": (not _is_gil_enabled()) if _is_gil_enabled else False,
        # Runtime execution state:
        # Is the GIL enabled right now?
        "is_gil_enabled": _is_gil_enabled() if _is_gil_enabled else True,
    }
    sys_info.update(RUNTIME_INFO)
    return sys_info


def _get_dep_info():
    """
    Overview of the installed version of main dependencies.

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    dep_info: dict
        version information on relevant Python libraries

    """
    # Import __version__ here
    from scikitplot import __version__

    core_deps = [
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
    ]
    optional_deps = [
        # "bokeh",
        # "plotly",
        # "streamlit",
        # "gradio",
        # "pyyaml",
        # "meson-python",
    ]
    dep_info = {}
    dep_info["scikitplot"] = __version__
    for pkg in core_deps + optional_deps:
        try:
            dep_info[pkg] = version(pkg)
        except PackageNotFoundError:
            dep_info[pkg] = None
    return dep_info


def show_versions(mode: str = "stdout") -> Optional[dict[str, any]]:
    """
    Print or return debugging information about the system, Python, dependencies, and hardware.

    Parameters
    ----------
    mode : {'stdout', 'dict', 'yaml', 'rich'}, default='stdout'
        - 'stdout': prints information to console using `rich` (if available) or plain text.
        - 'dict': returns the information as a nested dictionary.
        - 'yaml': returns the information in YAML format (requires PyYAML).
        - 'rich': prints formatted output using rich library.

    Returns
    -------
    version_data : str, dict or None
        If `mode='dict'`, returns a dictionary of version information.
        If `mode='yaml'`, returns a string of version information.
        Otherwise, returns None.

    Notes
    -----
    Useful for debugging and issue reporting.

    - is_free_threaded_build -> capability / ABI
    - is_gil_enabled         -> runtime state
    - is_running_no_gil      -> effective execution mode

    | Python                 | `is_free_threaded_build` | `is_gil_enabled` | `is_running_no_gil` |
    | ---------------------- | -----------------------: | ---------------: | ------------------: |
    | 3.8-3.12               |                    False |             True |               False |
    | 3.13-3.14 regular      |                    False |             True |               False |
    | 3.13-3.14 FT + GIL ON  |                     True |             True |               False |
    | 3.13-3.14 FT + GIL OFF |                     True |            False |                True |
    | 3.15+ FT + GIL OFF     |                     True |            False |                True |

    Examples
    --------
    .. jupyter-execute::

      >>> import scikitplot
      >>> scikitplot.show_versions()

    .. jupyter-execute::

      >>> import scikitplot
      >>> scikitplot.show_versions(mode="dict")

    .. jupyter-execute::

      >>> # !scikitplot show_versions -j
      >>> !scikitplot show_versions || true
    """
    sys_info = _get_system_info()  # Returns dict of system info
    dep_info = _get_dep_info()  # Returns dict of dependency info
    env_info = _get_env_info()
    gpu_info = _get_cuda_info()
    threadpool_info_ = threadpool_info()  # List of dict
    data = {
        "system": sys_info,
        "dependencies": dep_info,
        "environment": env_info,
        "gpu": gpu_info,
        "threadpoolctl": threadpool_info_,
    }
    if mode == "dict":
        # dict
        return data
    if mode == "yaml":
        try:
            # Lazy import for performance
            import yaml  # noqa: PLC0415
        except ImportError:
            logger.warn("PyYAML is not installed! Install with: `pip install PyYAML`")
            mode = "stdout"  # fallback stdout
        else:
            # str
            return yaml.safe_dump(data, sort_keys=False)
    # Try to use rich formatting for display
    if mode == "rich":
        try:
            # Lazy import for performance
            from rich.console import Console  # noqa: PLC0415
            from rich.table import Table  # noqa: PLC0415
        except ImportError:
            logger.warn("rich is not installed! Install with: `pip install rich`")
            mode = "stdout"  # fallback stdout
        else:
            console = Console()

            # System info
            console.print("[bold cyan]System Information:[/bold cyan]")
            table_sys = Table(show_header=True, header_style="bold magenta")
            table_sys.add_column("Key")
            table_sys.add_column("Value")
            for k, v in sys_info.items():
                table_sys.add_row(k, str(v))
            console.print(table_sys)

            # Dependency info
            console.print("\n[bold cyan]Python Dependencies:[/bold cyan]")
            table_deps = Table(show_header=True, header_style="bold magenta")
            table_deps.add_column("Library")
            table_deps.add_column("Version")
            for k, v in dep_info.items():
                table_deps.add_row(k, str(v))
            console.print(table_deps)

            # ENV info
            console.print("\n[bold cyan]Environment Variables:[/bold cyan]")
            for k, v in env_info.items():
                console.print(f"{k}: {v}")

            # GPU info
            console.print("\n[bold cyan]GPU Information:[/bold cyan]")
            for k, v in gpu_info.items():
                console.print(f"{k}: {v}")

            # Threadpoolctl threadpool_info(),  # List of dict
            if threadpool_info_:
                console.print("\n[bold cyan]Threadpoolctl Information:[/bold cyan]")
                for info_dict in threadpool_info_:
                    for k, v in info_dict.items():
                        console.print(f"[dim]{k}[/dim]: {v}")
                    console.print()
            return

    # Fallback to plain Print formatted output
    print("\nSystem Information:")
    for k, v in sys_info.items():
        print(f"{k:>25}: {v}")

    print("\nPython Dependencies:")
    for k, v in dep_info.items():
        print(f"{k:>25}: {v}")

    # if any(env_info.values()):
    print("\nEnvironment Variables:")
    for k, val in env_info.items():
        # if val is not None:
        print(f"{k:>25}: {val}")

    # if gpu_info:
    print("\nGPU Information:")
    for k, v in gpu_info.items():
        print(f"{k:>25}: {v}")

    if threadpool_info_:
        print("\nThreadpoolctl Information:")
        for i, info_dict in enumerate(threadpool_info_):
            for k, v in info_dict.items():
                print(f"{k:>25}: {v}")
            print()
