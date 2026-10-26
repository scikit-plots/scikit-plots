# scikitplot/_testing/_runtime_report.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Say which native ML stack and model cache a test session runs with.

A pytest plugin, registered by the root ``scikitplot/conftest.py``. It prints
two lines in the session header (the versions of TensorFlow, torch, triton
and the model libraries; the model cache and the offline switches), and it
names the test that loaded TensorFlow into a session where that is known to
crash a later import of triton.

Notes
-----
**User notes.** The header answers "what was installed, was the network
off, which models were already cached" from the log of a run, without a
rerun. The warning answers "which test made this session unsafe": with
TensorFlow loaded first, a later import of triton (``torch._dynamo``,
``transformers``, ``sentence-transformers``) crashes the process with
SIGSEGV and no Python error (measured with tensorflow 2.21.0 and triton
3.8.0; the reverse order works). TensorFlow tests run in a session of their
own: ``SKPLT_TEST_TENSORFLOW=1 python -m pytest scikitplot/visualkeras``.

**Developer notes.** Versions come from installed metadata
(:py:func:`importlib.metadata.version`), so the header imports none of the
libraries it names. The cache location follows ``huggingface_hub``
(checked on 2.3.0): ``HF_HUB_CACHE``, else ``HUGGINGFACE_HUB_CACHE``, else
``$HF_HOME/hub``, where ``HF_HOME`` defaults to
``$XDG_CACHE_HOME/huggingface`` and that to ``~/.cache/huggingface``; a
model is a ``models--<owner>--<name>`` directory there. The per-test check
is one lookup in :data:`sys.modules` after each teardown report, until
TensorFlow is seen; the warning is written there because what a test writes
while it runs is captured and never shown when it passes.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "TENSORFLOW_SESSION_ENV",
    "RuntimeReport",
    "cached_models",
    "describe_runtime",
    "hub_cache_dir",
    "register",
    "tensorflow_conflict",
]

#: Set to ``1`` to declare a session that may load TensorFlow.
TENSORFLOW_SESSION_ENV = "SKPLT_TEST_TENSORFLOW"

#: The distributions named in the header, in this order.
DISTRIBUTIONS = (
    "tensorflow",
    "torch",
    "triton",
    "transformers",
    "sentence-transformers",
    "huggingface-hub",
    "openai-whisper",
    "faster-whisper",
    "open_clip_torch",
)

#: ``huggingface_hub``'s spelling of a true switch (``ENV_VARS_TRUE_VALUES``).
_TRUE = frozenset({"1", "ON", "YES", "TRUE"})

_PLUGIN_NAME = "skplt-runtime-report"

#: How many cached model names the header shows before "...".
_SHOWN_MODELS = 8


def _version(name: str) -> str | None:
    """
    Return the installed version of distribution ``name``, or ``None``.

    Parameters
    ----------
    name : str
        A distribution name, as pip spells it.

    Returns
    -------
    str or None
        Its version from installed metadata; ``None`` when not installed.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tensorflow_version() -> str | None:
    """
    Return the TensorFlow version, also when only ``tensorflow-cpu`` is installed.

    Returns
    -------
    str or None
        The version of ``tensorflow``, else of ``tensorflow-cpu``, else ``None``.
    """
    return _version("tensorflow") or _version("tensorflow-cpu")


def hub_cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """
    Return the Hugging Face Hub cache directory, as ``huggingface_hub`` finds it.

    Parameters
    ----------
    environ : mapping of str to str or None, default=None
        The environment to read; ``None`` reads :data:`os.environ`.

    Returns
    -------
    pathlib.Path
        The directory, whether or not it exists.
    """
    env = os.environ if environ is None else environ
    explicit = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(os.path.expandvars(os.path.expanduser(explicit)))
    home = env.get("HF_HOME")
    if not home:
        xdg = env.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        home = os.path.join(xdg, "huggingface")
    return Path(os.path.expandvars(os.path.expanduser(home))) / "hub"


def cached_models(cache_dir: Path) -> list[str]:
    """
    Return the model repositories present in a Hub cache, as ``owner/name``.

    Parameters
    ----------
    cache_dir : pathlib.Path
        A Hub cache directory (:func:`hub_cache_dir`).

    Returns
    -------
    list of str
        Sorted; empty when the directory does not exist.
    """
    try:
        entries = list(cache_dir.iterdir())
    except OSError:
        return []
    return sorted(
        entry.name[len("models--") :].replace("--", "/", 1)
        for entry in entries
        if entry.is_dir() and entry.name.startswith("models--")
    )


def describe_runtime(environ: Mapping[str, str] | None = None) -> list[str]:
    """
    Return the header lines: the ML stack, the model cache and the switches.

    Parameters
    ----------
    environ : mapping of str to str or None, default=None
        The environment to read; ``None`` reads :data:`os.environ`.

    Returns
    -------
    list of str
        Two lines, ready for ``pytest_report_header``.
    """
    env = os.environ if environ is None else environ
    present, absent = [], []
    for name in DISTRIBUTIONS:
        version = _tensorflow_version() if name == "tensorflow" else _version(name)
        if version is None:
            absent.append(name)
        else:
            present.append(f"{name} {version}")
    stack = "ML runtime: " + (", ".join(present) or "none of the ML libraries")
    if absent:
        stack += " (absent: " + ", ".join(absent) + ")"

    cache = hub_cache_dir(env)
    models = cached_models(cache)
    offline = any(
        env.get(name, "").upper() in _TRUE
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    shown = ", ".join(models[:_SHOWN_MODELS]) + (
        ", ..." if len(models) > _SHOWN_MODELS else ""
    )
    session = "yes" if env.get(TENSORFLOW_SESSION_ENV, "") == "1" else "no"
    models_part = f"{len(models)} model(s): {shown}" if models else "no model cached"
    cache_line = (
        f"model cache: {cache.as_posix()} ({models_part}); "
        f"Hugging Face offline: {'yes' if offline else 'no'}; "
        f"TensorFlow session ({TENSORFLOW_SESSION_ENV}): {session}"
    )
    return [stack, cache_line]


def tensorflow_conflict(
    modules: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    find_spec: Any = None,
) -> bool:
    """
    Return whether this process holds TensorFlow where triton can still load.

    Parameters
    ----------
    modules : mapping or None, default=None
        The loaded modules; ``None`` reads :data:`sys.modules`.
    environ : mapping of str to str or None, default=None
        The environment; ``None`` reads :data:`os.environ`.
    find_spec : callable or None, default=None
        ``importlib.util.find_spec``, replaceable in tests.

    Returns
    -------
    bool
        ``True`` when TensorFlow is loaded, triton is installed but its
        native library is not loaded yet, and the session is not a declared
        TensorFlow session.
    """
    loaded = sys.modules if modules is None else modules
    env = os.environ if environ is None else environ
    spec = importlib.util.find_spec if find_spec is None else find_spec
    return (
        "tensorflow" in loaded
        and "triton._C.libtriton" not in loaded
        and env.get(TENSORFLOW_SESSION_ENV, "") != "1"
        and spec("triton") is not None
    )


class RuntimeReport:
    """
    The pytest plugin: header lines, and who loaded TensorFlow.

    Parameters
    ----------
    environ : mapping of str to str or None, default=None
        The environment; ``None`` reads :data:`os.environ` when asked.

    Attributes
    ----------
    loaded_by : str or None
        ``"collection"``, the node id of the test after which TensorFlow was
        first seen in :data:`sys.modules`, or ``None``.
    conflict : bool
        Whether TensorFlow was loaded where triton can still load
        (:func:`tensorflow_conflict`).
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ
        self.loaded_by: str | None = None
        self.conflict = False
        self._config: Any = None

    def pytest_configure(self, config: Any) -> None:  # noqa: D102
        self._config = config

    def pytest_report_header(self, config: Any) -> list[str]:  # noqa: D102
        return describe_runtime(self._environ)

    def pytest_collection_finish(self, session: Any) -> None:  # noqa: D102
        self._check("collection")

    def pytest_runtest_logreport(self, report: Any) -> None:  # noqa: D102
        # After the teardown report, not in a teardown hook: output written
        # while a test runs (setup, call, teardown) goes to its captured
        # output, which a passing test never shows.
        if self.loaded_by is None and report.when == "teardown":
            self._check(report.nodeid)

    def _check(self, where: str) -> None:
        if "tensorflow" not in sys.modules:
            return
        self.loaded_by = where
        self.conflict = tensorflow_conflict(environ=self._environ)
        if self.conflict:
            self._write(self.message())

    def message(self) -> str:
        """
        Return the warning about the test that loaded TensorFlow.

        Returns
        -------
        str
            One line naming :attr:`loaded_by`, what will crash and how to run
            TensorFlow tests instead.
        """
        where = (
            "during collection"
            if self.loaded_by == "collection"
            else f"by {self.loaded_by}"
        )
        return (
            f"TensorFlow was loaded into this test process {where}. With "
            "triton installed, a later import of triton (torch._dynamo, "
            "transformers, sentence-transformers) crashes the process with "
            "SIGSEGV and no Python error (tensorflow 2.21.0 with triton 3.8.0). "
            f"Run TensorFlow tests on their own: {TENSORFLOW_SESSION_ENV}=1 "
            "python -m pytest <their path>"
        )

    def _write(self, text: str) -> None:
        reporter = (
            self._config.pluginmanager.get_plugin("terminalreporter")
            if self._config is not None
            else None
        )
        if reporter is not None:
            reporter.write_line("")
            reporter.write_line(f"WARNING {text}", yellow=True, bold=True)
        else:
            sys.stderr.write(f"WARNING {text}\n")

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:  # noqa: D102
        if self.loaded_by is None:
            return
        terminalreporter.section("native runtime")
        if self.conflict:
            terminalreporter.write_line(self.message(), yellow=True)
        else:
            where = (
                "during collection"
                if self.loaded_by == "collection"
                else f"by {self.loaded_by}"
            )
            terminalreporter.write_line(f"TensorFlow was loaded {where}.")


def register(config: Any, environ: Mapping[str, str] | None = None) -> RuntimeReport:
    """
    Register the plugin once per pytest session.

    Parameters
    ----------
    config : pytest.Config
        The session's configuration.
    environ : mapping of str to str or None, default=None
        The environment; ``None`` reads :data:`os.environ`.

    Returns
    -------
    RuntimeReport
        The registered plugin (the existing one when already registered).
    """
    existing = config.pluginmanager.get_plugin(_PLUGIN_NAME)
    if existing is not None:
        return existing
    plugin = RuntimeReport(environ)
    plugin.pytest_configure(config)
    config.pluginmanager.register(plugin, _PLUGIN_NAME)
    return plugin
