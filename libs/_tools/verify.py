# libs/_tools/verify.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Build the partial distributions and prove they install, alone and together.

``python -m libs._tools verify`` runs three groups of checks and exits non-zero
if any of them fails.

Artefact checks (no environment needed)
    * generated packaging files are up to date;
    * no file under ``scikitplot/`` is owned by two distributions;
    * every wheel ships exactly the files its distribution owns, byte for
      byte equal to the source tree;
    * no two wheels ship the same file, and only the core ships a console
      script;
    * a wheel built from the source distribution equals one built from the
      repository;
    * the build leaves nothing behind in the lib directories.

Environment checks, per Python version
    Each distribution is installed alone into a clean environment, then all
    of them together. In each: ``import scikitplot`` is silent, the reported
    flavor is ``partial``, ``scikitplot doctor`` finds no problem, every
    shipped module imports (or needs only an optional third-party package),
    the documented example runs, and the part's own test suite passes. A
    distribution with third-party dependencies is checked twice: with their
    newest versions, and with the lowest versions it declares.

Coexistence checks
    Every spelling of a project name installs the same project; uninstalling
    one distribution leaves every other one working.

Notes
-----
**User.** Requires `uv <https://docs.astral.sh/uv/>`_ on ``PATH``: it creates
the environments, provides the Python versions, and can resolve dependencies
to their *lowest* allowed versions, which pip cannot. ``python -m build`` must
be available to the interpreter that runs this module.

**Developer.** Every check is a function that returns :class:`Result` rows and
never raises for a failed expectation, so one failure does not hide the next.
Third-party packages come from the configured index; the distributions under
test come only from ``--outdir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

from . import generate, registry, staging

__all__ = ["Result", "build", "main", "run"]

#: Seconds allowed for one build, one installation or one test run.
TIMEOUT = 1800

#: Installed to run every part's test suite. Test tooling, not a dependency
#: of any distribution; the floor is the root project's own (its ``test``
#: extras). What a single part needs on top is in ``registry.Package``.
TEST_REQUIREMENTS: tuple[str, ...] = ("pytest>=7.1.2",)

#: The pytest plugin (a module beside this one) the suites run with.
PLUGIN = "pytest_partial"

#: Exit code of the ``scikitplot`` command line for an unavailable capability
#: (``scikitplot._cli.exit_codes.UNAVAILABLE``).
EXIT_UNAVAILABLE = 69

#: Exit code of pytest when no test was collected.
PYTEST_NO_TESTS = 5

#: Longest detail printed in the result table; failures are printed in full
#: below it.
DETAIL_WIDTH = 150

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Result(NamedTuple):
    """
    The outcome of one check.

    Parameters
    ----------
    check : str
        What was checked.
    target : str
        The distribution, combination or file it was checked on.
    python : str
        The Python version of the environment, or ``"-"`` for artefact checks.
    status : {"PASS", "FAIL", "SKIP"}
        The outcome. ``SKIP`` always carries the reason in ``detail``.
    detail : str
        Evidence: what was observed.
    """

    check: str
    target: str
    python: str
    status: str
    detail: str


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _run(
    command: Sequence[str], *, cwd: Path | None = None, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Run a command and capture its output as text; never raises on non-zero exit."""
    return subprocess.run(  # noqa: S603 - arguments are built here, not user input
        list(command),
        cwd=None if cwd is None else str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
        check=False,
    )


def _tail(text: str, lines: int = 12) -> str:
    """Return the last lines of a block of output, on one line."""
    kept = [line for line in text.strip().splitlines() if line.strip()][-lines:]
    return " | ".join(kept)


def _clean_env() -> dict:
    """Return the process environment without variables that redirect imports."""
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"):
        env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _require_uv() -> str:
    """Return the path of ``uv``, or stop with what to do about its absence."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "verify needs `uv` on PATH to create the test environments "
            "(https://docs.astral.sh/uv/). Install it, or run `python -m "
            "libs._tools build` to build without verifying."
        )
    return uv


def _running_python() -> str:
    """Return the version of the interpreter running the tooling, as ``"X.Y"``."""
    return f"{sys.version_info[0]}.{sys.version_info[1]}"


def _buildable_here(package, meta) -> bool:
    """
    Return whether a distribution can be built by the running interpreter.

    A pure-Python wheel is the same whichever interpreter builds it. A
    compiled one is built *for* the running interpreter, so it can only be
    built by one its ``requires-python`` admits: a wheel for any other could
    never be installed.
    """
    if not package.extensions:
        return True
    floor = package.requires_python or meta.requires_python
    return registry.python_satisfies(floor, _running_python())


def _venv_python(venv: Path) -> Path:
    """Return the interpreter of a virtual environment on this platform."""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _selected(names: Sequence[str] | None):
    """Return the distributions to work on, validated, in declaration order."""
    distributions = staging.load_distributions()
    if not names:
        return list(distributions.DISTRIBUTIONS)
    wanted = {distributions.get(name).name for name in names}
    return [dist for dist in distributions.DISTRIBUTIONS if dist.name in wanted]


def build(names: Sequence[str] | None, outdir: Path) -> list[Path]:
    """
    Build the source distribution and the wheel of each distribution.

    Parameters
    ----------
    names : sequence of str or None
        Distribution names in any spelling; all of them when ``None``.
    outdir : pathlib.Path
        Where the artefacts are written. Created if missing; artefacts of the
        selected distributions already there are replaced.

    Returns
    -------
    list of pathlib.Path
        The built files, sdist then wheel for each distribution.

    Raises
    ------
    SystemExit
        If ``python -m build`` is unavailable, a build fails (the message
        carries the end of the build output), or a compiled distribution is
        named that the running interpreter is too old to build.

    Notes
    -----
    **Developer.** ``python -m build`` builds the sdist from the repository
    and the wheel *from that sdist*, so every wheel produced here has already
    proved that its source distribution is complete.

    A compiled distribution is built for the running interpreter. When
    everything is being built, one that does not support this interpreter is
    left out, since a wheel for it could never be installed.
    """
    root = staging.repo_root()
    outdir = (
        (root / outdir).resolve() if not Path(outdir).is_absolute() else Path(outdir)
    )
    outdir.mkdir(parents=True, exist_ok=True)
    probe = _run([sys.executable, "-m", "build", "--version"])
    if probe.returncode != 0:
        raise SystemExit(
            "`python -m build` is not available to this interpreter. "
            "Run: python -m pip install build"
        )
    packages = registry.by_distribution()
    meta = generate.read_root_metadata(root)
    built: list[Path] = []
    for dist in _selected(names):
        stem = dist.name.replace("-", "_")
        for stale in outdir.glob(f"{stem}-*"):
            stale.unlink()
        if not _buildable_here(packages[dist.name], meta):
            floor = packages[dist.name].requires_python or meta.requires_python
            if names:
                raise SystemExit(
                    f"{dist.name} is compiled and requires Python {floor}; it "
                    f"cannot be built with Python {_running_python()}."
                )
            continue  # building everything: skip what this interpreter cannot build
        lib_dir = root / "libs" / registry.directory_of(dist.name)
        done = _run(
            [sys.executable, "-m", "build", "--outdir", str(outdir), str(lib_dir)],
            cwd=root,
            env=_clean_env(),
        )
        if done.returncode != 0:
            raise SystemExit(
                f"building {dist.name} failed:\n{_tail(done.stdout + done.stderr, 40)}"
            )
        produced = sorted(outdir.glob(f"{stem}-*"))
        if not any(p.suffix == ".whl" for p in produced) or not any(
            p.name.endswith(".tar.gz") for p in produced
        ):
            raise SystemExit(
                f"building {dist.name} produced {[p.name for p in produced]}"
            )
        built.extend(produced)
    return built


def _wheel_of(dist_name: str, outdir: Path) -> Path:
    """Return the single wheel of a distribution in ``outdir``."""
    wheels = sorted(outdir.glob(f"{dist_name.replace('-', '_')}-*.whl"))
    if len(wheels) != 1:
        raise FileNotFoundError(
            f"expected exactly one wheel of {dist_name} in {outdir}, found "
            f"{[w.name for w in wheels]}; run `python -m libs._tools build`"
        )
    return wheels[0]


# ---------------------------------------------------------------------------
# Artefact checks
# ---------------------------------------------------------------------------


def _wheel_files(wheel: Path) -> dict[str, str]:
    """
    Return ``{path: sha256}`` for every file of the package inside a wheel.

    The digests are computed here from the archive's bytes, not read from the
    wheel's own ``RECORD``, so a wheel cannot vouch for itself.
    """
    digests: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir() or ".dist-info/" in info.filename:
                continue
            digests[info.filename] = hashlib.sha256(archive.read(info)).hexdigest()
    return digests


def _wheel_metadata(wheel: Path, member: str) -> str:
    """Return a ``*.dist-info`` member of a wheel as text, or ``""`` if absent."""
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith(f".dist-info/{member}"):
                return archive.read(name).decode("utf-8")
    return ""


def _is_native(path: str) -> bool:
    """Return whether a wheel member is a compiled extension module."""
    return path.endswith((".so", ".pyd", ".dylib"))


def _wheel_metadata_problems(wheel: Path, package, meta) -> tuple[list[str], str]:
    """
    Compare a wheel's metadata with what its distribution should declare.

    Parameters
    ----------
    wheel : pathlib.Path
        The wheel to read.
    package : registry.Package
        The packaging facts of the wheel's distribution.
    meta : generate.RootMetadata
        Inherited root metadata.

    Returns
    -------
    problems : list of str
        What is wrong; empty when the metadata is right.
    summary : str
        A one-line description of the metadata, for the result when it is right.
    """
    distributions = staging.load_distributions()
    name = package.distribution
    floor = package.requires_python or meta.requires_python
    fields = [
        line for line in _wheel_metadata(wheel, "METADATA").splitlines() if ": " in line
    ]
    prefix = "Requires-Dist: "
    requires = [line[len(prefix) :] for line in fields if line.startswith(prefix)]
    base = [line for line in requires if "extra ==" not in line]
    has_core = any(generate.requirement_name(r) == distributions.CORE for r in base)
    has_scripts = "[console_scripts]" in _wheel_metadata(wheel, "entry_points.txt")
    is_pure = "-py3-none-any.whl" in wheel.name
    expectations = [
        (f"Name: {name}" in fields, "Name is not the canonical project name"),
        (f"Version: {meta.version}" in fields, f"Version is not {meta.version}"),
        (f"Requires-Python: {floor}" in fields, f"Requires-Python is not {floor}"),
        (
            has_core == (name != distributions.CORE),
            "core requirement missing or unexpected",
        ),
        (
            not any(
                generate.requirement_name(r) == distributions.FULL for r in requires
            ),
            "requires the full distribution",
        ),
        (has_scripts == package.scripts, "console script missing or unexpected"),
        (is_pure != bool(package.extensions), "wheel tag does not match pure/compiled"),
    ]
    problems = [message for holds, message in expectations if not holds]
    return problems, f"Requires-Python {floor}, {len(base)} base requirement(s)"


def check_generated() -> list[Result]:
    """Check that the generated packaging files match their inputs."""
    stale = generate.check()
    return [
        Result(
            "generated files up to date",
            "libs/",
            "-",
            FAIL if stale else PASS,
            (
                f"stale: {stale}; run python -m libs._tools generate"
                if stale
                else "0 stale"
            ),
        )
    ]


def check_artefacts(outdir: Path) -> list[Result]:
    """
    Check every wheel against the ownership map and the source tree.

    Parameters
    ----------
    outdir : pathlib.Path
        Directory holding the built wheels.

    Returns
    -------
    list of Result
    """
    root = staging.repo_root()
    distributions = staging.load_distributions(root)
    package_dir = root / staging.PACKAGE_NAME
    packages = registry.by_distribution()
    meta = generate.read_root_metadata(root)
    results: list[Result] = []
    shipped_by: dict[str, str] = {}

    try:
        owned = staging.check_ownership(distributions.DISTRIBUTIONS, package_dir)
    except ValueError as exc:
        return [Result("one owner per file", "scikitplot/", "-", FAIL, str(exc))]
    total = sum(len(files) for files in owned.values())
    results.append(
        Result(
            "one owner per file", "scikitplot/", "-", PASS, f"{total} files, 0 shared"
        )
    )

    for dist in distributions.DISTRIBUTIONS:
        try:
            wheel = _wheel_of(dist.name, outdir)
        except FileNotFoundError:
            if _buildable_here(packages[dist.name], meta):
                raise
            floor = packages[dist.name].requires_python or meta.requires_python
            results.append(
                Result(
                    "wheel ships exactly the owned files",
                    dist.name,
                    "-",
                    SKIP,
                    f"compiled; requires Python {floor}, so it is not built "
                    f"with Python {_running_python()}",
                )
            )
            continue
        files = _wheel_files(wheel)
        expected = {f"{staging.PACKAGE_NAME}/{path}" for path in owned[dist.name]}
        native = {path for path in files if _is_native(path)}
        extensions = packages[dist.name].extensions
        actual = set(files) - native

        missing, extra = sorted(expected - actual), sorted(actual - expected)
        results.append(
            Result(
                "wheel ships exactly the owned files",
                wheel.name,
                "-",
                FAIL if missing or extra else PASS,
                (
                    f"missing {missing[:5]} extra {extra[:5]}"
                    if missing or extra
                    else f"{len(actual)} files, {wheel.stat().st_size / 1024:.0f} KiB"
                ),
            )
        )
        wanted_native = {ext.name.replace(".", "/") for ext in extensions}
        got_native = {path.split(".", 1)[0] for path in native}
        results.append(
            Result(
                "wheel ships exactly the declared extensions",
                wheel.name,
                "-",
                PASS if wanted_native == got_native else FAIL,
                f"declared {sorted(wanted_native)} built {sorted(native)}",
            )
        )
        differing = [
            path
            for path in sorted(actual & expected)
            if files[path]
            != hashlib.sha256((root / Path(*path.split("/"))).read_bytes()).hexdigest()
        ]
        results.append(
            Result(
                "wheel files equal the source tree",
                wheel.name,
                "-",
                FAIL if differing else PASS,
                (
                    f"differ: {differing[:5]}"
                    if differing
                    else f"{len(actual & expected)} identical"
                ),
            )
        )
        for path in files:
            if path in shipped_by:
                results.append(
                    Result(
                        "no file in two wheels",
                        path,
                        "-",
                        FAIL,
                        f"shipped by {shipped_by[path]} and {dist.name}",
                    )
                )
            shipped_by[path] = dist.name

        problems, summary = _wheel_metadata_problems(wheel, packages[dist.name], meta)
        results.append(
            Result(
                "wheel metadata",
                wheel.name,
                "-",
                FAIL if problems else PASS,
                "; ".join(problems) if problems else summary,
            )
        )

    if not any(r.check == "no file in two wheels" for r in results):
        results.append(
            Result(
                "no file in two wheels",
                "all wheels",
                "-",
                PASS,
                f"{len(shipped_by)} files",
            )
        )
    return results


def check_wheel_from_tree(outdir: Path) -> list[Result]:
    """
    Check that a wheel built straight from the repository equals the built one.

    Parameters
    ----------
    outdir : pathlib.Path
        Directory holding the wheels that were built from source distributions.

    Returns
    -------
    list of Result

    Notes
    -----
    **Developer.** Only pure-Python wheels are compared: their content is a
    function of the sources alone. A compiled extension is not reproducible
    byte for byte across two compiler runs, and its sources are compared by
    the other artefact checks.
    """
    root = staging.repo_root()
    packages = registry.by_distribution()
    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="skplt-libs-tree-") as tmp:
        for dist in staging.load_distributions(root).DISTRIBUTIONS:
            if packages[dist.name].extensions:
                continue
            lib_dir = root / "libs" / registry.directory_of(dist.name)
            done = _run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    tmp,
                    str(lib_dir),
                ],
                cwd=root,
                env=_clean_env(),
            )
            if done.returncode != 0:
                results.append(
                    Result(
                        "wheel from tree equals wheel from sdist",
                        dist.name,
                        "-",
                        FAIL,
                        _tail(done.stdout + done.stderr),
                    )
                )
                continue
            from_sdist = _wheel_files(_wheel_of(dist.name, outdir))
            from_tree = _wheel_files(_wheel_of(dist.name, Path(tmp)))
            same = from_sdist == from_tree
            results.append(
                Result(
                    "wheel from tree equals wheel from sdist",
                    dist.name,
                    "-",
                    PASS if same else FAIL,
                    (
                        f"{len(from_tree)} files identical"
                        if same
                        else f"differ: {sorted(set(from_sdist.items()) ^ set(from_tree.items()))[:4]}"
                    ),
                )
            )
    return results


def check_no_residue() -> list[Result]:
    """Check that building left no staged file or build directory behind."""
    root = staging.repo_root()
    left: list[str] = []
    for dist in staging.load_distributions(root).DISTRIBUTIONS:
        lib_dir = root / "libs" / registry.directory_of(dist.name)
        names = [staging.PACKAGE_NAME, staging.LICENSE_NAME, *staging.RESIDUE_DIR_NAMES]
        left += [
            str((lib_dir / n).relative_to(root))
            for n in names
            if (lib_dir / n).exists()
        ]
        left += [str(p.relative_to(root)) for p in lib_dir.glob("*.egg-info")]
    return [
        Result(
            "build leaves nothing behind",
            "libs/",
            "-",
            FAIL if left else PASS,
            f"left: {left}" if left else "lib directories are clean",
        )
    ]


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

#: Runs inside a test environment. Prints one JSON document describing how
#: ``scikitplot`` behaves there. It is a string, not a module, because the
#: environment has only the distributions under test installed.
_PROBE = textwrap.dedent(
    """
    import importlib, io, json, logging, sys, traceback
    from importlib import metadata

    names = json.loads(sys.argv[1])
    out = {"import_error": None, "log": "", "modules": {}}
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logging.getLogger().addHandler(handler)
    old_stderr, sys.stderr = sys.stderr, stream
    try:
        import scikitplot
    except BaseException as exc:
        out["import_error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        sys.stderr = old_stderr
    out["log"] = stream.getvalue()
    if out["import_error"] is None:
        from scikitplot import _distributions
        out["version"] = scikitplot.__version__
        out["built_with_meson"] = scikitplot._BUILT_WITH_MESON
        out["report"] = _distributions.report()
        out["numpy_loaded"] = "numpy" in sys.modules
        try:
            out["dir"] = len(dir(scikitplot))
        except BaseException as exc:
            out["dir"] = "%s: %s" % (type(exc).__name__, exc)
        for name in names:
            for record in metadata.files(name) or ():
                parts = record.parts
                if record.suffix != ".py" or parts[0] != "scikitplot":
                    continue
                if "tests" in parts or parts[-1] == "conftest.py":
                    continue
                located = record.locate()
                # A module only if every directory from the file up to the
                # package root is a package; otherwise it is a data file that
                # happens to end in .py (a template, a fixture).
                if not all(
                    (located.parents[depth] / "__init__.py").is_file()
                    for depth in range(len(parts) - 1)
                ):
                    continue
                dotted = ".".join(parts)[: -len(".py")]
                if dotted.endswith(".__init__"):
                    dotted = dotted[: -len(".__init__")]
                if dotted.endswith("__main__"):
                    continue  # executing an entry point is the example's job
                try:
                    importlib.import_module(dotted)
                    out["modules"][dotted] = None
                except ModuleNotFoundError as exc:
                    out["modules"][dotted] = {"kind": "missing", "name": exc.name or ""}
                except BaseException as exc:
                    out["modules"][dotted] = {
                        "kind": "error",
                        "name": "%s: %s" % (type(exc).__name__, str(exc)[:200]),
                    }
    print("@@PROBE@@" + json.dumps(out))
    """,
)


class _Session(NamedTuple):
    """
    Where and how the test environments of one Python version are created.

    Parameters
    ----------
    uv : str
        Path of the ``uv`` executable.
    python : str
        Python version of the environments, e.g. ``"3.12"``.
    outdir : pathlib.Path
        Directory holding the built artefacts under test.
    base : pathlib.Path
        Directory the environments are created in.
    """

    uv: str
    python: str
    outdir: Path
    base: Path


class _Environment:
    """A clean virtual environment that installs only from ``outdir`` and the index."""

    def __init__(self, session: _Session, label: str) -> None:
        self.uv, self.version, self.outdir = session.uv, session.python, session.outdir
        self.path = session.base / f"{label}-py{session.python}"
        self.env = _clean_env()

    def create(self) -> str | None:
        """Create the environment; return an error message, or ``None``."""
        if self.path.exists():
            shutil.rmtree(self.path)
        done = _run(
            [self.uv, "venv", "--quiet", "--python", self.version, str(self.path)],
            env=self.env,
        )
        return None if done.returncode == 0 else _tail(done.stdout + done.stderr)

    @property
    def python(self) -> Path:
        return _venv_python(self.path)

    def pip(self, *args: str) -> subprocess.CompletedProcess:
        """Run ``uv pip <args>`` against this environment."""
        return _run(
            [self.uv, "pip", args[0], "--python", str(self.python), *args[1:]],
            env=self.env,
        )

    def install(self, *requirements: str, extra: Sequence[str] = ()) -> str | None:
        """
        Install requirements, taking the distributions under test from ``outdir``.

        Notes
        -----
        **Developer.** ``uv`` caches a wheel under its file name. A rebuilt
        wheel has the same name as the one it replaces, so without a refresh
        the *previous* build is installed and every check runs against stale
        files. The cache is therefore bypassed for exactly the distributions
        under test, and kept for third-party packages.
        """
        refresh: list[str] = []
        for dist in staging.load_distributions().DISTRIBUTIONS:
            refresh += ["--refresh-package", dist.name]
        done = self.pip(
            "install", "--find-links", str(self.outdir), *refresh, *extra, *requirements
        )
        return None if done.returncode == 0 else _tail(done.stdout + done.stderr)

    def run(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        """Run the environment's interpreter, from a directory outside the repository."""
        return _run([str(self.python), *args], cwd=cwd or self.path, env=self.env)


def _probe(env: _Environment, names: Sequence[str]) -> tuple[dict | None, str]:
    """Run the probe in an environment; return ``(data, raw_output)``."""
    done = env.run("-c", _PROBE, json.dumps(list(names)))
    for line in done.stdout.splitlines():
        if line.startswith("@@PROBE@@"):
            return json.loads(line[len("@@PROBE@@") :]), done.stdout + done.stderr
    return None, done.stdout + done.stderr


def _check_probe(
    data: dict | None,
    raw: str,
    names: Sequence[str],
    *,
    label: str,
    python: str,
    version: str,
) -> list[Result]:
    """Turn a probe document into results."""
    distributions = staging.load_distributions()
    if data is None:
        return [Result("probe ran", label, python, FAIL, _tail(raw))]
    if data["import_error"]:
        return [Result("import scikitplot", label, python, FAIL, data["import_error"])]
    results = [
        Result(
            "import scikitplot is silent",
            label,
            python,
            PASS if not data["log"].strip() else FAIL,
            data["log"].strip()[:300] or "no log output, no stderr",
        ),
        Result(
            "version matches",
            label,
            python,
            PASS if data["version"] == version else FAIL,
            f"scikitplot.__version__ = {data['version']!r}",
        ),
        Result(
            "dir(scikitplot) works",
            label,
            python,
            PASS if isinstance(data["dir"], int) else FAIL,
            (
                f"{data['dir']} names"
                if isinstance(data["dir"], int)
                else str(data["dir"])
            ),
        ),
    ]
    report = data["report"]
    expected = {distributions.get(name).name for name in names}
    installed = set(report["installed"])
    coherent = (
        report["flavor"] == "partial"
        and not report["problems"]
        and expected <= installed
        and distributions.FULL not in installed
    )
    results.append(
        Result(
            "doctor: partial flavor, no problem",
            label,
            python,
            PASS if coherent else FAIL,
            f"flavor={report['flavor']} installed={sorted(installed)} "
            f"problems={report['problems']}",
        )
    )

    modules = data["modules"]
    packages = registry.by_distribution()
    # Modules of an optional tier that needs a newer Python than this one
    # cannot be imported here, by the part's own definition of that tier.
    gated = {
        module: floor
        for name in names
        for module, floor in packages[distributions.get(name).name].python_gated
        if not registry.python_satisfies(floor, python)
    }
    failed = {m: v for m, v in modules.items() if v and m not in gated}
    errors = {m: v["name"] for m, v in failed.items() if v["kind"] == "error"}
    missing = {m: v["name"] for m, v in failed.items() if v["kind"] == "missing"}
    # A module of the package itself that cannot be found is a sibling part
    # imported at module level, or a file missing from the wheel: both faults.
    sibling = {
        m: n for m, n in missing.items() if n.split(".")[0] == staging.PACKAGE_NAME
    }
    optional = sorted({n.split(".")[0] for m, n in missing.items() if m not in sibling})
    bad = dict(errors, **sibling)
    results.append(
        Result(
            "every shipped module imports",
            label,
            python,
            FAIL if bad else PASS,
            (
                f"{len(bad)} of {len(modules)} fail: "
                + "; ".join(f"{m} -> {why}" for m, why in sorted(bad.items())[:6])
                if bad
                else f"{len(modules) - len(missing) - len(gated)} imported, {len(missing)} "
                f"need an optional package {optional}"
                + (
                    f", {len(gated)} gated by Python version {sorted(gated.items())}"
                    if gated
                    else ""
                )
            ),
        )
    )
    roots = [
        f"{staging.PACKAGE_NAME}.{tree.replace('/', '.')}"
        for name in names
        for tree in distributions.get(name).trees
    ]
    # A part the probe never saw is as much a failure as one that raised: its
    # package is not among the installed files.
    broken_roots = {
        root: (
            "not among the installed modules"
            if root not in modules
            else modules[root]["name"]
        )
        for root in roots
        if root not in modules or modules[root] is not None
    }
    results.append(
        Result(
            "each part imports with base dependencies only",
            label,
            python,
            FAIL if broken_roots else PASS,
            (
                "; ".join(f"{root}: {why}" for root, why in broken_roots.items())
                if broken_roots
                else f"{len(roots)} part(s): {', '.join(roots)}"
            ),
        )
    )
    return results


def _check_example(env: _Environment, package, label: str) -> Result:
    """Run a distribution's documented example in an environment."""
    if not package.example.strip():
        return Result("documented example runs", label, env.version, SKIP, "no example")
    if package.example_language == "python":
        done = env.run("-c", package.example)
    else:
        words = package.example.split()
        if words[0] == "python":
            done = env.run(*words[1:])
        else:
            script = env.python.with_name(
                words[0] + (".exe" if os.name == "nt" else "")
            )
            done = _run([str(script), *words[1:]], cwd=env.path, env=env.env)
    ok = done.returncode == 0
    return Result(
        "documented example runs",
        label,
        env.version,
        PASS if ok else FAIL,
        _tail(done.stdout, 2) if ok else _tail(done.stdout + done.stderr),
    )


def _check_cli(env: _Environment, label: str, version: str) -> list[Result]:
    """Run the ``scikitplot`` console script in an environment."""
    script = env.python.with_name("scikitplot" + (".exe" if os.name == "nt" else ""))
    results = []
    done = _run([str(script), "--version"], cwd=env.path, env=env.env)
    ok = done.returncode == 0 and done.stdout.strip() == f"scikitplot {version}"
    results.append(
        Result(
            "scikitplot --version",
            label,
            env.version,
            PASS if ok else FAIL,
            (done.stdout + done.stderr).strip()[:200],
        )
    )
    done = _run([str(script), "doctor", "--format", "json"], cwd=env.path, env=env.env)
    try:
        status = json.loads(done.stdout)["status"]
    except (ValueError, KeyError):
        status = None
    ok = done.returncode == 0 and status == "ok" and not done.stderr.strip()
    results.append(
        Result(
            "scikitplot doctor",
            label,
            env.version,
            PASS if ok else FAIL,
            f"status={status} exit={done.returncode} stderr={done.stderr.strip()[:200]!r}",
        )
    )
    # A command that needs the full distribution must say so, not crash.
    done = _run([str(script), "show-versions"], cwd=env.path, env=env.env)
    ok = (
        done.returncode == EXIT_UNAVAILABLE
        and "pip install scikit-plots" in done.stderr
    )
    results.append(
        Result(
            "full-only command is actionable",
            label,
            env.version,
            PASS if ok else FAIL,
            f"exit={done.returncode} {_tail(done.stderr, 2)}",
        )
    )
    return results


def _test_requirements(package) -> list[str]:
    """Return what must be installed to run a distribution's test suite."""
    name = package.distribution
    if package.test_extras:
        name = f"{name}[{','.join(package.test_extras)}]"
    return [name, *TEST_REQUIREMENTS, *package.test_requires]


def _ignore_options(package, tree: str) -> list[str]:
    """
    Return the pytest options that leave out a distribution's ``test_ignore``.

    Parameters
    ----------
    package : registry.Package
        The distribution's packaging facts.
    tree : str
        The owned tree being tested, e.g. ``"_externals/_sphinx_ext"``.

    Returns
    -------
    list of str
        One ``--ignore-glob`` option for each ``test_ignore`` entry under
        ``tree``, written with the path separator of the running platform.

    Notes
    -----
    **Developer.** The suite is addressed by module name (``--pyargs``), so the
    directory it is installed in is not known here. A leading ``*`` stands for
    that directory; the rest of the pattern is the full path below
    ``site-packages``, which cannot match any other file.
    """
    options = []
    for entry in package.test_ignore:
        if entry == tree or entry.startswith(tree + "/"):
            parts = [staging.PACKAGE_NAME, *entry.split("/")]
            options.append("--ignore-glob=*" + os.sep + os.sep.join(parts))
    return options


def _check_tests(env: _Environment, dist, label: str) -> list[Result]:
    """
    Run each owned part's own test suite from the installed files.

    Notes
    -----
    **Developer.** The suite runs with the ``pytest_partial`` plugin, which
    reports a test as skipped when, and only when, it needs a part of
    ``scikitplot`` that is not installed. The plugin is copied into a directory
    of its own before it is put on the path, so that nothing else from the
    tooling becomes importable in the environment under test.
    """
    package = registry.by_distribution()[dist.name]
    if package.test_extras_python and not registry.python_satisfies(
        package.test_extras_python, env.version
    ):
        return [
            Result(
                "part's own tests pass",
                label,
                env.version,
                SKIP,
                f"its test suite needs the extra(s) {list(package.test_extras)}, "
                f"which exist for Python {package.test_extras_python} only",
            )
        ]
    if package.test_python and not registry.python_satisfies(
        package.test_python, env.version
    ):
        return [
            Result(
                "part's own tests pass",
                label,
                env.version,
                SKIP,
                f"its test suite runs on Python {package.test_python} only",
            )
        ]
    plugin_dir = env.path / "skplt-pytest-plugin"
    plugin_dir.mkdir(exist_ok=True)
    shutil.copy2(
        Path(__file__).with_name(PLUGIN + ".py"), plugin_dir / (PLUGIN + ".py")
    )
    run_env = dict(env.env, PYTHONPATH=str(plugin_dir))
    results = []
    for tree in dist.trees:
        target = f"{staging.PACKAGE_NAME}.{tree.replace('/', '.')}"
        done = _run(
            [
                str(env.python),
                "-m",
                "pytest",
                "--pyargs",
                target,
                "-q",
                "-p",
                PLUGIN,
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "--no-header",
                *_ignore_options(package, tree),
            ],
            cwd=env.path,
            env=run_env,
        )
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        counts = next(
            (
                line
                for line in reversed(lines)
                if " in " in line
                and any(
                    word in line for word in ("passed", "failed", "error", "skipped")
                )
            ),
            "",
        )
        partial = next((line for line in lines if line.startswith(PLUGIN + ":")), "")
        summary = "; ".join(part for part in (counts.strip("= "), partial) if part)
        # pytest exits 0 when everything passed and 5 when nothing was collected.
        if done.returncode == PYTEST_NO_TESTS:
            status, summary = SKIP, f"no tests collected for {target}"
        else:
            status = PASS if done.returncode == 0 else FAIL
            if status == FAIL:
                summary = _tail(done.stdout + done.stderr, 14)
        results.append(
            Result(
                "part's own tests pass",
                f"{label}:{target}",
                env.version,
                status,
                summary,
            )
        )
    return results


def _third_party_requirements(package, meta) -> list[str]:
    """Return a distribution's base requirements on projects outside this repository."""
    distributions = staging.load_distributions()
    requirements = generate.inherit_requirements(
        package.inherit,
        meta.dependencies,
    ) + list(package.requires)
    return [
        line
        for line in requirements
        if generate.requirement_name(line) != distributions.CORE
    ]


def _declares_floor(requirement: str) -> bool:
    """
    Return whether a requirement string declares a lowest supported version.

    Parameters
    ----------
    requirement : str
        A dependency specifier, with or without an environment marker.

    Returns
    -------
    bool
        ``True`` when the version part (everything before ``;``) holds a
        ``>=`` or ``~=`` clause.

    Notes
    -----
    **Developer.** The project declares most requirements without a version.
    Such a requirement makes no claim about old releases, so there is nothing
    to test at its low end: its "lowest version" is the first release ever
    published, which nobody supports (measured: a version-free
    ``typing_extensions`` resolved to a 2017 release that installs a
    ``typing.py`` shadowing the standard library). Only a declared floor is a
    claim, and only claims are tested.
    """
    version_part = requirement.split(";", 1)[0]
    return ">=" in version_part or "~=" in version_part


def check_single(
    session: _Session, dist, *, run_tests: bool, lowest: bool = False
) -> list[Result]:
    """
    Install one distribution alone and check it.

    Parameters
    ----------
    session : _Session
        Where and how the environment is created.
    dist : Distribution
        The distribution to install.
    run_tests : bool
        Whether to run the part's own test suite.
    lowest : bool, optional
        For each base dependency that declares a floor, install the lowest
        version the distribution allows that has a wheel for this interpreter,
        instead of the newest. Dependencies declared without a version stay at
        the newest release (see ``_declares_floor``). The distribution's
        ``lowest_constraints`` (``registry.Package``) are applied on top.

    Returns
    -------
    list of Result

    Notes
    -----
    **Developer.** For the lowest-version run the base requirements that
    declare a floor are passed to the installer as direct requirements, because
    "lowest" is applied to direct requirements only: resolving *every*
    transitive dependency to its oldest release selects versions nobody
    declares support for. Versions
    without a wheel for the interpreter are skipped (``--only-binary``); what
    is tested is the lowest version that can actually be installed.

    The test tooling is then installed with those versions held fixed. If it
    cannot be installed beside them, the test run is reported as skipped with
    that reason, rather than run against versions that were silently raised.
    """
    python = session.python
    distributions = staging.load_distributions()
    package = registry.by_distribution()[dist.name]
    root = staging.repo_root()
    meta_version = generate.read_version(root)
    label = dist.name + (" [lowest]" if lowest else "")
    env = _Environment(
        session, registry.directory_of(dist.name) + ("-lowest" if lowest else "")
    )
    error = env.create()
    if error:
        return [Result("create environment", label, python, FAIL, error)]
    extra: list[str] = []
    requirements = [dist.name]
    if lowest:
        extra = ["--resolution", "lowest-direct", "--only-binary", ":all:"]
        if package.lowest_constraints:
            constraints = env.path.parent / (env.path.name + "-constraints.txt")
            constraints.write_text(
                "\n".join(package.lowest_constraints) + "\n", encoding="utf-8"
            )
            extra += ["--constraint", str(constraints)]
        requirements += [
            line
            for line in _third_party_requirements(
                package, generate.read_root_metadata(root)
            )
            if _declares_floor(line)
        ]
    error = env.install(*requirements, extra=extra)
    if error:
        return [Result("install", label, python, FAIL, error)]
    listing = env.pip("list", "--format", "json")
    installed = {
        p["name"].lower().replace("_", "-"): p["version"]
        for p in json.loads(listing.stdout or "[]")
    }
    results = [
        Result(
            "install",
            label,
            python,
            PASS,
            f"{len(installed)} package(s): "
            + ", ".join(f"{n} {v}" for n, v in sorted(installed.items())),
        )
    ]
    names = sorted({distributions.CORE, dist.name})
    data, raw = _probe(env, names)
    results += _check_probe(
        data, raw, names, label=label, python=python, version=meta_version
    )
    results.append(_check_example(env, package, label))
    results += _check_cli(env, label, meta_version)
    if run_tests:
        # Installed after every other check, so that those ran with the base
        # dependencies alone.
        extra = []
        if lowest:
            held = env.path / "held-versions.txt"
            held.write_text(
                "".join(
                    f"{name}=={version}\n"
                    for name, version in sorted(installed.items())
                    if not name.startswith(distributions.FULL)
                ),
                encoding="utf-8",
            )
            extra = ["--constraint", str(held)]
        if package.test_extras_python and not registry.python_satisfies(
            package.test_extras_python, python
        ):
            # Reported as a skip, with the reason, by ``_check_tests``.
            return results + _check_tests(env, dist, label)
        error = env.install(*_test_requirements(package), extra=extra)
        if error and lowest:
            results.append(
                Result(
                    "part's own tests pass",
                    label,
                    python,
                    SKIP,
                    "the test tooling cannot be installed beside the lowest "
                    f"dependency versions: {error}",
                )
            )
        elif error:
            results.append(Result("install test tooling", label, python, FAIL, error))
        else:
            results += _check_tests(env, dist, label)
    return results


def check_refused(session: _Session, dist, floor: str) -> list[Result]:
    """
    Check that an installer refuses a distribution on a Python it excludes.

    Parameters
    ----------
    session : _Session
        Where and how the environment is created; its Python is one the
        distribution's ``requires-python`` excludes.
    dist : Distribution
        The distribution to try to install.
    floor : str
        The distribution's ``requires-python``.

    Returns
    -------
    list of Result

    Notes
    -----
    **Developer.** ``requires-python`` is a promise in both directions. On a
    Python the range admits, the distribution is installed and tested. On one
    it excludes, the range must actually keep the distribution out: a refused
    install is a clear message, while an accepted one is an ``ImportError``
    at some later point.
    """
    python = session.python
    env = _Environment(session, registry.directory_of(dist.name) + "-refused")
    error = env.create()
    if error:
        return [Result("create environment", dist.name, python, FAIL, error)]
    error = env.install(dist.name)
    return [
        Result(
            "installer refuses an unsupported Python",
            dist.name,
            python,
            PASS if error else FAIL,
            (
                f"requires Python {floor}; refused"
                if error
                else f"requires Python {floor} but installed on {python}"
            ),
        )
    ]


def check_together(
    session: _Session, dists: Sequence, *, run_tests: bool = False
) -> list[Result]:
    """
    Install several distributions together, then uninstall them one at a time.

    Parameters
    ----------
    session : _Session
        Where and how the environment is created.
    dists : sequence of Distribution
        The distributions to install together.
    run_tests : bool, optional
        Also run the test suites of the parts that integrate with a sibling
        part, now that the sibling is installed.

    Returns
    -------
    list of Result

    Notes
    -----
    **Developer.** A part that can use a sibling (``scikitplot.mcp`` searching
    a ``scikitplot.corpus`` index) has tests for that integration. Alone, they
    are skipped because the sibling is absent; here they run for real.

    The uninstall loop is the test of the one-owner-per-file
    invariant on a real installation: after each removal the parts that remain
    must still import, and the removed one must be reported as available
    again rather than as broken.
    """
    distributions = staging.load_distributions()
    meta_version = generate.read_version(staging.repo_root())
    names = [dist.name for dist in dists]
    label = "+".join(registry.directory_of(name) for name in names)
    python = session.python
    env = _Environment(session, "together")
    error = env.create()
    if error:
        return [Result("create environment", label, python, FAIL, error)]
    error = env.install(*names)
    if error:
        return [Result("install together", label, python, FAIL, error)]
    results = [
        Result("install together", label, python, PASS, f"{len(names)} distributions")
    ]
    data, raw = _probe(env, names)
    results += _check_probe(
        data, raw, names, label=label, python=python, version=meta_version
    )
    results += _check_cli(env, label, meta_version)

    if run_tests:
        packages = registry.by_distribution()
        for dist in dists:
            package = packages[dist.name]
            if not any(
                set(siblings) <= set(names) for _extra, siblings in package.siblings
            ):
                continue
            if package.test_extras_python and not registry.python_satisfies(
                package.test_extras_python, python
            ):
                results += _check_tests(
                    env, dist, "together"
                )  # a skip, with the reason
                continue
            error = env.install(*_test_requirements(package))
            if error:
                results.append(
                    Result(
                        "install test tooling",
                        f"{label}:{dist.name}",
                        python,
                        FAIL,
                        error,
                    )
                )
                continue
            results += _check_tests(env, dist, "together")

    remaining = list(names)
    for name in [n for n in names if n != distributions.CORE]:
        done = env.pip("uninstall", name)
        remaining.remove(name)
        if done.returncode != 0:
            results.append(
                Result(
                    "uninstall one, the rest still work",
                    name,
                    python,
                    FAIL,
                    _tail(done.stdout + done.stderr),
                )
            )
            continue
        data, raw = _probe(env, remaining)
        after = _check_probe(
            data,
            raw,
            remaining,
            label=f"after removing {name}",
            python=python,
            version=meta_version,
        )
        failed = [r for r in after if r.status == FAIL]
        gone = (
            data is not None
            and not data["import_error"]
            and name in data["report"]["available"]
        )
        results.append(
            Result(
                "uninstall one, the rest still work",
                name,
                python,
                PASS if not failed and gone else FAIL,
                (
                    "; ".join(f"{r.check}: {r.detail}" for r in failed)[:400]
                    if failed
                    else f"{len(remaining)} remaining import; {name} reported as "
                    + ("available" if gone else "NOT available")
                ),
            )
        )
    return results


def check_spellings(session: _Session) -> list[Result]:
    """
    Check that every spelling of a project name installs the same project.

    Parameters
    ----------
    session : _Session
        Where and how the environments are created.

    Returns
    -------
    list of Result

    Notes
    -----
    **Developer.** Installed with ``--no-deps --no-index`` so the only thing
    exercised is how the installer matches the name against the files in
    ``outdir``.
    """
    python, outdir = session.python, session.outdir
    packages = registry.by_distribution()
    results = []
    for dist in staging.load_distributions().DISTRIBUTIONS:
        if (
            "-" not in registry.directory_of(dist.name)
            or packages[dist.name].extensions
        ):
            continue  # only names with a separator after the prefix are interesting
        tail = registry.directory_of(dist.name)
        spellings = [
            dist.name,
            dist.name.replace("-", "_"),
            f"scikit-plots-{tail.replace('-', '_')}",
            f"scikit-plots_{tail.replace('-', '_')}",
            f"Scikit.Plots.{tail.replace('-', '.').upper()}",
        ]
        for spelling in spellings:
            env = _Environment(session, "spelling")
            error = env.create()
            if error:
                results.append(
                    Result("create environment", spelling, python, FAIL, error)
                )
                continue
            done = env.pip(
                "install",
                "--no-deps",
                "--no-index",
                "--find-links",
                str(outdir),
                "--refresh-package",
                dist.name,
                spelling,
            )
            listing = json.loads(env.pip("list", "--format", "json").stdout or "[]")
            got = sorted(p["name"].lower().replace("_", "-") for p in listing)
            ok = done.returncode == 0 and got == [dist.name]
            results.append(
                Result(
                    "every spelling installs the same project",
                    spelling,
                    python,
                    PASS if ok else FAIL,
                    f"installed {got}" if ok else _tail(done.stdout + done.stderr),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    outdir: Path,
    pythons: Sequence[str],
    *,
    names: Sequence[str] | None = None,
    skip_build: bool = False,
    skip_tests: bool = False,
) -> list[Result]:
    """
    Run every check.

    Parameters
    ----------
    outdir : pathlib.Path
        Where artefacts are built, or already are when ``skip_build`` is set.
    pythons : sequence of str
        Python versions to create environments for.
    names : sequence of str, optional
        Distributions to install and test, in any spelling; all of them when
        omitted. The core is always included, because every other one needs
        it. The artefact checks always cover every distribution: they are
        statements about the whole set.
    skip_build : bool, optional
        Reuse the artefacts already in ``outdir``.
    skip_tests : bool, optional
        Do not run each part's own test suite.

    Returns
    -------
    list of Result
        Every outcome, in the order the checks ran.
    """
    root = staging.repo_root()
    outdir = (
        (root / outdir).resolve() if not Path(outdir).is_absolute() else Path(outdir)
    )
    uv = _require_uv()
    results = check_generated()
    if not skip_build:
        build(None, outdir)
    results += check_artefacts(outdir)
    if not skip_build:
        results += check_wheel_from_tree(outdir)
    results += check_no_residue()

    distributions = staging.load_distributions(root)
    packages = registry.by_distribution()
    meta = generate.read_root_metadata(root)
    running = _running_python()
    chosen = _selected(names)
    if names and distributions.CORE not in {dist.name for dist in chosen}:
        chosen.insert(0, distributions.get(distributions.CORE))
    with tempfile.TemporaryDirectory(prefix="skplt-libs-env-") as tmp:
        base = Path(tmp)
        for python in pythons:
            session = _Session(uv, python, outdir, base)
            dists = list(chosen)
            floors = {
                d.name: packages[d.name].requires_python or meta.requires_python
                for d in dists
            }
            supported = [
                d for d in dists if registry.python_satisfies(floors[d.name], python)
            ]
            # A compiled wheel exists only for the interpreter that built it.
            usable = [
                d
                for d in supported
                if not packages[d.name].extensions or python == running
            ]
            for dist in dists:
                if dist not in supported:
                    # Checked with pure-Python wheels only. A compiled one is
                    # not built for a Python it does not support, so there is
                    # nothing for the installer to refuse.
                    if not packages[dist.name].extensions:
                        results += check_refused(session, dist, floors[dist.name])
                    continue
                if dist not in usable:
                    results.append(
                        Result(
                            "install",
                            dist.name,
                            python,
                            SKIP,
                            f"compiled wheel was built for Python {running} only",
                        )
                    )
                    continue
                results += check_single(session, dist, run_tests=not skip_tests)
                package = packages[dist.name]
                # Nothing to test at a low end when no requirement states one.
                if any(
                    _declares_floor(line)
                    for line in _third_party_requirements(package, meta)
                ):
                    results += check_single(
                        session, dist, run_tests=not skip_tests, lowest=True
                    )
            results += check_together(session, usable, run_tests=not skip_tests)
        results += check_spellings(_Session(uv, pythons[0], outdir, base))
    return results


def _print(results: Iterable[Result]) -> None:
    """Print results as an aligned table, failures last and in full."""
    rows = list(results)
    write = sys.stdout.write
    for row in rows:
        detail = row.detail
        if len(detail) > DETAIL_WIDTH:
            detail = detail[: DETAIL_WIDTH - 3] + "..."
        write(
            f"{row.status:<4} py{row.python:<5} {row.check:<46} "
            f"{row.target:<34} {detail}\n"
        )
    failed = [row for row in rows if row.status == FAIL]
    counts = {s: sum(1 for row in rows if row.status == s) for s in (PASS, FAIL, SKIP)}
    write(f"\n{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped.\n")
    for row in failed:
        write(f"\nFAIL py{row.python} {row.check} [{row.target}]\n  {row.detail}\n")


def main(args: argparse.Namespace) -> int:
    """
    Run ``verify`` from parsed command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        ``names``, ``outdir``, ``python``, ``skip_build``, ``skip_tests`` and
        ``report``, as defined in ``libs/_tools/__main__.py``.

    Returns
    -------
    int
        0 when every check passed or was skipped with a reason, 1 otherwise.
    """
    pythons = args.python or [_running_python()]
    results = run(
        Path(args.outdir),
        pythons,
        names=args.names or None,
        skip_build=args.skip_build,
        skip_tests=args.skip_tests,
    )
    _print(results)
    if args.report:
        Path(args.report).write_text(
            json.dumps([row._asdict() for row in results], indent=1) + "\n",
            encoding="utf-8",
        )
    return 1 if any(row.status == FAIL for row in results) else 0
