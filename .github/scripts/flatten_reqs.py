"""
Flatten and validate requirements files for use with pipenv.

Recursively resolves all ``-r``/``-c`` include directives, evaluates
environment markers against a target Python version to exclude requirements
that do not apply to that version, validates remaining lines for known invalid
patterns (e.g. bare glob specifiers), and writes two flat output files that are
consumed directly by ``pipenv install``.

WHY THIS FILE EXISTS
--------------------
Pipenv processes ``-r`` includes by copying them into a temporary constraints
file at a runner-controlled path.  Relative paths in nested ``-r`` directives
(e.g. ``default.txt`` → ``-r legacy.txt``) are resolved from *that* temp path,
not from the original file's directory, which causes "file not found" failures
for any multi-level include tree.  Flattening collapses the entire include
graph into two single files with all paths already resolved.

WHY MARKER FILTERING IS REQUIRED
---------------------------------
Pipenv's internal constraint-file generator does not correctly handle
environment markers that evaluate to ``False`` for the target Python version.
Instead of omitting such packages it writes an invalid glob specifier (e.g.
``toml*``) into its temporary constraints file, which pip rejects with::

    ERROR: Invalid requirement: 'toml*': Expected semicolon ...

Pre-filtering removes every requirement whose marker evaluates to ``False``
for the target Python version *before* pipenv runs, so the resolver never
encounters the invalid constraint.

Usage
-----
::

    python flatten_reqs.py \\
        <runtime_in> <dev_in> <runtime_out> <dev_out> <python_version>

Arguments
---------
runtime_in : path
    Top-level runtime requirements file (e.g. ``requirements/default.txt``).
dev_in : path
    Top-level dev requirements file (e.g. ``requirements/all.txt``).
runtime_out : path
    Destination for the flattened, validated runtime requirements.
dev_out : path
    Destination for the flattened, validated dev requirements.
python_version : str
    Target Python version string used to evaluate environment markers
    (e.g. ``"3.11"``, ``"3.12"``, ``"3.13t"``).  Free-threaded suffix
    ``t`` is stripped before version comparison.

Exit codes
----------
0
    Success — both output files written; all lines valid.
1
    One or more invalid requirement specifiers found (all reported to stderr
    before exit), or wrong number of CLI arguments.
2
    A referenced requirements file does not exist (reported to stderr).
"""
from __future__ import annotations

import os
import re
import sys
from typing import Generator


# ── Compiled patterns ─────────────────────────────────────────────────────────

# Matches blank lines and pure comment lines (safe to skip).
RE_BLANK_OR_COMMENT = re.compile(r"^\s*(#.*)?$")

# Matches -r / -c include directives; group(1) = flag, group(2) = path token.
# Handles optional whitespace between the flag and the path.
RE_INCLUDE = re.compile(r"^\s*(-r|-c)\s+(\S+)")

# Invalid PEP 508 specifier: a bare package name ending in ``*`` with no
# preceding version operator (e.g. ``toml*``, ``numpy*``).
# Valid compatible-release syntax (``~=X.Y.*``) is NOT matched because the
# ``*`` is preceded by a version number and a dot.
RE_INVALID_BARE_GLOB = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\*")


# ── Marker evaluation ─────────────────────────────────────────────────────────

def _parse_major_minor(python_version: str) -> str:
    """
    Normalise a Python version string to ``MAJOR.MINOR`` form.

    Parameters
    ----------
    python_version : str
        Raw version string such as ``"3.11"``, ``"3.11.5"``, or ``"3.13t"``
        (free-threaded variant).

    Returns
    -------
    str
        Normalised ``"MAJOR.MINOR"`` string, e.g. ``"3.11"``.

    Raises
    ------
    ValueError
        When the string cannot be parsed into at least MAJOR.MINOR parts.
    """
    # Strip free-threaded suffix (e.g. "3.13t" → "3.13").
    clean = python_version.rstrip("t")
    parts = clean.split(".")
    if len(parts) < 2:  # noqa: PLR2004
        raise ValueError(
            f"Cannot parse python_version {python_version!r}: "
            "expected at least MAJOR.MINOR (e.g. '3.11')."
        )
    return f"{parts[0]}.{parts[1]}"


def _build_marker_env(python_version_mm: str) -> "dict[str, str] | None":
    """
    Build a PEP 508 marker environment dict with ``python_version`` overridden
    to *python_version_mm*.

    Uses ``packaging.markers.default_environment()`` so that platform-specific
    variables (``sys_platform``, ``platform_machine``, …) reflect the actual
    CI runner.  Only the Python-version fields are overridden.

    Falls back to ``pip._vendor.packaging`` when the standalone ``packaging``
    package is not installed.

    Parameters
    ----------
    python_version_mm : str
        Target Python version in ``MAJOR.MINOR`` form, e.g. ``"3.11"``.

    Returns
    -------
    dict of str to str or None
        Marker environment dict on success; ``None`` when ``packaging`` is
        unavailable (caller treats this as "keep all lines").
    """
    default_environment = None

    try:
        from packaging.markers import default_environment  # type: ignore[assignment]
    except ImportError:
        pass

    if default_environment is None:
        try:
            # pip vendors packaging; use it as a fallback.
            from pip._vendor.packaging.markers import (  # type: ignore[no-redef]
                default_environment,
            )
        except ImportError:
            return None

    env: dict[str, str] = default_environment()

    # Override the Python-version fields to match the target version.
    env["python_version"] = python_version_mm

    # Keep python_full_version consistent with python_version.
    # If the runner's full version happens to share the same major.minor we
    # leave it; otherwise we synthesise a plausible value (MAJOR.MINOR.0).
    full = env.get("python_full_version", "")
    if not full.startswith(python_version_mm):
        env["python_full_version"] = f"{python_version_mm}.0"

    return env


def marker_applies(line: str, marker_env: "dict[str, str] | None") -> bool:
    """
    Return ``False`` only when the line has an environment marker that
    evaluates to ``False`` for the given marker environment.

    Lines without a marker always return ``True`` (keep them).
    Lines whose marker cannot be parsed return ``True`` (safe default).
    When *marker_env* is ``None`` (packaging unavailable) every line is kept.

    Parameters
    ----------
    line : str
        A single requirement line, possibly containing a PEP 508 environment
        marker after a semicolon (e.g.
        ``"tomli;python_version<'3.11'"``).
    marker_env : dict of str to str or None
        Marker environment produced by :func:`_build_marker_env`, or ``None``
        when ``packaging`` is unavailable.

    Returns
    -------
    bool
        ``True``  → requirement applies to the target Python; **keep**.
        ``False`` → requirement does not apply to the target Python; **drop**.

    Notes
    -----
    Developer note
        Platform-specific markers (``sys_platform``, ``platform_machine``,
        etc.) are evaluated against the actual CI runner values supplied by
        ``packaging.markers.default_environment()``.  This is intentional: the
        CI runner and the target Docker image share the same platform, so the
        evaluation is correct.
    """
    if marker_env is None:
        return True  # Cannot evaluate; keep line to avoid silent drops.

    if ";" not in line:
        return True  # No marker — unconditionally applicable.

    marker_str = line.split(";", 1)[1].strip()
    if not marker_str:
        return True  # Empty marker field — treat as unconditional.

    try:
        # Import here so the module-level import failure above is still caught.
        try:
            from packaging.markers import Marker
        except ImportError:
            from pip._vendor.packaging.markers import Marker  # type: ignore[no-redef]

        return bool(Marker(marker_str).evaluate(environment=marker_env))
    except Exception:  # noqa: BLE001
        # Unparseable or un-evaluable marker — keep; safe default.
        return True


# ── Core logic ────────────────────────────────────────────────────────────────

def flatten(
    req_file: str,
    visited: set[str] | None = None,
) -> Generator[tuple[str, int, str], None, None]:
    """
    Recursively yield ``(source_file, lineno, line)`` for every resolved line.

    Include directives (``-r`` / ``-c``) are followed recursively.  Each
    included path is resolved relative to the *including* file's directory,
    matching pip's own resolution semantics.

    Parameters
    ----------
    req_file : str
        Absolute (or resolvable) path to the requirements file to process.
    visited : set of str or None
        Accumulated set of already-visited absolute paths.  Passed through
        recursive calls to prevent infinite loops caused by circular includes.
        Pass ``None`` on the initial call; the function initialises it.

    Yields
    ------
    source_file : str
        Absolute path of the originating file for each yielded line.
    lineno : int
        1-based line number within *source_file*.
    line : str
        Stripped requirement line (trailing whitespace removed; leading
        whitespace preserved for alignment validation if needed downstream).

    Raises
    ------
    SystemExit(2)
        When a referenced file does not exist on disk.

    Notes
    -----
    Developer note
        Circular-include detection uses the absolute path as the deduplication
        key.  Symlinks that resolve to the same inode are therefore deduplicated
        correctly.
    """
    if visited is None:
        visited = set()

    req_file = os.path.abspath(req_file)

    if not os.path.isfile(req_file):
        print(f"ERROR: Requirements file not found: {req_file}", file=sys.stderr)
        sys.exit(2)

    if req_file in visited:
        # Already processed in this traversal — skip to break circular chain.
        return
    visited.add(req_file)

    file_dir = os.path.dirname(req_file)

    with open(req_file, encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip()

            if RE_BLANK_OR_COMMENT.match(line):
                continue  # Skip blanks and comments.

            m = RE_INCLUDE.match(line)
            if m:
                included = m.group(2)
                if not os.path.isabs(included):
                    # Resolve relative to the *current* file's directory,
                    # not to cwd — same semantics as pip itself.
                    included = os.path.join(file_dir, included)
                yield from flatten(included, visited=visited)
            else:
                yield (req_file, lineno, line)


def validate_and_write(
    input_file: str,
    output_file: str,
    label: str,
    marker_env: "dict[str, str] | None",
) -> list[str]:
    """
    Flatten *input_file*, filter by environment markers, validate each
    remaining line for invalid specifiers, and write passing lines to
    *output_file*.

    All validation errors are collected before returning so the caller can
    report every problem at once instead of failing on the first.

    Parameters
    ----------
    input_file : str
        Top-level requirements file (may contain ``-r``/``-c`` includes).
    output_file : str
        Destination path for the flattened, validated output.  Parent
        directories are created automatically.
    label : str
        Human-readable label (``"runtime"`` or ``"dev"``) used in log and
        error messages.
    marker_env : dict of str to str or None
        Marker environment for :func:`marker_applies`.  ``None`` disables
        marker filtering (all lines pass through).

    Returns
    -------
    list of str
        Validation error strings.  Empty when every line passes.

    Notes
    -----
    User note
        Output is written *only* when no errors are found, so a partial flat
        file is never produced.  If errors are found the caller reports them
        all, then exits 1.

    Developer note
        The output file always ends with a trailing newline so that tools
        consuming it (e.g. ``wc -l``, ``cat``) behave predictably.
    """
    entries = list(flatten(input_file))
    errors: list[str] = []
    valid_lines: list[str] = []
    skipped = 0

    for source, lineno, line in entries:
        # ── Filter: drop requirements inapplicable to target Python ───────
        if not marker_applies(line, marker_env):
            skipped += 1
            py_ver = (marker_env or {}).get("python_version", "?")
            print(
                f"  [{label}] skipped (marker false for py{py_ver}): "
                f"{os.path.basename(source)}:{lineno}: {line!r}"
            )
            continue

        # ── Validate: reject bare-glob specifiers (e.g. "toml*") ─────────
        if RE_INVALID_BARE_GLOB.match(line):
            pkg_name = line.rstrip("*")
            errors.append(
                f"  [{label}] {source}:{lineno}: "
                f"invalid specifier (bare glob) → {line!r}\n"
                f"    Fix: replace with a valid PEP 508 specifier, e.g.:\n"
                f"      '{pkg_name}'              (unpinned)\n"
                f"      '{pkg_name}>=<version>'   (lower-bound pin)\n"
                f"      '~={pkg_name}1.0'         (compatible-release pin)"
            )
        else:
            valid_lines.append(line)

    if not errors:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(valid_lines) + "\n")
        print(
            f"✅ [{label}] {len(valid_lines)} requirement(s) written, "
            f"{skipped} skipped by marker → {output_file}"
        )

    return errors


def main() -> None:
    """
    Entry point: parse CLI arguments, run :func:`validate_and_write` for both
    runtime and dev requirement files, and exit with the appropriate code.

    Exits
    -----
    0
        Both flat files written successfully; all specifiers valid.
    1
        Wrong argument count, unparseable Python version, or one or more
        invalid specifiers found.  All errors are reported before exit.
    2
        A referenced requirements file does not exist.
    """
    if len(sys.argv) != 6:  # noqa: PLR2004
        print(
            "Usage: flatten_reqs.py "
            "<runtime_in> <dev_in> <runtime_out> <dev_out> <python_version>",
            file=sys.stderr,
        )
        sys.exit(1)

    runtime_in, dev_in, runtime_out, dev_out, python_version_raw = sys.argv[1:]

    # ── Normalise and validate the Python version argument ─────────────────
    try:
        python_version_mm = _parse_major_minor(python_version_raw)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Build the marker evaluation environment ────────────────────────────
    marker_env = _build_marker_env(python_version_mm)
    if marker_env is None:
        print(
            "WARNING: 'packaging' is not available; environment markers will "
            "not be evaluated and all lines will be passed to pipenv.",
            file=sys.stderr,
        )
    else:
        print(f"Target Python version : {python_version_mm}")
        print(f"Marker env python_version : {marker_env['python_version']}")

    # ── Flatten, filter, validate, and write both files ───────────────────
    all_errors: list[str] = []
    all_errors.extend(
        validate_and_write(runtime_in, runtime_out, "runtime", marker_env)
    )
    all_errors.extend(
        validate_and_write(dev_in, dev_out, "dev", marker_env)
    )

    if all_errors:
        print(
            "\n❌ Invalid requirement specifiers detected.\n"
            "   Fix the offending lines in the source requirements files,\n"
            "   then re-run this workflow.\n",
            file=sys.stderr,
        )
        for err in all_errors:
            print(err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
