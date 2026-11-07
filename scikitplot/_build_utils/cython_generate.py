#!/usr/bin/env python3
"""
Cython Template Code Generator for Annoy.

Processes ``.pyx.in`` and ``.pxd.in`` template files using Tempita to
generate the concrete ``.pyx`` / ``.pxd`` files with all type combinations
expanded.

Usage::

    # Auto-detect .in files in current directory
    python cython_generate.py

    # Explicit source directory
    python cython_generate.py --src-dir scikitplot/annoy/_annoy

    # Verbose + validation
    python cython_generate.py -v --validate

    # Debug mode
    python cython_generate.py --debug

    # Dry-run (no files written)
    python cython_generate.py --dry-run

Template Engine:

Uses ``Cython.Tempita`` (shipped with Cython, no extra dependency).
Falls back to standalone ``tempita`` package if Cython is unavailable.

Template Files:

Input  : ``*.in``            e.g. ``annoylib.pxd.in``, ``annoylib.pyx.in``
Output : strip ``.in`` ext   e.g. ``annoylib.pxd``,    ``annoylib.pyx``

Notes
-----
Template syntax uses ``{{py: ... }}`` for configuration blocks and
``{{for ...}}`` / ``{{endfor}}`` for code generation loops.  See
`Cython.Tempita` or the standalone `tempita` documentation.

Security:

* Only reads ``.in`` files from the specified source directory.
* No shell execution – Tempita templates are processed in-process.
* Output is written atomically to the same directory as the template.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Import Tempita – Cython ships its own copy; fall back to standalone package
# ---------------------------------------------------------------------------
try:
    from Cython import Tempita as _tempita_mod

    _Template = _tempita_mod.Template
    _tempita_source = "Cython.Tempita"
except ImportError:
    try:
        import tempita as _tempita_mod  # type: ignore[no-redef]

        _Template = _tempita_mod.Template
        _tempita_source = "tempita (standalone)"
    except ImportError:
        print(
            "ERROR: Tempita not available.\n"
            "Install Cython (preferred) or the standalone tempita package:\n"
            "    pip install cython\n"
            "    # or\n"
            "    pip install tempita",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_template_content(
    content: str,
    filepath: Path,
) -> Tuple[bool, Optional[str]]:
    """
    Perform lightweight sanity checks on a template file before processing.

    Parameters
    ----------
    content : str
        Full text of the template file.
    filepath : Path
        Path to the template file (used in error messages).

    Returns
    -------
    is_valid : bool
        ``True`` if the template passes all checks.
    error_msg : str or None
        Human-readable error description, or ``None`` when valid.

    Notes
    -----
    Checks performed:

    1. **Balanced delimiters** – the number of ``{{`` tokens must equal the
       number of ``}}`` tokens.  Imbalanced delimiters cause cryptic Tempita
       parse errors.

    2. **Non-empty content** – an empty template is almost certainly a
       read error.
    """
    if not content.strip():
        return False, f"Template file is empty: {filepath}"

    open_count = content.count("{{")
    close_count = content.count("}}")

    if open_count != close_count:
        return False, (
            f"Unbalanced template delimiters in {filepath}: "
            f"{open_count} '{{{{' but {close_count} '}}}}'"
        )

    return True, None


def process_template(
    template_path: Path,
    output_path: Path,
    verbose: bool = False,
    debug: bool = False,
) -> Tuple[int, int]:
    """
    Process a single Tempita template file and write the result.

    Parameters
    ----------
    template_path : Path
        Path to the ``.in`` template file.
    output_path : Path
        Path to write the generated output.
    verbose : bool, default=False
        Print processing details.
    debug : bool, default=False
        Print additional debug information.

    Returns
    -------
    template_lines : int
        Number of lines in the input template.
    generated_lines : int
        Number of lines in the generated output.

    Raises
    ------
    FileNotFoundError
        If ``template_path`` does not exist.
    ValueError
        If template validation fails (e.g. unbalanced delimiters).
    PermissionError
        If the template cannot be read or the output cannot be written.
    RuntimeError
        If Tempita substitution fails.

    Notes
    -----
    The output directory is created automatically if it does not exist.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not template_path.is_file():
        raise ValueError(f"Template path is not a regular file: {template_path}")

    if verbose:
        print(f"  Processing : {template_path}")
        print(f"  Output     : {output_path}")

    # Read template
    try:
        template_content = template_path.read_text(encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(f"Cannot read template {template_path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read template {template_path}: {exc}") from exc

    template_lines = template_content.count("\n")

    if debug:
        print(
            f"  Template   : {len(template_content):,} bytes, "
            f"{template_lines:,} lines"
        )

    # Validate
    is_valid, error_msg = validate_template_content(template_content, template_path)
    if not is_valid:
        raise ValueError(f"Template validation failed: {error_msg}")

    # Substitute
    generated_content = _render(template_content, template_path)
    if debug:
        print("  Substituted: ok")

    generated_lines = generated_content.count("\n")

    # Write output (create parent dirs if needed), atomically.
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(output_path, generated_content)
    except PermissionError as exc:
        raise PermissionError(f"Cannot write output {output_path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to write output {output_path}: {exc}") from exc

    if verbose:
        ratio = generated_lines / template_lines if template_lines else float("nan")
        print(
            f"  Lines      : {template_lines:,} → {generated_lines:,} "
            f"(expansion {ratio:.1f}x)"
        )

    return template_lines, generated_lines


def _render(template_content: str, template_path: Path) -> str:
    """
    Substitute one template's text and return the generated text.

    Parameters
    ----------
    template_content : str
        The template's text.
    template_path : Path
        Its path, for messages and Tempita's name.

    Returns
    -------
    str
        The generated text; never blank.

    Raises
    ------
    RuntimeError
        If Tempita fails, or the output is blank.
    """
    try:
        generated = _Template(template_content, name=str(template_path)).substitute()
    except Exception as exc:
        raise RuntimeError(f"Tempita failed for {template_path}: {exc}") from exc
    if not generated.strip():
        raise RuntimeError(f"Template produced empty output for {template_path}")
    return generated


def _write_atomically(output_path: Path, content: str) -> None:
    """
    Write *content* to *output_path* so a reader never sees a partial file.

    Parameters
    ----------
    output_path : Path
        Final path; its directory must exist.
    content : str
        Text, written as UTF-8 with the platform's newline translation (as
        :meth:`pathlib.Path.write_text` does).

    Notes
    -----
    The text goes to a temporary file in the same directory, which is then
    renamed over the target with :func:`os.replace` (atomic on POSIX and on
    Windows for paths on one volume). On any failure the temporary file is
    removed and the previous output, if any, is left as it was.
    """
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, str(output_path))
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def find_template_files(directory: Path) -> List[Path]:
    """
    Find all ``.in`` template files in ``directory`` (non-recursive).

    Parameters
    ----------
    directory : Path
        Directory to search.

    Returns
    -------
    list of Path
        Sorted list of ``.in`` files found.

    Notes
    -----
    Only direct children of ``directory`` are returned; subdirectories
    are not recursed into.
    """
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.in"))


def generate_all(
    src_dir: Path,
    verbose: bool = False,
    dry_run: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Discover and process all ``.in`` template files in ``src_dir``.

    Parameters
    ----------
    src_dir : Path
        Source directory containing ``.in`` files.
    verbose : bool, default=False
        Print per-file details.
    dry_run : bool, default=False
        Report what would be done without writing any files.
    debug : bool, default=False
        Print detailed debug output for each template.

    Returns
    -------
    dict
        Summary with keys:

        * ``success`` – ``True`` iff no errors occurred.
        * ``template_files`` – list of discovered template paths.
        * ``generated_files`` – list of successfully generated output paths.
        * ``total_template_lines`` – total template lines processed.
        * ``total_generated_lines`` – total generated lines written.
        * ``errors`` – list of error messages.

    Raises
    ------
    FileNotFoundError
        If ``src_dir`` does not exist.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    template_files = find_template_files(src_dir)

    if not template_files:
        print(f"WARNING: No .in files found in {src_dir}", file=sys.stderr)

    print()
    print("=" * 70)
    print(f"Annoy Cython Template Generator  (engine: {_tempita_source})")
    print("=" * 70)
    print(f"Source dir : {src_dir}")
    print(f"Templates  : {len(template_files)} found")
    print()

    generated_files: List[Path] = []
    errors: List[str] = []
    total_template_lines = 0
    total_generated_lines = 0

    for template_path in template_files:
        output_path = template_path.parent / template_path.stem  # strip .in

        if dry_run:
            print(f"  [DRY RUN] {template_path.name} → {output_path.name}")
            continue

        try:
            t_lines, g_lines = process_template(
                template_path,
                output_path,
                verbose=verbose,
                debug=debug,
            )
            total_template_lines += t_lines
            total_generated_lines += g_lines
            generated_files.append(output_path)
            print(f"  ✓  {output_path.name}  ({g_lines:,} lines)")
        except Exception as exc:
            msg = f"  ✗  {template_path.name}: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            # Continue so all templates are attempted

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    if not dry_run:
        ok_count = len(template_files) - len(errors)
        print(f"  Processed  : {ok_count}/{len(template_files)} templates")
        print(f"  Template ↑ : {total_template_lines:,} lines")
        print(f"  Generated ↓: {total_generated_lines:,} lines")
        if total_template_lines > 0:
            ratio = total_generated_lines / total_template_lines
            reduction = (1 - total_template_lines / total_generated_lines) * 100
            print(
                f"  Expansion  : {ratio:.1f}x  ({reduction:.0f}% less hand-written code)"
            )

    if errors:
        print(f"\n  ⚠  {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"    {err}", file=sys.stderr)

    return {
        "success": len(errors) == 0,
        "template_files": template_files,
        "generated_files": generated_files,
        "total_template_lines": total_template_lines,
        "total_generated_lines": total_generated_lines,
        "errors": errors,
    }


def validate_generated_files(src_dir: Path, verbose: bool = False) -> bool:
    """
    Verify that every generated file is exactly what its template produces.

    Parameters
    ----------
    src_dir : Path
        Directory containing the ``.in`` templates and their outputs.
    verbose : bool, default=False
        Print per-file results.

    Returns
    -------
    bool
        ``True`` if every template's output passes; ``False`` if any fails or
        no template is found.

    Notes
    -----
    For each ``X.in`` template in *src_dir*, ``X`` must exist, be valid
    UTF-8, and equal the template rendered again now -- so the check proves
    both that templates were expanded and that the outputs are current.

    **Developer notes.** Earlier versions searched every ``.pyx``/``.pxd``
    for ``{{`` or ``}}`` and rejected generated sources that legitimately
    contain doubled braces (CSS and JavaScript in docstrings, ``f"{{"``
    escapes). Comparing with a fresh rendering has no such false positive and
    also catches a stale or hand-edited output. Files with no template are
    not the generator's and are not checked.
    """
    print()
    print("Validating generated files...")

    templates = find_template_files(src_dir)
    if not templates:
        print(f"  WARNING: No .in templates found in {src_dir}", file=sys.stderr)
        return False

    all_valid = True
    for template_path in templates:
        output_path = template_path.parent / template_path.stem
        problem = None
        try:
            expected = _render(template_path.read_text(encoding="utf-8"), template_path)
            actual = output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            problem = "not generated (file missing)"
        except UnicodeDecodeError as exc:
            problem = f"encoding error – {exc}"
        except (OSError, RuntimeError) as exc:
            problem = f"{exc}"
        else:
            if actual != expected:
                problem = "differs from its template's output (stale or edited)"
        if problem is None:
            if verbose:
                print(f"  ✓  {output_path.name}")
        else:
            print(f"  ✗  {output_path.name}: {problem}", file=sys.stderr)
            all_valid = False

    return all_valid


def _auto_detect_src_dir() -> Optional[Path]:
    """
    Search common locations for a directory containing ``.in`` files.

    Returns
    -------
    Path or None
        First matching directory, or ``None`` if none found.
    """
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "scikitplot" / "annoy" / "_annoy",
        cwd / "annoy" / "_annoy",
        cwd / "src" / "annoy",
        cwd / "annoy",
        cwd.parent / "src" / "annoy",
    ]
    for candidate in candidates:
        if candidate.is_dir() and list(candidate.glob("*.in")):
            return candidate
    return None


def main() -> None:
    """
    CLI entry point.

    Raises
    ------
    SystemExit
        Exit code 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Generate Cython code from Tempita templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cython_generate.py                           # auto-detect
  python cython_generate.py --src-dir path/to/annoy   # explicit
  python cython_generate.py -v --validate             # verbose + validate
  python cython_generate.py --debug                   # extra diagnostics
  python cython_generate.py --dry-run                 # show, don't generate
""",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing .in template files (auto-detected if omitted)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-file processing details",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information (implies --verbose)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing any files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated files after creation",
    )

    args = parser.parse_args()

    # Debug implies verbose
    if args.debug:
        args.verbose = True

    # Resolve source directory
    if args.src_dir is not None:
        src_dir = args.src_dir
    else:
        # Auto-detect: look for .in files in current dir or parent dirs
        src_dir = _auto_detect_src_dir()
        if src_dir is None:
            print(
                "ERROR: Cannot find a directory containing .in files.\n"
                "Use --src-dir to specify the directory explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.verbose:
            print(f"Auto-detected source directory: {src_dir}")

    if not src_dir.exists():
        print(
            f"ERROR: Source directory does not exist: {src_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate
    try:
        result = generate_all(
            src_dir,
            verbose=args.verbose,
            dry_run=args.dry_run,
            debug=args.debug,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"FATAL ERROR: {exc}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Optional validation pass
    if args.validate and not args.dry_run:
        if validate_generated_files(src_dir, verbose=args.verbose):
            print("\n  ✓  All generated files passed validation.")
        else:
            print("\n  ✗  Validation failed!", file=sys.stderr)
            sys.exit(1)

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
