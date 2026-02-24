#!/usr/bin/env python3
# scikitplot/_build_utils/tempita.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Process a Tempita-templated file and write the result to the output file.

Usage (standalone CLI)
----------------------
    python tempita.py <infile.in> -o <outfile>

Usage (meson custom_target)
---------------------------
    custom_target('name',
        output: '@BASENAME@',          # e.g. annoylib.pxd from annoylib.pxd.in
        input:  'annoylib.pxd.in',
        command: [tempita_cli, '@INPUT@', '-o', '@OUTPUT@'],
    )

Usage (meson generator)
-----------------------
    gen_tempita = generator(cli_tempita,
        arguments: ['@INPUT@', '-o', '@OUTPUT@'],
        output:    '@BASENAME@',
    )
    gen_tempita.process('annoylib.pxd.in')

Template Engine
---------------
Tries ``Cython.Tempita`` first (no extra dependency, ships with Cython),
then falls back to the standalone ``tempita`` package.

Parameters
----------
infile : str (positional)
    Path to the template file.  Must end in ``.in``.
-o / --outfile : str (required)
    Path to write the generated output.  May be absolute or relative to
    the working directory.
-i / --ignore : str (optional)
    Ignored.  Provided so meson can declare a dependency between custom
    targets without affecting output.

Raises
------
SystemExit
    Exit code 1 on any error (file not found, template error, IO error).
"""

import os
import sys

# ---------------------------------------------------------------------------
# Import Tempita – Cython ships its own copy; fall back to standalone package
# ---------------------------------------------------------------------------
# XXX: If this import ever fails (does it really?), vendor either
# cython.tempita or numpy/npy_tempita.
try:
    from Cython import Tempita as _tempita
except ImportError:  # pragma: no cover
    try:
        import tempita as _tempita  # type: ignore[no-redef]
    except ImportError:
        print(
            "ERROR: Tempita not available.  Install Cython (preferred) or "
            "the standalone 'tempita' package:\n"
            "    pip install cython\n"
            "    # or\n"
            "    pip install tempita",
            file=sys.stderr,
        )
        sys.exit(1)


def process_tempita(fromfile: str, outfile: str) -> None:
    """
    Process a Tempita template and write the result.

    Parameters
    ----------
    fromfile : str
        Path to the input template file (must end in ``.in``).
    outfile : str
        Absolute or relative path to write the generated output.

    Raises
    ------
    FileNotFoundError
        If ``fromfile`` does not exist.
    ValueError
        If ``fromfile`` does not end in ``.in``.
    PermissionError
        If the output file cannot be written.
    RuntimeError
        If template substitution fails.

    Notes
    -----
    The output directory is created automatically if it does not exist.

    Examples
    --------
    >>> process_tempita("annoylib.pxd.in", "annoylib.pxd")
    """
    if not fromfile.endswith(".in"):
        raise ValueError(f"Input file must end in '.in', got: {fromfile!r}")

    if not os.path.isfile(fromfile):
        raise FileNotFoundError(f"Template file not found: {fromfile!r}")

    # Read template
    try:
        with open(fromfile, encoding="utf-8") as fh:
            template_content = fh.read()
    except OSError as exc:
        raise PermissionError(f"Cannot read template {fromfile!r}: {exc}") from exc

    # Substitute
    try:
        template = _tempita.Template(template_content, name=fromfile)
        generated = template.substitute()
    except Exception as exc:
        raise RuntimeError(
            f"Template processing failed for {fromfile!r}: {exc}"
        ) from exc

    if not generated.strip():
        raise RuntimeError(f"Template produced empty output for {fromfile!r}")

    # Ensure output directory exists
    out_dir = os.path.dirname(outfile)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write output
    try:
        with open(outfile, "w", encoding="utf-8") as fh:
            fh.write(generated)
    except OSError as exc:
        raise PermissionError(f"Cannot write output {outfile!r}: {exc}") from exc


def main() -> None:
    """
    CLI entry point.

    Raises
    ------
    SystemExit
        Exit code 0 on success, 1 on failure.

    Notes
    -----
    The ``--outfile`` argument accepts either:

    * A direct output file path (e.g. ``annoylib.pxd`` or
      ``/build/annoylib.pxd``) – the generated content is written there.

    This matches both the meson ``@OUTPUT@`` convention (direct file path)
    and standalone CLI usage.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a Tempita-templated file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standalone
  python tempita.py annoylib.pxd.in -o annoylib.pxd

  # Meson custom_target (called internally – not by users)
  python tempita.py /src/annoylib.pxd.in -o /build/annoylib.pxd
""",
    )
    parser.add_argument(
        "infile",
        help="Path to the input template file (must end in .in)",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        required=True,
        help="Path to write the generated output file",
    )
    parser.add_argument(
        "-i",
        "--ignore",
        default=None,
        help="Ignored argument (used to declare meson dependencies only)",
    )

    args = parser.parse_args()

    if not args.infile.endswith(".in"):
        print(
            f"ERROR: Input file must end in '.in', got: {args.infile!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve output path:
    # - Absolute paths are used as-is (meson passes absolute @OUTPUT@).
    # - Relative paths are resolved against the current working directory.
    outfile: str
    if os.path.isabs(args.outfile):
        outfile = args.outfile
    else:
        outfile = os.path.join(os.getcwd(), args.outfile)

    try:
        process_tempita(args.infile, outfile)
    except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        import traceback

        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
