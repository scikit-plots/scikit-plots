#!/usr/bin/env python

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Cython Template Code Generator for Annoy.

This script processes .in template files using Tempita to generate
the actual .pyx and .pxd files with all type combinations expanded.

Usage:
    python cython_generate.py

Requirements:
    pip install tempita

Based on NumPy's code generation approach.
"""

import sys
import os
from pathlib import Path
import argparse
from typing import List, Tuple

try:
    from tempita import Template
except ImportError:
    print("ERROR: tempita module not found.")
    print("Install it with: pip install tempita")
    sys.exit(1)


def process_template(
    template_path: Path,
    output_path: Path,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Process a single template file.

    Parameters
    ----------
    template_path : Path
        Path to .in template file.
    output_path : Path
        Path to write generated output.
    verbose : bool
        Print detailed information.

    Returns
    -------
    tuple[int, int]
        (template_lines, generated_lines)
    """
    if verbose:
        print(f"Processing: {template_path}")
        print(f"  Output: {output_path}")

    # Read template
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
    except Exception as e:
        print(f"ERROR: Failed to read template {template_path}: {e}")
        sys.exit(1)

    template_lines = len(template_content.splitlines())

    # Process template
    try:
        template = Template(template_content, name=str(template_path))
        generated_content = template.substitute()
    except Exception as e:
        print(f"ERROR: Failed to process template {template_path}: {e}")
        print(f"\nTemplate error details:")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    generated_lines = len(generated_content.splitlines())

    # Write output
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_content)
    except Exception as e:
        print(f"ERROR: Failed to write output {output_path}: {e}")
        sys.exit(1)

    if verbose:
        print(f"  Template lines: {template_lines:,}")
        print(f"  Generated lines: {generated_lines:,}")
        print(f"  Expansion ratio: {generated_lines / template_lines:.1f}x")

    return template_lines, generated_lines


def find_template_files(directory: Path) -> List[Path]:
    """
    Find all .in template files in directory.

    Parameters
    ----------
    directory : Path
        Directory to search.

    Returns
    -------
    list[Path]
        List of template files.
    """
    template_files = list(directory.glob("*.in"))
    template_files.sort()
    return template_files


def generate_all(
    src_dir: Path,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Generate all Cython files from templates.

    Parameters
    ----------
    src_dir : Path
        Source directory containing .in files.
    verbose : bool
        Print detailed information.
    dry_run : bool
        Don't write files, just report what would be done.
    """
    print("=" * 70)
    print("Annoy Cython Code Generator")
    print("=" * 70)
    print()

    # Find all template files
    template_files = find_template_files(src_dir)

    if not template_files:
        print(f"WARNING: No .in template files found in {src_dir}")
        return

    print(f"Found {len(template_files)} template file(s):")
    for tf in template_files:
        print(f"  - {tf.name}")
    print()

    # Process each template
    total_template_lines = 0
    total_generated_lines = 0

    for template_path in template_files:
        # Determine output path (.in → .pyx or .pxd)
        output_name = template_path.stem  # Remove .in extension
        output_path = template_path.parent / output_name

        if dry_run:
            print(f"[DRY RUN] Would generate: {output_path}")
            continue

        # Process template
        t_lines, g_lines = process_template(
            template_path,
            output_path,
            verbose=verbose,
        )

        total_template_lines += t_lines
        total_generated_lines += g_lines

        print(f"✓ Generated: {output_path.name} ({g_lines:,} lines)")

    # Summary
    print()
    print("=" * 70)
    print("Generation Complete")
    print("=" * 70)

    if not dry_run:
        print(f"Template lines: {total_template_lines:,}")
        print(f"Generated lines: {total_generated_lines:,}")
        print(
            f"Code reduction: {(1 - total_template_lines / total_generated_lines) * 100:.1f}%"
        )
        print(f"Expansion ratio: {total_generated_lines / total_template_lines:.1f}x")


def validate_generated_files(src_dir: Path) -> bool:
    """
    Validate that generated files are syntactically correct.

    Parameters
    ----------
    src_dir : Path
        Directory containing generated files.

    Returns
    -------
    bool
        True if all files valid.
    """
    print()
    print("Validating generated files...")

    # Check .pxd files
    pxd_files = list(src_dir.glob("*.pxd"))
    pyx_files = list(src_dir.glob("*.pyx"))

    if not pxd_files and not pyx_files:
        print("WARNING: No generated files found to validate")
        return False

    all_valid = True

    for filepath in pxd_files + pyx_files:
        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Basic syntax checks
            if len(content) == 0:
                print(f"✗ ERROR: {filepath.name} is empty")
                all_valid = False
            elif "{{" in content or "}}" in content:
                print(f"✗ ERROR: {filepath.name} contains unprocessed template markers")
                all_valid = False
            else:
                print(f"✓ {filepath.name} looks valid")

        except Exception as e:
            print(f"✗ ERROR: Failed to validate {filepath.name}: {e}")
            all_valid = False

    return all_valid


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Cython code from templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all files
  python cython_generate.py

  # Verbose mode
  python cython_generate.py -v

  # Dry run (show what would be generated)
  python cython_generate.py --dry-run

  # Specify custom source directory
  python cython_generate.py --src-dir /path/to/annoy/src
        """,
    )

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="Source directory containing .in files (default: auto-detect)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually generating files",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated files after creation",
    )

    args = parser.parse_args()

    # Determine source directory
    if args.src_dir:
        src_dir = args.src_dir
    else:
        # Auto-detect: look for .in files in current dir or parent dirs
        current = Path.cwd()
        candidates = [
            current,
            current / "scikitplot" / "annoy" / "_annoy",
            current.parent / "scikitplot" / "annoy" / "_annoy",
        ]

        src_dir = None
        for candidate in candidates:
            if candidate.exists() and list(candidate.glob("*.in")):
                src_dir = candidate
                break

        if src_dir is None:
            print("ERROR: Could not find source directory with .in files")
            print("Please specify --src-dir")
            sys.exit(1)

    if not src_dir.exists():
        print(f"ERROR: Source directory does not exist: {src_dir}")
        sys.exit(1)

    # Generate files
    generate_all(src_dir, verbose=args.verbose, dry_run=args.dry_run)

    # Validate if requested
    if args.validate and not args.dry_run:
        if not validate_generated_files(src_dir):
            print("\n✗ Validation failed!")
            sys.exit(1)
        else:
            print("\n✓ All files validated successfully!")


if __name__ == "__main__":
    main()
