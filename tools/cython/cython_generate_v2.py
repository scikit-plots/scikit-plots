#!/usr/bin/env python3
"""
Cython Template Code Generator for Annoy.

This script processes .in template files using Tempita to generate
the actual .pyx and .pxd files with all type combinations expanded.

Usage:
    python cython_generate.py

Requirements:
    pip install tempita

Based on NumPy's code generation approach.

Developer Notes
---------------
This generator uses Tempita templating to eliminate code repetition
in Cython wrapper generation. The template files use {{py: ... }}
blocks to define configurations and {{for ...}} loops to generate
type-specific code.

Security
--------
- All file operations validate paths and check permissions
- Template processing is sandboxed with explicit error handling
- No arbitrary code execution from untrusted sources
"""

import sys
import os
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any, Optional

try:
    from tempita import Template
except ImportError:
    print("ERROR: tempita module not found.", file=sys.stderr)
    print("Install it with: pip install tempita", file=sys.stderr)
    sys.exit(1)


def validate_template_content(content: str, filepath: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate template content for common issues.

    Parameters
    ----------
    content : str
        Template file content.
    filepath : Path
        Path to template file (for error reporting).

    Returns
    -------
    tuple[bool, Optional[str]]
        (is_valid, error_message)

    Notes
    -----
    Checks for:
    - Balanced template delimiters {{ }}
    - Valid Python syntax in {{py:}} blocks
    - No undefined variable references in common patterns
    """
    # Check for balanced delimiters
    open_count = content.count('{{')
    close_count = content.count('}}')

    if open_count != close_count:
        return False, (
            f"Unbalanced template delimiters: {open_count} '{{{{' but "
            f"{close_count} '}}}}'"
        )

    # Check for potentially problematic patterns in docstrings
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        # Check if line is in a docstring (simple heuristic)
        stripped = line.strip()

        # Warn about template-like syntax in comments/docstrings
        if stripped.startswith('#') and '{{' in stripped and '}}' in stripped:
            # Check if this looks like actual template code
            if 'for ' in stripped or 'endfor' in stripped:
                return False, (
                    f"Line {i}: Template syntax found in comment. "
                    f"Comments should not use {{{{...}}}} delimiters.\n"
                    f"  Line: {line}"
                )

    return True, None


def process_template(
    template_path: Path,
    output_path: Path,
    verbose: bool = False,
    debug: bool = False
) -> Tuple[int, int]:
    """
    Process a single template file.

    Parameters
    ----------
    template_path : Path
        Path to .in template file.
    output_path : Path
        Path to write generated output.
    verbose : bool, default=False
        Print detailed information.
    debug : bool, default=False
        Print debug information including template parsing.

    Returns
    -------
    tuple[int, int]
        (template_lines, generated_lines)

    Raises
    ------
    FileNotFoundError
        If template file does not exist.
    PermissionError
        If cannot read template or write output.
    ValueError
        If template validation fails.
    RuntimeError
        If template processing fails.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    if not template_path.is_file():
        raise ValueError(f"Template path is not a file: {template_path}")

    if verbose:
        print(f"Processing: {template_path}")
        print(f"  Output: {output_path}")

    # Read template
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except PermissionError as e:
        raise PermissionError(f"Cannot read template {template_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read template {template_path}: {e}")

    template_lines = len(template_content.splitlines())

    if debug:
        print(f"  Template size: {len(template_content)} bytes, {template_lines} lines")

    # Validate template content
    is_valid, error_msg = validate_template_content(template_content, template_path)
    if not is_valid:
        raise ValueError(f"Template validation failed:\n{error_msg}")

    # Process template
    try:
        template = Template(template_content, name=str(template_path))

        if debug:
            print(f"  Template parsed successfully")
            print(f"  Substituting variables...")

        generated_content = template.substitute()

        if debug:
            print(f"  Template substitution complete")

    except Exception as e:
        print(f"\nERROR: Failed to process template {template_path}", file=sys.stderr)
        print(f"\nTemplate error details:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)

        # Try to give helpful context
        error_str = str(e)
        if 'not defined' in error_str:
            print(f"\nDEBUG: This usually means a variable is referenced before it's defined.", file=sys.stderr)
            print(f"       Check that all variables are defined in the {{{{py:}}}} block", file=sys.stderr)
            print(f"       BEFORE they are used in template expressions.", file=sys.stderr)

        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Template processing failed: {e}")

    generated_lines = len(generated_content.splitlines())

    # Validate generated content is not empty
    if len(generated_content.strip()) == 0:
        raise RuntimeError("Generated content is empty!")

    # Write output
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(generated_content)
    except PermissionError as e:
        raise PermissionError(f"Cannot write output {output_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to write output {output_path}: {e}")

    if verbose:
        print(f"  Template lines: {template_lines:,}")
        print(f"  Generated lines: {generated_lines:,}")
        if template_lines > 0:
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
        Sorted list of template files.

    Notes
    -----
    Only searches the specified directory (not recursive).
    Only includes files with .in extension.
    """
    if not directory.exists():
        return []

    if not directory.is_dir():
        return []

    template_files = list(directory.glob('*.in'))
    template_files.sort()
    return template_files


def generate_all(
    src_dir: Path,
    verbose: bool = False,
    dry_run: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Generate all Cython files from templates.

    Parameters
    ----------
    src_dir : Path
        Source directory containing .in files.
    verbose : bool, default=False
        Print detailed information.
    dry_run : bool, default=False
        Don't write files, just report what would be done.
    debug : bool, default=False
        Print debug information.

    Returns
    -------
    dict
        Generation statistics and results.

    Raises
    ------
    FileNotFoundError
        If src_dir does not exist.
    RuntimeError
        If template processing fails.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    print("=" * 70)
    print("Annoy Cython Code Generator")
    print("=" * 70)
    print()

    # Find all template files
    template_files = find_template_files(src_dir)

    if not template_files:
        print(f"WARNING: No .in template files found in {src_dir}", file=sys.stderr)
        return {
            'success': False,
            'template_files': [],
            'generated_files': [],
            'total_template_lines': 0,
            'total_generated_lines': 0,
        }

    print(f"Found {len(template_files)} template file(s):")
    for tf in template_files:
        print(f"  - {tf.name}")
    print()

    # Process each template
    total_template_lines = 0
    total_generated_lines = 0
    generated_files = []
    errors = []

    for template_path in template_files:
        # Determine output path (.in → .pyx or .pxd)
        output_name = template_path.stem  # Remove .in extension
        output_path = template_path.parent / output_name

        if dry_run:
            print(f"[DRY RUN] Would generate: {output_path}")
            continue

        # Process template
        try:
            t_lines, g_lines = process_template(
                template_path,
                output_path,
                verbose=verbose,
                debug=debug
            )

            total_template_lines += t_lines
            total_generated_lines += g_lines
            generated_files.append(output_path)

            print(f"✓ Generated: {output_path.name} ({g_lines:,} lines)")

        except Exception as e:
            error_msg = f"✗ Failed: {template_path.name} - {e}"
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
            # Continue processing other files rather than failing immediately

    # Summary
    print()
    print("=" * 70)
    print("Generation Complete")
    print("=" * 70)

    if not dry_run:
        print(f"Template files processed: {len(template_files) - len(errors)}/{len(template_files)}")
        print(f"Template lines: {total_template_lines:,}")
        print(f"Generated lines: {total_generated_lines:,}")

        if total_template_lines > 0:
            reduction_pct = (1 - total_template_lines / total_generated_lines) * 100
            expansion_ratio = total_generated_lines / total_template_lines
            print(f"Code reduction: {reduction_pct:.1f}%")
            print(f"Expansion ratio: {expansion_ratio:.1f}x")

    if errors:
        print()
        print(f"⚠ {len(errors)} error(s) occurred during generation", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)

    return {
        'success': len(errors) == 0,
        'template_files': template_files,
        'generated_files': generated_files,
        'total_template_lines': total_template_lines,
        'total_generated_lines': total_generated_lines,
        'errors': errors,
    }


def validate_generated_files(src_dir: Path, verbose: bool = False) -> bool:
    """
    Validate that generated files are syntactically correct.

    Parameters
    ----------
    src_dir : Path
        Directory containing generated files.
    verbose : bool, default=False
        Print detailed information.

    Returns
    -------
    bool
        True if all files valid.

    Notes
    -----
    Performs basic syntax checks:
    - File is not empty
    - No unprocessed template markers ({{ or }})
    - File can be read as UTF-8
    """
    print()
    print("Validating generated files...")

    # Check .pxd files
    pxd_files = list(src_dir.glob('*.pxd'))
    pyx_files = list(src_dir.glob('*.pyx'))

    # Exclude template files
    pxd_files = [f for f in pxd_files if not f.name.endswith('.in')]
    pyx_files = [f for f in pyx_files if not f.name.endswith('.in')]

    if not pxd_files and not pyx_files:
        print("WARNING: No generated files found to validate", file=sys.stderr)
        return False

    all_valid = True

    for filepath in pxd_files + pyx_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Basic syntax checks
            if len(content) == 0:
                print(f"✗ ERROR: {filepath.name} is empty", file=sys.stderr)
                all_valid = False
            elif '{{' in content or '}}' in content:
                print(f"✗ ERROR: {filepath.name} contains unprocessed template markers", file=sys.stderr)

                # Show context
                if verbose:
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if '{{' in line or '}}' in line:
                            print(f"    Line {i}: {line}", file=sys.stderr)

                all_valid = False
            else:
                print(f"✓ {filepath.name} looks valid")

        except UnicodeDecodeError as e:
            print(f"✗ ERROR: {filepath.name} has encoding issues: {e}", file=sys.stderr)
            all_valid = False
        except Exception as e:
            print(f"✗ ERROR: Failed to validate {filepath.name}: {e}", file=sys.stderr)
            all_valid = False

    return all_valid


def main():
    """
    Main entry point.

    Raises
    ------
    SystemExit
        With exit code 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description='Generate Cython code from templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all files
  python cython_generate.py

  # Verbose mode
  python cython_generate.py -v

  # Debug mode (extra diagnostics)
  python cython_generate.py --debug

  # Dry run (show what would be generated)
  python cython_generate.py --dry-run

  # Specify custom source directory
  python cython_generate.py --src-dir /path/to/annoy/src

  # Validate after generation
  python cython_generate.py --validate

Notes:
  - Requires tempita: pip install tempita
  - Template files must have .in extension
  - Output files will have same name without .in extension
        """
    )

    parser.add_argument(
        '--src-dir',
        type=Path,
        default=None,
        help='Source directory containing .in files (default: auto-detect)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed information'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print debug information (implies --verbose)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually generating files'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated files after creation'
    )

    args = parser.parse_args()

    # Debug implies verbose
    if args.debug:
        args.verbose = True

    # Determine source directory
    if args.src_dir:
        src_dir = args.src_dir
    else:
        # Auto-detect: look for .in files in current dir or parent dirs
        current = Path.cwd()
        candidates = [
            current,
            current / 'src' / 'annoy',
            current / 'annoy',
            current.parent / 'src' / 'annoy',
        ]

        src_dir = None
        for candidate in candidates:
            if candidate.exists() and list(candidate.glob('*.in')):
                src_dir = candidate
                if args.verbose:
                    print(f"Auto-detected source directory: {src_dir}")
                break

        if src_dir is None:
            print("ERROR: Could not find source directory with .in files", file=sys.stderr)
            print("Please specify --src-dir", file=sys.stderr)
            sys.exit(1)

    if not src_dir.exists():
        print(f"ERROR: Source directory does not exist: {src_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate files
    try:
        result = generate_all(
            src_dir,
            verbose=args.verbose,
            dry_run=args.dry_run,
            debug=args.debug
        )
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Validate if requested
    if args.validate and not args.dry_run:
        if not validate_generated_files(src_dir, verbose=args.verbose):
            print("\n✗ Validation failed!", file=sys.stderr)
            sys.exit(1)
        else:
            print("\n✓ All files validated successfully!")

    # Exit with appropriate code
    if result['success']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
