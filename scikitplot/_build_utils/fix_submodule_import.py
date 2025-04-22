#!/usr/bin/env python
"""
Script to convert absolute imports to relative imports within a Python package.

This tool helps refactor code to make intra-package imports relative, aiding portability,
modularity, and avoiding issues during module packaging or testing.

Features:
- Converts `import module` to `from . import module`
- Converts `from module.sub import name` to relative equivalents
- Optionally replaces unquoted type hints like `Quantity` with quoted versions
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import re


def get_relative_import_level(file_path, root):
    """
    Calculate the relative import level from a file path to the package root.

    Parameters
    ----------
    file_path : str
        The path of the source file using the import.
    root : str
        The root directory of the package.

    Returns
    -------
    str
        A string of dots representing the relative import level (e.g., '..', '.', etc.).
    """
    root = os.path.abspath(root)
    depth = os.path.relpath(file_path, root).count(os.sep)
    return "." * (depth + 1) if depth > 0 else "."


def replace_quantity(match):
    """
    Replace unquoted 'Quantity' with a quoted version in type hints.

    This avoids NameError issues when type hints reference types not in scope.

    Parameters
    ----------
    match : re.Match
        A regex match object containing the type hint.

    Returns
    -------
    str
        The type hint with 'Quantity' replaced by "'Quantity'".
    """
    return re.sub(r'\b(?<!["\'])Quantity(?!["\'])\b', "'Quantity'", match.group(0))


def convert_to_relative_imports(directory, library_name, fix_type_hints=False):
    """
    Recursively convert absolute imports of a library to relative imports in all files.

    Parameters
    ----------
    directory : str
        The root directory of the codebase to scan.
    library_name : str
        The name of the top-level package/module to be replaced (e.g., 'astropy').
    fix_type_hints : bool, optional
        Whether to quote 'Quantity' in type hints and return annotations.

    Returns
    -------
    None
        The files are updated in-place.
    """
    import_pattern = re.compile(r"^(\s*)import (" + library_name + r")$", re.MULTILINE)
    import_as_pattern = re.compile(
        r"^(\s*)import (" + library_name + r")\.(.+) as (\w+)", re.MULTILINE
    )
    from_import_pattern = re.compile(
        r"^(\s*)from (" + library_name + r")\.(.+) import (.+)", re.MULTILINE
    )
    from_module_import_pattern = re.compile(
        r"^(\s*)from (" + library_name + r") import (.+)", re.MULTILINE
    )
    from_import_as_pattern = re.compile(
        r"^(\s*)from (" + library_name + r") import (.+) as (\w+)", re.MULTILINE
    )

    # cimport_pattern = re.compile(r'^(\s*)cimport (' + library_name + r')\.(.+)', re.MULTILINE)

    # Match function parameter type hints containing "Quantity", avoiding already quoted cases
    type_hint_pattern = re.compile(
        r'(\b\w+\s*:\s*[^=,]*\b(?<!["\'])Quantity(?!["\'])\b[^=,]*)',
    )

    # Match function return types containing "Quantity", avoiding already quoted cases
    return_type_pattern = re.compile(
        r'(->\s*[^:]*\b(?<!["\'])Quantity(?!["\'])\b.*)',
    )

    for root, _, files in os.walk(os.path.abspath(directory)):
        for file in files:
            if file.endswith((".py", ".pyx", ".pxd", ".pxi", ".pyi")):
                file_path = os.path.join(root, file)
                relative_import_prefix = get_relative_import_level(file_path, directory)

                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Replace various import styles with relative versions
                # Convert absolute imports to relative imports
                # Convert `import astropy` to `from . import __init__` with correct relative depth
                content = import_pattern.sub(
                    r"\1from " + relative_import_prefix + " import __init__", content
                )
                # Convert `import astropy.units as u` to `from . import units as u`
                content = import_as_pattern.sub(
                    r"\1from " + relative_import_prefix + r" import \3 as \4", content
                )
                # Convert `from astropy.utils.exceptions import AstropyUserWarning` to `from ..utils.exceptions import AstropyUserWarning`
                content = from_import_pattern.sub(
                    r"\1from " + relative_import_prefix + r"\3 import \4", content
                )
                # Convert `from astropy import utils` to `from . import utils`
                content = from_module_import_pattern.sub(
                    r"\1from " + relative_import_prefix + r" import \3", content
                )
                # Convert `from astropy import units as u` to `from . import units as u`
                content = from_import_as_pattern.sub(
                    r"\1from " + relative_import_prefix + r" import \3 as \4", content
                )

                # Convert `cimport astropy.something` to `from . cimport something`
                # content = cimport_pattern.sub(r'\1from ' + relative_import_prefix + r' cimport \3', content)

                # Replace Quantity in function signatures and return types if enabled
                if fix_type_hints:
                    content = type_hint_pattern.sub(replace_quantity, content)
                    content = return_type_pattern.sub(replace_quantity, content)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"ABS import to REL: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix submodule absolute imports and type hints."
    )
    parser.add_argument(
        "-m",
        "--module",
        default="seaborn",
        help="Top-level module name to convert (e.g., 'mypkg')",
    )
    parser.add_argument(
        "-r", "--root", default=".", help="Root directory of the codebase"
    )
    parser.add_argument(
        "--fix-type-hints",
        action="store_true",
        help="Replace Quantity in type hints and return annotations with quoted 'Quantity'",
    )
    args = parser.parse_args()

    # Define the target module that should be converted to relative imports
    ROOT = args.root
    TARGET_MODULE = args.module

    convert_to_relative_imports(ROOT, TARGET_MODULE, fix_type_hints=args.fix_type_hints)
