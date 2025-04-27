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

# pylint: disable=broad-exception-caught

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


# def replace_quantity(match):
#     """
#     Replace unquoted Quantity with a quoted version "Quantity" in type hints.

#     This avoids NameError issues when type hints reference types not in scope.

#     Parameters
#     ----------
#     match : re.Match
#         A regex match object containing the type hint.

#     Returns
#     -------
#     str
#         The type hint with Quantity replaced by "Quantity".
#     """
#     return re.sub(r'\b(?<!["\'])Quantity(?!["\'])\b', "'Quantity'", match.group(0))

# def replace_quantity(match, raw_types_pattern):
#     """
#     Replace unquoted type hints matching specific types with quoted versions.

#     This function wraps specific type hints (such as "Quantity" or "Optional[Quantity]")
#     with double quotes in function signatures. It avoids quoting already quoted types
#     and targets both parameter and return type hints.

#     Useful for making type hints valid at runtime (e.g., for forward references)
#     to avoid NameError when types are not yet imported.

#     Parameters
#     ----------
#     match : re.Match
#         A regex match object containing the portion of function signature
#         (parameter or return type) that potentially includes types to be quoted.

#     Returns
#     -------
#     str
#         The modified string with matched types wrapped in double quotes.
#         If an error occurs, returns the original matched string unchanged.

#     Notes
#     -----
#     - Only types listed in `raw_types` (and built into `raw_types_pattern`) are processed.
#     - Already-quoted types are skipped (i.e., no double quoting).
#     - The matching respects nested generics like `Optional[Quantity]` or `List[Quantity]`.
#     - Errors during processing are caught and ignored safely.

#     Examples
#     --------
#     Given `raw_types = ['Quantity', 'Optional[Quantity]']`:

#     Before:
#         def foo(x: Quantity, y: Optional[Quantity]) -> Quantity:
#             pass

#     After:
#         def foo(x: "Quantity", y: "Optional[Quantity]") -> "Quantity":
#             pass

#     """
#     text = match.group(0)

#     try:
#         def replacer(m):
#             type_text = m.group(1)
#             return f'"{type_text}"'  # wrap matched type in quotes

#         ## Do the replacement
#         return re.sub(
#             rf'(?<!["\'])({raw_types_pattern})(?!["\'])',
#             replacer,
#             text
#         )
#     except Exception:
#         # Optional: log the error if you want
#         # print(f"Error processing type hint: {e}")

#         # Fail safely: return original text unchanged
#         return text


def replace_type_to_str(match, type_matcher):
    """
    Wraps the type portion of the type hint in double quotes
    if it contains any of the specified raw types.

    Parameters
    ----------
    match : re.Match
        Regex match object for the type hint (parameter or return type).

    type_matcher : re.Pattern
        The precompiled regex pattern to match the raw types within the type hint.

    Returns
    -------
    str
        The type hint with the type part wrapped in double quotes if it contains a matching type;
        otherwise, returns the original string unchanged.
    """
    text = match.group(0)

    try:
        # Check for parameter type hints
        if ":" in text:
            # Split at the first colon to avoid quoting the variable name
            param_name, type_part = text.split(":", 1)
            # Apply the replacement for the type part
            if type_matcher.search(type_part):
                type_part = f' "{type_part.strip()}"'
            return (param_name + ":" + type_part).strip()  # Rebuild the string

        # Check for return type hints (matching "->")
        elif "->" in text:
            # Split at "->" to avoid quoting the "->" part
            return_arrow, type_part = text.split("->", 1)
            # Apply the replacement for the type part
            if type_matcher.search(type_part):
                type_part = f' "{type_part.strip()}"'
            return (return_arrow + "->" + type_part).strip()  # Rebuild the string

        # If no colon or arrow is present, just return the text unchanged
        return text
    except Exception:
        # Optional: log warning
        # print(f"Warning: failed to process type hint: {e}")
        return text


def convert_to_relative_imports(
    directory,
    library_name,
    fix_type_hints=False,
    raw_types=None,
):
    """
    Recursively convert absolute imports of a library to relative imports in all files.

    Parameters
    ----------
    directory : str
        The root directory of the codebase to scan.
    library_name : str
        The name of the top-level package/module to be replaced (e.g., 'astropy').
    fix_type_hints : bool, optional
        Whether to quote "Quantity" in type hints and return annotations.

    Returns
    -------
    None
        The files are updated in-place.
    """
    ## Precompile regex pattern for matching import
    import_matcher = re.compile(r"^(\s*)import (" + library_name + r")$", re.MULTILINE)
    import_as_matcher = re.compile(
        r"^(\s*)import (" + library_name + r")\.(.+) as (\w+)", re.MULTILINE
    )
    from_import_matcher = re.compile(
        r"^(\s*)from (" + library_name + r")\.(.+) import (.+)", re.MULTILINE
    )
    from_module_import_matcher = re.compile(
        r"^(\s*)from (" + library_name + r") import (.+)", re.MULTILINE
    )
    from_import_as_matcher = re.compile(
        r"^(\s*)from (" + library_name + r") import (.+) as (\w+)", re.MULTILINE
    )

    ## Precompile regex pattern for matching cimport
    # cimport_matcher = re.compile(r'^(\s*)cimport (' + library_name + r')\.(.+)', re.MULTILINE)

    ## Precompile regex pattern for matching types to quote (outside the function)
    ## Expand this list as needed (e.g., Optional[Quantity], List[Quantity], etc.)
    raw_types = raw_types or ["Quantity"]
    raw_types_pattern = f"({ '|'.join(re.escape(t) for t in raw_types) })"
    type_matcher = re.compile(rf'(?<!["\'])({raw_types_pattern})(?!["\'])')
    type_hint_matcher = re.compile(
        ## Match function parameter type hints containing "Quantity", avoiding already quoted cases
        ## Only Quantity to "Quantity"
        # rf"(\b\w+\s*:\s*[^=,]*\b(?<![\"\'])Quantity(?![\"\'])\b[^=,]*)",
        # rf"(\b\w+\s*:\s*[^=,]*\b(?<![\"\']){raw_types_pattern}(?![\"\'])\b[^=,]*)",
        ## Regex to capture the type part after ":" in parameter hints (ignore the variable name)
        r"(:\s*(?:[A-Za-z0-9_<>[\],| ]+\b)+)"
    )
    return_type_matcher = re.compile(
        ## Match function return types containing "Quantity", avoiding already quoted cases
        ## Only Quantity to "Quantity"
        # r"(->\s*[^:]*\b(?<![\"\'])Quantity(?![\"\'])\b.*)",
        # rf"(->\s*[^:]*\b(?<![\"\']){raw_types_pattern}(?![\"\'])\b[^:]*)"
        ## Regex to capture the type part after "->" in return types (ignore the "->")
        r"(->\s*[A-Za-z0-9_<>[\],| ]+)"
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
                content = import_matcher.sub(
                    r"\1from " + relative_import_prefix + " import __init__", content
                )
                # Convert `import astropy.units as u` to `from . import units as u`
                content = import_as_matcher.sub(
                    r"\1from " + relative_import_prefix + r" import \3 as \4", content
                )
                # Convert `from astropy.utils.exceptions import AstropyUserWarning` to `from ..utils.exceptions import AstropyUserWarning`
                content = from_import_matcher.sub(
                    r"\1from " + relative_import_prefix + r"\3 import \4", content
                )
                # Convert `from astropy import utils` to `from . import utils`
                content = from_module_import_matcher.sub(
                    r"\1from " + relative_import_prefix + r" import \3", content
                )
                # Convert `from astropy import units as u` to `from . import units as u`
                content = from_import_as_matcher.sub(
                    r"\1from " + relative_import_prefix + r" import \3 as \4", content
                )

                # Convert `cimport astropy.something` to `from . cimport something`
                # content = cimport_matcher.sub(r'\1from ' + relative_import_prefix + r' cimport \3', content)

                # Replace Quantity in function signatures and return types if enabled
                if fix_type_hints:
                    # Example usage: using type_matcher in type hint pattern substitution
                    content = type_hint_matcher.sub(
                        lambda m: replace_type_to_str(m, type_matcher),
                        content,
                    )
                    content = return_type_matcher.sub(
                        lambda m: replace_type_to_str(m, type_matcher),
                        content,
                    )
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
