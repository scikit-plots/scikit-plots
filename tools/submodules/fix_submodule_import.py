#!/usr/bin/env python3
import argparse
import os
import re


def get_relative_import_level(file_path, root):
    """Calculate relative import level based on the package root."""
    root = os.path.abspath(root)
    depth = os.path.relpath(file_path, root).count(os.sep)
    return "." * (depth + 1) if depth > 0 else "."


def replace_quantity(match):
    """Replace Quantity occurrences while preserving syntax."""
    return re.sub(r'\b(?<!["\'])Quantity(?!["\'])\b', "'Quantity'", match.group(0))


def convert_to_relative_imports(directory, library_name, fix_type_hints=False):
    import_pattern = re.compile(r"^(\s*)import (" + library_name + r")$", re.MULTILINE)
    from_import_pattern = re.compile(
        r"^(\s*)from (" + library_name + r")\.(.+) import (.+)", re.MULTILINE
    )
    import_as_pattern = re.compile(
        r"^(\s*)import (" + library_name + r")\.(.+) as (\w+)", re.MULTILINE
    )
    from_import_as_pattern = re.compile(
        r"^(\s*)from (" + library_name + r") import (.+) as (\w+)", re.MULTILINE
    )
    from_module_import_pattern = re.compile(
        r"^(\s*)from (" + library_name + r") import (.+)", re.MULTILINE
    )

    # cimport_pattern = re.compile(r'^(\s*)cimport (' + library_name + r')\.(.+)', re.MULTILINE)

    # Match function parameter type hints containing "Quantity", avoiding already quoted cases
    type_hint_pattern = re.compile(
        r'(\b\w+\s*:\s*[^=,]*\b(?<!["\'])Quantity(?!["\'])\b[^=,]*)'
    )

    # Match function return types containing "Quantity", avoiding already quoted cases
    return_type_pattern = re.compile(r'(->\s*[^:]*\b(?<!["\'])Quantity(?!["\'])\b.*)')

    for root, _, files in os.walk(os.path.abspath(directory)):
        for file in files:
            if file.endswith((".py", ".pyx", ".pxd", ".pxi", ".pyi")):
                file_path = os.path.join(root, file)
                relative_import_prefix = get_relative_import_level(file_path, directory)

                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Convert absolute imports to relative imports
                # Convert `import astropy` to `from . import __init__` with correct relative depth
                content = import_pattern.sub(
                    r"\1from " + relative_import_prefix + " import __init__", content
                )
                # Convert `from astropy.utils.exceptions import AstropyUserWarning` to `from ..utils.exceptions import AstropyUserWarning`
                content = from_import_pattern.sub(
                    r"\1from " + relative_import_prefix + r"\3 import \4", content
                )
                # Convert `import astropy.units as u` to `from . import units as u`
                content = import_as_pattern.sub(
                    r"\1from " + relative_import_prefix + r" import \3 as \4", content
                )
                # Convert `from astropy import units as u` to `from . import units as u`
                content = from_import_as_pattern.sub(
                    r"\1from " + relative_import_prefix + r" import \3 as \4", content
                )
                # Convert `from astropy import utils` to `from . import utils`
                content = from_module_import_pattern.sub(
                    r"\1from " + relative_import_prefix + r" import \3", content
                )

                # Convert `cimport astropy.something` to `from . cimport something`
                # content = cimport_pattern.sub(r'\1from ' + relative_import_prefix + r' cimport \3', content)

                # Replace Quantity in function signatures and return types if enabled
                if fix_type_hints:
                    content = type_hint_pattern.sub(replace_quantity, content)
                    content = return_type_pattern.sub(replace_quantity, content)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Updated: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module import fix.")
    parser.add_argument("-m", "--module", default="seaborn", help="Module name")
    parser.add_argument("-r", "--root", default=".", help="root")
    parser.add_argument(
        "--fix-type-hints",
        action="store_true",
        help="Apply Quantity replacement in type hints",
    )
    args = parser.parse_args()

    # Define the target module that should be converted to relative imports
    ROOT = args.root
    TARGET_MODULE = args.module

    convert_to_relative_imports(ROOT, TARGET_MODULE, fix_type_hints=args.fix_type_hints)
