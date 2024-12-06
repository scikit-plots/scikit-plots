#!/usr/bin/env python
"""
check_license.py [MODULE]

Check the presence of a LICENSE.txt in the installed module directory,
and that it appears to contain text prevalent for a scikitplot binary
distribution.
"""
import os
import sys
import re
import argparse
import pathlib
# from glob import glob

def check_text(text):
    # Define the expected text fragments you want to check
    ok = "Copyright (c)" in text and re.search(
        r"This binary distribution of \w+ also bundles the following software",
        text, re.IGNORECASE
    )
    return ok


def main():    
    p = argparse.ArgumentParser(usage=__doc__.rstrip())
    p.add_argument("mod_name", nargs="?", default='scikitplot')  # import name format
    p.add_argument("package_name", nargs="?", default='scikit-plots')  # Package name format
    p.add_argument("license_name", nargs="?", default='LICENSE')  # LICENSE file name format
    args = p.parse_args()

    # Drop '' from sys.path
    sys.path.pop(0)

    try:
        # Try to import the module dynamically
        mod = __import__(args.mod_name)
        # Access the imported module via sys.modules
        # mod = sys.modules[args.mod_name]
        print(f"Module {args.mod_name} imported successfully.")
    except ImportError as e:
        # Catch ImportError and raise a more specific error with context
        raise RuntimeError(f"Failed to import the module '{args.mod_name}'. Please check if the module is installed correctly.") from e
    except Exception as e:
        # Catch any other unexpected exceptions and raise them
        raise RuntimeError(f"An unexpected error occurred while importing '{args.mod_name}'.") from e

    # Locate the LICENSE.txt file
    # Try to find the .dist-info directory associated with the package, so find it there by Package name
    sitepkgs = pathlib.Path(mod.__file__).parent.parent  # This should give you the site-packages path
    print(f"Looking for .dist-info directory in: {sitepkgs}")

    distinfo_paths = list(sitepkgs.glob(f"{args.package_name.replace('-', '_')}-*.dist-info"))  # Package name format

    if not distinfo_paths:
        print(f"ERROR: No .dist-info directory found for module '{args.mod_name}' in {sitepkgs}")
        sys.exit(1)

    distinfo_path = distinfo_paths[0]
    # Use glob pattern to find LICENSE files including subdirectories
    license_files = list(sitepkgs.glob(f"{args.package_name.replace('-', '_')}-*.dist-info/{args.license_name}*"))
    print(license_files)
    license_txt = distinfo_path / args.license_name
    # license_txt = os.path.join(os.path.dirname(mod.__file__), args.license_name)

    # Check if LICENSE.txt exists
    if not license_txt.exists():
        print(f"ERROR: {args.license_name} not found at {license_txt}")
        sys.exit(1)

    # Read and check the content of LICENSE.txt
    with open(license_txt, encoding="utf-8") as f:
        text = f.read()

    # Check if the license text contains the expected fragments
    ok = check_text(text)
    if not ok:
        print(
            "ERROR: License text {} does not contain expected "
            "text fragments\n".format(license_txt)
        )
        print(text)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()