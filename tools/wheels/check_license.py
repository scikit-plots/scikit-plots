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


def check_text(text):
    # Define the expected text fragments you want to check
    ok = "Copyright (c)" in text and re.search(
        r"This binary distribution of \w+ also bundles the following software",
        text, re.IGNORECASE
    )
    return ok


def main():
    p = argparse.ArgumentParser(usage=__doc__.rstrip())
    p.add_argument("module", nargs="?", default="scikitplot")
    args = p.parse_args()

    # Drop '' from sys.path
    sys.path.pop(0)

    # Find module path
    __import__(args.module)
    mod = sys.modules[args.module]

    # Locate the LICENSE.txt file
    # Try to find the .dist-info directory associated with the package, so find it there
    sitepkgs = pathlib.Path(mod.__file__).parent.parent  # This should give you the site-packages path
    distinfo_path = list(sitepkgs.glob(f"{args.module.replace('-', '_')}-*.dist-info"))[0]  # Handling package name format
    license_txt = distinfo_path / "LICENSE.txt"
    # license_txt = os.path.join(os.path.dirname(mod.__file__), "LICENSE.txt")

    # Check if LICENSE.txt exists
    if not license_txt.exists():
        print(f"ERROR: LICENSE.txt not found at {license_txt}")
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