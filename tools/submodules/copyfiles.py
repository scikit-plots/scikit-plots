#!/usr/bin/env python
"""
Platform-independent file and folder copier script.

Consistency with Unix tools like `cp`.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import shutil
import sys


def copy_item(src, dest, recursive=False):
    """Copy a file or directory to the destination."""
    if not os.path.exists(src):
        print(f"Error: Source '{src}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(src):
        if not recursive:
            print(
                f"Error: '{src}', '{dest}' are a directory. Use '-r' flag to copy recursively.",
                file=sys.stderr,
            )
            sys.exit(1)
        # if os.path.exists(dest):
        #     shutil.rmtree(dest)  # Remove existing directory to avoid copy errors
        # Ensure destination directory exists before copying
        os.makedirs(dest, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        # Ensure parent directory of dest exists
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)

    # ## Create a dummy file for Meson tracking
    # meson_done_file = os.path.join(dest, "meson_copy_done.txt")
    # ## Ensure the parent directory exists
    # os.makedirs(os.path.dirname(meson_done_file), exist_ok=True)
    # with open(meson_done_file, "w") as f:
    #     f.write(f"Copy completed: '{src}' to '{dest}'")


def main():
    parser = argparse.ArgumentParser(
        description="Copy files or directories to an output directory."
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Enable recursive copying for directories",
    )
    parser.add_argument("src", help="Path to the source file or directory")
    parser.add_argument("dest", help="Path to the destination directory")
    args = parser.parse_args()

    copy_item(args.src, args.dest, recursive=args.recursive)


if __name__ == "__main__":
    main()
