#!/usr/bin/env python3

# Authors: Spotify AB
# SPDX-License-Identifier: Apache-2.0

"""
Compile and run the C++ Annoy with examples
===========================================

Compile and run the C++ Annoy precision example.

Usage::

    >>> python plot_s_compile_cpp.py                 # default run
    >>> python plot_s_compile_cpp.py 40 100000 0     # f, n, seed
    >>> python plot_s_compile_cpp.py --compile-only  # only compile

Designed so Sphinx can execute it directly using::

    >>> python plot_s_compile_cpp.py 10 1000 123
"""

# %%

import contextlib
import subprocess
import sys
import shutil
from pathlib import Path

# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
# with contextlib.suppress(NameError, TypeError, ValueError):
#     ROOT = Path(__file__).resolve().parent
# else:
#     ROOT = Path.cwd()
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    # Sphinx-Gallery fallback
    ROOT = Path.cwd()

CPP_FILE = ROOT / "precision_test.cpp"
BIN_FILE = ROOT / "precision_test"


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

def run_cmd(cmd, **kwargs):
    """Run a system command and stream output."""
    print(f"[CMD] {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, **kwargs)
    except subprocess.CalledProcessError as e:
        print("\n❌ Command failed:", " ".join(cmd))
        sys.exit(e.returncode)


def compile_cpp():
    """Compile the C++ precision binary."""
    if not CPP_FILE.exists():
        print(f"❌ C++ file not found: {CPP_FILE}")
        sys.exit(1)

    # Detect compiler
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        print("❌ No C++ compiler (g++/clang++) found in PATH.")
        sys.exit(1)

    print("🔨 Compiling precision_test.cpp ...")
    cmd = [
        cxx,
        str(CPP_FILE),
        "-DANNOYLIB_MULTITHREADED_BUILD",
        # "-std=c++17",
        "-std=c++14",  # non-copyable (because of std::atomic)
        "-pthread",
        "-o",
        str(BIN_FILE)
    ]
    run_cmd(cmd)
    print("✅ Compile done.\n")


def run_binary(args):
    """Run compiled binary with optional arguments."""
    if not BIN_FILE.exists():
        print("⚠ No binary found; compiling first...\n")
        compile_cpp()

    cmd = [str(BIN_FILE)] + args
    print("🚀 Running binary...\n")
    run_cmd(cmd)


# --------------------------------------------------------------
# Main logic
# --------------------------------------------------------------

def main():
    args = sys.argv[1:]

    # Handle flags
    if "--compile-only" in args:
        compile_cpp()
        return

    # Compile first
    compile_cpp()

    # If no arguments → default run
    if len(args) == 0:
        run_binary([])
    else:
        run_binary(args)


if __name__ == "__main__":
    # with contextlib.suppress(Exception, NameError, TypeError, ValueError):
    #     main()
    pass

# %%
