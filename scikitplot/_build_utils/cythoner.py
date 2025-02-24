#!/usr/bin/env python
"""
Scipy variant of Cython command

Cython, as applied to single pyx file.

Expects two arguments, infile and outfile.

Other options passed through to cython command line parser.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys


def main():
    f_in, f_out = (os.path.abspath(p) for p in sys.argv[1:3])
    subprocess.run(
        [
            "cython",
            "-3",
            "--fast-fail",
            "--output-file",
            f_out,
            "--include-dir",
            os.getcwd(),
        ]
        + sys.argv[3:]
        + [f_in],
        check=True,
    )


if __name__ == "__main__":
    main()
