"""
Scipy variant of Cython command

Cython, as applied to single pyx file.

Expects two arguments, infile and outfile.

Other options passed through to cython command line parser.
"""
#!/usr/bin/env python
import os
import sys
import subprocess

def main():
  f_in, f_out = ( os.path.abspath(p) for p in sys.argv[1:3] )
  subprocess.run(
    ['cython', '-3', '--fast-fail', '--output-file', f_out, '--include-dir', os.getcwd(),] +
    sys.argv[3:] + [f_in],
    check=True,
  )

if __name__ == '__main__':
    main()