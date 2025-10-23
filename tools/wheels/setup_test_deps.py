#!/usr/bin/env python

# ==========================  ================================  =======================  ===============================  ====================================
# Function                    Purpose                           Captures Output          Raises Error on Failure          Typical Use Case
# ==========================  ================================  =======================  ===============================  ====================================
# subprocess.run()            General-purpose                   Optional                 Optional (with ``check=True``)   Flexible modern default
# subprocess.check_call()     Run command and ensure success    No                       Yes                              Simple “run and fail if error”
# subprocess.check_output()   Run and return command output     Yes (stdout only)        Yes                              When output is required
# subprocess.Popen()          Low-level interface               Not by default           No                               Advanced control / streaming
# ==========================  ================================  =======================  ===============================  ====================================

# import os, platform, subprocess, sys
# print("Implementation\t:", platform.python_implementation(), sys.implementation.name)
# print("Version\t\t:", sys.version)
# print("Compiler\t:", platform.python_compiler())

## Python prints -VV, by os
# os.system("python -VV")
# with os.popen("python -VV") as f: output=f.read().strip()

## Python prints -VV, by subprocess
# output = subprocess.Popen(["python", "-VV"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).communicate()[0].strip()
# output = subprocess.run(["python", "-VV"], capture_output=True, text=True).stdout.strip()
# output = subprocess.check_output("python -VV", shell=True, text=True).strip()
# print(output)

import platform
import sys
import subprocess
from pathlib import Path
import platform
import sys

# Check if a project path was given as argument
if len(sys.argv) > 1:
    project_root = Path(sys.argv[1]).resolve()
else:
    # fallback: assume script is in project root
    project_root = Path(__file__).resolve().parent.parent.parent  # or your specific


## manylinux	Precompiled wheels compatible with glibc-based Linux	Ubuntu, Debian, Fedora, CentOS
## musllinux	Precompiled wheels for musl-based Linux	Alpine Linux, musllinux_x86_64 wheels
## WASM / Emscripten / WASI	Pure Python only (no native C extensions)	wasm32-emscripten, wasm32-wasi
# is_wasm = sys.platform in ("emscripten", "wasi") or "emscripten" in sys.executable.lower()
# 🔹 glibc-based distributions (like Ubuntu, Debian, Fedora)
# 🔹 musl-based distributions (like Alpine Linux, musllinux wheels)
def detect_linux_wheel_type():
    machine = platform.machine().lower()  # e.g., 'x86_64'
    system = platform.system().lower()    # e.g., 'linux'

    # Get libc information
    libc, _ = platform.libc_ver()
    is_musl = libc.lower() == "musl"

    # Detect WebAssembly (emscripten or wasi)
    is_wasm = sys.platform.lower() in ("emscripten", "wasi") or "emscripten" in sys.executable.lower()

    # Determine wheel type / platform category
    return {
        "is_wasm": is_wasm,
        "is_musl": is_musl,
        "machine": machine,
        "system": system,
    }


# Example usage
is_musllinux = detect_linux_wheel_type()["is_musl"]

# if sys.version_info >= (3, 13):
# Detect free-threaded Python (nogil)
is_nogil = getattr(sys, "is_free_threaded", False)

if is_nogil:
    # Free-threaded builds (3.13t, 3.14t) should skip 'h5py' completely.
    req_file = project_root / "requirements" / "test_nogil.txt"
    print(f"Detected free-threaded Python: installing from {req_file}")
# elif is_musllinux:
#     # skip 'h5py' completely.
#     req_file = project_root / "requirements" / "test_nogil.txt"
#     print(f"Detected free-threaded Python: installing from {req_file}")
else:
    req_file = project_root / "requirements" / "test.txt"
    print(f"Installing standard dependencies from {req_file}")

try:
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)
except:
    pass
