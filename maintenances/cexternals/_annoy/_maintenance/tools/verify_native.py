#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Compile/run the fixed standalone native regression registry for _annoy."""
from __future__ import annotations

import json
import shlex
import platform
import pathlib
import shutil
import subprocess
import sys
import tempfile


def repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for p in here.parents:
        if (p / "scikitplot").is_dir() and (p / "maintenances").is_dir() and (p / "skills").is_dir():
            return p
    raise RuntimeError("wide repository root not found")


def run(cmd: list[str], *, timeout: int = 60) -> tuple[bool, str]:
    """Use bounded diagnostics; classify timeouts as failures without a traceback."""
    with tempfile.TemporaryFile() as log:
        try:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout,
                                  check=False, shell=False)
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout} seconds"
        except OSError as exc:
            return False, f"could not run probe: {exc}"
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 6000))
        return proc.returncode == 0, log.read().decode("utf-8", errors="replace").strip()


def overall(results):
    """Only actually executed GREEN probes can establish a complete PASS."""
    statuses = [item["status"] for item in results.values()]
    if "RED" in statuses:
        return "FAIL"
    return "PASS" if statuses and all(s == "GREEN" for s in statuses) else "UNAVAILABLE"


def main() -> int:
    compiler = shutil.which("g++")
    if compiler is None:
        print(json.dumps({"status": "UNAVAILABLE", "reason": "g++ not found"}, indent=2))
        return 1
    repo = repo_root()
    src = repo / "scikitplot/cexternals/_annoy/src"
    tests = repo / "scikitplot/cexternals/_annoy/tests"
    registry = [
        ("int_cmp", "test_annoy_int_cmp.cpp", []),
        ("dealloc_noexcept", "test_dealloc_noexcept.cpp", []),
        ("float16_fallback", "test_float16_portable_fallback.cpp", []),
        ("float_capability", "test_float_capability_report.cpp", []),
        ("mman_ftruncate", "test_mman_ftruncate.cpp", []),
        ("type_support", "test_type_support.cpp", []),
        ("w_bridge_errors", "test_w_bridge_errors.cpp", []),
        (
            "warning_budget",
            "test_warning_budget.cpp",
            ["-Wall", "-Wextra", "-Werror=unused-result", "-Werror=return-type", "-Werror=uninitialized"],
        ),
    ]
    results: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="annoy-native-") as td:
        out = pathlib.Path(td)
        for name, source, extra in registry:
            binary = out / name
            compile_cmd = [compiler, "-std=c++17", "-O2", *extra, f"-I{src}", str(tests / source), "-o", str(binary)]
            compiled, compile_output = run(compile_cmd)
            if not compiled:
                results[name] = {"status": "RED", "phase": "compile", "output": compile_output[-3000:]}
                continue
            passed, runtime_output = run([str(binary)])
            results[name] = {"status": "GREEN" if passed else "RED", "phase": "run", "output": runtime_output[-1000:]}

        pyconfig = shutil.which(f"python{sys.version_info.major}.{sys.version_info.minor}-config")
        if pyconfig is None:
            results["pyconv"] = {"status": "UNAVAILABLE", "reason": "python3-config not found"}
        else:
            inc = shlex.split(subprocess.check_output([pyconfig, "--includes"], text=True, timeout=60))
            ld = shlex.split(subprocess.check_output([pyconfig, "--ldflags", "--embed"], text=True, timeout=60))
            binary = out / "pyconv"
            cmd = [compiler, "-std=c++17", "-O2", *inc, f"-I{src}", str(tests / "test_annoy_pyconv.cpp"), *ld, "-o", str(binary)]
            compiled, compile_output = run(cmd)
            if compiled:
                passed, runtime_output = run([str(binary)])
                results["pyconv"] = {"status": "GREEN" if passed else "RED", "phase": "run", "output": runtime_output[-1000:]}
            else:
                results["pyconv"] = {"status": "RED", "phase": "compile", "output": compile_output[-3000:]}

        if platform.machine().lower() in {"x86_64", "amd64", "i386", "i686"}:
            f16 = out / "f16-hardware"
            compiled, output = run([
                compiler, "-std=c++17", "-O2", "-mf16c", f"-I{src}",
                str(tests / "test_float16_scalar_matches_hardware.cpp"), "-o", str(f16),
            ])
            results["float16_hardware"] = {
                "status": "GREEN_COMPILE_ONLY" if compiled else "RED",
                "phase": "compile-only",
                "output": output[-1000:],
            }

        else:
            results["float16_hardware"] = {"status": "UNAVAILABLE", "reason": "x86-only probe on another architecture"}

        reloc_script = tests / "check_no_cpu_feature_reloc.sh"
        bash = shutil.which("bash")
        if bash is None or platform.system() != "Linux" or not all(shutil.which(t) for t in ("nm", "readelf")):
            results["cpu_feature_reloc"] = {"status": "UNAVAILABLE", "reason": "bash not found"}
        else:
            passed, output = run([bash, str(reloc_script)])
            results["cpu_feature_reloc"] = {"status": "GREEN" if passed else "RED", "phase": "run", "output": output[-1000:]}

    red = [name for name, item in results.items() if item["status"] == "RED"]
    green = [name for name, item in results.items() if str(item["status"]).startswith("GREEN")]
    unavailable = [name for name, item in results.items() if item["status"] == "UNAVAILABLE"]
    payload = {
        "status": overall(results),
        "green": len(green),
        "red": len(red),
        "unavailable": len(unavailable),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        code = main()
        sys.stdout.flush()
    except BrokenPipeError:
        import os
        sys.stdout = open(os.devnull, "w")
        code = 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "ERROR", "detail": str(exc)}), file=sys.stderr)
        code = 2
    raise SystemExit(code)
