// libs/_tools/pyodide_probe.mjs
//
// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Load pure-Python partial distributions in Pyodide (CPython compiled to
// WebAssembly, the interpreter behind JupyterLite) and import every module
// they ship.
//
// Why: Pyodide has no threads, no subprocesses and no compiler. A pure-Python
// wheel installs there unchanged, so the only question is whether its modules
// import. This script answers it for the wheels it is given.
//
// Usage (needs Node.js). Pyodide is taken from the directory the command is
// run in, so that the version is the one a committed lockfile names:
//
//     cd libs/_tools/pyodide/v314.0.0      # package.json + package-lock.json
//     npm ci
//     node ../../pyodide_probe.mjs [--packages numpy,click] WHEEL...
//
// One directory per Pyodide version lives under libs/_tools/pyodide/. To add
// a version, copy a directory, change the version in package.json and run
// `npm install --package-lock-only` there.
//
// --packages  Pyodide packages to load first (downloaded from the Pyodide
//             index, so this needs network access). Without it only the
//             standard library is available.
//
// Result: one JSON document on standard output, and the exit status.
//
//   0  `import scikitplot` worked and every module either imported or needs
//      a third-party package that was not loaded (listed under "needs").
//   1  `import scikitplot` failed, or a module failed for any other reason
//      (listed under "failed").
//   2  wrong usage.
//
// A module that needs a package you did not load is reported, not failed:
// whether numpy is available is a property of the page, not of the wheel.

import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

// A bare `import "pyodide"` is resolved from the directory of *this file*, and
// there is deliberately no node_modules above it: which Pyodide runs is decided
// by the directory the command is started in (see "Usage").
function loadPyodideModule() {
  const here = createRequire(path.join(process.cwd(), "package.json"));
  let entry;
  try {
    entry = here.resolve("pyodide");
  } catch (error) {
    console.error(
      `Pyodide is not installed in ${process.cwd()}.\n` +
        "Run this from a directory under libs/_tools/pyodide/ after `npm ci` there.",
    );
    process.exit(2);
  }
  // The package's ES module sits beside its CommonJS entry point.
  return import(pathToFileURL(path.join(path.dirname(entry), "pyodide.mjs")).href);
}

const args = process.argv.slice(2);
let packages = [];
const wheels = [];
for (let i = 0; i < args.length; i += 1) {
  if (args[i] === "--packages") {
    i += 1;
    if (i >= args.length) {
      console.error("--packages needs a comma-separated list");
      process.exit(2);
    }
    packages = args[i].split(",").filter(Boolean);
  } else {
    wheels.push(args[i]);
  }
}
if (wheels.length === 0) {
  console.error("usage: node pyodide_probe.mjs [--packages a,b] WHEEL...");
  process.exit(2);
}

const PROBE = `
import importlib, json, pkgutil, platform, sys, time

result = {
    "python": platform.python_version(),
    "platform": sys.platform,
    "machine": platform.machine(),
    "imported": 0,
    "needs": {},
    "failed": {},
}
try:
    import _thread
    _thread.start_new_thread(lambda: None, ())
    result["threads"] = True
except RuntimeError:
    result["threads"] = False

started = time.perf_counter()
try:
    import scikitplot
except BaseException as exc:
    result["failed"]["scikitplot"] = f"{type(exc).__name__}: {exc}"
else:
    result["import_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["version"] = scikitplot.__version__
    result["numpy_imported_by_import_scikitplot"] = "numpy" in sys.modules
    from scikitplot import _distributions

    report = _distributions.report()
    result["flavor"] = report["flavor"]
    result["installed"] = report["installed"]
    result["problems"] = report["problems"]
    names = [
        info.name
        for info in pkgutil.walk_packages(
            scikitplot.__path__, "scikitplot.", onerror=lambda name: None
        )
    ]
    for name in sorted(names):
        parts = name.split(".")
        if "tests" in parts or parts[-1] in ("conftest", "__main__"):
            continue
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing and missing != "scikitplot":
                result["needs"].setdefault(missing, []).append(name)
            else:
                result["failed"][name] = f"{type(exc).__name__}: {exc}"
        except BaseException as exc:
            result["failed"][name] = f"{type(exc).__name__}: {exc}"
        else:
            result["imported"] += 1
json.dumps(result)
`;

const { loadPyodide } = await loadPyodideModule();
const pyodide = await loadPyodide();
if (packages.length > 0) {
  await pyodide.loadPackage(packages);
}
const sitePackages = pyodide.runPython("import site; site.getsitepackages()[0]");
for (const wheel of wheels) {
  // A wheel is a zip archive; unpacking it into site-packages is what an
  // installer does for a pure-Python wheel.
  pyodide.unpackArchive(new Uint8Array(fs.readFileSync(wheel)), "zip", {
    extractDir: sitePackages,
  });
}
const result = JSON.parse(pyodide.runPython(PROBE));
result.pyodide = pyodide.version;
result.wheels = wheels.length;
result.packages = packages;
console.log(JSON.stringify(result, null, 2));
process.exit(Object.keys(result.failed).length === 0 ? 0 : 1);
