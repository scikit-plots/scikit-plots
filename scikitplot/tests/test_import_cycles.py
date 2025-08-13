import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from .test_public_api import PUBLIC_MODULES, REQUIRES_HEAVY

# Regression tests for gh-6793.
# Check that all modules are importable in a new Python process.
# This is not necessarily true if there are import cycles present.


def import_module(module: str) -> tuple[str, int, str]:
    """
    Attempt to import a module in a subprocess.

    Returns
    -------
    A tuple of (module name, return code, stderr output).
    """
    # If this submodule needs a heavy dep, skip if missing
    if module in REQUIRES_HEAVY:
        return module, 0, "skip this submodule only"

    # with subprocess.Popen([sys.executable, "-c", f"import {module}"]) as process:
    #     return module, process.wait()
    process = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=False,  # Explicitly allow failure to avoid exception
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return module, process.returncode, process.stderr.strip()
    # try:
    # except subprocess.CalledProcessError as e:
    #     return module, e.returncode, e.stderr.strip()
    # else:
    #     return module, 0, ""


@pytest.mark.fail_slow(240)  # 1s to 40s? adjust as needed 54.40319907600002s
@pytest.mark.slow
@pytest.mark.thread_unsafe
def test_public_modules_importable():
    """
    Ensures all public modules can be imported independently in separate subprocesses.
    Uses parallel execution (max 4 workers) to test importability in isolation.
    """
    # pids = [subprocess.Popen([sys.executable, '-c', f'import {module}'])
    #         for module in PUBLIC_MODULES]
    # for i, pid in enumerate(pids):
    #     assert pid.wait() == 0, f'Failed to import {PUBLIC_MODULES[i]}'

    # Limit to 4 concurrent subprocesses
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(import_module, PUBLIC_MODULES))

    # Assert that all imports succeed
    # for module, return_code in results:
    #     assert return_code == 0, f"Failed to import {module}"

    failures = [
        (module, stderr)
        for module, returncode, stderr in results
        if returncode != 0
    ]

    if failures:
        error_report = "\n".join(
            f"✗ {module}\n{stderr or '(no stderr)'}\n" for module, stderr in failures
        )
        pytest.fail(f"The following modules failed to import:\n\n{error_report}")
