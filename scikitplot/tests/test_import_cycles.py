import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from .test_public_api import PUBLIC_MODULES

# Regression tests for gh-6793.
# Check that all modules are importable in a new Python process.
# This is not necessarily true if there are import cycles present.


def import_module(module):
    with subprocess.Popen([sys.executable, "-c", f"import {module}"]) as process:
        return module, process.wait()


@pytest.mark.fail_slow(200)  # 99s
@pytest.mark.slow
@pytest.mark.thread_unsafe
def test_public_modules_importable():
    # pids = [subprocess.Popen([sys.executable, '-c', f'import {module}'])
    #         for module in PUBLIC_MODULES]
    # for i, pid in enumerate(pids):
    #     assert pid.wait() == 0, f'Failed to import {PUBLIC_MODULES[i]}'

    # Limit to 4 concurrent subprocesses
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(import_module, PUBLIC_MODULES))

    # Assert that all imports succeed
    for module, return_code in results:
        assert return_code == 0, f"Failed to import {module}"
