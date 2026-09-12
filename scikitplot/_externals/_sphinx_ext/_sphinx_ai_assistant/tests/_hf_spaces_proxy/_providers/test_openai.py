# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Canonical tests for the corresponding runtime module.

Feature-sized cases live under '_cases.openai' and are not collected directly.
"""
# Large-contract case fragments are collected only through this canonical owner.
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._case_loader import export_case_tests as _export_case_tests

_export_case_tests(globals(), package=__package__, case_package='_cases.openai', cases=('resource_executor', 'responses_execution'))
del _export_case_tests
