# scikitplot/mlflow/tests/test__extended.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
DECOMPOSED — this file is intentionally empty.

All tests that previously lived here have been redistributed into the canonical
per-module test files following the ``test__<module_name>.py`` naming convention.

Migration map
-------------
+----------------------------------+---------------------------------------+
| Former class in test__extended   | Canonical file                        |
+==================================+=======================================+
| TestEnvSnapshot                  | test__env.py                          |
| TestParseDotenv                  | test__env.py                          |
| TestApplyEnv                     | test__env.py                          |
+----------------------------------+---------------------------------------+
| TestResolveDownloadArtifacts     | test__compat.py                       |
+----------------------------------+---------------------------------------+
| TestSessionConfigValidation      | test__config.py                       |
| TestServerConfigValidation       | test__config.py                       |
+----------------------------------+---------------------------------------+
| TestRunningInDocker              | test__container.py                    |
+----------------------------------+---------------------------------------+
| TestMlflowVersion (utils)        | test__utils.py                        |
+----------------------------------+---------------------------------------+
| TestCliCaps                      | test__cli_caps.py                     |
+----------------------------------+---------------------------------------+
| TestArtifactsFacade              | test__facade.py                       |
| TestModelsFacade                 | test__facade.py                       |
+----------------------------------+---------------------------------------+
| TestLocalPathHelpers             | test__project.py                      |
| TestProjectMarkers               | test__project.py                      |
| TestProjectConfigIO              | test__project.py                      |
+----------------------------------+---------------------------------------+
| TestWaitTrackingReady            | test__readiness.py                    |
+----------------------------------+---------------------------------------+
| TestBuildServerArgs              | test__server.py                       |
+----------------------------------+---------------------------------------+
| TestSessionModuleStructure       | test__session.py                      |
| TestSessionContextManager        | test__session.py                      |
+----------------------------------+---------------------------------------+
| TestWorkflowHelpers              | test__workflow.py                     |
+----------------------------------+---------------------------------------+

Notes
-----
- Do NOT add tests to this file.
- New gap coverage belongs in the matching canonical file.
- This file is kept to preserve git history and to signal clearly that the
  decomposition is intentional (not accidental deletion).
"""
