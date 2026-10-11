.. _mlflow-providers-and-workflows:

Providers and workflows
=======================

:class:`MlflowProvider` is the customization seam for applications that need to
replace selected upstream interactions without forking the integration layer.
Use ``use_provider`` for scoped overrides; avoid changing global provider state
from reusable library code.

Workflow helpers export built-in configuration, patch an experiment name, build
default project paths and run the demonstration flow. They are conveniences on
top of the lower-level config/session/server APIs, not separate configuration
systems.
