.. _mlflow-sessions-and-facades:

Sessions and facades
====================

:class:`MlflowHandle` represents integration/session state. The ``session``
helpers own configuration application and cleanup so scripts can share one
lifecycle rather than configuring MLflow globally in several unrelated places.

``ArtifactsFacade`` and ``ModelsFacade`` wrap selected upstream operations so
Scikit-Plots can handle compatibility differences and provide one project-level
entry point.

A facade is not a cache of truth about the remote tracking server. Handle
upstream authentication, permissions, artifact availability and server errors
as runtime conditions.
