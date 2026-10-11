.. _mlflow-project-configuration:

Project configuration
=====================

Project discovery walks upward from a starting path looking for configured
markers. ``DEFAULT_PROJECT_MARKERS`` can be overridden explicitly or through
project-marker configuration helpers.

``ProjectConfig`` and the TOML/YAML loaders normalize paths relative to the
project root so multiple scripts can share tracking/database/artifact settings
without depending on their current working directory.

Environment files
-----------------

``parse_dotenv`` and ``apply_env`` make environment application explicit.
Treat env files as configuration with the same sensitivity as the credentials
or endpoints they contain; do not copy secrets into generated documentation or
logs.

Local store helpers create/normalize MLflow store locations. They do not make a
remote URI local and should not silently reinterpret unsupported schemes.
