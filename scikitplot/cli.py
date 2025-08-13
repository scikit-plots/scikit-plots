# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=undefined-variable
# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# pylint: disable=unreachable

# ruff: noqa: F401

# This module was copied from the mlflow project.
# https://github.com/mlflow/mlflow/blob/master/mlflow/cli.py

"""
cli.py.

Copied from mlflow.
https://github.com/mlflow/mlflow/blob/master/mlflow/cli.py
"""

import contextlib
import json

# import logging
import os
import re
import sys
import warnings
from datetime import timedelta

import click
from click import UsageError

import scikitplot

from . import __version__
from . import logger as _logger
from .environment_variables import SKPLT_EXPERIMENT_ID, SKPLT_EXPERIMENT_NAME
from .exceptions import InvalidUrlException, ScikitplotException
from .utils import cli_args
from .utils.logging_utils import eprint
from .utils.os import is_windows
from .utils.process import ShellCommandException

__all__ = [
    "cli",
]

INVALID_PARAMETER_VALUE = 0

######################################################################
## Accept aliased command.
######################################################################


class AliasedGroup(click.Group):
    """Accept aliased command."""

    def get_command(self, ctx, cmd_name):
        """get_command."""
        # `scikitplot ui` is an alias for `scikitplot server`
        cmd_name = "server" if cmd_name == "ui" else cmd_name
        return super().get_command(ctx, cmd_name)


######################################################################
## Main Entry-point: cli()
######################################################################


@click.group(cls=AliasedGroup)
@click.help_option(
    "--help",
    "-h",  # param_decls (only positional)
    "-H",  # param_decls (only positional)
)
# Automatically provides --version (or -v) as a command-line flag
# to print the program version and exit.
@click.version_option(
    __version__,  # version
    "--version",
    "-v",  # param_decls (only positional)
    "-V",  # param_decls (only positional)
    # package_name="scikitplot",  # optional, use version from installed `scikitplot`
    # prog_name="scikit-plots",  # optional, default display name in the version message
    # The message to show.
    # The values %(prog)s, %(package)s, and %(version)s are available.
    # Defaults to "%(prog)s, version %(version)s".
    # message="%(prog)s, %(package)s, version %(version)s",
)
# You're also manually defining --version / -v as an option
# ⚠️ This is a name conflict — click.version_option() reserves --version/-v,
# so you cannot reuse --version or -v for another purpose.
# for something else (a git commit ref).
# @click.version_option(version=__version__) ← remove this
# @click.option("--version", "-v", ...):
def cli():
    """scikit-plots main CLI entrypoint helper."""


######################################################################
## Reset all global state
######################################################################


@cli.command()
def reset():
    """Reset all global state (sklearn, matplotlib, seaborn, numpy)."""
    from . import _reset

    click.echo("Resetting modules...")
    _reset.reset()
    click.echo("Modules reset complete.")


######################################################################
## Add a COMMAND to Entry-point: doctor()
######################################################################


@cli.command(
    short_help="Prints out useful information for debugging issues with Scikit-plots."
)
@click.option(
    "--mask-envs",
    is_flag=True,
    help=(
        "If set (the default behavior without setting this flag is not to obfuscate information), "
        'mask the Scikit-plots environment variable values (e.g. `"SKPLT_ENV_VAR": "***"`) '
        "in the output to prevent leaking sensitive information."
    ),
)
def doctor(mask_envs):
    """Doctor."""
    raise NotImplementedError
    scikitplot.doctor(mask_envs)


######################################################################
## Add a COMMAND to Entry-point: gc()
######################################################################


# @cli.command(short_help="Permanently delete runs in the `deleted` lifecycle stage.")
# @click.option(
#     "--older-than",
#     default=None,
#     help="Optional. Remove run(s) older than the specified time limit. "
#     "Specify a string in #d#h#m#s format. Float values are also supported. "
#     "For example: --older-than 1d2h3m4s, --older-than 1.2d3h4m5s",
# )
# @click.option(
#     "--backend-store-uri",
#     metavar="PATH",
#     default=os.environ.get("DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH"),
#     help="URI of the backend store from which to delete runs. Acceptable URIs are "
#     "SQLAlchemy-compatible database connection strings "
#     "(e.g. 'sqlite:///path/to/file.db') or local filesystem URIs "
#     "(e.g. 'file:///absolute/path/to/directory'). By default, data will be deleted "
#     "from the ./mlruns directory.",
# )
# @click.option(
#     "--artifacts-destination",
#     envvar="SKPLT_ARTIFACTS_DESTINATION",
#     metavar="URI",
#     default=None,
#     help=(
#         "The base artifact location from which to resolve artifact upload/download/list requests "
#         "(e.g. 's3://my-bucket'). This option only applies when the tracking server is configured "
#         "to stream artifacts and the experiment's artifact root location is http or "
#         "scikitplot-artifacts URI. Otherwise, the default artifact location will be used."
#     ),
# )
# @click.option(
#     "--run-ids",
#     default=None,
#     help="Optional comma separated list of runs to be permanently deleted. If run ids"
#     " are not specified, data is removed for all runs in the `deleted`"
#     " lifecycle stage.",
# )
# @click.option(
#     "--experiment-ids",
#     default=None,
#     help="Optional comma separated list of experiments to be permanently deleted including "
#     "all of their associated runs. If experiment ids are not specified, data is removed for all "
#     "experiments in the `deleted` lifecycle stage.",
# )
# @click.option(
#     "--tracking-uri",
#     default=os.environ.get("SKPLT_TRACKING_URI"),
#     help="Tracking URI to use for deleting 'deleted' runs e.g. http://127.0.0.1:8080",
# )
# def gc(
#     older_than,
#     backend_store_uri,
#     artifacts_destination,
#     run_ids,
#     experiment_ids,
#     tracking_uri,
# ):
#     """
#     Permanently delete runs in the `deleted` lifecycle stage from the specified backend store.

#     This command deletes all artifacts and metadata associated with the specified runs.
#     If the provided artifact URL is invalid, the artifact deletion will be bypassed,
#     and the gc process will continue.

#     .. attention::

#         If you are running an Scikit-plots tracking server with artifact proxying enabled,
#         you **must** set the ``SKPLT_TRACKING_URI`` environment variable before running
#         this command. Otherwise, the ``gc`` command will not be able to resolve
#         artifact URIs and will not be able to delete the associated artifacts.

#     """
#     from .utils.time import get_current_time_millis

#     # raise NotImplementedError
#     time_delta = 0
#     if older_than is not None:
#         regex = re.compile(
#             r"^((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)"
#             r"?((?P<seconds>[\.\d]+?)s)?$"
#         )
#         parts = regex.match(older_than)
#         if parts is None:
#             raise ScikitplotException(
#                 f"Could not parse any time information from '{older_than}'. "
#                 "Examples of valid strings: '8h', '2d8h5m20s', '2m4s'",
#                 error_code=INVALID_PARAMETER_VALUE,
#             )
#         time_params = {
#             name: float(param) for name, param in parts.groupdict().items() if param
#         }
#         time_delta = int(timedelta(**time_params).total_seconds() * 1000)
#         _logger.info(time_delta)

#     time_threshold = get_current_time_millis() - time_delta
#     try:
#         raise ScikitplotException
#     except InvalidUrlException as iue:
#         click.echo(
#             click.style(
#                 f"An exception {iue!r} was raised during the deletion of a model artifact",
#                 fg="yellow",
#             )
#         )
#         click.echo(
#             click.style(
#                 f"Unable to resolve the provided artifact URL: '{time_threshold}'. "
#                 "The gc process will continue and bypass artifact deletion. "
#                 "Please ensure that the artifact exists "
#                 "and consider manually deleting any unused artifacts. ",
#                 fg="yellow",
#             ),
#         )
#     click.echo(f"Run with ID {time_threshold} has been permanently deleted.")


######################################################################
## Add a COMMAND to Entry-point: run()
######################################################################


# @cli.command()
# @click.help_option(
#     "--help",
#     "-h",  # param_decls (only positional)
# )
# @click.argument("uri")
# @click.option(
#     "--entry-point",
#     "-e",
#     metavar="NAME",
#     default="main",
#     help="Entry point within project. [default: main]. If the entry point is not found, "
#     "attempts to run the project file with the specified name as a script, "
#     "using 'python' to run .py files and the default shell (specified by "
#     "environment variable $SHELL) to run .sh files",
# )
# @click.option(
#     "--version",
#     "-v",
#     metavar="VERSION",
#     help="Version of the project to run, as a Git commit reference for Git projects.",
# )
# @click.option(
#     "--param-list",
#     "-P",
#     metavar="NAME=VALUE",
#     multiple=True,
#     help="A parameter for the run, of the form -P name=value. Provided parameters that "
#     "are not in the list of parameters for an entry point will be passed to the "
#     "corresponding entry point as command-line arguments in the form `--name value`",
# )
# @click.option(
#     "--docker-args",
#     "-A",
#     metavar="NAME=VALUE",
#     multiple=True,
#     help="A `docker run` argument or flag, of the form -A name=value (e.g. -A gpus=all) "
#     "or -A name (e.g. -A t). The argument will then be passed as "
#     "`docker run --name value` or `docker run --name` respectively. ",
# )
# @click.option(
#     "--experiment-name",
#     envvar=SKPLT_EXPERIMENT_NAME.name,
#     help="Name of the experiment under which to launch the run. If not "
#     "specified, 'experiment-id' option will be used to launch run.",
# )
# @click.option(
#     "--experiment-id",
#     envvar=SKPLT_EXPERIMENT_ID.name,
#     type=click.STRING,
#     help="ID of the experiment under which to launch the run.",
# )
# # TODO: Add tracking server argument once we have it working.
# @click.option(
#     "--backend",
#     "-b",
#     metavar="BACKEND",
#     default="local",
#     help="Execution backend to use for run. Supported values: 'local', 'databricks', "
#     "kubernetes (experimental). Defaults to 'local'. If running against "
#     "Databricks, will run against a Databricks workspace determined as follows: "
#     "if a Databricks tracking URI of the form 'databricks://profile' has been set "
#     "(e.g. by setting the SKPLT_TRACKING_URI environment variable), will run "
#     "against the workspace specified by <profile>. Otherwise, runs against the "
#     "workspace specified by the default Databricks CLI profile. See "
#     "https://github.com/databricks/databricks-cli for more info on configuring a "
#     "Databricks CLI profile.",
# )
# @click.option(
#     "--backend-config",
#     "-c",
#     metavar="FILE",
#     help="Path to JSON file (must end in '.json') or JSON string which will be passed "
#     "as config to the backend. The exact content which should be "
#     "provided is different for each execution backend and is documented "
#     "at https://www.mlflow.org/docs/latest/projects.html.",
# )
# @cli_args.ENV_MANAGER_PROJECTS
# @click.option(
#     "--storage-dir",
#     envvar="SKPLT_TMP_DIR",
#     help="Only valid when ``backend`` is local. "
#     "Scikit-plots downloads artifacts from distributed URIs passed to parameters of "
#     "type 'path' to subdirectories of storage_dir.",
# )
# @click.option(
#     "--run-id",
#     metavar="RUN_ID",
#     help="If specified, the given run ID will be used instead of creating a new run. "
#     "Note: this argument is used internally by the Scikit-plots project APIs "
#     "and should not be specified.",
# )
# @click.option(
#     "--run-name",
#     metavar="RUN_NAME",
#     help="The name to give the Scikit-plots Run associated with the project execution. If not specified, "
#     "the Scikit-plots Run name is left unset.",
# )
# @click.option(
#     "--build-image",
#     is_flag=True,
#     default=False,
#     show_default=True,
#     help=(
#         "Only valid for Docker projects. If specified, build a new Docker image that's based on "
#         "the image specified by the `image` field in the MLproject file, and contains files in the "
#         "project directory."
#     ),
# )
# def run(
#     uri,
#     entry_point,
#     version,
#     param_list,
#     docker_args,
#     experiment_name,
#     experiment_id,
#     backend,
#     backend_config,
#     env_manager,
#     storage_dir,
#     run_id,
#     run_name,
#     build_image,
# ):
#     """
#     Run an Scikit-plots project from the given URI.

#     For local runs, the run will block until it completes.
#     Otherwise, the project will run asynchronously.

#     If running locally (the default), the URI can be either a Git repository URI or a local path.
#     If running on Databricks, the URI must be a Git repository.

#     By default, Git projects run in a new working directory with the given parameters, while
#     local projects run from the project's root directory.
#     """

#     raise NotImplementedError
#     if experiment_id is not None and experiment_name is not None:
#         eprint("Specify only one of 'experiment-name' or 'experiment-id' options.")
#         sys.exit(1)

#     param_dict = _user_args_to_dict(param_list)
#     args_dict = _user_args_to_dict(docker_args, argument_type="A")

#     if backend_config is not None and os.path.splitext(backend_config)[-1] != ".json":
#         try:
#             backend_config = json.loads(backend_config)
#         except ValueError as e:
#             eprint(f"Invalid backend config JSON. Parse error: {e}")
#             raise
#     if backend == "kubernetes":  # noqa: SIM102
#         if backend_config is None:
#             eprint("Specify 'backend_config' when using kubernetes mode.")
#             sys.exit(1)
#     try:
#         projects.run(  # noqa: F821
#             uri,
#             entry_point,
#             version,
#             experiment_name=experiment_name,
#             experiment_id=experiment_id,
#             parameters=param_dict,
#             docker_args=args_dict,
#             backend=backend,
#             backend_config=backend_config,
#             env_manager=env_manager,
#             storage_dir=storage_dir,
#             synchronous=backend in ("local", "kubernetes") or backend is None,
#             run_id=run_id,
#             run_name=run_name,
#             build_image=build_image,
#         )
#     except projects.ExecutionException as e:  # noqa: F821
#         _logger.error("=== %s ===", e)
#         sys.exit(1)


# def _user_args_to_dict(arguments, argument_type="P"):
#     user_dict = {}
#     for arg in arguments:
#         split = arg.split("=", maxsplit=1)
#         # Docker arguments such as `t` don't require a value -> set to True if specified
#         if len(split) == 1 and argument_type == "A":
#             name = split[0]
#             value = True
#         elif len(split) == 2:  # noqa: PLR2004
#             name = split[0]
#             value = split[1]
#         else:
#             eprint(
#                 f"Invalid format for -{argument_type} parameter: '{arg}'. "
#                 f"Use -{argument_type} name=value."
#             )
#             sys.exit(1)
#         if name in user_dict:
#             eprint(f"Repeated parameter: '{name}'")
#             sys.exit(1)
#         user_dict[name] = value
#     return user_dict


# def _validate_server_args(gunicorn_opts=None, workers=None, waitress_opts=None):
#     if sys.platform == "win32":
#         if gunicorn_opts is not None or workers is not None:
#             raise NotImplementedError(
#                 "waitress replaces gunicorn on Windows, cannot specify --gunicorn-opts or --workers"
#             )
#     else:  # noqa: PLR5501
#         if waitress_opts is not None:
#             raise NotImplementedError(
#                 "gunicorn replaces waitress on non-Windows platforms, "
#                 "cannot specify --waitress-opts"
#             )


# def _validate_static_prefix(ctx, param, value):
#     """
#     Validate that the static_prefix option starts with a "/" and does not end in a "/".

#     Conforms to the callback interface of click documented at
#     http://click.pocoo.org/5/options/#callbacks-for-validation.
#     """
#     if value is not None:
#         if not value.startswith("/"):
#             raise UsageError("--static-prefix must begin with a '/'.")
#         if value.endswith("/"):
#             raise UsageError("--static-prefix should not end with a '/'.")
#     return value


######################################################################
## Add a COMMAND to Entry-point: server()
######################################################################


# @cli.command()
# @click.option(
#     "--backend-store-uri",
#     envvar="SKPLT_BACKEND_STORE_URI",
#     metavar="PATH",
#     default=os.environ.get("DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH"),
#     help="URI to which to persist experiment and run data. Acceptable URIs are "
#     "SQLAlchemy-compatible database connection strings "
#     "(e.g. 'sqlite:///path/to/file.db') or local filesystem URIs "
#     "(e.g. 'file:///absolute/path/to/directory'). By default, data will be logged "
#     "to the ./mlruns directory.",
# )
# @click.option(
#     "--registry-store-uri",
#     envvar="SKPLT_REGISTRY_STORE_URI",
#     metavar="URI",
#     default=None,
#     help="URI to which to persist registered models. Acceptable URIs are "
#     "SQLAlchemy-compatible database connection strings (e.g. 'sqlite:///path/to/file.db'). "
#     "If not specified, `backend-store-uri` is used.",
# )
# @click.option(
#     "--default-artifact-root",
#     envvar="SKPLT_DEFAULT_ARTIFACT_ROOT",
#     metavar="URI",
#     default=None,
#     help="Directory in which to store artifacts for any new experiments created. For tracking "
#     "server backends that rely on SQL, this option is required in order to store artifacts. "
#     "Note that this flag does not impact already-created experiments with any previous "
#     "configuration of an Scikit-plots server instance. "
#     f"By default, data will be logged to the {'DEFAULT_ARTIFACTS_URI'} uri proxy if "
#     "the --serve-artifacts option is enabled. Otherwise, the default location will "
#     f"be {'DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH'}.",
# )
# @cli_args.SERVE_ARTIFACTS
# @click.option(
#     "--artifacts-only",
#     envvar="SKPLT_ARTIFACTS_ONLY",
#     is_flag=True,
#     default=False,
#     help="If specified, configures the scikitplot server to be used only for proxied artifact serving. "
#     "With this mode enabled, functionality of the scikitplot tracking service (e.g. run creation, "
#     "metric logging, and parameter logging) is disabled. The server will only expose "
#     "endpoints for uploading, downloading, and listing artifacts. "
#     "Default: False",
# )
# @cli_args.ARTIFACTS_DESTINATION
# @cli_args.HOST
# @cli_args.PORT
# @cli_args.WORKERS
# @click.option(
#     "--static-prefix",
#     envvar="SKPLT_STATIC_PREFIX",
#     default=None,
#     callback=_validate_static_prefix,
#     help="A prefix which will be prepended to the path of all static paths.",
# )
# @click.option(
#     "--gunicorn-opts",
#     envvar="SKPLT_GUNICORN_OPTS",
#     default=None,
#     help="Additional command line options forwarded to gunicorn processes.",
# )
# @click.option(
#     "--waitress-opts",
#     default=None,
#     help="Additional command line options for waitress-serve.",
# )
# @click.option(
#     "--expose-prometheus",
#     envvar="SKPLT_EXPOSE_PROMETHEUS",
#     default=None,
#     help="Path to the directory where metrics will be stored. If the directory "
#     "doesn't exist, it will be created. "
#     "Activate prometheus exporter to expose metrics on /metrics endpoint.",
# )
# @click.option(
#     "--app-name",
#     default=None,
#     type=click.Choice(list('get_entry_points("scikitplot.app")')),
#     show_default=True,
#     help=(
#         "Application name to be used for the tracking server. "
#         "If not specified, 'scikitplot.server:app' will be used."
#     ),
# )
# @click.option(
#     "--dev",
#     is_flag=True,
#     default=False,
#     show_default=True,
#     help=(
#         "If enabled, run the server with debug logging and auto-reload. "
#         "Should only be used for development purposes. "
#         "Cannot be used with '--gunicorn-opts'. "
#         "Unsupported on Windows."
#     ),
# )
# def server(
#     backend_store_uri,
#     registry_store_uri,
#     default_artifact_root,
#     serve_artifacts,
#     artifacts_only,
#     artifacts_destination,
#     host,
#     port,
#     workers,
#     static_prefix,
#     gunicorn_opts,
#     waitress_opts,
#     expose_prometheus,
#     app_name,
#     dev,
# ):
#     """
#     Run the Scikit-plots tracking server.

#     The server listens on http://localhost:5000 by default and only accepts connections
#     from the local machine. To let the server accept connections from other machines, you will need
#     to pass ``--host 0.0.0.0`` to listen on all network interfaces
#     (or a specific interface address).
#     """
#     raise NotImplementedError
#     from .server import _run_server
#     from .server.handlers import initialize_backend_stores

#     if dev and is_windows():
#         raise click.UsageError("'--dev' is not supported on Windows.")

#     if dev and gunicorn_opts:
#         raise click.UsageError(
#             "'--dev' and '--gunicorn-opts' cannot be specified together."
#         )

#     gunicorn_opts = "--log-level debug --reload" if dev else gunicorn_opts
#     _validate_server_args(
#         gunicorn_opts=gunicorn_opts, workers=workers, waitress_opts=waitress_opts
#     )

#     # Ensure that both backend_store_uri and default_artifact_uri are set correctly.
#     if not backend_store_uri:
#         backend_store_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH  # noqa: F821

#     # the default setting of registry_store_uri is same as backend_store_uri
#     if not registry_store_uri:
#         registry_store_uri = backend_store_uri

#     default_artifact_root = resolve_default_artifact_root(  # noqa: F821
#         serve_artifacts, default_artifact_root, backend_store_uri
#     )
#     artifacts_only_config_validation(artifacts_only, backend_store_uri)  # noqa: F821

#     try:
#         initialize_backend_stores(
#             backend_store_uri, registry_store_uri, default_artifact_root
#         )
#     except NotImplementedError as e:
#         _logger.error("Error initializing backend store")
#         _logger.exception(e)
#         sys.exit(1)

#     try:
#         _run_server(
#             backend_store_uri,
#             registry_store_uri,
#             default_artifact_root,
#             serve_artifacts,
#             artifacts_only,
#             artifacts_destination,
#             host,
#             port,
#             static_prefix,
#             workers,
#             gunicorn_opts,
#             waitress_opts,
#             expose_prometheus,
#             app_name,
#         )
#     except ShellCommandException:
#         eprint(
#             "Running the scikitplot server failed. Please see the logs above for details."
#         )
#         sys.exit(1)

#     raise NotImplementedError


######################################################################
## Add a COMMAND to Entry-point: st()
######################################################################


@cli.command(
    short_help="Launch the Streamlit app with the provided configuration options.",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def st2(args):
    """
    Launch the Streamlit app with the provided configuration options.

    For Docker or WSL 2 environments::

        # Inside Docker: set address to "0.0.0.0" instead of "localhost"

        python -m scikitplot.streamlit.run_app st --address 0.0.0.0

        # (optionally persist) ~/.wslconfig

        localhostforwarding=true
    """
    from ._ui_app.streamlit.run_ui_app_st import run_ui_app_st

    run_ui_app_st(args)


@cli.command()
@click.option(
    "--file_path",
    default="template_ui_app_st.py",
    help="Streamlit app file, default 'template_ui_app_st.py'.",
)
@click.option(
    "--address",
    "-a",
    default="localhost",
    help="Streamlit Host address, default 'localhost'.",
)
@click.option("--port", "-p", default="8501", help="Streamlit Port, default '8501'.")
@click.option(
    "--dark_theme",
    "-d",
    is_flag=True,
    default=False,
    help="Streamlit Enable dark theme",
)
@click.option(
    "--lib_sample",
    "-ls",
    is_flag=True,
    default=True,
    help="Use lib sample app, default True.",
)
def st(file_path, address, port, dark_theme, lib_sample):
    """
    Launch the Streamlit app with the provided configuration options.

    For Docker or WSL 2 environments::

        # Inside Docker: set address to "0.0.0.0" instead of "localhost"

        python -m scikitplot.streamlit.run_app st --address 0.0.0.0

        # (optionally persist) ~/.wslconfig

        localhostforwarding=true
    """
    from ._ui_app.streamlit.run_ui_app_st import launch_streamlit  # run_ui_app_st

    try:
        click.echo(f"Launching {file_path} on port {port}")
        launch_streamlit(
            file_path=file_path,
            address=address,
            port=port,
            dark_theme=dark_theme,
            lib_sample=lib_sample,
        )
    except ShellCommandException:
        eprint(
            "Running the scikitplot streamlit UI app failed. "
            "Please see the logs above for details."
        )
        sys.exit(1)


######################################################################
## Add a COMMAND to Entry-point: gr()
######################################################################


@cli.command()
@click.option(
    "--share",
    "-s",
    is_flag=True,
    default=True,
    help="Gradio public link serve.",
)
def gr(share):
    """
    Launch the gradio app with the provided configuration options.

    For Docker or WSL 2 environments::

        # Inside Docker: set address to "0.0.0.0" instead of "localhost"

        python -m scikitplot.streamlit.run_app st --address 0.0.0.0

        # (optionally persist) ~/.wslconfig

        localhostforwarding=true
    """
    from ._ui_app.gradio.template_ui_app_gr import ui_app_gr

    try:
        ui_app_gr.launch(share=True)
    except ShellCommandException:
        eprint(
            "Running the scikitplot gradio UI app failed. "
            "Please see the logs above for details."
        )
        sys.exit(1)


######################################################################
## Add a COMMAND to Entry-point: from defined py
######################################################################

with contextlib.suppress(
    AttributeError,
    ModuleNotFoundError,
    NameError,
):
    cli.add_command(scikitplot.runs.commands)  # noqa: F821
    cli.add_command(scikitplot.db.commands)  # noqa: F821

# We are conditional loading these commands since the skinny client does
# not support them due to the pandas and numpy dependencies of Scikit-plots Models
with contextlib.suppress(
    ImportError,
):
    from .gateway import cli  # type: ignore  # noqa: PGH003

    cli.add_command(scikitplot.gateway.cli.commands)


if __name__ == "__main__":
    cli()
