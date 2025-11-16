# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

"""
shared options and groups

The principle here is to define options once, but *not* instantiate them
globally. One reason being that options with action='append' can carry state
between parses. pip parses general options twice internally, and shouldn't
pass on state. To be consistent, all options will follow this design.
"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False
from __future__ import annotations

import os
import pathlib
import textwrap
from functools import partial

# https://docs.python.org/3/library/optparse.html
# getopt -> optparse -> argparse
from optparse import SUPPRESS_HELP, Option, OptionGroup, OptionParser, Values
from textwrap import dedent
from typing import Any, Callable

from ..exceptions import CommandError
from ..externals import _appdirs
from ._misc import strtobool

# Application Directories
USER_CACHE_DIR = _appdirs.user_cache_dir("scikitplot")


def raise_option_error(parser: OptionParser, option: Option, msg: str) -> None:
    """
    Raise an option parsing error using parser.error().

    parser: an OptionParser instance.
    option: an Option instance.
    msg: the error text.
    """
    msg = f"{option} error: {msg}"
    msg = textwrap.fill(" ".join(msg.split()))
    parser.error(msg)


def _path_option_check(option: Option, opt: str, value: str) -> str:
    return os.path.expanduser(value)


class PipOption(Option):
    TYPES = Option.TYPES + ("path", "package_name")
    TYPE_CHECKER = Option.TYPE_CHECKER.copy()
    TYPE_CHECKER["path"] = _path_option_check


###########
# options #
###########

help_: Callable[..., Option] = partial(
    Option,
    "-h",
    "--help",
    dest="help",
    action="help",
    help="Show help.",
)

debug_mode: Callable[..., Option] = partial(
    Option,
    "--debug",
    dest="debug_mode",
    action="store_true",
    default=False,
    help=(
        "Let unhandled exceptions propagate outside the main subroutine, "
        "instead of logging them to stderr."
    ),
)

isolated_mode: Callable[..., Option] = partial(
    Option,
    "--isolated",
    dest="isolated_mode",
    action="store_true",
    default=False,
    help=(
        "Run pip in an isolated mode, ignoring environment variables and user "
        "configuration."
    ),
)

require_virtualenv: Callable[..., Option] = partial(
    Option,
    "--require-virtualenv",
    "--require-venv",
    dest="require_venv",
    action="store_true",
    default=False,
    help=(
        "Allow pip to only run in a virtual environment; exit with an error otherwise."
    ),
)

override_externally_managed: Callable[..., Option] = partial(
    Option,
    "--break-system-packages",
    dest="override_externally_managed",
    action="store_true",
    help="Allow pip to modify an EXTERNALLY-MANAGED Python installation",
)

python: Callable[..., Option] = partial(
    Option,
    "--python",
    dest="python",
    help="Run pip with the specified Python interpreter.",
)

verbose: Callable[..., Option] = partial(
    Option,
    "-v",
    "--verbose",
    dest="verbose",
    action="count",
    default=0,
    help="Give more output. Option is additive, and can be used up to 3 times.",
)

no_color: Callable[..., Option] = partial(
    Option,
    "--no-color",
    dest="no_color",
    action="store_true",
    default=False,
    help="Suppress colored output.",
)

version: Callable[..., Option] = partial(
    Option,
    "-V",
    "--version",
    dest="version",
    action="store_true",
    help="Show version and exit.",
)

quiet: Callable[..., Option] = partial(
    Option,
    "-q",
    "--quiet",
    dest="quiet",
    action="count",
    default=0,
    help=(
        "Give less output. Option is additive, and can be used up to 3"
        " times (corresponding to WARNING, ERROR, and CRITICAL logging"
        " levels)."
    ),
)

progress_bar: Callable[..., Option] = partial(
    Option,
    "--progress-bar",
    dest="progress_bar",
    type="choice",
    choices=["auto", "on", "off", "raw"],
    default="auto",
    help=(
        "Specify whether the progress bar should be used. In 'auto'"
        " mode, --quiet will suppress all progress bars."
        " [auto, on, off, raw] (default: auto)"
    ),
)

log: Callable[..., Option] = partial(
    PipOption,
    "--log",
    "--log-file",
    "--local-log",
    dest="log",
    metavar="path",
    type="path",
    help="Path to a verbose appending log.",
)

no_input: Callable[..., Option] = partial(
    Option,
    # Don't ask for input
    "--no-input",
    dest="no_input",
    action="store_true",
    default=False,
    help="Disable prompting for input.",
)

keyring_provider: Callable[..., Option] = partial(
    Option,
    "--keyring-provider",
    dest="keyring_provider",
    choices=["auto", "disabled", "import", "subprocess"],
    default="auto",
    help=(
        "Enable the credential lookup via the keyring library if user input is allowed."
        " Specify which mechanism to use [auto, disabled, import, subprocess]."
        " (default: %default)"
    ),
)

proxy: Callable[..., Option] = partial(
    Option,
    "--proxy",
    dest="proxy",
    type="str",
    default="",
    help="Specify a proxy in the form scheme://[user:passwd@]proxy.server:port.",
)

retries: Callable[..., Option] = partial(
    Option,
    "--retries",
    dest="retries",
    type="int",
    default=5,
    help="Maximum attempts to establish a new HTTP connection. (default: %default)",
)

resume_retries: Callable[..., Option] = partial(
    Option,
    "--resume-retries",
    dest="resume_retries",
    type="int",
    default=5,
    help="Maximum attempts to resume or restart an incomplete download. "
    "(default: %default)",
)

timeout: Callable[..., Option] = partial(
    Option,
    "--timeout",
    "--default-timeout",
    metavar="sec",
    dest="timeout",
    type="float",
    default=15,
    help="Set the socket timeout (default %default seconds).",
)


def exists_action() -> Option:
    return Option(
        # Option when path already exist
        "--exists-action",
        dest="exists_action",
        type="choice",
        choices=["s", "i", "w", "b", "a"],
        default=[],
        action="append",
        metavar="action",
        help="Default action when a path already exists: "
        "(s)witch, (i)gnore, (w)ipe, (b)ackup, (a)bort.",
    )


cert: Callable[..., Option] = partial(
    PipOption,
    "--cert",
    dest="cert",
    type="path",
    metavar="path",
    help=(
        "Path to PEM-encoded CA certificate bundle. "
        "If provided, overrides the default. "
        "See 'SSL Certificate Verification' in pip documentation "
        "for more information."
    ),
)

client_cert: Callable[..., Option] = partial(
    PipOption,
    "--client-cert",
    dest="client_cert",
    type="path",
    default=None,
    metavar="path",
    help="Path to SSL client certificate, a single file containing the "
    "private key and the certificate in PEM format.",
)

index_url: Callable[..., Option] = partial(
    Option,
    "-i",
    "--index-url",
    "--pypi-url",
    dest="index_url",
    metavar="URL",
    default="PyPI.simple_url",
    help="Base URL of the Python Package Index (default %default). "
    "This should point to a repository compliant with PEP 503 "
    "(the simple repository API) or a local directory laid out "
    "in the same format.",
)


def extra_index_url() -> Option:
    return Option(
        "--extra-index-url",
        dest="extra_index_urls",
        metavar="URL",
        action="append",
        default=[],
        help="Extra URLs of package indexes to use in addition to "
        "--index-url. Should follow the same rules as "
        "--index-url.",
    )


no_index: Callable[..., Option] = partial(
    Option,
    "--no-index",
    dest="no_index",
    action="store_true",
    default=False,
    help="Ignore package index (only looking at --find-links URLs instead).",
)


def find_links() -> Option:
    return Option(
        "-f",
        "--find-links",
        dest="find_links",
        action="append",
        default=[],
        metavar="url",
        help="If a URL or path to an html file, then parse for links to "
        "archives such as sdist (.tar.gz) or wheel (.whl) files. "
        "If a local path or file:// URL that's a directory, "
        "then look for archives in the directory listing. "
        "Links to VCS project URLs are not supported.",
    )


def trusted_host() -> Option:
    return Option(
        "--trusted-host",
        dest="trusted_hosts",
        action="append",
        metavar="HOSTNAME",
        default=[],
        help="Mark this host or host:port pair as trusted, even though it "
        "does not have valid or any HTTPS.",
    )


def constraints() -> Option:
    return Option(
        "-c",
        "--constraint",
        dest="constraints",
        action="append",
        default=[],
        metavar="file",
        help="Constrain versions using the given constraints file. "
        "This option can be used multiple times.",
    )


def build_constraints() -> Option:
    return Option(
        "--build-constraint",
        dest="build_constraints",
        action="append",
        type="str",
        default=[],
        metavar="file",
        help=(
            "Constrain build dependencies using the given constraints file. "
            "This option can be used multiple times."
        ),
    )


def requirements() -> Option:
    return Option(
        "-r",
        "--requirement",
        dest="requirements",
        action="append",
        default=[],
        metavar="file",
        help="Install from the given requirements file. "
        "This option can be used multiple times.",
    )


def editable() -> Option:
    return Option(
        "-e",
        "--editable",
        dest="editables",
        action="append",
        default=[],
        metavar="path/url",
        help=(
            "Install a project in editable mode (i.e. setuptools "
            '"develop mode") from a local project path or a VCS url.'
        ),
    )


def _handle_src(option: Option, opt_str: str, value: str, parser: OptionParser) -> None:
    value = os.path.abspath(value)
    setattr(parser.values, option.dest, value)


src: Callable[..., Option] = partial(
    PipOption,
    "--src",
    "--source",
    "--source-dir",
    "--source-directory",
    dest="src_dir",
    type="path",
    metavar="dir",
    default="get_src_prefix()",
    action="callback",
    callback=_handle_src,
    help="Directory to check out editable projects into. "
    'The default in a virtualenv is "<venv path>/src". '
    'The default for global installs is "<current dir>/src".',
)


def _get_format_control(values: Values, option: Option) -> Any:
    """Get a format_control object."""
    return getattr(values, option.dest)


platforms: Callable[..., Option] = partial(
    Option,
    "--platform",
    dest="platforms",
    metavar="platform",
    action="append",
    default=None,
    help=(
        "Only use wheels compatible with <platform>. Defaults to the "
        "platform of the running system. Use this option multiple times to "
        "specify multiple platforms supported by the target interpreter."
    ),
)


# This was made a separate function for unit-testing purposes.
def _convert_python_version(value: str) -> tuple[tuple[int, ...], str | None]:
    """
    Convert a version string like "3", "37", or "3.7.3" into a tuple of ints.

    :return: A 2-tuple (version_info, error_msg), where `error_msg` is
        non-None if and only if there was a parsing error.
    """
    if not value:
        # The empty string is the same as not providing a value.
        return (None, None)

    parts = value.split(".")
    if len(parts) > 3:
        return ((), "at most three version parts are allowed")

    if len(parts) == 1:
        # Then we are in the case of "3" or "37".
        value = parts[0]
        if len(value) > 1:
            parts = [value[0], value[1:]]

    try:
        version_info = tuple(int(part) for part in parts)
    except ValueError:
        return ((), "each version part must be an integer")

    return (version_info, None)


def _handle_python_version(
    option: Option, opt_str: str, value: str, parser: OptionParser
) -> None:
    """
    Handle a provided --python-version value.
    """
    version_info, error_msg = _convert_python_version(value)
    if error_msg is not None:
        msg = f"invalid --python-version value: {value!r}: {error_msg}"
        raise_option_error(parser, option=option, msg=msg)

    parser.values.python_version = version_info


python_version: Callable[..., Option] = partial(
    Option,
    "--python-version",
    dest="python_version",
    metavar="python_version",
    action="callback",
    callback=_handle_python_version,
    type="str",
    default=None,
    help=dedent(
        """\
    The Python interpreter version to use for wheel and "Requires-Python"
    compatibility checks. Defaults to a version derived from the running
    interpreter. The version can be specified using up to three dot-separated
    integers (e.g. "3" for 3.0.0, "3.7" for 3.7.0, or "3.7.3"). A major-minor
    version can also be given as a string without dots (e.g. "37" for 3.7.0).
    """
    ),
)


implementation: Callable[..., Option] = partial(
    Option,
    "--implementation",
    dest="implementation",
    metavar="implementation",
    default=None,
    help=(
        "Only use wheels compatible with Python "
        "implementation <implementation>, e.g. 'pp', 'jy', 'cp', "
        " or 'ip'. If not specified, then the current "
        "interpreter implementation is used.  Use 'py' to force "
        "implementation-agnostic wheels."
    ),
)


abis: Callable[..., Option] = partial(
    Option,
    "--abi",
    dest="abis",
    metavar="abi",
    action="append",
    default=None,
    help=(
        "Only use wheels compatible with Python abi <abi>, e.g. 'pypy_41'. "
        "If not specified, then the current interpreter abi tag is used. "
        "Use this option multiple times to specify multiple abis supported "
        "by the target interpreter. Generally you will need to specify "
        "--implementation, --platform, and --python-version when using this "
        "option."
    ),
)


def add_target_python_options(cmd_opts: OptionGroup) -> None:
    cmd_opts.add_option(platforms())
    cmd_opts.add_option(python_version())
    cmd_opts.add_option(implementation())
    cmd_opts.add_option(abis())


def prefer_binary() -> Option:
    return Option(
        "--prefer-binary",
        dest="prefer_binary",
        action="store_true",
        default=False,
        help=(
            "Prefer binary packages over source packages, even if the "
            "source packages are newer."
        ),
    )


cache_dir: Callable[..., Option] = partial(
    PipOption,
    "--cache-dir",
    dest="cache_dir",
    default=USER_CACHE_DIR,
    metavar="dir",
    type="path",
    help="Store the cache data in <dir>.",
)


def _handle_no_cache_dir(
    option: Option, opt: str, value: str, parser: OptionParser
) -> None:
    """
    Process a value provided for the --no-cache-dir option.

    This is an optparse.Option callback for the --no-cache-dir option.
    """
    # The value argument will be None if --no-cache-dir is passed via the
    # command-line, since the option doesn't accept arguments.  However,
    # the value can be non-None if the option is triggered e.g. by an
    # environment variable, like PIP_NO_CACHE_DIR=true.
    if value is not None:
        # Then parse the string value to get argument error-checking.
        try:
            strtobool(value)
        except ValueError as exc:
            raise_option_error(parser, option=option, msg=str(exc))

    # Originally, setting PIP_NO_CACHE_DIR to a value that strtobool()
    # converted to 0 (like "false" or "no") caused cache_dir to be disabled
    # rather than enabled (logic would say the latter).  Thus, we disable
    # the cache directory not just on values that parse to True, but (for
    # backwards compatibility reasons) also on values that parse to False.
    # In other words, always set it to False if the option is provided in
    # some (valid) form.
    parser.values.cache_dir = False


no_cache: Callable[..., Option] = partial(
    Option,
    "--no-cache-dir",
    dest="cache_dir",
    action="callback",
    callback=_handle_no_cache_dir,
    help="Disable the cache.",
)

no_deps: Callable[..., Option] = partial(
    Option,
    "--no-deps",
    "--no-dependencies",
    dest="ignore_dependencies",
    action="store_true",
    default=False,
    help="Don't install package dependencies.",
)


def _handle_dependency_group(
    option: Option, opt: str, value: str, parser: OptionParser
) -> None:
    """
    Process a value provided for the --group option.

    Splits on the rightmost ":", and validates that the path (if present) ends
    in `pyproject.toml`. Defaults the path to `pyproject.toml` when one is not given.

    `:` cannot appear in dependency group names, so this is a safe and simple parse.

    This is an optparse.Option callback for the dependency_groups option.
    """
    path, sep, groupname = value.rpartition(":")
    if not sep:
        path = "pyproject.toml"
    # check for 'pyproject.toml' filenames using pathlib
    elif pathlib.PurePath(path).name != "pyproject.toml":
        msg = "group paths use 'pyproject.toml' filenames"
        raise_option_error(parser, option=option, msg=msg)

    parser.values.dependency_groups.append((path, groupname))


dependency_groups: Callable[..., Option] = partial(
    Option,
    "--group",
    dest="dependency_groups",
    default=[],
    type=str,
    action="callback",
    callback=_handle_dependency_group,
    metavar="[path:]group",
    help='Install a named dependency-group from a "pyproject.toml" file. '
    'If a path is given, the name of the file must be "pyproject.toml". '
    'Defaults to using "pyproject.toml" in the current directory.',
)

ignore_requires_python: Callable[..., Option] = partial(
    Option,
    "--ignore-requires-python",
    dest="ignore_requires_python",
    action="store_true",
    help="Ignore the Requires-Python information.",
)

no_build_isolation: Callable[..., Option] = partial(
    Option,
    "--no-build-isolation",
    dest="build_isolation",
    action="store_false",
    default=True,
    help="Disable isolation when building a modern source distribution. "
    "Build dependencies specified by PEP 518 must be already installed "
    "if this option is used.",
)

check_build_deps: Callable[..., Option] = partial(
    Option,
    "--check-build-dependencies",
    dest="check_build_deps",
    action="store_true",
    default=False,
    help="Check the build dependencies.",
)


use_pep517: Any = partial(
    Option,
    "--use-pep517",
    dest="use_pep517",
    action="store_true",
    default=True,
    help=SUPPRESS_HELP,
)


def _handle_config_settings(
    option: Option, opt_str: str, value: str, parser: OptionParser
) -> None:
    key, sep, val = value.partition("=")
    if sep != "=":
        parser.error(f"Arguments to {opt_str} must be of the form KEY=VAL")
    dest = getattr(parser.values, option.dest)
    if dest is None:
        dest = {}
        setattr(parser.values, option.dest, dest)
    if key in dest:
        if isinstance(dest[key], list):
            dest[key].append(val)
        else:
            dest[key] = [dest[key], val]
    else:
        dest[key] = val


config_settings: Callable[..., Option] = partial(
    Option,
    "-C",
    "--config-settings",
    dest="config_settings",
    type=str,
    action="callback",
    callback=_handle_config_settings,
    metavar="settings",
    help="Configuration settings to be passed to the build backend. "
    "Settings take the form KEY=VALUE. Use multiple --config-settings options "
    "to pass multiple keys to the backend.",
)

no_clean: Callable[..., Option] = partial(
    Option,
    "--no-clean",
    action="store_true",
    default=False,
    help="Don't clean up build directories.",
)

pre: Callable[..., Option] = partial(
    Option,
    "--pre",
    action="store_true",
    default=False,
    help="Include pre-release and development versions. By default, "
    "pip only finds stable versions.",
)

json: Callable[..., Option] = partial(
    Option,
    "--json",
    action="store_true",
    default=False,
    help="Output data in a machine-readable JSON format.",
)

disable_pip_version_check: Callable[..., Option] = partial(
    Option,
    "--disable-pip-version-check",
    dest="disable_pip_version_check",
    action="store_true",
    default=False,
    help="Don't periodically check PyPI to determine whether a new version "
    "of pip is available for download. Implied with --no-index.",
)

root_user_action: Callable[..., Option] = partial(
    Option,
    "--root-user-action",
    dest="root_user_action",
    default="warn",
    choices=["warn", "ignore"],
    help="Action if pip is run as a root user [warn, ignore] (default: warn)",
)


require_hashes: Callable[..., Option] = partial(
    Option,
    "--require-hashes",
    dest="require_hashes",
    action="store_true",
    default=False,
    help="Require a hash to check each requirement against, for "
    "repeatable installs. This option is implied when any package in a "
    "requirements file has a --hash option.",
)


list_path: Callable[..., Option] = partial(
    PipOption,
    "--path",
    dest="path",
    type="path",
    action="append",
    help="Restrict to the specified installation path for listing "
    "packages (can be used multiple times).",
)


def check_list_path_option(options: Values) -> None:
    if options.path and (options.user or options.local):
        raise CommandError("Cannot combine '--path' with '--user' or '--local'")


list_exclude: Callable[..., Option] = partial(
    PipOption,
    "--exclude",
    dest="excludes",
    action="append",
    metavar="package",
    type="package_name",
    help="Exclude specified package from the output",
)


no_python_version_warning: Callable[..., Option] = partial(
    Option,
    "--no-python-version-warning",
    dest="no_python_version_warning",
    action="store_true",
    default=False,
    help=SUPPRESS_HELP,  # No-op, a hold-over from the Python 2->3 transition.
)


# Features that are now always on. A warning is printed if they are used.
ALWAYS_ENABLED_FEATURES = [
    "truststore",  # always on since 24.2
    "no-binary-enable-wheel-cache",  # always on since 23.1
]

use_new_feature: Callable[..., Option] = partial(
    Option,
    "--use-feature",
    dest="features_enabled",
    metavar="feature",
    action="append",
    default=[],
    choices=[
        "fast-deps",
        "build-constraint",
    ]
    + ALWAYS_ENABLED_FEATURES,
    help="Enable new functionality, that may be backward incompatible.",
)

use_deprecated_feature: Callable[..., Option] = partial(
    Option,
    "--use-deprecated",
    dest="deprecated_features_enabled",
    metavar="feature",
    action="append",
    default=[],
    choices=[
        "legacy-resolver",
        "legacy-certs",
    ],
    help=("Enable deprecated functionality, that will be removed in the future."),
)

##########
# groups #
##########

general_group: dict[str, Any] = {
    "name": "General Options",
    "options": [
        help_,
        debug_mode,
        # isolated_mode,
        # require_virtualenv,
        # python,
        verbose,
        version,
        quiet,
        log,
        no_input,
        keyring_provider,
        proxy,
        retries,
        timeout,
        # exists_action,
        # trusted_host,
        # cert,
        # client_cert,
        cache_dir,
        no_cache,
        # disable_pip_version_check,
        no_color,
        # no_python_version_warning,
        # use_new_feature,
        # use_deprecated_feature,
        resume_retries,
    ],
}

index_group: dict[str, Any] = {
    "name": "Package Index Options",
    "options": [
        index_url,
        extra_index_url,
        no_index,
        find_links,
    ],
}
