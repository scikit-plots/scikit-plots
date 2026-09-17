"""
Grant regressions for a non-local bind (slice S-24).

An unauthenticated bind to a non-local interface widens exposure. It was
grantable by ``SCIKITPLOT_MCP_ALLOW_UNAUTHENTICATED_REMOTE`` or by
``SCIKITPLOT_MCP_DOCKER`` alone, so an inherited environment could open the
port with no operator action at the command line.

See Also
--------
scikitplot.mcp.__main__._resolve_config
"""

import pytest

from .. import __main__ as entry

REMOTE = ["--transport", "streamable-http", "--host", "0.0.0.0"]
GRANTING_VARIABLES = [
    "SCIKITPLOT_MCP_ALLOW_UNAUTHENTICATED_REMOTE",
    "SCIKITPLOT_MCP_DOCKER",
]


def _resolve(argv, env=None):
    """
    Resolve a runtime configuration from an argument vector and environment.

    Notes
    -----
    The non-local bind guard lives inside ``_resolve_config``, so a refused
    bind surfaces here as :class:`SystemExit` rather than as a returned
    configuration.
    """
    return entry._resolve_config(entry._parser().parse_args(argv), environ=env or {})


@pytest.mark.parametrize("variable", GRANTING_VARIABLES)
def test_environment_alone_does_not_grant(variable):
    """An inherited variable configures; it does not authorise, so the bind is refused."""
    with pytest.raises(SystemExit):
        _resolve(REMOTE, {variable: "1"})


@pytest.mark.parametrize("flag", ["--allow-unauthenticated-remote", "--docker"])
def test_an_invocation_flag_grants(flag):
    """The operator can still grant it, where the grant is visible."""
    assert _resolve([*REMOTE, flag]).allow_unauthenticated_remote is True


@pytest.mark.parametrize("variable", GRANTING_VARIABLES)
def test_refusal_names_the_flag_to_add(variable):
    """A refused bind says exactly what to pass, not just that it was refused."""
    with pytest.raises(SystemExit) as excinfo:
        _resolve(REMOTE, {variable: "1"})
    message = str(excinfo.value)
    assert "--allow-unauthenticated-remote" in message
    assert "--docker" in message


def test_local_bind_is_unaffected_by_the_environment():
    """A loopback bind never needed the grant and still does not."""
    config = _resolve(["--transport", "streamable-http", "--host", "127.0.0.1"],
                      {"SCIKITPLOT_MCP_ALLOW_UNAUTHENTICATED_REMOTE": "1"})
    assert config.host == "127.0.0.1"
    assert config.allow_unauthenticated_remote is False


def test_docker_mode_still_selects_its_transport_from_the_environment():
    """The variable keeps its configuring role; only the authorising role is removed."""
    config = _resolve(["--host", "127.0.0.1"], {"SCIKITPLOT_MCP_DOCKER": "1"})
    assert config.docker is True
    assert config.transport == "streamable-http"
    assert config.allow_unauthenticated_remote is False
