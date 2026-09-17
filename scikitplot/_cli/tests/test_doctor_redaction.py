"""
Redaction regressions for ``scikitplot doctor`` (slice S-26).

``doctor`` collects every environment variable under the ``SKPLT_`` and
``SCIKITPLOT_`` prefixes, which is where an integration token or a database
password lives. The contract these checks pin is that values are redacted
unless the caller asks for them at invocation, that an environment variable
cannot make that request on the caller's behalf, and that the output says
which of the two happened.

See Also
--------
scikitplot._cli._commands.doctor.run
"""

import io
import json
import os

import pytest

from .._commands import doctor
from ..context import Context
from ..registry import BUILTIN_COMMANDS

SECRET = "tok_live_PROBE_VALUE"
PLANTED = {
    "SCIKITPLOT_AI_API_TOKEN": SECRET,
    "SKPLT_DB_PASSWORD": "probe-password",
}


@pytest.fixture
def planted_environment(monkeypatch):
    """Place secret-looking variables in the namespace ``doctor`` collects."""
    for key, value in PLANTED.items():
        monkeypatch.setenv(key, value)
    return PLANTED


def _run(**kwargs):
    """Run ``doctor`` into a buffer and return the parsed JSON payload."""
    ctx = Context(stdout=io.StringIO(), stderr=io.StringIO(), fmt="json")
    assert doctor.run(ctx, fmt="json", **kwargs) == 0
    return json.loads(ctx.stdout.getvalue())


def test_values_are_redacted_by_default(planted_environment):
    """No collected value reaches the output unless it was asked for."""
    data = _run()
    rendered = json.dumps(data)
    assert SECRET not in rendered
    assert "probe-password" not in rendered
    assert set(PLANTED) <= set(data["environment"])
    assert all(value == doctor._MASK for value in data["environment"].values())


def test_output_declares_that_values_were_redacted(planted_environment):
    """A consumer can tell a redacted report from a revealed one."""
    assert _run()["environment_values_redacted"] is True
    assert _run(reveal_env_values=True)["environment_values_redacted"] is False


def test_values_are_revealed_only_on_explicit_request(planted_environment):
    """The opt-in still works, and reveals exactly what it says it does."""
    data = _run(reveal_env_values=True)
    assert data["environment"]["SCIKITPLOT_AI_API_TOKEN"] == SECRET


def test_explicit_masking_wins_over_reveal(planted_environment):
    """Asking to mask and to reveal in the same invocation resolves to masking."""
    data = _run(reveal_env_values=True, mask_envs=True)
    assert data["environment"]["SCIKITPLOT_AI_API_TOKEN"] == doctor._MASK
    assert data["environment_values_redacted"] is True


@pytest.mark.parametrize(
    "variable",
    ["SCIKITPLOT_CLI_SHOW_ENV_VALUES", "SKPLT_SHOW_ENV_VALUES",
     "SCIKITPLOT_REVEAL_ENV_VALUES"],
)
def test_environment_cannot_request_exposure(planted_environment, monkeypatch, variable):
    """An inherited variable does not widen exposure on the caller's behalf."""
    monkeypatch.setenv(variable, "1")
    data = _run()
    assert SECRET not in json.dumps(data)
    assert data["environment_values_redacted"] is True


def test_command_spec_exposes_the_opt_in():
    """The shared spec advertises the opt-in, so both frontends get it."""
    spec = next(command for command in BUILTIN_COMMANDS if command.name == "doctor")
    by_dest = {param.dest: param for param in spec.params}
    assert "reveal_env_values" in by_dest
    assert by_dest["reveal_env_values"].default is False
    assert "--show-env-values" in by_dest["reveal_env_values"].flags


def test_capabilities_are_unaffected(planted_environment):
    """Redaction changes values only; the rest of the report is untouched."""
    data = _run()
    assert data["status"] == "ok"
    assert set(data["capabilities"]) >= {"click", "rich", "toml_read", "toml_write"}
    for capability in data["capabilities"].values():
        assert set(capability) == {"available", "provider"}
