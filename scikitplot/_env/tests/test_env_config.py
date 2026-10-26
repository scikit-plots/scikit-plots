# scikitplot/_env/tests/test_env_config.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Requests and configuration documents: validation, layering, formats, batches."""

from __future__ import annotations

import json
import pickle

import pytest

from .._config import _formats
from .._config._schema import find_config, load_batch, load_request
from .._core._errors import ConfigError
from .._core._request import EnvironmentRequest, Policy, SourceRef, infer_source_kind
from ._helpers import write

TOML = """
schema = "scikitplot.env/v1"
name = "demo"
python = "3.12"
packages = ["numpy>=2", "pandas"]
sources = ["requirements.txt"]

[policy]
network = "offline"

[verify]
imports = ["numpy"]
"""

JSON = {
    "schema": "scikitplot.env/v1",
    "name": "demo",
    "python": "3.12",
    "packages": ["numpy>=2", "pandas"],
    "sources": ["requirements.txt"],
    "policy": {"network": "offline"},
    "verify": {"imports": ["numpy"]},
}


class TestRequest:
    def test_exactly_one_of_name_and_prefix(self):
        with pytest.raises(ConfigError, match="exactly one"):
            EnvironmentRequest()
        with pytest.raises(ConfigError, match="exactly one"):
            EnvironmentRequest(name="a", prefix="/tmp/a")

    @pytest.mark.parametrize("name", ["", "-x", "a b", "a/b", "x" * 101])
    def test_bad_names(self, name):
        with pytest.raises(ConfigError):
            EnvironmentRequest(name=name)

    def test_relative_prefix_refused(self):
        with pytest.raises(ConfigError, match="absolute"):
            EnvironmentRequest(prefix="relative/env")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("index_url", "https://user:pw@example.org/simple"),
            ("extra_index_urls", ["https://tok@example.org/simple"]),
            ("channels", ["https://me:secret@conda.example.org/x"]),
        ],
    )
    def test_credentials_in_urls_refused(self, field, value):
        with pytest.raises(ConfigError):
            EnvironmentRequest(name="a", **{field: value})

    def test_bad_python_and_verify(self):
        with pytest.raises(ConfigError, match="python"):
            EnvironmentRequest(name="a", python="three")
        with pytest.raises(ConfigError, match="module name"):
            EnvironmentRequest(name="a", verify_imports=["os; rm"])

    def test_value_object(self):
        request = EnvironmentRequest(name="a", packages=["x"])
        same = EnvironmentRequest(name="a", packages=("x",))
        assert request == same
        assert hash(request) == hash(same)
        assert pickle.loads(pickle.dumps(request)) == request
        assert request.digest == same.digest

    def test_digest_ignores_provenance(self):
        a = EnvironmentRequest(name="a", origins=(("name", "x"),), config_digest="1")
        b = EnvironmentRequest(name="a")
        assert a.digest == b.digest
        assert "origins" in a.to_dict(provenance=True)
        assert "origins" not in a.to_dict()

    def test_policy_choices(self):
        with pytest.raises(ConfigError, match="network"):
            Policy(network="prefer-offline")  # removed on purpose (not deterministic)
        assert Policy(network="offline").offline

    def test_source_kind_inferred_and_checked(self, tmp_path):
        assert infer_source_kind(str(tmp_path / "Pipfile")) == "pipfile"
        assert infer_source_kind(str(tmp_path / "env.yaml")) == "environment"
        assert infer_source_kind("dev-requirements.txt") == "requirements"
        with pytest.raises(ConfigError):
            infer_source_kind("deps.cfg")
        with pytest.raises(ConfigError, match="absolute"):
            SourceRef("relative.txt", "requirements")
        with pytest.raises(ConfigError, match="type"):
            SourceRef(str(tmp_path / "x.txt"), "wheel")


class TestFormats:
    def test_same_request_from_toml_json_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml  # noqa: PLC0415

        files = [
            write(tmp_path / "a" / "scikitplot-env.toml", TOML),
            write(tmp_path / "b" / "scikitplot-env.json", json.dumps(JSON)),
            write(tmp_path / "c" / "scikitplot-env.yaml", yaml.safe_dump(JSON)),
        ]
        requests = [load_request(f) for f in files]
        # relative sources resolve against each file's directory
        assert [r.sources[0].path for r in requests] == [
            str(f.parent / "requirements.txt") for f in files
        ]
        stripped = [
            EnvironmentRequest(**{**r.__dict__, "sources": (), "origins": (),
                                  "config_digest": None})
            for r in requests
        ]
        assert len({r.digest for r in stripped}) == 1
        assert requests[0].policy.network == "offline"
        assert requests[0].verify_imports == ("numpy",)

    def test_pyproject_table(self, tmp_path):
        path = write(
            tmp_path / "pyproject.toml",
            '[project]\nname = "p"\n[tool.scikitplot.env]\nname = "proj"\n',
        )
        assert load_request(path).name == "proj"
        (tmp_path / "sub").mkdir()
        assert find_config(tmp_path / "sub") == path

    def test_pyproject_without_table_is_not_a_config(self, tmp_path):
        write(tmp_path / "pyproject.toml", '[project]\nname = "p"\n')
        assert find_config(tmp_path) is None

    def test_json_duplicate_and_nan_refused(self):
        with pytest.raises(ConfigError, match="duplicate"):
            _formats.loads('{"name": "a", "name": "b"}', "json")
        with pytest.raises(ConfigError, match="non-standard"):
            _formats.loads('{"x": NaN}', "json")

    def test_yaml_aliases_and_duplicates_refused(self):
        pytest.importorskip("yaml")
        with pytest.raises(ConfigError, match="alias"):
            _formats.loads("a: &x [1]\nb: *x\n", "yaml")
        with pytest.raises(ConfigError, match="duplicate"):
            _formats.loads("a: 1\na: 2\n", "yaml")

    def test_unknown_and_authority_keys(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown key"):
            load_request({"name": "a", "pkgs": []}, base_dir=tmp_path)
        with pytest.raises(ConfigError) as info:
            load_request({"name": "a", "approve": ["remove"]}, base_dir=tmp_path)
        assert info.value.code == "config_authority"
        with pytest.raises(ConfigError) as info:
            load_request({"name": "a", "policy": {"yes": True}}, base_dir=tmp_path)
        assert info.value.code == "config_authority"


class TestLayers:
    def test_precedence_and_origins(self, tmp_path):
        path = write(
            tmp_path / "scikitplot-env.toml",
            'name = "doc"\nprofile = "airgap"\n[policy]\ninteraction = "always"\n',
        )
        request = load_request(path, name="cli", policy={"bootstrap": "if-missing"})
        origins = dict(request.origins)
        assert request.name == "cli"
        assert origins["name"] == "invocation"
        assert request.policy.network == "offline"
        assert origins["policy.network"] == "preset:airgap"
        assert request.policy.interaction == "always"
        assert origins["policy.interaction"].startswith("config:")
        assert request.policy.bootstrap == "if-missing"
        assert request.config_digest is not None

    def test_unknown_profile(self, tmp_path):
        with pytest.raises(ConfigError, match="profile"):
            load_request({"name": "a", "profile": "fast"}, base_dir=tmp_path)

    def test_verify_imports_spelling(self):
        request = load_request(None, name="a", verify_imports=["json"])
        assert request.verify_imports == ("json",)


class TestBatch:
    def test_load_batch(self, tmp_path):
        lines = [
            {"schema": "scikitplot.env.request/v1", "id": "one",
             "request": {"name": "one", "sources": ["r.txt"]}},
            {"schema": "scikitplot.env.request/v1", "id": "two",
             "request": {"name": "two"}},
        ]
        path = write(tmp_path / "b.jsonl", "\n".join(json.dumps(x) for x in lines))
        out = load_batch(path)
        assert [rid for rid, _ in out] == ["one", "two"]
        assert out[0][1].sources[0].path == str(tmp_path / "r.txt")

    @pytest.mark.parametrize(
        ("line", "match"),
        [
            ({"schema": "x", "id": "a", "request": {"name": "a"}}, "schema"),
            ({"schema": "scikitplot.env.request/v1", "request": {"name": "a"}}, "id"),
            ({"schema": "scikitplot.env.request/v1", "id": "a", "request": {"x": 1}},
             "unknown key"),
        ],
    )
    def test_bad_lines_name_the_line(self, tmp_path, line, match):
        path = write(tmp_path / "b.jsonl", json.dumps(line) + "\n")
        with pytest.raises(ConfigError, match=match) as info:
            load_batch(path)
        assert f"{path}:1" in str(info.value)

    def test_duplicate_ids(self, tmp_path):
        line = {"schema": "scikitplot.env.request/v1", "id": "a",
                "request": {"name": "a"}}
        path = write(tmp_path / "b.jsonl", (json.dumps(line) + "\n") * 2)
        with pytest.raises(ConfigError, match=":2"):
            load_batch(path)


def test_discovery_stops_at_the_project_root(tmp_path):
    write(tmp_path / "scikitplot-env.toml", 'name = "someone-else"\n')
    write(tmp_path / "proj" / "pyproject.toml", '[project]\nname = "p"\n')
    (tmp_path / "proj" / "sub").mkdir()
    assert find_config(tmp_path / "proj" / "sub") is None
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    assert find_config(tmp_path / "repo") is None
