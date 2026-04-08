from __future__ import annotations

"""Tests for project config discovery and config loading.

These tests also cover deterministic project marker customization (env + context manager).
"""

from pathlib import Path
import pytest

from scikitplot.mlflow._project import (
    find_project_root,
    load_project_config_toml,
    load_project_config,
)


def test_find_project_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path.resolve()


def test_find_project_root_raises(tmp_path: Path) -> None:
    d = tmp_path / "x"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        find_project_root(d, markers=("does_not_exist.marker",))


def test_load_toml_normalizes_paths_and_empty_strings(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    toml_path = cfg_dir / "mlflow.toml"

    toml_path.write_text(
        '''
[profiles.local]
start_server = true

[profiles.local.session]
tracking_uri = "http://127.0.0.1:5001"
registry_uri = ""
startup_timeout_s = 1.0

[profiles.local.server]
host = "127.0.0.1"
port = 5001
backend_store_uri = "sqlite:///./.mlflow/mlflow.db"
default_artifact_root = "./.mlflow/artifacts"
serve_artifacts = true
strict_cli_compat = false
''',
        encoding="utf-8",
    )

    cfg = load_project_config_toml(toml_path, profile="local")
    assert cfg.start_server is True
    assert cfg.session.registry_uri is None

    assert cfg.server is not None
    assert cfg.server.backend_store_uri is not None
    assert cfg.server.backend_store_uri.startswith("sqlite:///")

    db_path = Path(cfg.server.backend_store_uri[len("sqlite:///"):])
    assert db_path.is_absolute()
    assert str(tmp_path.resolve()) in str(db_path)

    assert cfg.server.default_artifact_root is not None
    art_path = Path(cfg.server.default_artifact_root)
    assert art_path.is_absolute()
    assert art_path.exists()


def test_load_project_config_dispatch_unsupported(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_project_config(p)


def test_project_markers_env_override(monkeypatch) -> None:
    """Environment variable override is strict JSON list."""
    import json as _json
    import scikitplot.mlflow._project as p

    monkeypatch.setenv("SCIKITPLOT_PROJECT_MARKERS", _json.dumps(["pyproject.toml", ".git", "configs/mlflow.toml"]))
    markers = p.get_project_markers()
    assert markers == ("pyproject.toml", ".git", "configs/mlflow.toml")


def test_project_markers_env_invalid_json_raises(monkeypatch) -> None:
    import pytest
    import scikitplot.mlflow._project as p

    monkeypatch.setenv("SCIKITPLOT_PROJECT_MARKERS", "not-json")
    with pytest.raises(ValueError):
        p.get_project_markers()


def test_project_markers_context_manager_restores(monkeypatch) -> None:
    import scikitplot.mlflow._project as p

    base = p.get_project_markers()
    with p.project_markers(["X.marker"]):
        assert p.get_project_markers() == ("X.marker",)
    assert p.get_project_markers() == base


def test_project_markers_setter_reset() -> None:
    import scikitplot.mlflow._project as p

    p.set_project_markers(["A", "B"])
    assert p.get_project_markers() == ("A", "B")
    p.set_project_markers(None)
    assert p.get_project_markers() == p.DEFAULT_PROJECT_MARKERS


# ===========================================================================
# Gap-fill: _validate_markers error paths (lines 133, 137, 140)
# ===========================================================================


class TestValidateMarkers:
    """Tests for _validate_markers validation logic."""

    def test_valid_markers_returns_tuple(self) -> None:
        from scikitplot.mlflow._project import _validate_markers
        result = _validate_markers(["pyproject.toml", ".git"])
        assert result == ("pyproject.toml", ".git")

    def test_string_raises_type_error(self) -> None:
        """Passing a bare string (not a list) must raise TypeError (line 133)."""
        from scikitplot.mlflow._project import _validate_markers
        with pytest.raises(TypeError):
            _validate_markers("pyproject.toml")  # type: ignore[arg-type]

    def test_blank_string_in_list_raises_type_error(self) -> None:
        """Blank strings in the list must raise TypeError (line 137)."""
        from scikitplot.mlflow._project import _validate_markers
        with pytest.raises(TypeError):
            _validate_markers(["  "])

    def test_non_string_item_raises_type_error(self) -> None:
        from scikitplot.mlflow._project import _validate_markers
        with pytest.raises(TypeError):
            _validate_markers([123])  # type: ignore[list-item]

    def test_empty_list_raises_value_error(self) -> None:
        """Empty list must raise ValueError (line 140)."""
        from scikitplot.mlflow._project import _validate_markers
        with pytest.raises(ValueError):
            _validate_markers([])

    def test_single_valid_marker_accepted(self) -> None:
        from scikitplot.mlflow._project import _validate_markers
        assert _validate_markers(["pyproject.toml"]) == ("pyproject.toml",)


# ===========================================================================
# Gap-fill: _is_windows_drive_path (lines 346-351)
# ===========================================================================


class TestIsWindowsDrivePath:
    """Tests for the Windows drive path detection helper."""

    def test_valid_windows_drive_accepted(self) -> None:
        from scikitplot.mlflow._project import _is_windows_drive_path
        assert _is_windows_drive_path(r"C:\path\to\file") is True

    def test_windows_drive_forward_slash(self) -> None:
        from scikitplot.mlflow._project import _is_windows_drive_path
        assert _is_windows_drive_path("C:/path") is True

    def test_drive_letter_only(self) -> None:
        from scikitplot.mlflow._project import _is_windows_drive_path
        assert _is_windows_drive_path("C:") is True

    def test_single_char_string_is_false(self) -> None:
        """Length < 2 must return False (line 346)."""
        from scikitplot.mlflow._project import _is_windows_drive_path
        assert _is_windows_drive_path("C") is False

    def test_empty_string_is_false(self) -> None:
        from scikitplot.mlflow._project import _is_windows_drive_path
        assert _is_windows_drive_path("") is False

    def test_non_alpha_first_char_is_false(self) -> None:
        """First character not alphabetic must return False (line 348)."""
        from scikitplot.mlflow._project import _is_windows_drive_path
        assert _is_windows_drive_path("1:/path") is False

    def test_no_colon_at_index_1_is_false(self) -> None:
        from scikitplot.mlflow._project import _is_windows_drive_path
        assert _is_windows_drive_path("Cx/path") is False

    def test_posix_path_is_false(self) -> None:
        from scikitplot.mlflow._project import _is_windows_drive_path
        assert _is_windows_drive_path("/usr/local") is False


# ===========================================================================
# Gap-fill: _is_probably_local_path edge cases (lines 366, 368)
# ===========================================================================


class TestIsProbablyLocalPathEdgeCases:
    """Tests for the remaining _is_probably_local_path branches."""

    def test_windows_drive_path_is_local(self) -> None:
        """Windows drive paths like C:\\ must be treated as local (line 366/368)."""
        from scikitplot.mlflow._project import _is_probably_local_path
        # Windows drive paths should NOT match the scheme regex as remote URIs
        # Even if on Linux, the logic path for 'C:' should return True
        assert _is_probably_local_path(r"C:\mlruns") is True

    def test_path_without_scheme_is_local(self) -> None:
        from scikitplot.mlflow._project import _is_probably_local_path
        assert _is_probably_local_path("relative/path") is True

    def test_double_slash_uri_is_not_local(self) -> None:
        from scikitplot.mlflow._project import _is_probably_local_path
        assert _is_probably_local_path("s3://bucket") is False

    def test_single_colon_scheme_is_not_local(self) -> None:
        """e.g. 'dbfs:/path' — has scheme but no '//' (line 366)."""
        from scikitplot.mlflow._project import _is_probably_local_path
        assert _is_probably_local_path("dbfs:/path/to") is False

    def test_file_uri_is_not_local(self) -> None:
        from scikitplot.mlflow._project import _is_probably_local_path
        assert _is_probably_local_path("file:///tmp/x") is False


# ===========================================================================
# Gap-fill: _normalize_sqlite_uri bad prefix raises (line 394)
# ===========================================================================


class TestNormalizeSqliteUri:
    """Tests for _normalize_sqlite_uri error path."""

    def test_bad_prefix_raises_value_error(self, tmp_path: Path) -> None:
        """Non-sqlite:/// URI must raise ValueError (line 394)."""
        from scikitplot.mlflow._project import _normalize_sqlite_uri
        with pytest.raises(ValueError, match="sqlite"):
            _normalize_sqlite_uri("postgresql://user/db", base_dir=tmp_path)

    def test_relative_path_becomes_absolute(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import _normalize_sqlite_uri
        result = _normalize_sqlite_uri("sqlite:///./mlflow.db", base_dir=tmp_path)
        assert result.startswith("sqlite:///")
        assert Path(result[len("sqlite:///"):]).is_absolute()

    def test_already_absolute_path_preserved(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import _normalize_sqlite_uri
        abs_path = str(tmp_path / "mlflow.db")
        result = _normalize_sqlite_uri(f"sqlite:///{abs_path}", base_dir=tmp_path)
        assert abs_path in result


# ===========================================================================
# Gap-fill: normalize_mlflow_store_values resolve paths (lines 400, 417, 459-460, 498-499)
# ===========================================================================


class TestNormalizeMlflowStoreValues:
    """Tests for normalize_mlflow_store_values path resolution branches."""

    def test_sqlite_backend_relative_becomes_absolute(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import normalize_mlflow_store_values
        b, a = normalize_mlflow_store_values(
            backend_store_uri="sqlite:///./mlflow.db",
            default_artifact_root=None,
            base_dir=tmp_path,
        )
        assert b is not None
        assert Path(b[len("sqlite:///"):]).is_absolute()

    def test_local_artifact_root_becomes_absolute(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import normalize_mlflow_store_values
        b, a = normalize_mlflow_store_values(
            backend_store_uri=None,
            default_artifact_root="./artifacts",
            base_dir=tmp_path,
        )
        assert a is not None and Path(a).is_absolute()

    def test_local_backend_store_uri_becomes_absolute(self, tmp_path: Path) -> None:
        """A bare local backend_store_uri path must be resolved (line 459-460)."""
        from scikitplot.mlflow._project import normalize_mlflow_store_values
        b, a = normalize_mlflow_store_values(
            backend_store_uri="mlruns",
            default_artifact_root=None,
            base_dir=tmp_path,
        )
        assert b is not None and Path(b).is_absolute()

    def test_local_artifact_root_without_sqlite(self, tmp_path: Path) -> None:
        """Local artifact root that is not sqlite:// (line 498-499)."""
        from scikitplot.mlflow._project import normalize_mlflow_store_values
        b, a = normalize_mlflow_store_values(
            backend_store_uri=None,
            default_artifact_root="my_artifacts",
            base_dir=tmp_path,
        )
        assert a is not None and Path(a).is_absolute()

    def test_remote_values_untouched(self) -> None:
        from scikitplot.mlflow._project import normalize_mlflow_store_values
        b, a = normalize_mlflow_store_values(
            backend_store_uri="s3://bucket/mlflow",
            default_artifact_root="gs://bucket/artifacts",
            base_dir=Path("."),
        )
        assert b == "s3://bucket/mlflow"
        assert a == "gs://bucket/artifacts"


# ===========================================================================
# Gap-fill: _safe_construct non-dataclass fallback (lines 571-579)
# ===========================================================================


class TestSafeConstruct:
    """Tests for _safe_construct filtering kwargs to dataclass and non-dataclass."""

    def test_dataclass_path_filters_unknown_kwargs(self) -> None:
        """Unknown kwargs must be silently dropped (line 571-574)."""
        from scikitplot.mlflow._project import _construct_dataclass as _safe_construct
        from scikitplot.mlflow._config import SessionConfig
        obj = _safe_construct(
            SessionConfig,
            tracking_uri="http://x:5000",
            unknown_field_xyz="should_be_dropped",
        )
        assert obj.tracking_uri == "http://x:5000"
        assert not hasattr(obj, "unknown_field_xyz")

    def test_non_dataclass_fallback(self) -> None:
        """Non-dataclass class must use signature introspection (lines 576-579)."""
        from scikitplot.mlflow._project import _construct_dataclass as _safe_construct

        class Simple:
            def __init__(self, a: int, b: str = "default") -> None:
                self.a = a
                self.b = b

        obj = _safe_construct(Simple, a=42, b="hello", unknown="ignored")
        assert obj.a == 42
        assert obj.b == "hello"
        assert not hasattr(obj, "unknown")


# ===========================================================================
# Gap-fill: _norm_str type error (line 606) and _coerce_mapping error (line 634)
# ===========================================================================


class TestNormStr:
    """Tests for _norm_str strict type enforcement."""

    def test_non_str_non_none_raises_type_error(self) -> None:
        """Passing an integer must raise TypeError (line 606)."""
        from scikitplot.mlflow._project import _norm_str
        with pytest.raises(TypeError):
            _norm_str(42)  # type: ignore[arg-type]

    def test_empty_string_returns_none(self) -> None:
        from scikitplot.mlflow._project import _norm_str
        assert _norm_str("") is None

    def test_none_returns_none(self) -> None:
        from scikitplot.mlflow._project import _norm_str
        assert _norm_str(None) is None

    def test_valid_string_returned_unchanged(self) -> None:
        from scikitplot.mlflow._project import _norm_str
        assert _norm_str("hello") == "hello"


class TestCoerceMapping:
    """Tests for _coerce_mapping type enforcement."""

    def test_list_raises_type_error(self) -> None:
        """Non-dict value must raise TypeError (line 634)."""
        from scikitplot.mlflow._project import _coerce_mapping
        with pytest.raises(TypeError, match="mapping"):
            _coerce_mapping([1, 2], name="test")

    def test_none_returns_empty_dict(self) -> None:
        from scikitplot.mlflow._project import _coerce_mapping
        assert _coerce_mapping(None, name="x") == {}

    def test_dict_returned_unchanged(self) -> None:
        from scikitplot.mlflow._project import _coerce_mapping
        d = {"a": 1}
        assert _coerce_mapping(d, name="x") == {"a": 1}


# ===========================================================================
# Gap-fill: _build_project_config missing 'profiles' table (line 672)
# ===========================================================================


class TestBuildProjectConfigFromMapping:
    """Tests for _build_project_config_from_mapping error paths."""

    def test_missing_profiles_table_raises_key_error(self) -> None:
        """No 'profiles' key → KeyError (line 672)."""
        from scikitplot.mlflow._project import _build_project_config_from_mapping
        with pytest.raises(KeyError, match="profiles"):
            import tempfile
            from pathlib import Path
            _build_project_config_from_mapping({"other": {}}, profile="local",
                                               project_root=Path(tempfile.mkdtemp()))

    def test_missing_profile_entry_raises_key_error(self) -> None:
        """profiles dict has no 'local' key → KeyError."""
        from scikitplot.mlflow._project import _build_project_config_from_mapping
        with pytest.raises(KeyError):
            import tempfile
            from pathlib import Path
            _build_project_config_from_mapping(
                {"profiles": {"dev": {}}}, profile="local",
                project_root=Path(tempfile.mkdtemp()),
            )

    def test_extra_env_not_dict_raises_type_error(self, tmp_path: Path) -> None:
        """extra_env must be a mapping; list raises TypeError (line 685)."""
        from scikitplot.mlflow._project import _build_project_config_from_mapping
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        with pytest.raises(TypeError):
            _build_project_config_from_mapping(
                {
                    "profiles": {
                        "local": {
                            "start_server": False,
                            "session": {"extra_env": ["bad"]},
                        }
                    }
                },
                profile="local",
                project_root=tmp_path,
            )

    def test_default_run_tags_not_dict_raises_type_error(self, tmp_path: Path) -> None:
        """default_run_tags must be a mapping; string raises TypeError (line 690)."""
        from scikitplot.mlflow._project import _build_project_config_from_mapping
        with pytest.raises(TypeError):
            _build_project_config_from_mapping(
                {
                    "profiles": {
                        "local": {
                            "session": {"default_run_tags": "bad"},
                        }
                    }
                },
                profile="local",
                project_root=tmp_path,
            )


# ===========================================================================
# Gap-fill: load_project_config_yaml missing file / bad YAML (lines 794, 836, 840-841, 849)
# ===========================================================================


class TestLoadProjectConfigYaml:
    """Tests for load_project_config_yaml error paths."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent path must raise FileNotFoundError (line 794)."""
        from scikitplot.mlflow._project import load_project_config_yaml
        with pytest.raises(FileNotFoundError):
            load_project_config_yaml(tmp_path / "ghost.yaml", profile="local")

    def test_yaml_not_mapping_raises_value_error(self, tmp_path: Path) -> None:
        """YAML that parses to a list must raise ValueError (line 849)."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")
        from scikitplot.mlflow._project import load_project_config_yaml
        f = tmp_path / "bad.yaml"
        f.write_text("- item1\n- item2\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_project_config_yaml(f, profile="local")


# ===========================================================================
# Gap-fill: dump_project_config_yaml explicit + convenience paths (lines 947-955, 984-985)
# ===========================================================================


class TestDumpProjectConfigYaml:
    """Tests for dump_project_config_yaml error paths."""

    def test_explicit_mode_cfg_without_path_raises(self, tmp_path: Path) -> None:
        """cfg provided without path must raise ValueError (line 947)."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")
        from scikitplot.mlflow._project import dump_project_config_yaml, load_project_config_toml
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        toml_path = cfg_dir / "mlflow.toml"
        toml_path.write_text(
            "[profiles.local]\nstart_server = false\n[profiles.local.session]\ntracking_uri = \"http://127.0.0.1:5000\"\n",
            encoding="utf-8",
        )
        cfg = load_project_config_toml(toml_path, profile="local")
        with pytest.raises(ValueError, match="path"):
            dump_project_config_yaml(cfg, path=None)

    def test_roundtrip_toml_to_yaml(self, tmp_path: Path) -> None:
        """TOML→YAML roundtrip must produce a file that YAML can load."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")
        from scikitplot.mlflow._project import (
            dump_project_config_yaml,
            load_project_config_toml,
            load_project_config_yaml,
        )
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        toml_path = cfg_dir / "mlflow.toml"
        toml_path.write_text(
            "[profiles.local]\nstart_server = false\n[profiles.local.session]\ntracking_uri = \"http://127.0.0.1:5000\"\n",
            encoding="utf-8",
        )
        yaml_path = cfg_dir / "mlflow.yaml"
        cfg = load_project_config_toml(toml_path, profile="local")
        dump_project_config_yaml(cfg, path=yaml_path)
        assert yaml_path.exists()
        # Load back to verify it's valid YAML with correct content
        cfg2 = load_project_config_yaml(yaml_path, profile="local")
        assert cfg2.session.tracking_uri == "http://127.0.0.1:5000"


# ===========================================================================
# Gap-fill: _markers_from_toml (lines 196-203) and get_project_markers
#           with config_path (lines 227-229)
# ===========================================================================


class TestMarkersFromToml:
    """Tests for _markers_from_toml and get_project_markers with config_path."""

    def test_markers_from_toml_returns_tuple_when_present(
        self, tmp_path: Path
    ) -> None:
        """[project].markers in pyproject.toml must be returned as a tuple (line 203)."""
        from scikitplot.mlflow._project import _markers_from_toml

        cfg = tmp_path / "pyproject.toml"
        cfg.write_text(
            '[project]\nmarkers = ["pyproject.toml", ".git"]\n', encoding="utf-8"
        )
        result = _markers_from_toml(cfg)
        assert result == ("pyproject.toml", ".git")

    def test_markers_from_toml_returns_none_when_no_project_section(
        self, tmp_path: Path
    ) -> None:
        """Missing [project] section returns None (line 198-199)."""
        from scikitplot.mlflow._project import _markers_from_toml

        cfg = tmp_path / "pyproject.toml"
        cfg.write_text("[tool.ruff]\nline-length = 100\n", encoding="utf-8")
        assert _markers_from_toml(cfg) is None

    def test_markers_from_toml_returns_none_when_no_markers_key(
        self, tmp_path: Path
    ) -> None:
        """[project] without markers key returns None (line 201-202)."""
        from scikitplot.mlflow._project import _markers_from_toml

        cfg = tmp_path / "pyproject.toml"
        cfg.write_text('[project]\nname = "myproject"\n', encoding="utf-8")
        assert _markers_from_toml(cfg) is None

    def test_get_project_markers_reads_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_project_markers(config_path=...) must read markers from the TOML (lines 227-229)."""
        from scikitplot.mlflow._project import get_project_markers

        cfg = tmp_path / "pyproject.toml"
        cfg.write_text(
            '[project]\nmarkers = ["mymarker.yaml"]\n', encoding="utf-8"
        )
        # Suppress env override
        monkeypatch.delenv("SCIKITPLOT_PROJECT_MARKERS", raising=False)
        result = get_project_markers(config_path=cfg)
        assert result == ("mymarker.yaml",)

    def test_get_project_markers_falls_through_when_toml_has_no_markers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If config_path has no markers, fall through to env/default (lines 228-229)."""
        from scikitplot.mlflow._project import get_project_markers, DEFAULT_PROJECT_MARKERS

        cfg = tmp_path / "pyproject.toml"
        cfg.write_text("[tool.myapp]\nfoo = 1\n", encoding="utf-8")
        monkeypatch.delenv("SCIKITPLOT_PROJECT_MARKERS", raising=False)
        result = get_project_markers(config_path=cfg)
        assert result == DEFAULT_PROJECT_MARKERS


# ===========================================================================
# Gap-fill: _normalize_path_like absolute path branch (line 417)
# ===========================================================================


class TestNormalizePathLikeAbsolute:
    """Tests for the absolute-path branch in _normalize_path_like (line 417)."""

    def test_absolute_path_is_resolved(self, tmp_path: Path) -> None:
        """An already-absolute path must be resolved (line 417)."""
        from scikitplot.mlflow._project import _normalize_path_like

        abs_path = str(tmp_path.resolve())
        result = _normalize_path_like(abs_path, base_dir=Path("/other"))
        # Result must equal the resolved absolute path
        assert Path(result) == tmp_path.resolve()

    def test_relative_path_resolved_against_base(self, tmp_path: Path) -> None:
        """Relative path must be resolved relative to base_dir."""
        from scikitplot.mlflow._project import _normalize_path_like

        result = _normalize_path_like("sub/dir", base_dir=tmp_path)
        assert Path(result) == (tmp_path / "sub" / "dir").resolve()


# ===========================================================================
# Gap-fill: ensure_local_store_layout bare local path (lines 498-499)
# ===========================================================================


class TestEnsureLocalStoreLayoutLocalPath:
    """Tests for the bare local path mkdir branch (lines 498-499)."""

    def test_bare_local_backend_store_creates_directory(self, tmp_path: Path) -> None:
        """
        When backend_store_uri is a bare local path (not sqlite:///) and not remote,
        ensure_local_store_layout must create the directory (lines 498-499).
        """
        from scikitplot.mlflow._project import ensure_local_store_layout
        import os

        local_store = str(tmp_path / "my_mlruns")
        ensure_local_store_layout(
            backend_store_uri=local_store,
            default_artifact_root=None,
        )
        assert (tmp_path / "my_mlruns").exists()


# ===========================================================================
# Gap-fill: load_project_config_yaml missing file (line 794) and yaml
#           ImportError (lines 840-841)
# ===========================================================================


class TestLoadProjectConfigYamlErrors:
    """Tests for load_project_config_yaml error branches."""

    def test_missing_yaml_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """File must exist; if not → FileNotFoundError (line 794)."""
        from scikitplot.mlflow._project import load_project_config_yaml

        with pytest.raises(FileNotFoundError):
            load_project_config_yaml(tmp_path / "ghost.yaml", profile="local")

    def test_missing_pyyaml_raises_import_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If PyYAML is not importable, raise ImportError (lines 840-841)."""
        import builtins

        real_import = builtins.__import__

        def _blocking_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        from scikitplot.mlflow._project import load_project_config_yaml

        f = tmp_path / "mlflow.yaml"
        f.write_text("profiles:\n  local:\n    session: {}\n", encoding="utf-8")
        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        with pytest.raises(ImportError, match="PyYAML"):
            load_project_config_yaml(f, profile="local")


# ===========================================================================
# Gap-fill: dump_project_config_yaml convenience FileNotFoundError (line 955)
#           and yaml ImportError (lines 984-985)
# ===========================================================================


class TestDumpProjectConfigYamlErrors:
    """Tests for dump_project_config_yaml error branches."""

    def test_convenience_mode_missing_source_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When cfg=None and source_config_path is absent, raise FileNotFoundError
        (line 955).  Requires find_project_root to return tmp_path.
        """
        from scikitplot.mlflow._project import dump_project_config_yaml
        import scikitplot.mlflow._project as _proj

        # No configs/mlflow.toml exists in tmp_path
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        monkeypatch.setattr(_proj, "find_project_root", lambda *a, **k: tmp_path)

        with pytest.raises(FileNotFoundError):
            dump_project_config_yaml(cfg=None, path=None)

    def test_explicit_mode_missing_pyyaml_raises_import_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit mode (cfg+path) without PyYAML raises ImportError (lines 984-985)."""
        import builtins
        from scikitplot.mlflow._project import (
            dump_project_config_yaml,
            load_project_config_toml,
        )

        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        toml_path = cfg_dir / "mlflow.toml"
        toml_path.write_text(
            "[profiles.local]\nstart_server = false\n"
            "[profiles.local.session]\ntracking_uri = \"http://127.0.0.1:5000\"\n",
            encoding="utf-8",
        )
        cfg = load_project_config_toml(toml_path, profile="local")
        yaml_out = tmp_path / "out.yaml"

        real_import = builtins.__import__

        def _blocking_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        with pytest.raises(ImportError, match="PyYAML"):
            dump_project_config_yaml(cfg, path=yaml_out)


# ===========================================================================
# Gap-fill: dump_project_config_yaml with server config (lines 1022-1023)
# ===========================================================================


class TestDumpProjectConfigYamlWithServer:
    """Tests for dump serialising a ServerConfig section (lines 1022-1023)."""

    def test_dump_includes_server_section_when_server_present(
        self, tmp_path: Path
    ) -> None:
        """When cfg.server is not None, the output YAML must contain a server dict."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")

        from scikitplot.mlflow._project import (
            dump_project_config_yaml,
            load_project_config_toml,
        )

        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        toml_path = cfg_dir / "mlflow.toml"
        toml_path.write_text(
            "[profiles.local]\nstart_server = true\n"
            "[profiles.local.session]\ntracking_uri = \"http://127.0.0.1:5000\"\n"
            "[profiles.local.server]\nhost = \"127.0.0.1\"\nport = 5000\n"
            "strict_cli_compat = false\n",
            encoding="utf-8",
        )
        cfg = load_project_config_toml(toml_path, profile="local")
        assert cfg.server is not None  # precondition

        yaml_out = tmp_path / "mlflow.yaml"
        dump_project_config_yaml(cfg, path=yaml_out)

        content = yaml_out.read_text(encoding="utf-8")
        assert "server" in content
        assert yaml_out.exists()
