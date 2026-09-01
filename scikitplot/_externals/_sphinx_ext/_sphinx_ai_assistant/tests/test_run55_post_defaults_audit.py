from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "_hf_spaces_proxy" / "app.py"
SCHEMA = ROOT / "_hf_spaces_proxy" / "_utils" / "_dataset_schema.py"


def test_runtime_rejects_retired_feedback_lineage_aliases_instead_of_reading_them():
    app = APP.read_text(encoding="utf-8")
    validator = app[app.index("def _validate_feedback_lineage_fields("):]
    validator = validator[: validator.index("\n\ndef ", 10)]
    assert 'if "sessionId" in container or "prevSessionId" in container:' in validator
    assert 'container.get("sessionId")' not in validator
    assert 'container.get("prevSessionId")' not in validator


def test_offline_historical_normalizer_drops_but_does_not_migrate_feedback_aliases():
    schema = SCHEMA.read_text(encoding="utf-8")
    assert 'out.pop("sessionId", None)' in schema
    assert 'out.pop("prevSessionId", None)' in schema
    assert 'out["feedbackId"] = out.pop("sessionId")' not in schema
    assert 'out["prevFeedbackId"] = out.pop("prevSessionId")' not in schema
