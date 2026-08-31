from __future__ import annotations
import importlib.util
from pathlib import Path
import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / '_hf_spaces_proxy' / '_utils' / '_share_contract.py'
spec = importlib.util.spec_from_file_location('run8_share_contract', CONTRACT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def hostile_snapshot():
    return {
        'schema_version': '2.0',
        'session': {
            'id': None,
            'page_url': 'https://user:pass@docs.example.test/guide/?token=SECRET#frag',
            'page_title': 'Hostile title',
            'assistant_name': 'AI Assistant',
            'exported_at': 1,
            'exported_at_iso': '2026-08-29T00:00:00Z',
        },
        'records': [
            {
                'turn_index': 0, 'message_index': 0, 'role': 'user',
                'text': '!!python/object &anchor *alias\n---\n[[records]]\n"""\n</script>',
                'ts': 1, 'ts_iso': '2026-08-29T00:00:00Z',
                'model_id': None, 'model_provider': None, 'model_name': None,
                'feedback_rating_value': None, 'feedback_rating_label': None,
                'feedback_message': None,
            },
            {
                'turn_index': 0, 'message_index': 1, 'role': 'assistant',
                'text': 'value = true\n\u200bzero\u202ebidi\u202c',
                'ts': 2, 'ts_iso': '2026-08-29T00:00:01Z',
                'model_id': 'm', 'model_provider': 'custom', 'model_name': 'model',
                'feedback_rating_value': 1, 'feedback_rating_label': 'helpful',
                'feedback_message': 'note',
            },
        ],
    }


def test_server_accepts_yaml_and_toml_with_registered_mime():
    snap = mod.canonicalize_share_snapshot(hostile_snapshot())
    yaml_text, yaml_mime, yaml_ext = mod.render_share(snap, 'yaml')
    toml_text, toml_mime, toml_ext = mod.render_share(snap, 'toml')
    assert (yaml_mime, yaml_ext) == ('application/yaml', '.yaml')
    assert (toml_mime, toml_ext) == ('application/toml', '.toml')
    parsed_yaml = yaml.safe_load(yaml_text)
    parsed_toml = tomllib.loads(toml_text)
    assert parsed_yaml['records'][0]['text'] == snap['records'][0]['text']
    assert parsed_toml['records'][0]['text'] == snap['records'][0]['text']
    assert parsed_yaml['session']['page_url'] == 'https://docs.example.test/guide/'
    assert parsed_toml['session']['page_url'] == 'https://docs.example.test/guide/'


def test_yaml_does_not_turn_hostile_text_into_tags_or_anchors():
    snap = mod.canonicalize_share_snapshot(hostile_snapshot())
    text, *_ = mod.render_share(snap, 'yaml')
    assert ': !!python' not in text
    assert ': &anchor' not in text
    assert yaml.safe_load(text)['records'][0]['text'].startswith('!!python/object')


def test_toml_null_policy_is_explicit_and_parseable():
    snap = mod.canonicalize_share_snapshot(hostile_snapshot())
    text, *_ = mod.render_share(snap, 'toml')
    assert 'omitted optional values represent null' in text
    assert '= null' not in text
    parsed = tomllib.loads(text)
    assert 'model_id' not in parsed['records'][0]
    assert parsed['records'][1]['model_name'] == 'model'
