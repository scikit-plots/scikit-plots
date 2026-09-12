from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT
import importlib.util
from pathlib import Path
import tomllib
import yaml

ROOT = RUNTIME_ROOT
CONTRACT = ROOT / '_hf_spaces_proxy' / '_utils' / '_share_contract.py'
spec = importlib.util.spec_from_file_location('run8_share_contract', CONTRACT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def hostile_snapshot():
    return {
        'schema_version': '2.1',
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
                'resources': {
                    'version': 2, 'totalCount': 1, 'includedCount': 1, 'localOnlyCount': 0,
                    'contextCount': 1, 'rawCount': 0, 'notSentCount': 0, 'pageCount': 1,
                    'replayCount': 0, 'totalBytes': 42, 'itemCount': 1, 'omittedCount': 0,
                    'complete': True, 'items': [{
                        'name': 'Docs', 'badge': 'PAGE', 'kind': 'page', 'size': 42, 'lineCount': 2,
                        'included': True, 'localOnly': False, 'delivery': 'context', 'modality': 'text',
                        'intent': 'context', 'replay': False, 'boundedExcerpt': False, 'status': 'Current page',
                        'type': '', 'relativePath': '', 'sourceKind': '', 'archiveName': '', 'contextRole': 'current',
                        'sourceUrl': 'https://user:pass@docs.example.test/resource/?token=SECRET#frag',
                    }],
                },
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
    assert snap['schema_version'] == '2.1'
    assert snap['turns'][0]['user']['resources']['items'][0]['sourceUrl'] == 'https://docs.example.test/resource/'
    assert parsed_yaml['records'][0]['resources']['items'][0]['name'] == 'Docs'
    assert parsed_toml['records'][0]['resources']['items'][0]['name'] == 'Docs'
    assert parsed_toml['turns'][0]['user']['resources']['items'][0]['name'] == 'Docs'


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


def test_schema_20_is_migration_input_but_canonical_output_is_21():
    raw = hostile_snapshot()
    raw['schema_version'] = '2.0'
    raw['records'][0].pop('resources', None)
    snap = mod.canonicalize_share_snapshot(raw)
    assert snap['schema_version'] == '2.1'
    assert snap['records'][0]['resources'] is None


def test_resource_source_urls_fail_closed_when_session_source_is_hidden():
    raw = hostile_snapshot()
    raw['session']['page_url'] = None
    raw['records'][0]['resources']['items'][0]['sourceUrl'] = 'file:///Users/private/project/index.html?token=SECRET'
    snap = mod.canonicalize_share_snapshot(raw)
    assert snap['session']['page_url'] is None
    assert snap['records'][0]['resources']['items'][0]['sourceUrl'] == ''


def test_text_and_html_preserve_human_metadata_without_active_content():
    snap = mod.canonicalize_share_snapshot(hostile_snapshot())
    text, *_ = mod.render_share(snap, 'txt')
    html_text, *_ = mod.render_share(snap, 'html')
    assert 'Schema: 2.1' in text
    assert '[Resources used for this question: 1 · context 1 · raw 0 · not-sent 0]' in text
    assert '[Rating: helpful · 1]' in text
    assert '[Feedback: note]' in text
    assert 'Resources used for this question: 1' in html_text
    assert 'Rating: helpful (1)' in html_text
    assert '<script' not in html_text.lower()
