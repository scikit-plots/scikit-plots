"""Focused regression checks; run with this package's parent on PYTHONPATH."""
import doctest
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import sys
import tempfile


_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)

from _sphinx_ext._sphinx_collection.select import parse_filter, group_records, Selection, FilterError
from _sphinx_ext._sphinx_youtube_gallery.directive import YouTubeGalleryDirective
from _sphinx_ext._sphinx_youtube_gallery.query import Query, apply_query, group_records as group_youtube_records, as_record
from _sphinx_ext._sphinx_youtube_gallery.model import (
    normalize_record, normalize_channel_record, normalize_gallery_catalog, derive_channel_records, CatalogError,
)
from _sphinx_ext._sphinx_collection.setup import ensure_assets

class RegressionTests(unittest.TestCase):
    def test_operator_in_value(self):
        for expression, expected in [('title="a~b"', 'a~b'),('title="x>=2"','x>=2'),('title~"a, b"','a, b'),('title="!warning"','!warning'),('title=""','')]:
            term, = parse_filter(expression)
            self.assertEqual(term.operand,expected)
            self.assertTrue(term.matches({'title':expected}))
    def test_bad_operator(self):
        for text in ['stars>>5','title=', 'title~"unfinished']:
            with self.assertRaises(FilterError):parse_filter(text)
    def test_group_dedup_zero_false(self):
        groups=group_records([{'tags':['a','a','b']},{'tags':0},{'tags':False}],Selection.from_text(group_by='tags'))
        self.assertEqual([(key,len(value)) for key,value in groups],[('a',1),('b',1),('0',1),('False',1)])
    def test_catalog_metadata_not_directives(self):
        directive=object.__new__(YouTubeGalleryDirective);directive.options={}
        record=normalize_record({'id':'JXtISpdDPNY','title':'Title\n.. raw:: html','description':'.. include:: secret.txt\n\n<script>bad</script>'})
        for rst in [True,False]:
            card=directive._card(record,'embed',rst)
            self.assertNotIn('\n.. include::',card['content'])
            self.assertNotIn('<script>',card['content'])
            self.assertNotIn('\n',card['title'])
            self.assertNotIn('Watch on YouTube',card['content'])
            self.assertNotIn('secret.txt',card['content'])

    def test_channel_catalog_uses_plain_gallery_grid_link_card(self):
        directive=object.__new__(YouTubeGalleryDirective);directive.options={}
        record=normalize_channel_record({'url':'https://www.youtube.com/@youtube/videos','description':'search only'})
        card=directive._card(record,'thumbnail',True)
        self.assertEqual(card['title'], r'\@youtube')
        self.assertEqual(card['link'], 'https://www.youtube.com/@youtube')
        self.assertNotIn('content', card)
        self.assertNotIn('img-top', card)
        self.assertEqual(card['description'], 'search only')

    def test_video_catalog_can_project_stable_channels_offline(self):
        videos = [
            normalize_record({'id':'abcdefghijk','title':'A','channel':'Claude','channel_id':'UC'+'A'*22,'published':'2026-01-01','tags':['ai']}),
            normalize_record({'id':'abcdefghijl','title':'B','channel':'Claude','channel_id':'UC'+'A'*22,'published':'2026-02-01','tags':['tools']}),
            normalize_record({'id':'abcdefghijm','title':'C','channel':'Display only'}),
        ]
        channels = derive_channel_records(videos)
        self.assertEqual(len(channels), 1)
        self.assertEqual(channels[0].video_count, 2)
        self.assertEqual(channels[0].tags, ['ai','tools'])
        self.assertEqual(channels[0].year, '2026')
        self.assertTrue(channels[0].url.startswith('https://www.youtube.com/channel/UC'))

    def test_custom_fields_share_gallery_selection_vocabulary(self):
        first = normalize_record({
            'id':'abcdefghijk','title':'First',
            'fields': {'category':'Agents','audience':{'level':'advanced'}},
        })
        second = normalize_record({
            'id':'abcdefghijl','title':'Second',
            'fields': {'category':'Data','audience':{'level':'beginner'}},
        })
        self.assertEqual(as_record(first)['audience']['level'], 'advanced')
        ordered, total = apply_query([second, first], Query(sort_by='category'))
        self.assertEqual(total, 2)
        self.assertEqual([r.title for r in ordered], ['First', 'Second'])
        groups = group_youtube_records(ordered, Query(group_by='category'))
        self.assertEqual([name for name, _ in groups], ['Agents', 'Data'])
        with self.assertRaisesRegex(CatalogError, 'reserved'):
            normalize_record({'id':'abcdefghijm','fields':{'title':'must not override'}})
        with self.assertRaisesRegex(CatalogError, 'not present'):
            apply_query([first], Query(sort_by='titel'))

    def test_channel_projection_sorts_channels_after_deduplication(self):
        videos = [
            normalize_record({'id':'abcdefghijk','title':'A video','channel':'@zeta'}),
            normalize_record({'id':'abcdefghijl','title':'Z video','channel':'@alpha'}),
            normalize_record({'id':'abcdefghijm','title':'M video','channel':'@alpha'}),
        ]
        channels = derive_channel_records(videos)
        ordered, total = apply_query(channels, Query(sort_by='title'))
        self.assertEqual(total, 2)
        self.assertEqual([r.title for r in ordered], ['@alpha', '@zeta'])
        ordered, _ = apply_query(channels, Query(sort_by='-video_count'))
        self.assertEqual([r.video_count for r in ordered], [2, 1])

    def test_mixed_youtube_catalog_is_rejected(self):
        with self.assertRaises(CatalogError):
            normalize_gallery_catalog({'videos':['JXtISpdDPNY'],'channels':['@youtube']})
    def test_asset_failure_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=SimpleNamespace(outdir=tmp,css=[],js=[])
            app.add_css_file=lambda name:app.css.append(name)
            app.add_js_file=lambda name,**kw:app.js.append(name)
            with patch.object(Path,'write_text',side_effect=OSError('test failure')):
                ensure_assets(app)
            self.assertFalse(getattr(app,'_sk_collection_assets_registered',False))
            ensure_assets(app);ensure_assets(app)
            self.assertEqual(len(app.css),1);self.assertEqual(len(app.js),1)
    def test_catalog_link_validation(self):
        for url in ['javascript:alert(1)', 'https://example.com/', 'https://youtu.be/dQw4w9WgXcQ']:
            with self.assertRaises(CatalogError):normalize_record({'id':'JXtISpdDPNY','url':url})
        record=normalize_record({'id':'JXtISpdDPNY','url':'https://youtu.be/JXtISpdDPNY?si=tracking'})
        self.assertEqual(record.url,'https://www.youtube.com/watch?v=JXtISpdDPNY')

    def test_existing_doctests(self):
        failures=0
        for module in ['_sphinx_collection.select','_sphinx_youtube_gallery.model','_sphinx_youtube_gallery.query','_sphinx_youtube_core.reference']:
            result=doctest.testmod(importlib.import_module('_sphinx_ext.'+module))
            failures+=result.failed
        self.assertEqual(failures,0)

if __name__=='__main__':unittest.main()
