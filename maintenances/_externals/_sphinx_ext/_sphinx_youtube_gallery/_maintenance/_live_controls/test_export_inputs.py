"""Build actual browser-exported YAML in RST/MyST; validate stable gallery IDs.

Run test_saved_additions.cjs first, then pass its HTML directory to this script.
"""
from pathlib import Path
from html.parser import HTMLParser
import os
import subprocess
import sys
import tempfile


_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)

from _sphinx_ext._sphinx_collection._browser import collection_id
from _sphinx_ext._sphinx_collection._yaml import load_bounded_yaml
from _sphinx_ext._sphinx_youtube_gallery.model import normalize_gallery_catalog

exports = Path(sys.argv[1])
videos = (exports / 'test-export-videos.yaml').read_text()
channels = (exports / 'test-export-channels.yaml').read_text()
video_kind, video_records = normalize_gallery_catalog(load_bounded_yaml(videos, 'browser video export'))
channel_kind, channel_records = normalize_gallery_catalog(load_bounded_yaml(channels, 'browser channel export'))
assert video_kind == 'video' and len(video_records) == 2
assert channel_kind == 'channel' and len(channel_records) == 1
for value in ['stable-gallery', 'Videos_2', 'v' * 64]:
    assert collection_id(value) == value
for value in ['', 'with spaces', '../path', '1video', 'v' * 65]:
    try:
        collection_id(value)
    except ValueError:
        pass
    else:
        raise AssertionError(f'invalid collection-id accepted: {value!r}')

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.hrefs.append(dict(attrs).get('href', ''))

with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / 'src'; src.mkdir()
    out = Path(tmp) / 'out'
    (src / 'conf.py').write_text('extensions=["_sphinx_ext._sphinx_youtube_gallery","myst_parser"]\nmyst_enable_extensions=["colon_fence"]\nproject="Exports"\nmaster_doc="index"\n')
    (src / 'videos.yaml').write_text(videos)
    (src / 'channels.yaml').write_text(channels)
    (src / 'index.rst').write_text('Exports\n=======\n\n.. toctree::\n\n   rst\n   myst\n')
    (src / 'rst.rst').write_text('RST\n===\n\n.. youtube-gallery:: videos.yaml\n   :interactive:\n   :collection-id: videos\n\n.. youtube-gallery:: channels.yaml\n   :interactive:\n   :collection-id: channels\n')
    (src / 'myst.md').write_text('# MyST\n\n```{youtube-gallery} videos.yaml\n:interactive:\n:collection-id: videos\n```\n\n```{youtube-gallery} channels.yaml\n:interactive:\n:collection-id: channels\n```\n')
    result = subprocess.run([sys.executable, '-m', 'sphinx', '-E', '-b', 'html', '-W', '-q', str(src), str(out)], env=os.environ, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in ['rst', 'myst']:
        html = (out / (name + '.html')).read_text()
        links = Links(); links.feed(html)
        assert not any('evil.example' in href for href in links.hrefs), html[html.find('<section'):html.find('<div class="sphinxsidebar"')]
        assert '&lt;script&gt;' in html
        assert 'collectionId' in html and 'videos' in html and 'channels' in html
    (src / 'rst.rst').write_text('Invalid\n=======\n\n.. youtube-gallery:: videos.yaml\n   :collection-id: invalid value\n')
    result = subprocess.run([sys.executable, '-m', 'sphinx', '-E', '-b', 'html', '-W', '-q', str(src), str(out)], env=os.environ, capture_output=True, text=True)
    assert result.returncode != 0 and 'collection-id' in result.stderr
    assert 'Traceback' not in result.stderr
print('Typed browser-exported video/channel YAML builds in RST/MyST; collection-id validation passed')
