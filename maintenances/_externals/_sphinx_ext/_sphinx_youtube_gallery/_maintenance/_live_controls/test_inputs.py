"""Build inline list, inline videos wrapper, file-backed YAML, and alias conflicts."""
from pathlib import Path
import sys
import json, os, subprocess, sys, tempfile
_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)
with tempfile.TemporaryDirectory() as td:
    src=Path(td)/'src';out=Path(td)/'html';src.mkdir()
    (src/'conf.py').write_text('extensions=["_sphinx_ext._sphinx_youtube_gallery","myst_parser"]\nmyst_enable_extensions=["colon_fence"]\nproject="Inputs"\nmaster_doc="index"\n')
    (src/'videos.yaml').write_text('videos:\n  - id: JXtISpdDPNY\n    title: PCA\n    channel: Statistics Globe\n')
    (src/'cards.yaml').write_text('- title: External card\n  category: Example\n  link: https://example.com\n')
    (src/'channels.yaml').write_text('channels:\n  - "@youtube"\n  - handle: claude\n    title: "@claude"\n    description: searchable only\n')
    (src/'index.rst').write_text('''Input forms
===========

.. youtube-gallery::
   :grid-columns: 1 1 2 2
   :interactive:
   :filter-fields: channel

   - id: JXtISpdDPNY
     title: PCA
     channel: Statistics Globe

.. youtube-gallery::
   :grid-columns: 1 1 2 2
   :interactive:
   :filter-fields: channel

   videos:
     - id: JXtISpdDPNY
       title: PCA
       channel: Statistics Globe

.. youtube-gallery:: videos.yaml
   :columns: 1 1 2 2
   :grid-columns: 1 1 2 2
   :interactive:
   :filter-fields: channel

.. gallery-grid:: cards.yaml
   :interactive:
   :filter-fields: category

.. youtube-gallery:: channels.yaml
   :interactive:
   :collection-id: learning-channels
   :grid-columns: 1 1 2 2
''')
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT.parent)+os.pathsep+env.get('PYTHONPATH','')
    cmd=[sys.executable,'-m','sphinx','-E','-b','html','-W','-q',str(src),str(out)]
    result=subprocess.run(cmd,env=env,capture_output=True,text=True);assert result.returncode==0,result.stderr
    html=(out/'index.html').read_text();assert html.count('<iframe')==3;assert 'External card' in html
    assert html.count('aspect-ratio: 16 / 9') >= 3
    assert html.count('sk-collection-searchable')==5
    assert 'href="https://www.youtube.com/@youtube"' in html
    assert '>@youtube<' in html.replace('\n','')
    assert 'sd-card-text' not in html
    (src/'inline.md').write_text('---\norphan: true\n---\n# MyST inline\n\n```{youtube-gallery}\n:grid-columns: 1 1 2 2\n:interactive:\n:filter-fields: channel,tags\n\nvideos:\n  - id: JXtISpdDPNY\n    title: PCA\n    channel: Statistics Globe\n    tags: [pca]\n```\n')
    # User-controlled metadata cannot create markup or publish unrelated keys.
    (src/'cards.yaml').write_text('- title: Safe card\n  category: "</span><script>bad()</script>"\n  private_note: DO_NOT_PUBLISH\n')
    result=subprocess.run(cmd,env=env,capture_output=True,text=True);assert result.returncode==0,result.stderr
    assert 'sk-collection-searchable' in (out/'inline.html').read_text()
    html=(out/'index.html').read_text();assert '<script>bad()</script>' not in html;assert 'DO_NOT_PUBLISH' not in html
    (src/'index.rst').write_text('Conflict\n========\n\n.. youtube-gallery:: videos.yaml\n   :columns: 1\n   :grid-columns: 2\n')
    result=subprocess.run(cmd,env=env,capture_output=True,text=True)
    assert result.returncode!=0 and ':columns: and :grid-columns: disagree' in result.stderr,result.stderr
    assert 'Traceback' not in result.stderr
print('Inline list/wrapper, file YAML, column aliases, escaped metadata, and conflict checks passed')
