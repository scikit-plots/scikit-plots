"""Integration coverage for nested presentation forwarding in RST and MyST."""
import os, subprocess, sys, tempfile
from pathlib import Path
import sys
from html.parser import HTMLParser
_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)
class Tags(HTMLParser):
    def __init__(self):super().__init__();self.frames=[];self.divs=[]
    def handle_starttag(self,t,a):
        if t=='iframe':self.frames.append(dict(a))
        if t=='div':self.divs.append(dict(a))
with tempfile.TemporaryDirectory() as td:
    src=Path(td)/'src';out=Path(td)/'html';src.mkdir()
    (src/'conf.py').write_text('extensions=["_sphinx_ext._sphinx_youtube_gallery","myst_parser"]\nmyst_enable_extensions=["colon_fence"]\nproject="Nested options"\nmaster_doc="index"\n')
    options={
        'grid-columns':'1 1 2 2','grid-gutter':'3','grid-margin':'2','grid-padding':'1',
        'grid-class-container':'custom-container','grid-class-row':'custom-row',
        'card-class-card':'custom-card','card-class-body':'custom-body','card-shadow':'sm',
        'video-width':'100%','video-aspect':'4:3','video-align':'center',
        'video-privacy-mode':'true','video-url-parameters':'rel=0&start=90',
        'video-title':'Accessible custom player',
    }
    (src/'index.rst').write_text('Test\n====\n\n.. toctree::\n\n   rst\n   myst\n')
    rst='RST\n===\n\n.. youtube-gallery::\n'+''.join(f'   :{k}: {v}\n' for k,v in options.items())+'\n   videos:\n     - id: JXtISpdDPNY\n       title: PCA\n'
    md='# MyST\n\n```{youtube-gallery}\n'+''.join(f':{k}: {v}\n' for k,v in options.items())+'\nvideos:\n  - id: JXtISpdDPNY\n    title: PCA\n```\n'
    (src/'rst.rst').write_text(rst);(src/'myst.md').write_text(md)
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT.parent)+os.pathsep+env.get('PYTHONPATH','')
    cmd=[sys.executable,'-m','sphinx','-E','-b','html','-W','-q',str(src),str(out)]
    def build():return subprocess.run(cmd,env=env,text=True,capture_output=True)
    result=build();assert result.returncode==0,result.stderr
    for page in ['rst','myst']:
        tags=Tags();tags.feed((out/(page+'.html')).read_text());assert len(tags.frames)==1
        frame=tags.frames[0];assert frame['src']=='https://www.youtube-nocookie.com/embed/JXtISpdDPNY?rel=0&start=90',frame
        assert frame['title']=='Accessible custom player';assert 'width: 100%' in frame['style']
        classes=' '.join(d.get('class','') for d in tags.divs)
        for name in ['custom-container','custom-row','custom-card','custom-body']:assert name in classes
        assert any('aspect-ratio: 4 / 3' in d.get('style','') for d in tags.divs)
    # False is different from an empty/true privacy flag.
    (src/'rst.rst').write_text(rst.replace(':video-privacy-mode: true',':video-privacy-mode: false'))
    result=build();assert result.returncode==0,result.stderr
    assert 'https://www.youtube.com/embed/' in (out/'rst.html').read_text()
    (src/'rst.rst').write_text(rst.replace(':video-width: 100%', ':video-width: 100%\n   :video-height: 50%'))
    result=build();assert result.returncode==0,result.stderr
    percent=Tags();percent.feed((out/'rst.html').read_text());assert 'height: 50%' in percent.frames[0]['style']
    for option,value in [('video-aspect','0:9'),('video-width','bad'),('video-align','diagonal'),('grid-gutter','bad'),('card-link','javascript:alert(1)')]:
        (src/'rst.rst').write_text(f'Invalid\n=======\n\n.. youtube-gallery::\n   :{option}: {value}\n\n   - id: JXtISpdDPNY\n')
        result=build();assert result.returncode!=0 and 'Traceback' not in result.stderr,result.stderr
    # Direct leaf directives also report located errors instead of crashing.
    for option,value in [('aspect','0:9'),('width','bad')]:
        (src/'rst.rst').write_text(f'Invalid\n=======\n\n.. youtube:: JXtISpdDPNY\n   :{option}: {value}\n')
        result=build();assert result.returncode!=0 and 'Traceback' not in result.stderr,result.stderr
print('Nested grid/card/player settings, privacy false, and invalid-option checks passed')
