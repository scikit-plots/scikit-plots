from pathlib import Path
import sys
_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
activate(__file__)
extensions = ["_sphinx_ext._sphinx_youtube_gallery"]
