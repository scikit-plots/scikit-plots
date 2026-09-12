#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
p=Path(__file__).with_name('DEPENDENCY_GRAPH.json')
print(json.dumps(json.loads(p.read_text(encoding='utf-8')),indent=2))
