"""
Compatibility facade for the canonical YouTube reference grammar.

New code should import from ``.._sphinx_youtube_core.reference``.  This
module remains so existing private imports keep object identity and behavior.
"""

from .._sphinx_youtube_core.reference import *  # noqa: F401,F403
