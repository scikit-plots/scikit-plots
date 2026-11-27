- :class:`~scikitplot.lean.LeanIndex` gives declarations after
  ``section A.B`` … ``end A.B`` their namespace: the ``end`` closed one
  scope too many (105 declarations in 7 mathlib4 files were misnamed). A
  token inside a «quoted name» no longer raises an audit flag
  (``theorem «a sorry b»``). Text Lean refuses to read, an ``end`` that
  names another scope or closes none and a file ending inside a comment
  or string, is reported by file and line in the summary
  (``source_problems``) and in a dossier (check ``source_text``), the same
  for any number of workers.
