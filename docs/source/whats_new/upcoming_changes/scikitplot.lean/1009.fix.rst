- :mod:`scikitplot.lean` reads identifiers with Lean's own characters
  (ASCII letters and digits, Greek and other letter-like ranges,
  subscripts) instead of Python's ``\w``, so notation such as ``sᶜ``,
  ``Aᵀ`` and ``→ₗ`` no longer hides the name it follows; and it keeps a
  field after an expression (``(s⁻¹).encard``) and Lean 4's leading-dot
  names (``.inl``) as references. A dossier no longer misses a flagged
  definition named under notation; on mathlib4, 13,036 declarations gain
  14,742 dependencies and none loses one.
