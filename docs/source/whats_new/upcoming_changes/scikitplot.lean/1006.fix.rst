- Statement shapes in :meth:`~scikitplot.lean.LeanIndex.discover` treat a
  binder without a type in the signature (``{l}``, ``(hx)``) as bound, so
  ``{l} (h : a ∈ l)`` and ``{s} (h : a ∈ s)`` share a shape (7 more
  parallel pairs in mathlib4, such as ``List`` and ``Multiset``'s
  ``of_mem_filter``), and keep each string or char literal as one token,
  as numerals are kept as ``#``: ``f "a" = 1`` no longer shares the shape
  of ``f = 1``. Each statement is masked once instead of three times.
