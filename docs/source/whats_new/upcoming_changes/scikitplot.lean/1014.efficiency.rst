- :class:`~scikitplot.lean.LeanIndex` keeps the names each declaration
  uses as numbers in two arrays, each distinct name once, instead of one
  tuple of strings per declaration: on mathlib4 peak memory falls by
  27 MB with one worker and by about 60 MB with two, since a worker
  reading the arrays copies none of them (1 MB, against 124 MB for the
  tuples). Name resolution is cached by those numbers; reports, edges and
  speed are unchanged.
