- :meth:`~scikitplot.lean.LeanIndex.from_folder` sends files to its
  workers in batches of at most 64, so the memory of a batch no longer
  grows with the project: on a project of 175,187 files, reading with two
  workers peaks at 1.5 GB instead of 2.1 GB (each worker 140 MB instead of
  420 MB), as fast; on mathlib4 each worker takes 28 MB instead of 56 MB.
