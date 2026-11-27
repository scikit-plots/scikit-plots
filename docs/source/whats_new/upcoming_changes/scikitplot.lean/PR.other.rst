- The user guide and :class:`~scikitplot.lean.LeanIndex` give what
  ``workers`` costs as well as what it saves, measured on a project of a
  million declarations: with two workers a summary takes 186 s instead of
  305 s, and 3.9 GB of memory in all instead of 2.1 GB, since each worker
  copies the parts of the index it reads.
