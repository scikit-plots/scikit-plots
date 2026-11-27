- :meth:`~scikitplot.lean.LeanIndex.save` writes an index to one file and
  :meth:`~scikitplot.lean.LeanIndex.load` reads it back with the same
  reports (also ``--save`` and ``--load`` on the command line): mathlib4
  loads in 1.4 s instead of 30 s of reading. The format holds numbers and
  text only, carries a checksum and is checked number by number when
  loaded, gives the same bytes for the same files, and names files by
  label: they are read only inside the ``root`` a load is given.
