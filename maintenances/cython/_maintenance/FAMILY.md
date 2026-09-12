# Ownership and family boundaries

`scikitplot.cython` owns runtime compilation of caller-supplied source.  It is
intentionally independent of sibling `scikitplot` submodules.

Do **not** merge this owner mentally with `scikitplot.annoy`: Annoy's Cython
sources are part of the project's build graph, whereas this subsystem is a
runtime service invoked by users.  Do not route Annoy Tempita/Meson generation
issues here.

External Cython, setuptools, NumPy, pybind11 and the host compiler are toolchain
providers, not owned code.  They must remain lazy/optional where the public API
allows it.
