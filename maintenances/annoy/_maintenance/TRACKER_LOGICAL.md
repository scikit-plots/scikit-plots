# Logical tracker

- Public `scikitplot.annoy.Index` remains a high-level wrapper over the native
  `scikitplot.cexternals._annoy.Annoy` extension type.
- Private `scikitplot.annoy._annoy.Index` remains a separate Cython extension.
- Cython templates are authoritative; generated `.pyx/.pxd/C++` are not.
- Cython extern declarations point to the canonical cexternals headers by the
  expected relative paths.
- The Cython build must have an available deterministic Tempita generator.
- The inactive `_annoy/annoymodule.cpp` must not silently become a second build
  input without an explicit architecture change and review.
- Runtime code never imports maintenance or skill planes.
