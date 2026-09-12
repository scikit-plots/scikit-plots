# Maintenance model

## Source authority

For the Cython backend, the authoritative editable sources are
`annoylib.pyx.in` and `annoylib.pxd.in`. The generated `.pyx`, `.pxd` and
Cython-emitted C++ belong to the build directory. A clean build must be able to
recreate them.

`_annoy/annoymodule.cpp` is not used by the current Meson target and is not a
substitute for the Cython-generated C++ translation unit. Its presence is a
maintenance hazard because its name looks authoritative; keep that distinction
explicit until it is removed or deliberately wired into a redesigned build.

## Verification layers

1. **Static contract:** header paths, generation wiring, public ownership and
   plane separation.
2. **Generation:** Tempita tool exists and regenerates both build inputs.
3. **Cython compile:** generated Cython declarations agree sufficiently with the
   C++ headers to compile.
4. **Native/runtime:** import and focused Annoy behavior tests pass against the
   separately compiled native backend and Cython backend as applicable.
5. **Platforms:** especially Windows plus at least one Unix-like platform for
   mmap/error/float/concurrency behavior.

Static path checks do not prove C++ ABI/signature parity. A header can exist at
the right path while a Cython declaration is wrong.


The active repository build requires **C++17**. Template-local C++14 comments/directives are historical/stale and must not be treated as the current compiler contract.
