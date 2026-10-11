.. _cython-profiles-and-toolchains:

Profiles and toolchains
=======================

Profiles are named build defaults; capabilities describe what the current
runtime can actually support. Keep those concepts separate.

Useful inspection functions include:

* :func:`platform_capabilities`;
* :func:`runtime_capabilities`;
* :func:`resolved_toolchain`;
* :func:`compiler_capabilities`;
* :func:`list_compilers`; and
* :func:`api_stability` / :func:`list_api`.

An advertised profile does not guarantee that a compiler, Python headers,
Cython, NumPy headers, pybind11, or other optional prerequisites are installed.
Capability checks should drive error messages and CI skips; they should not
silently downgrade a requested native feature.
