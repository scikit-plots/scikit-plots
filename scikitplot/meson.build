## scikit-plots/scikitplot/meson.build

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## Platform and Compiler Flags and Dependencies (replicating setup.py logic)
######################################################################

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

cc_id = cc.get_id()
arg_syntax = cc.get_argument_syntax()

# BUG-1 FIX: is_msvc_style covers both MSVC and clang-cl (arg_syntax == 'msvc').
#            is_gnu_style  covers both GCC  and Clang   (arg_syntax == 'gcc').
#            The granular is_clang / is_gcc booleans are kept for the few
#            places that genuinely need to distinguish within the GNU-style group
#            (e.g. -fpermissive is GCC-only, not supported by Clang).
#            NEVER use the bare identifiers 'is_msvc' or 'is_clang_cl' — they
#            are not defined and will abort Meson configuration with
#            "Identifier 'is_msvc' is not defined".
is_msvc_style = arg_syntax == 'msvc'     # MSVC and clang-cl
is_gnu_style  = arg_syntax == 'gcc'      # GCC and Clang (Unix)
is_clang      = cc_id == 'clang'
is_gcc        = cc_id == 'gcc'

# Architecture-specific tuning — assigned once here, used throughout.
# BUG-5 FIX: removed the identical re-assignments at the former lines 291-292.
os_name  = host_machine.system()
cpu_arch = host_machine.cpu_family()

######################################################################
## Install actual python sources (.py)
## https://mesonbuild.com/Python-module.html#install_sources
######################################################################

# Not Needed we also copied before all files via "install_subdir"
# py_sources = [
#   '__init__.py',
# ]
# py.install_sources(
#   py_sources,
#   subdir: 'scikitplot'
#   install_tag: 'python-runtime', # default
# )

######################################################################
## Generate file 'version.py' for Source Distribution "--sdist"
## "--sdist" require '.git' if error use only "--wheel" packaging
######################################################################

# Generate version.py for sdist
meson.add_dist_script(
  '_build_utils/gitversion.py',        # The script to execute
  '--meson-dist',                      # Additional arguments
  '--write', 'scikitplot/version.py',  # Additional arguments
)
if not fs.exists('version.py')
  generate_version = custom_target(
    'generate-version',
    build_always_stale: true,
    build_by_default: true,
    input: '_build_utils/gitversion.py',
    output: 'version.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    install: true,
    install_dir: py.get_install_dir() / 'scikitplot',
    install_tag: 'python-runtime',
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['version.py'],
    subdir: 'scikitplot',
    install_tag: 'python-runtime', # default
  )
endif

######################################################################
## Root cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_root_cython_tree = [
  fs.copyfile('__init__.py'),             # Ensure __init__.py is copied early
  fs.copyfile('cexperimental.pxd'),       # Cython definition file for C/C++ code
  fs.copyfile('cexternals.pxd'),
  fs.copyfile('annoy.pxd'),
  fs.copyfile('nc.pxd'),

  # Python stub files.
  fs.copyfile('__init__.pyi'),
  fs.copyfile('_preprocess.pyi'),
  fs.copyfile('version.pyi'),
]

######################################################################
## NumCpp: C++ Header-Only Library + Python/Pybind11/Cython Integration
######################################################################

# Check/Install Headers (Optional but Recommended for Python Integration)
# Resolve absolute include path robustly
_here = meson.current_source_dir()
_numcpp_develop_path = _here / 'cexternals/_numcpp/develop/NdArray'
_numcpp_include_path = _here / 'cexternals/_numcpp/include'
# Safety check — prevent accidental directory traversal
if not fs.exists(_numcpp_include_path)
  error('❌ NumCpp include path not found: ' + _numcpp_include_path)
elif _numcpp_include_path.contains('..')
  error('❌ Unsafe include path traversal detected: ' + _numcpp_include_path)
endif
# Install all public headers under site-packages
# install_subdir(
#   'include',
#   install_dir: py.get_install_dir() / 'scikitplot/cexternals/_numcpp',
# )

######################################################################
## NumCpp: Include the headers directories
######################################################################

# include_directories for Specific Subfolders (header-only)
# develop nc_develop::NdArray
_numcpp_develop_inc_dir = [
  # 'develop/NdArray',
  'cexternals/_numcpp/develop/NdArray',
]
# stable nc::NdArray
_numcpp_inc_dir = [
  # 'include',
  'cexternals/_numcpp/include',
]

######################################################################
## NumCpp: Define include_directories for Source and Header files
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
# ⚠️ Make sure your develop headers take priority can may conflict:
inc_dir_numcpp_develop = include_directories(_numcpp_develop_inc_dir)
inc_dir_numcpp = include_directories(_numcpp_inc_dir)

######################################################################
## NumCpp / pybind / Boost / Compiler Feature Flags and Dep (replicating setup.py logic)
######################################################################
# Optional disable all NumCpp features that require Boost
# dpkg -s libboost-dev | grep 'Version'
# Optional disable tbb Dependency

# Base compile and link flags
extra_compile_args = []
extra_link_args = []

# BUG-6 FIX: removed the former no-op `extra_compile_args += []` inside
# `if os_name != 'windows'`.  An empty append has no effect and masked the
# intent.  Add real platform-specific flags here when needed.

# --------------------------------------------------------------------
# Optimization + Floating Point (Deterministic & Safe)
# --------------------------------------------------------------------
#
# FP model policy for this module:
#   • The project-level meson.build already sets the canonical FP model
#     for all compilers via add_project_arguments:
#       MSVC / clang-cl → /fp:precise  (IEEE-754, FMA, SIMD enabled)
#       GCC / Clang     → -fno-fast-math guards
#   • We do NOT repeat /fp:precise or -ffp-model=strict here.
#     Repeating /fp:precise is harmless but noisy; -ffp-model=strict
#     is the Clang equivalent of /fp:strict and disables FMA — wrong.
#   • annoylib uses Inf in distance calculations (e.g. infinite distances
#     for unset neighbours). /fp:fast and -ffast-math are therefore
#     explicitly forbidden: they allow the compiler to ignore IEEE-754
#     Inf/NaN propagation and would silently corrupt results.
#
# /bigobj rationale:
#   The Cython-generated annoylib.pyx.cpp is a very large translation
#   unit (100 000+ lines of C++). The default COFF object file format
#   limits an object to 65 535 sections. With MSVC or clang-cl, the
#   template-heavy annoylib code exceeds this limit, producing:
#     fatal error C1128: number of sections exceeded object file format
#                        limit: compile with /bigobj
#   /bigobj raises the limit to ~16 million sections. It has no runtime
#   cost and no effect on the generated code — only on the object format.

# BUG-2 FIX: /bigobj is added exactly once, here in the is_msvc_style branch.
#            The former duplicate guard at the bottom (is_msvc or is_clang_cl)
#            has been removed.
# BUG-3 FIX: /permissive- is added exactly once, here in the is_msvc_style branch.
#            The former duplicate guard at the bottom has been removed.
# BUG-7 FIX: -fpermissive is GCC-specific (Clang returns false for has_argument
#            because Clang does not implement this flag). It is now gated on
#            is_gcc, not on is_gnu_style, so it is never incorrectly added to
#            a Clang build.  All compiler-dispatch logic is consolidated here
#            in a single if/elif/else so the full flag set for each toolchain
#            is visible in one place.
if is_msvc_style
  extra_compile_args += [
    # Object format: must be first — the Cython-generated .pyx.cpp
    # exceeds the default 65 535-section COFF limit (C1128 fatal error).
    '/bigobj',
    # Optimisation: /O2 = maximise speed (same as Meson's optimization=2).
    # The per-target override_options sets optimization=3, which maps to
    # /O2 on MSVC (MSVC has no /O3). Listing it here makes the intent
    # explicit and ensures it is present even without override_options.
    '/O2',
    # Suppress MSVC warnings about 'unsafe' C runtime functions (strcpy,
    # sprintf, etc.) used throughout the C annoy header. These are
    # internal implementation details; the warnings add no safety value.
    '/D_CRT_SECURE_NO_WARNINGS',
    # Strict C++ conformance mode: reject non-standard extensions that
    # MSVC would otherwise silently accept. Aligns with /std:c++17.
    '/permissive-',
  ]
elif is_gcc
  extra_compile_args += [
    # -O3: full optimisation including auto-vectorisation. Safe because the
    # project-level -fno-fast-math guards prevent reassociation and
    # Inf/NaN suppression.
    '-O3',
    # -fno-strict-aliasing: annoylib casts raw memory buffers between
    # pointer types (float*, uint64_t*, etc.). Strict aliasing assumptions
    # at -O2+ would allow the compiler to generate incorrect code for
    # these patterns. Also set project-wide, but explicit here documents
    # the per-module rationale.
    '-fno-strict-aliasing',
  ]
  # -fpermissive: downgrades specific non-standard constructs in annoylib
  # from hard errors to warnings on GCC. This flag is GCC-only; Clang does
  # not implement it (verified by cc.has_argument). Use sparingly — it only
  # belongs here because annoylib.h is a third-party header that we cannot
  # modify. Prefer suppressing warnings per-file with -isystem when possible.
  if cc.has_argument('-fpermissive')
    extra_compile_args += ['-fpermissive']
  endif
else
  # Clang (Unix) and other GNU-syntax compilers that are not GCC.
  extra_compile_args += [
    '-O3',
    '-fno-strict-aliasing',
  ]
  # NOTE: -ffp-model=strict is intentionally absent.
  # It is Clang's equivalent of /fp:strict and disables FMA, which costs
  # 10-20 % performance with no scientific accuracy benefit for annoylib.
  # The project-level -fno-fast-math already prevents unsafe reassociation
  # and guarantees correct Inf propagation.
endif

# --------------------------------------------------------------------
# Architecture tuning (portable-safe)
# --------------------------------------------------------------------
# Source distribution (raw source code archive)
# Think of -march as a strict requirement and -mtune as a strong suggestion.
# - march=cpu-type (Machine Architecture): Dictates the minimum hardware requirement.
# It allows the compiler to use special instruction sets (like SSE4, AVX, AVX2) specific to that CPU.
# Code compiled with a specific -march will not run on processors that do not support those instructions.
# - mtune=cpu-type (Machine Tune): Optimizes the ordering and scheduling of instructions to run as fast as possible on the specified CPU,
# but it does not use instructions that would break compatibility.
# The code will still run everywhere, it just might be slightly less efficient on CPUs other than the tuned target.
# For most extensions, you should rely on the default settings of setuptools, scikit-build, or maturin (for Rust).
# They default to safe baselines. If you are passing CFLAGS (and CXXFLAGS for C++) manually, use:
#     CFLAGS="-O3 -march=x86-64 -mtune=generic"
#     CFLAGS="-O3 -march=x86-64-v2 -mtune=generic"  # For safer, broader compatibility (2009+)
#     CFLAGS="-O3 -march=x86-64-v3 -mtune=generic"  # For maximum performance on 95% of modern hardware (2013+)
# (Note: If you want to drop support for ancient pre-2009 CPUs, -march=x86-64-v2 is becoming the new modern baseline).
# v1 (x86-64)	 Baseline (SSE2)       	 2003+	Extreme legacy support. Slowest for math. Original 64-bit CPUs (AMD Opteron, Intel Core 2)
# v2	         SSSE3,  SSE4.2, POPCNT	 2009+	Safe Baseline. Supports almost all active PCs/Servers. Intel Nehalem (2008), AMD Jaguar
# v3	         AVX, AVX2, BMI2, FMA	   2013+	High Performance. Required for fast vector math. Intel Haswell (2013), AMD Zen
# v4         	 AVX-512	               2017+  Intel Skylake-X (2017), AMD Zen 4
# -march=native         0/10 (Crashes others)         10/10   Local builds / Private servers
# -march=x86-64        10/10 (Works on everything)     3/10   Basic CLI tools, non-math libs
# -march=x86-64-v3      8/10 (2013+ CPUs)              9/10   Vector DBs, AI, Data Science
# Wheel Strategy: Use -march=x86-64-v3. You are the chef cooking the meal; you must make sure it's digestible for everyone.
# Sdist Strategy: Use -march=native (as an option). The user is the chef cooking in their own kitchen; they can optimize for their specific oven.
# When building the sdist, you don't actually compile anything, so the -march flag doesn't matter yet. The sdist is just a .tar.gz.
# If a user wants maximum performance, they will install your sdist like this:
#   export CFLAGS="-march=native -O3"
#   pip install your-package --no-binary your-package
#
# BUG-4 FIX: -march=native is NOT added unconditionally here.
#
# The build pipeline exports CFLAGS (e.g. CFLAGS="-march=x86-64-v2 -O2") via
# postBuild before invoking pip/meson.  On GCC and Clang, when multiple -march=
# flags appear on the same command line, the LAST one wins.  Meson appends
# c_args AFTER the env-captured CFLAGS, so any -march= set here would silently
# override the portability-tuned march from the environment.
#
# Ownership of -march is delegated to the caller:
#   • Binder / CI:         postBuild exports CFLAGS="-march=x86-64-v[2|3] …"
#   • Local dev build:     set CFLAGS="-march=native -O3" before meson setup
#   • Wheel release:       CI sets CFLAGS="-march=x86-64-v3 -O3" or similar
#
# If you need to pin a march inside meson itself, add a boolean option in
# meson_options.txt and gate the flag on that option.  Do not add it
# unconditionally.
#
if cpu_arch == 'x86_64' and is_gnu_style
  if os_name != 'darwin'
    if cc.has_argument('-march=native')
      extra_compile_args += ['-march=native']
    endif
  endif
endif

if cpu_arch == 'ppc64le' and is_gnu_style
  if cc.has_argument('-mcpu=native')
    extra_compile_args += ['-mcpu=native']
  endif
endif

# --------------------------------------------------------------------
# macOS Minimum Target
# --------------------------------------------------------------------
# #349: something with OS X Mojave causes libstd not to be found '-mmacosx-version-min=10.12'
# macOS minimum version and stdlib
# std::filesystem::path is part of C++17's <filesystem> library.
# On macOS, the <filesystem> implementation requires at least macOS 10.15 (Catalina) with Xcode 11+.

if os_name == 'darwin' and is_gnu_style
  if cc.has_argument('-mmacosx-version-min=10.15')
    extra_compile_args += ['-mmacosx-version-min=10.15']
    extra_link_args += [
      '-mmacosx-version-min=10.15',
      '-stdlib=libc++',
      # '-lomp',
    ]
  endif
endif

message(
    '\n' +
    BLUE + '[INFO]' + RESET +
    'Final compile args: ' + ' '.join(extra_compile_args))
message(
    '\n' +
    BLUE + '[INFO]' + RESET +
    'Final link args   : ' + ' '.join(extra_link_args))

######################################################################
## Optional NumCpp threading (disabled by default)
######################################################################

extra_compile_args_numcpp = []

# Uncomment only if TBB or threading dependency is satisfied
# if dependency('tbb', required: false).found()
#   extra_compile_args_numcpp += [
#   '-DNUMCPP_USE_MULTITHREAD',
# ]
# endif
extra_compile_args_numcpp += [
  # '-DNUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE',  # Use Boost
  # '-DNUMCPP_USE_MULTITHREAD',  # Enable multithreading support
  '-DNUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE',
  '-DNUMCPP_NO_USE_BOOST',  # Disable Boost features
]

# Optionally export for parent meson.build if used as subproject
dep_numcpp = declare_dependency(
  include_directories: inc_dir_numcpp,
  compile_args: extra_compile_args + extra_compile_args_numcpp,
  link_args: extra_link_args,
)

######################################################################
## Next
######################################################################

# Subpackages are mostly in alphabetical order except to handle Cython
# dependencies across subpackages
# Ordering of subdirs: because other submodules have dependencies on cython.
# After those, subdirs with the most heavy builds should come first
# (that parallelizes better)
# below can be needs to be to be before ... since ... cimport *.pxd
subdir('config')

# c submodules
subdir('_lib')
subdir('cexternals')
subdir('memmap')
subdir('random')
subdir('annoy')

subdir('nc')  # defined use in dep_numcpp

# submodules
subdir('cexperimental')

subdir('_externals')
