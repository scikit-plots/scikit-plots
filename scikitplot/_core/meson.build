# scikitplot/_core/meson.build

inc_curdir = include_directories('.')


# Set common build flags for C and C++ code
# -----------------------------------------
# Common build flags
c_args_common = [
  # '-DNPY_INTERNAL_BUILD',
  # '-DHAVE_NPY_CONFIG_H',
  cflags_large_file_support,
]

# Same as NPY_CXX_FLAGS (TODO: extend for what ccompiler_opt adds)
cpp_args_common = c_args_common + [
]
if cc.get_argument_syntax() != 'msvc'
  cpp_args_common += [
    '-fno-exceptions',  # no exception support
    '-fno-rtti',  # no runtime type information
  ]
endif

# Build SIMD module
# -----------------
_simd_dispatch = []
_simd_baseline = []
foreach target : get_option('test-simd')
  target = target.strip().to_upper().split(',')
  mfeatures = []
  foreach fet_name : target
    if fet_name == 'BASELINE'
      _simd_baseline = CPU_BASELINE
      break
    endif
    if fet_name not in CPU_FEATURES
      error('Expected a valid feature name, got('+fet_name+')')
    endif
    mfeatures += CPU_FEATURES[fet_name]
  endforeach
  _simd_dispatch += [mfeatures]
endforeach


# _simd_mtargets = mod_features.multi_targets(
#   '_simd.dispatch.h',
#   [
#     src_file.process('src/_simd/_simd_inc.h.src'),
#     src_file.process('src/_simd/_simd_data.inc.src'),
#     src_file.process('src/_simd/_simd.dispatch.c.src'),
#   ],
#   # Skip validating the order of `_simd_dispatch` because we execute all these
#   # features, not just the highest interest one. The sorting doesn't matter
#   # here, given the nature of this testing unit.
#   keep_sort: true,
#   dispatch: _simd_dispatch,
#   baseline: _simd_baseline,
#   prefix: 'NPY_',
#   dependencies: [dep_all],
#   include_directories: [
#     'include',
#     'src/common',
#     'src/_simd',
#   ],
#   c_args: c_args_common,
#   cpp_args: cpp_args_common,
# )

## https://ermig1979.github.io/Simd/
# py.extension_module('_simd',
#   [
#     'src/common/npy_cpu_features.c',
#     'src/_simd/_simd.c',
#   ],
#   include_directories: [
#     'include',
#     'src/common',
#     'src/_simd',
#   ],
#   # c_args: c_args_common,
#   dependencies: [dep_all],
#   link_with: [_simd_mtargets.static_lib('_simd_mtargets')],
#   install: true,
#   subdir: 'scikitplot/_core',
#   install_tag: 'tests',
# )

# include paths needed for __cpu__ and common/
_core_inc = include_directories(
  # generated_inc,      # wherever Meson writes npy_cpu_dispatch_config.h
  'include',            # npy_cpu.h, npy_os.h live here under scikitplot/
  'src/common',         # npy_cpu_features.h, npy_cpu_dispatch.h
  'src/__cpu__',
)

# The common CPU detection C files (compiled once, shared)
_cpu_common_sources = files(
  'src/common/npy_cpu_features.c',
  'src/common/npy_cpu_dispatch.c',
)

# __cpu_mtargets = mod_features.multi_targets(
#   '__cpu.h',
#   'src/__cpu__/__cpu.c',
#   _cpu_common_sources,
#   dispatch: [
#     AVX512_SPR, AVX512_ICL,
#     X86_V4, X86_V3, X86_V2,
#     ASIMDHP, ASIMD, NEON,
#     VSX3, VSX2, VSX,
#     VXE, VX, RVV,
#   ],
#   baseline: CPU_BASELINE,
#   prefix: 'NPY_',
#   include_directories: [
#     'include',
#     'src/common',
#     'src/__cpu__',
#   ],
#   c_args: c_args_common,
#   cpp_args: cpp_args_common,
#   dependencies: [dep_all],
# )

# # The __cpu__ Python extension module
# py.extension_module(
#   '__cpu__',
#   sources: [
#     'src/__cpu__/__cpu.c',
#     # _cpu_common_sources,
#   ],
#   include_directories: _core_inc,
#   link_with: __cpu_mtargets.static_lib('__cpu_mtargets'),
#   dependencies: [dep_all],
#   install: true,
#   subdir: 'scikitplot/_core',
#   # gnu_symbol_visibility: 'default',
# )

subdir('include')
