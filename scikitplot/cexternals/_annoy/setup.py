#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from setuptools import setup, Extension
import os
import platform
import sys

readme_note = """\
.. note::

   For the latest source, discussion, etc, please visit the
   `GitHub repository <https://github.com/spotify/annoy>`_\n\n

.. image:: https://img.shields.io/github/stars/spotify/annoy.svg
    :target: https://github.com/spotify/annoy

"""

with open('README.rst', encoding='utf-8') as fobj:
    long_description = readme_note + fobj.read()

# Various platform-dependent extras
extra_compile_args = ['-D_CRT_SECURE_NO_WARNINGS', '-fpermissive']
extra_link_args = []

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
# v1 (x86-64)	Baseline (SSE2)     	2003+	Extreme legacy support. Slowest for math.
# v2	        SSE4.2, POPCNT	        2009+	Safe Baseline. Supports almost all active PCs/Servers.
# v3	        AVX, AVX2, BMI2, FMA	2013+	High Performance. Required for fast vector math.
# -march=native	    0/10 (Crashes others)	    10/10	Local builds / Private servers
# -march=x86-64	   10/10 (Works on everything)	3/10	Basic CLI tools, non-math libs
# -march=x86-64-v3	8/10 (2013+ CPUs)	        9/10	Vector DBs, AI, Data Science
# Wheel Strategy: Use -march=x86-64-v3. You are the chef cooking the meal; you must make sure it’s digestible for everyone.
# Sdist Strategy: Use -march=native (as an option). The user is the chef cooking in their own kitchen; they can optimize for their specific oven.
# When building the sdist, you don't actually compile anything, so the -march flag doesn't matter yet. The sdist is just a .tar.gz.
# If a user wants maximum performance, they will install your sdist like this:
#   export CFLAGS="-march=native -O3"
#   pip install your-package --no-binary your-package

if platform.machine() == 'ppc64le':
    extra_compile_args += ['-mcpu=native',]

if platform.machine() == 'x86_64':
    # do not apply march on Intel Darwin
    if platform.system() != 'Darwin':
        # Not all CPUs have march as a tuning parameter
        extra_compile_args += ['-march=native',]

if os.name != 'nt':
    extra_compile_args += ['-O3', '-ffast-math', '-fno-associative-math']

# Add multithreaded build flag for all platforms using Python 3 and
# for non-Windows Python 2 platforms
python_major_version = sys.version_info[0]
if python_major_version == 3 or (python_major_version == 2 and os.name != 'nt'):
    extra_compile_args += ['-DANNOYLIB_MULTITHREADED_BUILD']

    if os.name != 'nt':
        extra_compile_args += ['-std=c++14']

# #349: something with OS X Mojave causes libstd not to be found
if platform.system() == 'Darwin':
    extra_compile_args += ['-mmacosx-version-min=10.12']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.12']

# Manual configuration, you're on your own here.
manual_compiler_args = os.environ.get('ANNOY_COMPILER_ARGS', None)
if manual_compiler_args:
    extra_compile_args = manual_compiler_args.split(',')
manual_linker_args = os.environ.get('ANNOY_LINKER_ARGS', None)
if manual_linker_args:
    extra_link_args = manual_linker_args.split(',')

setup(name='annoy',
      version='1.17.3',
      description='Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk.',
      packages=['annoy'],
      package_data={'annoy': ['__init__.pyi', 'py.typed']},
      ext_modules=[
          Extension(
              'annoy.annoylib', ['src/annoymodule.cc'],
              depends=['src/annoylib.h', 'src/kissrandom.h', 'src/mman.h'],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
          )
      ],
      long_description=long_description,
      long_description_content_type='text/x-rst',
      author='Erik Bernhardsson',
      author_email='mail@erikbern.com',
      url='https://github.com/spotify/annoy',
      license='Apache License 2.0',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
      ],
      keywords='nns, approximate nearest neighbor search',
      setup_requires=['nose>=1.0'],
      tests_require=['numpy', 'h5py']
      )
