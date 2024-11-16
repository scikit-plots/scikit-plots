"""
Updated:  August 17 2024
Author :  Muhammed Çelik

Setup script for installing scikit-plots

For license information, see LICENSE and/or NOTICE.md
"""
from __future__ import (
    print_function
)
import os
import io
import sys
import codecs
import pathlib

from setuptools import (
    setup,
    find_packages,
    find_namespace_packages
)

import pytest
# import test_commands


##########################################################################
## Package Information
##########################################################################

## Basic information
## Get the directory where setup.py is located
HERE         = pathlib.Path(__file__).parent
NAME         = 'scikit-plots'
VERSION_PATH = os.path.join(HERE, 'scikitplot/__init__.py')
DESCRIPTION  = 'An intuitive library to add plotting functionality to scikit-learn objects.'
## Read the contents of the README file
README       = 'README.md'
PKG_DESCRIBE = (HERE / README).read_text(encoding='utf-8')
## Define the keywords
KEYWORDS = [
    "matplotlib", 
    "visualization", 
    "scikit-learn", 
    "xgboost", 
    "catboost", 
    "tensorflow", 
    "keras", 
    "pytorch", 
    "scikit-learn", 
    "machine learning", 
    "data science",
]
# LICENSE      = 'MIT License'  # deprecated
LICENSE_FILES= 'LICEN[CS]E*'
## If your name first as you're the current maintainer
AUTHOR       = 'Reiichiro Nakano et al.'
A_EMAIL      = 'reiichiro.s.nakano@gmail.com'
MAINTAINER   = 'Muhammed Çelik'
M_EMAIL      = 'muhammed.business.network@gmail.com'
REQUIRE_PATH = 'requirements.txt'

# Project homepage, often a link to GitHub or GitLab
# Often specified in the [project] table
HOMEPAGE      = "https://scikit-plots.github.io"  # alias URL
DOWNLOAD_URL  = "https://github.com/celik-muhammed/scikit-plot/tree/muhammed-dev"
DOCUMENTATION = "https://scikit-plots.github.io/stable/"
DONATE        = "https://github.com/celik-muhammed/scikit-plot#donate"
FORUM         = "https://github.com/celik-muhammed/scikit-plot/issues"
ISSUES        = "https://github.com/celik-muhammed/scikit-plot/issues"
REPO_FORKED   = "https://github.com/reiinakano/scikit-plot"
# The changelog, really useful for ongoing users of your project
CHANGELOG     = "https://scikit-plots.github.io/dev/whats_new/whats_new.html"

## Directories to ignore in find_packages
EXCLUDES = [
    "auto_building_tools",
    "docs", "docs.*",
    "examples", "examples.*",
    "notebooks", "notebooks.*",
    "tests", "tests.*",
    "paper", "paper.*",
    "binder", "binder.*",
    "register",
    "fixtures",
    "bin",
]
PACKAGE = find_packages(
    where='.', 
    include=['scikitplot', 'scikitplot.utils'], 
    exclude=EXCLUDES
)

##########################################################################
## Helper Functions
##########################################################################


def read(*filenames, **kwargs):
    """
    Assume UTF-8 encoding and return the contents of the file located at the
    absolute path from the REPOSITORY joined with *filenames.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(os.path.join(here, filename), encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


def get_version(rel_path=VERSION_PATH):
    """
    Reads the python file defined in the VERSION_PATH to find the get_version
    function.
    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError('Unable to find version string.')


def get_description_type(path=README):
    """
    Returns the long_description_content_type based on the extension of the
    package describe path (e.g. .txt, .rst, or .md).
    """
    ext = pathlib.Path(path).suffix
    return {'.rst': 'text/x-rst', '.txt': 'text/plain', '.md': 'text/markdown'}[ext]


def get_requires(path=REQUIRE_PATH):
    """
    Yields a generator of requirements as defined by the REQUIRE_PATH which
    should point to a requirements.txt output by `pip freeze`.
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line

##########################################################################
## Define the configuration
##########################################################################

## https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#metadata
## https://setuptools.pypa.io/en/latest/deprecated/distutils/apiref.html#distutils.core.setup
config = {
    # packages': find_packages(),  # Finds all packages automatically
    'packages': PACKAGE,
    'include_package_data': True,
    'name': NAME,
    'version': get_version(),
    'description': DESCRIPTION,
    'long_description': PKG_DESCRIBE,
    'long_description_content_type': get_description_type(),
    'keywords': KEYWORDS,
    # 'license': LICENSE,
    'license_files': LICENSE_FILES,
    'author': AUTHOR,
    'author_email': A_EMAIL,
    'maintainer': MAINTAINER,
    'maintainer_email': M_EMAIL,
    'url': HOMEPAGE,
    'download_url': DOWNLOAD_URL,
    'project_urls': {
        'Homepage   '  : HOMEPAGE,
        'Download'     : DOWNLOAD_URL,
        'Documentation': DOCUMENTATION,
        'Donate'       : DONATE,
        'Forum'        : FORUM,
        'Issues'       : ISSUES,
        'Repo_Forked'  : REPO_FORKED,
        'Repo_Forked'  : CHANGELOG,
    },
    'classifiers': [
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # 'entry_points': {"console_scripts": []},
    'python_requires': '>=3',
    'install_requires': list(get_requires()),
    'platforms': 'any',
    'extras_require': {
        'testing': ['pytest'],
    },
    # 'cmdclass': {'test': PyTest},
    'cmdclass': {'test': pytest},
}

##########################################################################
## Run setup script
##########################################################################

if __name__ == "__main__":
    setup(**config)