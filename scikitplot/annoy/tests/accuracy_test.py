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

from __future__ import print_function
import os
import sys
import uuid

import pytest

h5py = pytest.importorskip("h5py")
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve  # Python 3

# from annoy import AnnoyIndex
from scikitplot.annoy import AnnoyIndex

#: Opt-in: the directory that caches the ann-benchmarks data and the indexes
#: built from it. Unset (the default), every test here is skipped: the data
#: is hundreds of MB, downloaded once, and never written into the package
#: (Rule 30).
DATA_ENV = "SKPLT_ANNOY_ACCURACY_DATA"


@pytest.fixture(scope="module")
def data_dir():
    """Return the opt-in cache directory, or skip when it is not set."""
    value = os.environ.get(DATA_ENV, "")
    if not value:
        pytest.skip(f"set {DATA_ENV} to a cache directory to run the accuracy tests")
    value = os.path.abspath(os.path.expanduser(value))
    if not os.path.isdir(value):
        raise NotADirectoryError(f"{DATA_ENV}={value!r} is not a directory")
    return value

# np = pytest.importorskip("numpy")

# Unconditional Skipping
# # Skipping an Individual Test Function
# pytest.skip("Skipping this test at runtime")
# # Skipping All Tests in a Suite
# pytest.skip("Skipping this test file", allow_module_level=True)
# # Markers for Skipping Tests
# @pytest.mark.skip(reason="This test is under development")
# # Skipping a Test Class
# @pytest.mark.skip(reason="Skipping all tests in this class")
#
# Conditional Skipping
# # Skip Tests on Specific Platforms
# @pytest.mark.skipif(sys.platform == "win32", reason="Does not work on Windows")
# # Skip Based on Python Version
# @pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8 or higher")
#
# xfail (short for expected failure)
# # Mark a Test as xfail on Python < 3.8
# @pytest.mark.xfail(sys.version_info < (3, 8), reason="Fails on Python < 3.8")
# # Test expected to fail
# @pytest.mark.xfail(reason="Feature not implemented yet")
# # Failing if a supposedly broken test passes
# @pytest.mark.xfail(reason="Known bug in function", strict=True)
# # Using pytest.xfail() Inside a Test Function
# pytest.xfail("This test is currently broken")

def _complete(path, write):
    """
    Create ``path`` through ``write(partial_path)``, then rename it into place.

    A cache holds complete files only: an interrupted write is not mistaken
    for the data on the next run, and two processes (``pytest-xdist``)
    writing one file each use a partial file of their own. The partial name
    is only chosen here, not created (``mkstemp`` would make it mode 0600,
    and the cache could not be shared): ``write`` creates it.
    """
    partial = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.part"
    try:
        write(partial)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _get_index(data_dir, dataset, custom_distance=None, custom_dim=None):
    url = 'http://ann-benchmarks.com/%s.hdf5' % dataset
    vectors_fn = os.path.join(data_dir, dataset + ".hdf5")
    index_fn = os.path.join(data_dir, dataset + ".annoy")

    if not os.path.exists(vectors_fn):
        print("downloading", url, "->", vectors_fn)
        _complete(vectors_fn, lambda partial: urlretrieve(url, partial))

    dataset_f = h5py.File(vectors_fn, "r")
    try:
        return _build_index(
            dataset_f,
            data_dir,
            dataset,
            index_fn,
            custom_distance=custom_distance,
            custom_dim=custom_dim,
        )
    except BaseException:
        dataset_f.close()  # the caller owns the file only on success
        raise


def _build_index(dataset_f, data_dir, dataset, index_fn, *, custom_distance, custom_dim):
    distance = dataset_f.attrs["distance"]
    if custom_distance is not None:
        distance = custom_distance
    f = dataset_f["train"].shape[1]
    if custom_dim:
        f = custom_dim
    if custom_distance:
        dataset = dataset.rsplit('-', 2)[0] + "-%d-%s" % (f, custom_distance)
        index_fn = os.path.join(data_dir, dataset + '.annoy')


    annoy = AnnoyIndex(f, distance)

    if not os.path.exists(index_fn):
        print("adding items", distance, f)
        for i, v in enumerate(dataset_f["train"]):
            if len(v) > f:
                v = v[:f]
            annoy.add_item(i, v)

        print("building index")
        annoy.build(10)

        def save(partial):
            annoy.save(partial)
            annoy.unload()  # Windows: a mapped file cannot be renamed

        _complete(index_fn, save)
        annoy.load(index_fn)
    else:
        annoy.load(index_fn)
    return annoy, dataset_f, dataset


def _test_index(data_dir, dataset, exp_accuracy, custom_metric=None, custom_dim=None):
    annoy, dataset_f, dataset = _get_index(data_dir, dataset, custom_metric, custom_dim)
    with dataset_f:
        _check_accuracy(annoy, dataset_f, dataset, exp_accuracy, custom_dim)


def _check_accuracy(annoy, dataset_f, dataset, exp_accuracy, custom_dim):
    n, k = 0, 0

    for i, v in enumerate(dataset_f["test"]):
        if custom_dim:
            v = v[:custom_dim]
        js_fast = annoy.get_nns_by_vector(v, 10, 10000)
        js_real = dataset_f["neighbors"][i][:10]
        assert len(js_fast) == 10
        assert len(js_real) == 10

        n += 10
        k += len(set(js_fast).intersection(js_real))

    accuracy = 100.0 * k / n
    print(
        "%50s accuracy: %5.2f%% (expected %5.2f%%)" % (dataset, accuracy, exp_accuracy)
    )


    assert accuracy > exp_accuracy - 1.0  # should be within 1%


def test_glove_25(data_dir):
    # with pytest.raises(Exception) as e:
    _test_index(data_dir, "glove-25-angular", 69.00)


def test_nytimes_16(data_dir):
    _test_index(data_dir, "nytimes-16-angular", 80.00)


def test_lastfm_dot(data_dir):
    _test_index(data_dir, 'lastfm-64-dot', 60.00, 'dot', 64)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "builds a 65-dimensional index from 64-dimensional rows; add_item "
        "refuses the length (embedding length mismatch)"
    ),
)
def test_lastfm_angular(data_dir):
    _test_index(data_dir, 'lastfm-64-dot', 60.00, 'angular', 65)
