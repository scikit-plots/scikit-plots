// https://github.com/QunBB/fastannoy/blob/main/src/annoymodule.cpp
// https://raw.githubusercontent.com/QunBB/fastannoy/refs/heads/main/src/annoymodule.cpp

#include "annoylib.h"
#include "kissrandom.h"

#include <tuple>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <exception>
#if defined(_MSC_VER) && _MSC_VER == 1500
typedef signed __int32    int32_t;
#else
#include <stdint.h>
#endif


#if defined(ANNOYLIB_USE_AVX512)
#define AVX_INFO "Using 512-bit AVX instructions"
#elif defined(ANNOYLIB_USE_AVX128)
#define AVX_INFO "Using 128-bit AVX instructions"
#else
#define AVX_INFO "Not using AVX instructions"
#endif

#if defined(_MSC_VER)
#define COMPILER_INFO "Compiled using MSC"
#elif defined(__GNUC__)
#define COMPILER_INFO "Compiled on GCC"
#else
#define COMPILER_INFO "Compiled on unknown platform"
#endif

#define ANNOY_DOC (COMPILER_INFO ". " AVX_INFO ".")

using namespace Annoy;
namespace py = pybind11;

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  typedef AnnoyIndexMultiThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#else
  typedef AnnoyIndexSingleThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#endif

template class Annoy::AnnoyIndexInterface<int32_t, float>;

class HammingWrapper : public AnnoyIndexInterface<int32_t, float> {
  // Wrapper class for Hamming distance, using composition.
  // This translates binary (float) vectors into packed uint64_t vectors.
  // This is questionable from a performance point of view. Should reconsider this solution.
private:
  int32_t _f_external, _f_internal;
  AnnoyIndex<int32_t, uint64_t, Hamming, Kiss64Random, AnnoyIndexThreadedBuildPolicy> _index;
  void _pack(const float* src, uint64_t* dst) const {
    for (int32_t i = 0; i < _f_internal; i++) {
      dst[i] = 0;
      for (int32_t j = 0; j < 64 && i*64+j < _f_external; j++) {
	dst[i] |= (uint64_t)(src[i * 64 + j] > 0.5) << j;
      }
    }
  };
  void _unpack(const uint64_t* src, float* dst) const {
    for (int32_t i = 0; i < _f_external; i++) {
      dst[i] = (src[i / 64] >> (i % 64)) & 1;
    }
  };
public:
  HammingWrapper(int f) : _f_external(f), _f_internal((f + 63) / 64), _index((f + 63) / 64) {};
  bool add_item(int32_t item, const float* w, char**error) {
    vector<uint64_t> w_internal(_f_internal, 0);
    _pack(w, &w_internal[0]);
    return _index.add_item(item, &w_internal[0], error);
  };
  bool build(int q, int n_threads, char** error) { return _index.build(q, n_threads, error); };
  bool unbuild(char** error) { return _index.unbuild(error); };
  bool save(const char* filename, bool prefault, char** error) { return _index.save(filename, prefault, error); };
  void unload() { _index.unload(); };
  bool load(const char* filename, bool prefault, char** error) { return _index.load(filename, prefault, error); };
  float get_distance(int32_t i, int32_t j) const { return _index.get_distance(i, j); };
  void get_nns_by_item(int32_t item, size_t n, int search_k, vector<int32_t>* result, vector<float>* distances) const {
    if (distances) {
      vector<uint64_t> distances_internal;
      _index.get_nns_by_item(item, n, search_k, result, &distances_internal);
      distances->insert(distances->begin(), distances_internal.begin(), distances_internal.end());
    } else {
      _index.get_nns_by_item(item, n, search_k, result, NULL);
    }
  };
  void get_nns_by_vector(const float* w, size_t n, int search_k, vector<int32_t>* result, vector<float>* distances) const {
    vector<uint64_t> w_internal(_f_internal, 0);
    _pack(w, &w_internal[0]);
    if (distances) {
      vector<uint64_t> distances_internal;
      _index.get_nns_by_vector(&w_internal[0], n, search_k, result, &distances_internal);
      distances->insert(distances->begin(), distances_internal.begin(), distances_internal.end());
    } else {
      _index.get_nns_by_vector(&w_internal[0], n, search_k, result, NULL);
    }
  };
  int32_t get_n_items() const { return _index.get_n_items(); };
  int32_t get_n_trees() const { return _index.get_n_trees(); };
  void verbose(bool v) { _index.verbose(v); };
  void get_item(int32_t item, float* v) const {
    vector<uint64_t> v_internal(_f_internal, 0);
    _index.get_item(item, &v_internal[0]);
    _unpack(&v_internal[0], v);
  };
  void set_seed(uint64_t q) { _index.set_seed(q); };
  bool on_disk_build(const char* filename, char** error) { return _index.on_disk_build(filename, error); };
};


void check_constraints(int32_t n_items, int32_t item, bool building) {
    if (item < 0) {
        throw py::index_error("Item index can not be negative");
    } else if (!building && item >= n_items) {
        throw py::index_error("Item index larger than the largest item index");
    }
}


struct PyAnnoy {
    int f;
    AnnoyIndexInterface<int32_t, float>* ptr;

    PyAnnoy(int f, const char *metric) : f(f){
        if (!strcmp(metric, "angular")) {
            ptr = new AnnoyIndex<int32_t, float, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(f);
        } else if (!strcmp(metric, "euclidean")) {
            ptr = new AnnoyIndex<int32_t, float, Euclidean, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(f);
        } else if (!strcmp(metric, "manhattan")) {
            ptr = new AnnoyIndex<int32_t, float, Manhattan, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(f);
        } else if (!strcmp(metric, "hamming")) {
            ptr = new HammingWrapper(f);
        } else if (!strcmp(metric, "dot")) {
            ptr = new AnnoyIndex<int32_t, float, DotProduct, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(f);
        } else {
            throw py::value_error("No such metric");
        }
    }

    ~PyAnnoy(){
      delete ptr;
    }

    void load(const char* filename, bool prefault=false){
        char *error;

        if (!ptr->load(filename, prefault, &error)) {
            PyErr_SetString(PyExc_IOError, error);
            free(error);
            throw py::error_already_set();
        }
    }

    void save(const char* filename, bool prefault=false){
        char *error;

        if (!ptr->save(filename, prefault, &error)) {
            PyErr_SetString(PyExc_IOError, error);
            free(error);
            throw py::error_already_set();
        }
    }

    py::object get_nns_by_item(int32_t item, int32_t n, int32_t search_k, bool include_distances) {

        check_constraints(ptr->get_n_items(), item, false);

        vector<int32_t> result;
        vector<float> distances;

        Py_BEGIN_ALLOW_THREADS;
            ptr->get_nns_by_item(item, n, search_k, &result, include_distances ? &distances : NULL);
        Py_END_ALLOW_THREADS;

        if (include_distances){
          return py::make_tuple(result, distances);
        } else {
          return py::cast(result);
        }
    }

    py::object get_nns_by_vector(py::array_t<float> w, int32_t n, int32_t search_k, bool include_distances) {
      if (w.ndim() != 1 || w.size() != f){
        throw py::type_error("vector's dimension must be one and size must be equal to f");
      }
      vector<int32_t> result;
      vector<float> distances;

      Py_BEGIN_ALLOW_THREADS;
      ptr->get_nns_by_vector(w.data(), n, search_k, &result, include_distances ? &distances : NULL);
      Py_END_ALLOW_THREADS;

      if (include_distances){
          return py::make_tuple(result, distances);
        } else {
          return py::cast(result);
        }
    }

    void add_item(int32_t item, py::array_t<float> v) {
      check_constraints(ptr->get_n_items(), item, true);

      char* error;
      if (!ptr->add_item(item, v.data(), &error)) {
        PyErr_SetString(PyExc_Exception, error);
        free(error);
        throw py::error_already_set();
      }
    }

    void build(int n_trees, int n_jobs) {
      bool res;
      char* error;
      Py_BEGIN_ALLOW_THREADS;
      res = ptr->build(n_trees, n_jobs, &error);
      Py_END_ALLOW_THREADS;
      if (!res) {
        PyErr_SetString(PyExc_Exception, error);
        free(error);
        throw py::error_already_set();
      }
    }

    py::array_t<float> get_item_vector(int32_t item) {
      check_constraints(ptr->get_n_items(), item, false);

      vector<float> v(f);
      ptr->get_item(item, &v[0]);

      return py::array_t<float>(v.size(), v.data());
    }

    void on_disk_build(char *fn) {
      char *error;

      if (!ptr->on_disk_build(fn, &error)) {
        PyErr_SetString(PyExc_IOError, error);
        free(error);
        throw py::error_already_set();
      }
    }

    void unbuild() {
      char* error;
      if (!ptr->unbuild(&error)) {
        PyErr_SetString(PyExc_Exception, error);
        free(error);
        throw py::error_already_set();
      }
    }

    void unload() {
      ptr->unload();
    }

    double get_distance(int32_t i, int32_t j) {
      check_constraints(ptr->get_n_items(), i, false);
      check_constraints(ptr->get_n_items(), j, false);

      double d = ptr->get_distance(i,j);
      return d;
    }

    int32_t get_n_items() {

      int32_t n = ptr->get_n_items();
      return n;
    }

    int32_t get_n_trees() {

      int32_t n = ptr->get_n_trees();
      return n;
    }

    void verbose(bool verbose) {
      ptr->verbose(verbose);
    }

    void set_seed(int q) {

      ptr->set_seed(q);
    }

    py::object get_batch_nns_by_items(py::array_t<int32_t> items, int32_t n, int32_t search_k, bool include_distances, int n_threads) {
        auto size = items.size();

        if (size == 0){
          return py::tuple();
        }

        if (items.ndim() != 1){
          throw py::value_error("search items list's dimension must be one");
        }

        auto items_data = items.unchecked<1>();
        for (int i = 0; i < size; i++){
          check_constraints(ptr->get_n_items(), items_data(i), false);
        }
        vector<vector<int32_t>> result;
        vector<vector<float>> distances;

        // reserve for avoiding vector's non-thread-safe when dynamic allocation with multi-threads
        for (int i = 0; i < size; i++){
          result.emplace_back(vector<int32_t>());
          result[i].reserve(n);
          distances.emplace_back(vector<float>());
          if (include_distances){
            distances[i].reserve(n);
          }
        }

        if (n_threads > size){
          n_threads = size;
        }
        vector<std::thread> threads;
        int chunk_size = size / n_threads;
        int remainder = size % n_threads;
        int start = 0;

        Py_BEGIN_ALLOW_THREADS;
        for (int thread_idx = 0; thread_idx < n_threads; thread_idx++) {
          int end = start + chunk_size + (thread_idx < remainder ? 1 : 0);
          threads.emplace_back(
            [&, start, end](){
              for (int i = start; i < end; i++){
                ptr->get_nns_by_item(items_data(i), n, search_k, &(result[i]), include_distances ? &(distances[i]) : NULL);
              }
            }
          );
          start = end;
        }

        for (auto& thread : threads) {
          thread.join();
        }
        Py_END_ALLOW_THREADS;

        if (include_distances){
          return py::make_tuple(result, distances);
        } else {
          return py::cast(result);
        }
    }

    py::object get_batch_nns_by_vectors(py::array_t<float> vectors, int32_t n, int32_t search_k, bool include_distances, int n_threads) {
        auto size = vectors.shape(0);

        if (size == 0){
          return py::tuple();
        }

        if (vectors.ndim() != 2){
          throw py::value_error("search vectors's dimension must be two");
        }

        if (vectors.shape(1) != f){
            throw py::value_error("each vector's size must be equal to f");
        }
        vector<vector<int32_t>> result;
        vector<vector<float>> distances;

        // reserve for avoiding vector's non-thread-safe when dynamic allocation with multi-threads
        for (int i = 0; i < size; i++){
          result.emplace_back(vector<int32_t>());
          result[i].reserve(n);
          distances.emplace_back(vector<float>());
          if (include_distances){
            distances[i].reserve(n);
          }
        }

        if (n_threads > size){
          n_threads = size;
        }
        vector<std::thread> threads;
        int chunk_size = size / n_threads;
        int remainder = size % n_threads;
        int start = 0;

        Py_BEGIN_ALLOW_THREADS;
        for (int thread_idx = 0; thread_idx < n_threads; thread_idx++) {
          int end = start + chunk_size + (thread_idx < remainder ? 1 : 0);
          threads.emplace_back(
            [&, start, end](){
              for (int i = start; i < end; i++){
                // py::array vec = vectors[py::make_tuple(0, py::ellipsis())];
                // ptr->get_nns_by_vector((float*)vec.data(), n, search_k, &(result[i]), include_distances ? &(distances[i]) : NULL);
                ptr->get_nns_by_vector((vectors.data()+i*f), n, search_k, &(result[i]), include_distances ? &(distances[i]) : NULL);
              }
            }
          );
          start = end;
        }

        for (auto& thread : threads) {
          thread.join();
        }
        Py_END_ALLOW_THREADS;

        if (include_distances){
          return py::make_tuple(result, distances);
        } else {
          return py::cast(result);
        }
    }

};

PYBIND11_MODULE(annoylib, m) {
    py::class_<PyAnnoy>(m, "Annoy")
        .def(py::init<int, const char*>(), py::arg("f"), py::arg("metric"))
        .def("load", &PyAnnoy::load, "Loads (mmaps) an index from disk.", py::arg("fn"), py::arg("prefault") = false)
        .def("save", &PyAnnoy::save, "Saves the index to disk.", py::arg("fn"), py::arg("prefault") = false)
        .def("get_nns_by_item", &PyAnnoy::get_nns_by_item, "Returns the `n` closest items to item `i`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances.", py::arg("i"), py::arg("n"), py::arg("search_k") = -1, py::arg("include_distances") = false)
        .def("get_nns_by_vector", &PyAnnoy::get_nns_by_vector, "Returns the `n` closest items to vector `vector`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances.", py::arg("vector"), py::arg("n"), py::arg("search_k") = -1, py::arg("include_distances") = false)
        .def("get_item_vector", &PyAnnoy::get_item_vector, "Returns the vector for item `i` that was previously added.", py::arg("i"))
        .def("add_item", &PyAnnoy::add_item, "Adds item `i` (any nonnegative integer) with vector `v`.\n\nNote that it will allocate memory for `max(i)+1` items.", py::arg("i"), py::arg("vector"))
        .def("on_disk_build", &PyAnnoy::on_disk_build, "Build will be performed with storage on disk instead of RAM.", py::arg("fn"))
        .def("build", &PyAnnoy::build, "Builds a forest of `n_trees` trees.\n\nMore trees give higher precision when querying. After calling `build`,\nno more items can be added. `n_jobs` specifies the number of threads used to build the trees. `n_jobs=-1` uses all available CPU cores.", py::arg("fn"), py::arg("n_jobs") = -1)
        .def("unbuild", &PyAnnoy::unbuild, "Unbuilds the tree in order to allows adding new items.\n\nbuild() has to be called again afterwards in order to\nrun queries.")
        .def("unload", &PyAnnoy::unload, "Unloads an index from disk.")
        .def("get_distance", &PyAnnoy::get_distance, "Returns the distance between items `i` and `j`.", py::arg("i"), py::arg("j"))
        .def("get_n_items", &PyAnnoy::get_n_items, "Returns the number of items in the index.")
        .def("get_n_trees", &PyAnnoy::get_n_trees, "Returns the number of trees in the index.")
        .def("verbose", &PyAnnoy::verbose, py::arg("verbose"))
        .def("set_seed", &PyAnnoy::set_seed, "Sets the seed of Annoy's random number generator.", py::arg("seed"))
        .def("get_batch_nns_by_items", &PyAnnoy::get_batch_nns_by_items, "Returns the `n` closest items to each item `i` in query list `items`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances.\n\n:param n_threads: defaults to 1\n`n_threads` gives you multiple concurrent threads.", py::arg("items"), py::arg("n"), py::arg("search_k") = -1, py::arg("include_distances") = false, py::arg("n_threads") = 1)
        .def("get_batch_nns_by_vectors", &PyAnnoy::get_batch_nns_by_vectors, "Returns the `n` closest items to each vector in query list `vectors`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances.\n\n:param n_threads: defaults to 1\n`n_threads` gives you multiple concurrent threads.", py::arg("vectors"), py::arg("n"), py::arg("search_k") = -1, py::arg("include_distances") = false, py::arg("n_threads") = 1);
}
