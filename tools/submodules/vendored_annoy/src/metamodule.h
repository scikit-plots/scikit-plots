
// Moved from annoylib.h
// A simple POD that allows Python wrapper to slice NNS results safely.
// A simple POD that allows Python wrapper to slice NNS results safely.
// consruct `NNSIndex` `NNSSeries`, `NNSFrame`
// `NNSIndex` like pandas.Index like Immutable Index implementing a monotonic integer range.
// class pandas.Index(data=None, dtype=None, copy=False, name=None, tupleize_cols=True)
// class pandas.RangeIndex(start=None, stop=None, step=None, dtype=None, copy=False, name=None)[source]
// `NNSSeries` like pandas.Series -> pandas.Series.index pandas.Series.values pandas.Series.values pandas.Series.array
// class pandas.Series(data=None, index=None, dtype=None, name=None, copy=None, fastpath=<no_default>)[source]
// `NNSFrame` like pandas.DataFrame -> pandas.DataFrame.info pandas.DataFrame.shape pandas.DataFrame.index pandas.DataFrame.columns pandas.DataFrame.values pandas.DataFrame.to_numpy
// class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)[source]

// ======================= NNSMeta Python type ============================
// NNSMeta -> NNSFrame / NNSDistances / NNSEmbeddings
// These keep naming canonical:
// index / NNSIndex → pandas.Index / RangeIndex
// series / NNSSeries (NNSDistances / NNSEmbeddings) → pandas.Series
// frame / NNSFrame → pandas.DataFrame
// meta / NNSMeta → schema + metadata view
// ==========================================================================

// metamodule.h old annoyquerymeta.h
#pragma once

#include <Python.h>

// Canonical "meta" view of results: column-oriented, pandas-like.
extern PyTypeObject NNSMetaType;

// Initialize and attach NNSMetaType to a module.
// Returns 0 on success, -1 on failure.
int Annoy_InitMetaType(PyObject* module);


#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

namespace Annoy {

// ----------------------------------------------------------------------
// Result "shape" – how Python should interpret the result
// ----------------------------------------------------------------------
enum class ResultType {
    DISTANCES,      // SERIES as indice + distance (1D, Series-like)
    EMBEDDINGS,  // SERIES as indice + value (vectors)
    FRAME        // FRAME as full schema: index / indice / distance / value / metrics
};

// ----------------------------------------------------------------------
// Metadata for a query
// ----------------------------------------------------------------------
struct AnnoyQueryInfo {
    int query_item_id = -1;              // -1 if it's a pure vector query
    std::vector<float> query_vector;     // original query embedding (if any)
    std::string metric_name;             // e.g. "angular", "euclidean"
    int64_t search_k = -1;               // Annoy search_k parameter
    bool include_distances = false;      // true if distances were requested
};

// ======================================================================
// AnnoyQueryMeta – canonical, pandas-friendly schema + metadata
// ======================================================================
struct AnnoyQueryMeta {
  // Canonical data components
  std::vector<int32_t> index;   // 0..n-1
  std::vector<int32_t> indice;  // neighbor ids
  std::vector<float>   distance;
  std::vector<std::vector<float>> value;   // row payload (e.g. embeddings)

  std::vector<std::string>        metric_names;
  std::vector<std::vector<float>> metrics; // metrics[k][i]

  // Metadata + status
  AnnoyQueryInfo query;
  ResultType result_type = ResultType::DISTANCES;

  bool ok = true;
  std::string error;
  bool validated = false;

  // --- ctors ------------------------------------------------
  AnnoyQueryMeta() = default;

  explicit AnnoyQueryMeta(size_t n)
      : index(n), indice(n), distance(n), value(n) {}

  AnnoyQueryMeta(std::vector<int32_t>&& ids,
                 std::vector<float>&& dist)
      : index(ids.size()),
        indice(std::move(ids)),
        distance(std::move(dist)),
        value(index.size())
  {
      for (size_t i = 0; i < index.size(); ++i)
          index[i] = static_cast<int32_t>(i);
  }

  // --- payload assignment -----------------------------------
  void set_value_row(size_t row, std::vector<float>&& vec) {
      if (row >= value.size())
          throw std::runtime_error("AnnoyQueryMeta::set_value_row: row index out of range");
      value[row] = std::move(vec);
  }

  void set_value_table(std::vector<std::vector<float>>&& table) {
      if (table.size() != indice.size())
          throw std::runtime_error("AnnoyQueryMeta::set_value_table: size mismatch with indice");
      value = std::move(table);
  }

  // --- extra metrics ----------------------------------------
  void add_metric(const std::string& name,
                  std::vector<float>&& col)
  {
      if (col.size() != indice.size())
          throw std::runtime_error("AnnoyQueryMeta::add_metric: size mismatch with indice");
      metric_names.push_back(name);
      metrics.push_back(std::move(col));
  }

  // --- slicing (iloc-style) ---------------------------------
  AnnoyQueryMeta slice(size_t start, size_t end) const {
    AnnoyQueryMeta r;

    if (start >= end || start >= indice.size())
        return r;

    end = std::min(end, indice.size());
    const size_t len = end - start;

    r.index    = std::vector<int32_t>(len);
    r.indice   = std::vector<int32_t>(indice.begin()   + static_cast<std::ptrdiff_t>(start),
                                      indice.begin()   + static_cast<std::ptrdiff_t>(end));
    r.distance = std::vector<float>(distance.begin()   + static_cast<std::ptrdiff_t>(start),
                                    distance.begin()   + static_cast<std::ptrdiff_t>(end));

    for (size_t i = 0; i < len; ++i)
        r.index[i] = static_cast<int32_t>(i);

    r.value.reserve(len);
    for (size_t i = start; i < end; ++i)
        r.value.push_back(value[i]);

    r.metric_names = metric_names;
    for (const auto& col : metrics) {
        r.metrics.emplace_back(col.begin() + static_cast<std::ptrdiff_t>(start),
                               col.begin() + static_cast<std::ptrdiff_t>(end));
    }

    r.query       = query;
    r.result_type = result_type;
    r.ok          = ok;
    r.error       = error;
    r.validated   = false;

    return r;
  }

  // --- utilities --------------------------------------------
  size_t size()  const noexcept { return indice.size(); }
  bool   empty() const noexcept { return indice.empty(); }

  void clear() {
    index.clear();
    indice.clear();
    distance.clear();
    value.clear();
    metric_names.clear();
    metrics.clear();
    query = AnnoyQueryInfo{};
    result_type = ResultType::DISTANCES;
    ok = true;
    error.clear();
    validated = false;
  }

  // --- validation -------------------------------------------
  bool validate() {
      validated = true;
      ok = true;
      error.clear();

      const size_t n = indice.size();

      if (!index.empty() && index.size() != n) {
          ok = false;
          error = "AnnoyQueryMeta::validate: index size mismatch";
          return ok;
      }
      if (!distance.empty() && distance.size() != n) {
          ok = false;
          error = "AnnoyQueryMeta::validate: distance size mismatch";
          return ok;
      }
      if (!value.empty() && value.size() != n) {
          ok = false;
          error = "AnnoyQueryMeta::validate: value size mismatch";
          return ok;
      }
      if (!metric_names.empty() && metric_names.size() != metrics.size()) {
          ok = false;
          error = "AnnoyQueryMeta::validate: metric_names / metrics size mismatch";
          return ok;
      }
      for (size_t k = 0; k < metrics.size(); ++k) {
          if (metrics[k].size() != n) {
              ok = false;
              error = "AnnoyQueryMeta::validate: metric column size mismatch at index " +
                      std::to_string(k);
              return ok;
          }
      }
      if (!index.empty()) {
          for (size_t i = 0; i < index.size(); ++i) {
              if (index[i] != static_cast<int32_t>(i)) {
                  ok = false;
                  error = "AnnoyQueryMeta::validate: index is not 0..n-1 range";
                  return ok;
              }
          }
      }
      return ok;
  }

  void ensure_valid() {
      if (!validated || !ok) {
          if (!validated)
              validate();
          if (!ok)
              throw std::runtime_error(
                  error.empty()
                      ? "AnnoyQueryMeta::ensure_valid: invalid result"
                      : error
              );
      }
  }
};

} // namespace Annoy
