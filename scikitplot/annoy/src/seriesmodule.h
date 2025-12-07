// ======================= NNSSeries Python type ============================
//
// Wraps AnnoyQueryMeta as a Series-like object with:
// NNSDistances,   // SERIES as indice + distance (1D, Series-like)
// NNSEmbeddings,  // SERIES as indice + value (vectors)

// seriesmodule.h
#pragma once

#include <Python.h>
#include "metamodule.h"
#include "seriesmodule.h"

// Series-like 1D view: distance / value / single column.
extern PyTypeObject NNSSeriesType;

int Annoy_InitSeriesType(PyObject* module);
