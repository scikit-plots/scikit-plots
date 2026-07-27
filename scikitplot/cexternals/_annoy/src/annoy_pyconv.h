// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Checked Python integer conversion for the COUNT / SIZE semantic type
// (ANNOY-CONV-001, guide 6.11).
//
// Non-negative quantities such as get_n_items(), get_n_trees(), node counts and
// byte sizes are held as IndexDtype / uint64_t / size_t. Converting them with a
// *signed* PyLong_FromLongLong misrepresents any value above LLONG_MAX (the sign
// bit is reinterpreted). AnnoyCountToPy preserves the full UNSIGNED range, so a
// count up to UINT64_MAX round-trips exactly.
//
// This mirrors the existing AnnoyIdxToPy helper: counts and item indices share
// the IndexDtype width, so both use the unsigned wide constructor. If IndexDtype
// is switched to uint32_t (see the centralized dtype block in annoymodule.cc),
// swap PyLong_FromUnsignedLongLong -> PyLong_FromUnsignedLong here too.
#ifndef ANNOY_PYCONV_H
#define ANNOY_PYCONV_H

#include <Python.h>

// Active variant: IndexDtype = uint64_t -> unsigned long long ("K").
#define AnnoyCountToPy(val)  PyLong_FromUnsignedLongLong((unsigned long long)(val))

// 32-bit variant (use instead when IndexDtype = uint32_t):
// #define AnnoyCountToPy(val)  PyLong_FromUnsignedLong((unsigned long)(val))

#endif  // ANNOY_PYCONV_H
