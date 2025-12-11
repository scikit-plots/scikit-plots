// include/nc.hpp
#pragma once

/// @file nc.hpp
/// @brief Central include for scikitplot.nc C++ bindings.
///
/// This header aggregates all pybind11 binding helpers under the
/// scikitplot::nc_bindings namespace. It does not define or extend
/// the ::nc namespace from the upstream NumCpp library.

#include "nc/linalg.hpp"
// later: #include "nc/random.hpp", "nc/stats.hpp", ...
