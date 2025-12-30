// scikitplot/cexternals/_annoy/src/typenames.h
// Copyright 2022-2023 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Header-only helpers: avoid ODR issues and keep portability.
#pragma once

#include <climits>
#include <string>

template <typename T> inline const std::string typeName();

// template <> const std::string typeName<char>() { return "int8"; }
// template <> const std::string typeName<unsigned char>() { return "uint8"; }
// template <> const std::string typeName<short>() { return "int16"; }
// template <> const std::string typeName<unsigned short>() { return "uint16"; }
// template <> const std::string typeName<int>() { return "int32"; }
// template <> const std::string typeName<unsigned int>() { return "uint32"; }
// template <> const std::string typeName<float>() { return "float32"; }
// template <> const std::string typeName<long>() { return "int64"; }
// template <> const std::string typeName<unsigned long>() { return "uint64"; }
// template <> const std::string typeName<double>() { return "float64"; }

template <> inline const std::string typeName<char>() { return "int8"; }
template <> inline const std::string typeName<unsigned char>() { return "uint8"; }
template <> inline const std::string typeName<short>() { return "int16"; }
template <> inline const std::string typeName<unsigned short>() { return "uint16"; }
template <> inline const std::string typeName<int>() { return "int32"; }
template <> inline const std::string typeName<unsigned int>() { return "uint32"; }
template <> inline const std::string typeName<float>() { return "float32"; }

// long size is platform ABI dependent:
// - Windows (LLP64): long is 32-bit
// - Linux/macOS (LP64): long is 64-bit
#if ULONG_MAX == 0xffffffffUL
template <> inline const std::string typeName<long>() { return "int32"; }
template <> inline const std::string typeName<unsigned long>() { return "uint32"; }
#elif ULONG_MAX == 0xffffffffffffffffUL
template <> inline const std::string typeName<long>() { return "int64"; }
template <> inline const std::string typeName<unsigned long>() { return "uint64"; }
#else
#error Unsupported 'unsigned long' size (expected 32-bit or 64-bit)
#endif

template <> inline const std::string typeName<double>() { return "float64"; }
