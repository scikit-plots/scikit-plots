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
//
// | Type | Introduced | Exact Width Guaranteed? | Signedness | ISO C Minimum Width | Typical Width (Modern) | Minimum (Formula) | Maximum (Formula) | Overflow Semantics | Portability Risk | Notes |
// |------|------------|------------------------|------------|---------------------|------------------------|-------------------|-------------------|-------------------|------------------|------|
// | char | C89 | No | Implementation-defined | ≥8 bits | 8 bits | 0 OR -2^(n-1) | 2^n-1 OR 2^(n-1)-1 | Signed: UB, Unsigned: modulo | HIGH | Cannot assume signedness |
// | signed char | C89 | No | Signed | ≥8 bits | 8 bits | -2^(n-1) | 2^(n-1)-1 | UB on overflow | LOW | Distinct type from char |
// | unsigned char | C89 | No | Unsigned | ≥8 bits | 8 bits | 0 | 2^n-1 | Defined modulo 2^n | LOW | Only integer type guaranteed to represent raw bytes |
// | short | C89 | No | Signed | ≥16 bits | 16 bits | -2^(n-1) | 2^(n-1)-1 | UB on overflow | LOW | At least 16 bits |
// | unsigned short | C89 | No | Unsigned | ≥16 bits | 16 bits | 0 | 2^n-1 | Defined modulo 2^n | LOW | |
// | int | C89 | No | Signed | ≥16 bits | 32 bits | -2^(n-1) | 2^(n-1)-1 | UB on overflow | MED | Most efficient integer type for target |
// | unsigned int | C89 | No | Unsigned | ≥16 bits | 32 bits | 0 | 2^n-1 | Defined modulo 2^n | LOW | |
// | long | C89 | No | Signed | ≥32 bits | 32 or 64 bits | -2^(n-1) | 2^(n-1)-1 | UB on overflow | HIGH | 32-bit (Windows), 64-bit (Linux/macOS) |
// | unsigned long | C89 | No | Unsigned | ≥32 bits | 32 or 64 bits | 0 | 2^n-1 | Defined modulo 2^n | HIGH | Platform dependent |
// | long long | C99 | No | Signed | ≥64 bits | 64 bits | -2^(n-1) | 2^(n-1)-1 | UB on overflow | LOW | At least 64 bits |
// | unsigned long long | C99 | No | Unsigned | ≥64 bits | 64 bits | 0 | 2^n-1 | Defined modulo 2^n | LOW | |
// | int8_t | C99 | YES (if exists) | Signed | Exactly 8 bits | 8 bits | -2^7 | 2^7-1 | UB on overflow | NONE | Exists only if platform supports 8-bit type |
// | uint8_t | C99 | YES (if exists) | Unsigned | Exactly 8 bits | 8 bits | 0 | 2^8-1 | Defined modulo 2^8 | NONE | |
// | int16_t | C99 | YES (if exists) | Signed | Exactly 16 bits | 16 bits | -2^15 | 2^15-1 | UB on overflow | NONE | |
// | uint16_t | C99 | YES (if exists) | Unsigned | Exactly 16 bits | 16 bits | 0 | 2^16-1 | Defined modulo 2^16 | NONE | |
// | int32_t | C99 | YES (if exists) | Signed | Exactly 32 bits | 32 bits | -2^31 | 2^31-1 | UB on overflow | NONE | |
// | uint32_t | C99 | YES (if exists) | Unsigned | Exactly 32 bits | 32 bits | 0 | 2^32-1 | Defined modulo 2^32 | NONE | |
// | int64_t | C99 | YES (if exists) | Signed | Exactly 64 bits | 64 bits | -2^63 | 2^63-1 | UB on overflow | NONE | |
// | uint64_t | C99 | YES (if exists) | Unsigned | Exactly 64 bits | 64 bits | 0 | 2^64-1 | Defined modulo 2^64 | NONE | |
//
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | Type                 | Introduced | Exact Width Guaranteed | Signedness            | ISO C Minimum Width | Typical Width (Modern) | Minimum (Formula) | Maximum (Formula) | Overflow Semantics | Portability Risk | Notes                                      |
// +======================+============+========================+=======================+=====================+========================+===================+===================+===================+==================+=============================================+
// | char                 | C89        | No                     | Implementation-defined| >=8 bits            | 8 bits                 | 0 OR -2^(n-1)     | 2^n-1 OR 2^(n-1)-1| Signed: UB        | HIGH             | Cannot assume signedness                    |
// |                      |            |                        |                       |                     |                        |                   |                   | Unsigned: modulo  |                  |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | signed char          | C89        | No                     | Signed                | >=8 bits            | 8 bits                 | -2^(n-1)          | 2^(n-1)-1         | UB on overflow    | LOW              | Distinct type from char                     |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | unsigned char        | C89        | No                     | Unsigned              | >=8 bits            | 8 bits                 | 0                 | 2^n-1             | Modulo 2^n        | LOW              | Only type guaranteed to hold raw byte data  |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | short                | C89        | No                     | Signed                | >=16 bits           | 16 bits                | -2^(n-1)          | 2^(n-1)-1         | UB on overflow    | LOW              | At least 16 bits                            |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | unsigned short       | C89        | No                     | Unsigned              | >=16 bits           | 16 bits                | 0                 | 2^n-1             | Modulo 2^n        | LOW              |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | int                  | C89        | No                     | Signed                | >=16 bits           | 32 bits                | -2^(n-1)          | 2^(n-1)-1         | UB on overflow    | MED              | Most efficient native integer               |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | unsigned int         | C89        | No                     | Unsigned              | >=16 bits           | 32 bits                | 0                 | 2^n-1             | Modulo 2^n        | LOW              |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | long                 | C89        | No                     | Signed                | >=32 bits           | 32 or 64 bits          | -2^(n-1)          | 2^(n-1)-1         | UB on overflow    | HIGH             | 32-bit (Windows), 64-bit (Linux/macOS)      |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | unsigned long        | C89        | No                     | Unsigned              | >=32 bits           | 32 or 64 bits          | 0                 | 2^n-1             | Modulo 2^n        | HIGH             | Platform dependent                          |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | long long            | C99        | No                     | Signed                | >=64 bits           | 64 bits                | -2^63             | 2^63-1            | UB on overflow    | LOW              | At least 64 bits                            |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | unsigned long long   | C99        | No                     | Unsigned              | >=64 bits           | 64 bits                | 0                 | 2^64-1            | Modulo 2^64       | LOW              |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | int8_t               | C99        | Yes (if exists)        | Signed                | Exactly 8 bits      | 8 bits                 | -2^7              | 2^7-1             | UB on overflow    | NONE             | Requires exact-width support                |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | uint8_t              | C99        | Yes (if exists)        | Unsigned              | Exactly 8 bits      | 8 bits                 | 0                 | 2^8-1             | Modulo 2^8        | NONE             |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | int16_t              | C99        | Yes (if exists)        | Signed                | Exactly 16 bits     | 16 bits                | -2^15             | 2^15-1            | UB on overflow    | NONE             |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | uint16_t             | C99        | Yes (if exists)        | Unsigned              | Exactly 16 bits     | 16 bits                | 0                 | 2^16-1            | Modulo 2^16       | NONE             |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | int32_t              | C99        | Yes (if exists)        | Signed                | Exactly 32 bits     | 32 bits                | -2^31             | 2^31-1            | UB on overflow    | NONE             |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | uint32_t             | C99        | Yes (if exists)        | Unsigned              | Exactly 32 bits     | 32 bits                | 0                 | 2^32-1            | Modulo 2^32       | NONE             |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | int64_t              | C99        | Yes (if exists)        | Signed                | Exactly 64 bits     | 64 bits                | -2^63             | 2^63-1            | UB on overflow    | NONE             |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
// | uint64_t             | C99        | Yes (if exists)        | Unsigned              | Exactly 64 bits     | 64 bits                | 0                 | 2^64-1            | Modulo 2^64       | NONE             |                                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+-------------------+------------------+---------------------------------------------+
//
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
