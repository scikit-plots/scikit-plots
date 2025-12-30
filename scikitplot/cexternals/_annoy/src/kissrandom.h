// scikitplot/cexternals/_annoy/src/kissrandom.h
#ifndef ANNOY_KISSRANDOM_H
#define ANNOY_KISSRANDOM_H

// MSVC 2008 lacks <stdint.h>.
#if defined(_MSC_VER) && _MSC_VER == 1500
  typedef unsigned __int32    uint32_t;
  typedef unsigned __int64    uint64_t;
#else
  #include <stdint.h>
#endif

#include <stddef.h>  // size_t

namespace Annoy {

// KISS = "keep it simple, stupid", but high quality random number generator
// http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf -> "Use a good RNG and build it into your code"
// http://mathforum.org/kb/message.jspa?messageID=6627731
// https://de.wikipedia.org/wiki/KISS_(Zufallszahlengenerator)
//
// KISS = "keep it simple, stupid" — a compact, high-quality RNG family.
//
// References (background / rationale):
//   - http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf
//   - http://mathforum.org/kb/message.jspa?messageID=6627731
//   - https://de.wikipedia.org/wiki/KISS_(Zufallszahlengenerator)
//
// Design invariants:
//   - Seeds are normalized so that seed==0 maps to default_seed (non-zero).
//   - set_seed() resets the entire internal state (not just one field), so the
//     generator is fully deterministic given the seed.
//   - index(n) is defined for all n; when n==0 it returns 0 (avoids UB).
//
// Notes
// -----
// - KISS is designed for speed and reproducibility, not cryptographic security.
//   Do NOT use it for secrets, tokens, or security-sensitive randomness.
// - The implementation matches a Marsaglia-style KISS recipe: a combination
//   of a linear congruential generator (LCG), Xorshift, and multiply-with-carry
//   (MWC) component.

// ---------------------------------------------------------------------------
// 32-bit KISS
// ---------------------------------------------------------------------------
struct Kiss32Random {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t c;

  static const uint32_t default_seed = 123456789u;

  // Fixed non-zero initial state constants (Marsaglia-style KISS).
  static const uint32_t default_y = 362436000u;
  static const uint32_t default_z = 521288629u;
  static const uint32_t default_c = 7654321u;

#if __cplusplus < 201103L
  typedef uint32_t seed_type;
#endif

  // Normalize a user seed to a valid non-zero seed.
  //
  // Notes
  // -----
  // KISS combines multiple sub-generators (LCG, Xorshift, MWC). Certain
  // all-zero states are degenerate for these components, so we map seed==0
  // to a deterministic non-zero default seed.
  static inline uint32_t normalize_seed(uint32_t seed) {
    return seed ? seed : default_seed;
  }

  inline void reset_default() { reset(default_seed); }

  // seed must be != 0 (seed=0 is normalized to default_seed)
  explicit Kiss32Random(uint32_t seed = default_seed) {
    reset(seed);
  }

  inline void reset(uint32_t seed) {
    seed = normalize_seed(seed);
    x = seed;
    y = default_y;
    z = default_z;
    c = default_c;
  }

  inline void set_seed(uint32_t seed) {
    reset(seed);
  }

  inline uint32_t kiss() {
    // Linear congruence generator
    x = 69069u * x + 12345u;

    // Xor shift
    y ^= y << 13;
    y ^= y >> 17;
    y ^= y << 5;

    // Multiply-with-carry
    const uint64_t t = 698769069ULL * static_cast<uint64_t>(z) + static_cast<uint64_t>(c);
    c = static_cast<uint32_t>(t >> 32);
    z = static_cast<uint32_t>(t);

    return x + y + z;
  }

  inline int flip() {
    // Draw random 0 or 1
    return static_cast<int>(kiss() & 1u);
  }

  inline size_t index(size_t n) {
    // Draw integer in [0, n-1]. Defined for all n; n==0 returns 0.
    return n ? (static_cast<size_t>(kiss()) % n) : 0u;
  }
};

// ---------------------------------------------------------------------------
// 64-bit KISS
// ---------------------------------------------------------------------------
// Use this if you have more than about 2^24 data points ("big data" ;)).
struct Kiss64Random {
  uint64_t x;
  uint64_t y;
  uint64_t z;
  uint64_t c;

  static const uint64_t default_seed = 1234567890987654321ULL;

  // Fixed non-zero initial state constants.
  static const uint64_t default_y = 362436362436362436ULL;
  static const uint64_t default_z = 1066149217761810ULL;
  static const uint64_t default_c = 123456123456123456ULL;

#if __cplusplus < 201103L
  typedef uint64_t seed_type;
#endif

  // Normalize a user seed to a valid non-zero seed.
  // See Kiss32Random::normalize_seed for rationale.
  static inline uint64_t normalize_seed(uint64_t seed) {
    // bool ? truthy : falsy
    return seed ? seed : default_seed;
  }

  inline void reset_default() { reset(default_seed); }

  // seed must be != 0 (seed=0 is normalized to default_seed)
  explicit Kiss64Random(uint64_t seed = default_seed) {
    reset(seed);
  }

  inline void reset(uint64_t seed) {
    seed = normalize_seed(seed);
    x = seed;
    y = default_y;
    z = default_z;
    c = default_c;
  }

  inline void set_seed(uint64_t seed) {
    reset(seed);
  }

  inline uint64_t kiss() {
    // Linear congruence generator
    z = 6906969069ULL * z + 1234567ULL;

    // Xor shift
    y ^= (y << 13);
    y ^= (y >> 17);
    y ^= (y << 43);

    // Multiply-with-carry (MWC)
    const uint64_t t = (x << 58) + c;
    c = (x >> 6);
    x += t;
    c += (x < t);

    return x + y + z;
  }

  inline int flip() {
    // Draw random 0 or 1
    return static_cast<int>(kiss() & 1ULL);
  }

  inline size_t index(size_t n) {
    // Draw integer in [0, n-1]. Defined for all n; n==0 returns 0.
    return n ? (static_cast<size_t>(kiss() % static_cast<uint64_t>(n))) : 0u;
  }
};

}  // namespace Annoy

#endif  // ANNOY_KISSRANDOM_H

// vim: tabstop=2 shiftwidth=2
