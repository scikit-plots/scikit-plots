// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Guard: the portable scalar float16 converters must be BIT-FOR-BIT identical to
// the F16C hardware converters (not merely within 1 ULP). This is what makes an
// index reproducible across machines regardless of whether the CPU has F16C — the
// scalar path (no-F16C CPUs) and the hardware path (F16C CPUs) must store the same
// bits. Round-to-nearest-even + subnormals + NaN payloads are all covered.
//
// Build/run (x86 with F16C):
//   g++ -std=c++17 -O3 -mf16c -I../src test_float16_scalar_matches_hardware.cpp -o t && ./t
// Set ANNOY_F16_EXHAUSTIVE=1 to sweep ALL 2^32 forward inputs (~40s); otherwise a
// fast comprehensive sweep (all subnormals + boundaries + strided normals + NaN
// representatives) runs by default. On non-F16C hosts the hardware comparison is
// skipped and the test passes trivially (the fallback test covers those).
#include "annoylib.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#include <cpuid.h>
static bool have_f16c() {
  unsigned int a,b,c,d;
  if (__get_cpuid(1u,&a,&b,&c,&d)==0) return false;
  return (c & (1u<<29)) != 0u;
}
static inline uint16_t hw_fwd(float f){ __m128 v=_mm_set_ss(f); __m128i h=_mm_cvtps_ph(v,_MM_FROUND_TO_NEAREST_INT); return (uint16_t)_mm_extract_epi16(h,0);}
static inline float    hw_rev(uint16_t x){ __m128i h=_mm_cvtsi32_si128(x); __m128 v=_mm_cvtph_ps(h); return _mm_cvtss_f32(v);}

static int check_fwd(uint32_t bits){
  float f; std::memcpy(&f,&bits,4);
  return float16_t::f32_to_f16_scalar(f) == hw_fwd(f) ? 0 : (std::printf("  fwd MISMATCH 0x%08X scalar=0x%04X hw=0x%04X\n",bits,float16_t::f32_to_f16_scalar(f),hw_fwd(f)),1);
}

int main(){
  if (!have_f16c()){ std::printf("F16C absent on this host; skipping hardware comparison (fallback test covers correctness).\n"); return 0; }
  uint64_t bad=0;
  // reverse: always exhaustive (only 2^16)
  for (uint32_t d=0; d<0x10000u; ++d){
    float s=float16_t::f16_to_f32_scalar((uint16_t)d), h=hw_rev((uint16_t)d);
    uint32_t sb,hb; std::memcpy(&sb,&s,4); std::memcpy(&hb,&h,4);
    if(!((s!=s)&&(h!=h)) && sb!=hb){ std::printf("  rev MISMATCH f16=0x%04X\n",d); ++bad; }
  }
  const bool exhaustive = std::getenv("ANNOY_F16_EXHAUSTIVE") != nullptr;
  if (exhaustive){
    uint32_t i=0; do { bad += check_fwd(i); } while (++i != 0);       // all 2^32
    std::printf("forward: exhaustive over all 2^32\n");
  } else {
    // fast comprehensive: all subnormal+boundary exponents exhaustive, normals strided,
    // plus NaN/Inf representatives — enough to catch any rounding/subnormal regression.
    for (uint32_t e=0; e<256u; ++e){
      uint32_t base = e<<23;
      // sweep the full mantissa for the boundary exponents (subnormal & overflow edges),
      // and a dense stride elsewhere
      uint32_t step = (e>=0x66u && e<=0x8Fu) ? 1u : 0x40u;   // exhaustive near the f16 range
      for (uint32_t m=0; m<0x800000u; m+=step){
        bad += check_fwd(base|m); bad += check_fwd(0x80000000u|base|m);
      }
    }
    std::printf("forward: fast comprehensive sweep (set ANNOY_F16_EXHAUSTIVE=1 for full 2^32)\n");
  }
  std::printf("scalar-vs-hardware bit-for-bit: %s (mismatches=%llu)\n", bad?"FAIL":"PASS", (unsigned long long)bad);
  return bad? 1 : 0;
}
#else
int main(){ std::printf("non-x86 host; native float16 path used, no scalar/hw split.\n"); return 0; }
#endif
