#ifndef GYROLABE_TRANSFORMS_H
#define GYROLABE_TRANSFORMS_H

#include <stdint.h>

#include "gyrolabe_wht.h"

#if !defined(GYROLABE_EXPORT)
#if defined(_WIN32) || defined(_WIN64)
#  define GYROLABE_EXPORT __declspec(dllexport)
#else
#  define GYROLABE_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// === WHT64 (GyroLabe §3.2) ===
void gyrolabe_wht64_float(float *data);
void gyrolabe_wht64_batch_int32(int32_t ** arrays, int batch_size);
void gyrolabe_wht64_batch_float(float ** arrays, int batch_size);

// === Krawtchouk7 (GyroLabe §3.2) ===
void gyrolabe_krawtchouk7_int32(const int32_t shell_hist[7], int32_t spectral[7]);
void gyrolabe_krawtchouk7_inverse_int32(const int32_t spectral[7], int32_t shell_hist[7]);
GYROLABE_EXPORT void gyrolabe_krawtchouk7_float(
    const float shell_hist[7],
    float spectral[7]
);
GYROLABE_EXPORT void gyrolabe_krawtchouk7_inverse_float(
    const float spectral[7],
    float shell_hist[7]
);

// === K4Char4 (GyroLabe §3.2) ===
void gyrolabe_k4char4_int32(const int32_t family_hist[4], int32_t character[4]);
GYROLABE_EXPORT void gyrolabe_k4char4_float(const float family_hist[4], float character[4]);

// === K4 lattice arithmetic (QuBEC transform algebra Part II) ===
GYROLABE_EXPORT void gyrolabe_k4_decompose_int32(int32_t v, int16_t *L, int16_t *H);
GYROLABE_EXPORT void gyrolabe_k4_contract(
    const int32_t *q,
    const int32_t *k,
    int n,
    int64_t *D00,
    int64_t *D01,
    int64_t *D10,
    int64_t *D11
);
GYROLABE_EXPORT int64_t gyrolabe_k4_dot(const int32_t *q, const int32_t *k, int n);

#ifdef __cplusplus
}
#endif

#endif