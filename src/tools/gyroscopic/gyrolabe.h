#ifndef GYROLABE_CORE_H
#define GYROLABE_CORE_H

#include <stddef.h>
#include <stdint.h>

#include "gyrograph_types.h"

#if defined(_WIN32) || defined(_WIN64)
#  define GYROLABE_EXPORT __declspec(dllexport)
#else
#  define GYROLABE_EXPORT __attribute__((visibility("default")))
#endif

#if defined(_MSC_VER)
#  define GYRO_RESTRICT __restrict
#else
#  define GYRO_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gyromatmul_runtime_caps {
    uint32_t avx2_enabled;
    uint32_t f16c_enabled;
    uint32_t fma_enabled;
    uint32_t reserved;
} gyromatmul_runtime_caps;

typedef struct gyromatmul_block_q8_0 {
    uint16_t d;
    int8_t   qs[32];
} gyromatmul_block_q8_0;

GYROLABE_EXPORT void gyromatmul_runtime_query(
    gyromatmul_runtime_caps * out_caps
);

GYROLABE_EXPORT int gyromatmul_vec_dot_f32_ref(
    int n,
    const float * GYRO_RESTRICT x,
    const float * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
);

GYROLABE_EXPORT int gyromatmul_vec_dot_f32_avx2(
    int n,
    const float * GYRO_RESTRICT x,
    const float * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
);

GYROLABE_EXPORT int gyromatmul_vec_dot_q8_0_q8_0_ref(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
);

GYROLABE_EXPORT int gyromatmul_vec_dot_q8_0_q8_0_avx2(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
);

GYROLABE_EXPORT int gyromatmul_vec_dot_f32(
    int n,
    const float * GYRO_RESTRICT x,
    const float * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
);

GYROLABE_EXPORT int gyrolabe_analyze_operator_64(
    const float *W_block,
    float threshold,
    gyrolabe_operator_report *out
);

GYROLABE_EXPORT int gyrolabe_apply_structured_64(
    const gyrolabe_operator_report *report,
    const float *x,
    float *y
);

#include "gyrolabe_aperture.h"
#include "gyrolabe_canonical.h"
#include "gyrolabe_evolution.h"

#ifdef __cplusplus
}
#endif

#endif
