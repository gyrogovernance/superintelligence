#ifndef GYROLABE_CORE_H
#define GYROLABE_CORE_H

#include <stddef.h>
#include <stdint.h>

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

GYROLABE_EXPORT int gyromatmul_vec_dot_q8_0_q8_0(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
);

/* c column-major: c[i + j*ldc]; ldc >= m (leading dim in floats). */
GYROLABE_EXPORT int gyromatmul_gemm_f32(
    int m,
    int n,
    int k,
    const float * GYRO_RESTRICT a,
    int lda,
    const float * GYRO_RESTRICT b,
    int ldb,
    float * GYRO_RESTRICT c,
    int ldc
);

/* c column-major: c[i + j*ldc] = dot(row i of A, row j of B); ldc >= m (leading dim in floats). */
GYROLABE_EXPORT int gyromatmul_gemm_q8_0_q8_0(
    int m,
    int n,
    int k,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT a,
    int lda_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT b,
    int ldb_blocks,
    float * GYRO_RESTRICT c,
    int ldc
);

GYROLABE_EXPORT int gyromatmul_out_prod_f32(
    int rows,
    int cols,
    const float * GYRO_RESTRICT x,
    const float * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out,
    int ld_out
);

#ifdef __cplusplus
}
#endif

#endif
