#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Canonical QuBEC public surface: in-place 64-point Walsh-Hadamard on floats.
 * Implementation is shared with the generic in-tree butterfly (gyrolabe_wht.c).
 */
void gyrolabe_wht64_f32_inplace(float * x);

void gyrolabe_precond_fill_dr_dc(uint64_t seed, float dr[64], float dc[64]);
void gyrolabe_precond_apply_Dc_vec(float * x, const float dc[64]);
void gyrolabe_precond_apply_Dr_vec(float * x, const float dr[64]);
void gyrolabe_precond_apply_DrDc_mat(float * M /* 64x64 row-major */, const float dr[64], const float dc[64]);
void gyrolabe_hadamard_cols64(float * M);
void gyrolabe_hadamard_rows64(float * M);

#if defined(__AVX2__)
void gyrolabe_wht64_f32_inplace_avx2(float * x);
void gyrolabe_wht64_int32_avx2(int32_t * x);
#endif

void gyrolabe_wht64_int32(int32_t * x);
int gyrolabe_wht64_int32_safe(int32_t * x);
void gyrolabe_wht64_verify_self_inverse(void);

static inline int gyrolabe_is_pow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

#ifdef __cplusplus
}
#endif
