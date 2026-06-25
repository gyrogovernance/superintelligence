#pragma once

/* kernel/native.h — gyrocrypt native C API (compiled to gyrocrypt_native.dll). */

#include <stdint.h>

#if defined(_WIN32) || defined(_WIN64)
#  ifndef GYROSCOPIC_EXPORT
#    define GYROSCOPIC_EXPORT __declspec(dllexport)
#  endif
#else
#  ifndef GYROSCOPIC_EXPORT
#    define GYROSCOPIC_EXPORT __attribute__((visibility("default")))
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

GYROSCOPIC_EXPORT uint64_t gyroscopic_mul_mod_ladder(
    uint64_t y,
    uint64_t multiplier,
    uint64_t n
);

GYROSCOPIC_EXPORT uint64_t gyroscopic_exp_mod_ladder(
    uint64_t a,
    uint64_t x,
    uint64_t n
);

GYROSCOPIC_EXPORT int gyroscopic_exp_mod_ladder_limbs(
    const uint32_t *a_limbs,
    int a_n,
    const uint32_t *x_limbs,
    int x_n,
    const uint32_t *n_limbs,
    int n_n,
    uint32_t *out_limbs,
    int out_n
);

GYROSCOPIC_EXPORT int gyroscopic_sparse_cqft_peaks(
    const uint32_t *support,
    int n_support,
    uint32_t Q,
    int k_top,
    uint32_t *out_k,
    double *out_mag2,
    int out_cap
);

GYROSCOPIC_EXPORT uint32_t gyroscopic_shor_period_u64(
    uint64_t base,
    uint64_t n,
    uint64_t Q
);

GYROSCOPIC_EXPORT const char *gyroscopic_shor_last_path_tag(void);

GYROSCOPIC_EXPORT double gyroscopic_shor_dp_mag2_y1_u64(
    uint64_t base,
    uint64_t n,
    uint64_t Q,
    uint64_t k
);

GYROSCOPIC_EXPORT int gyroscopic_horizon_pack_keys_u64(
    uint64_t n,
    uint64_t *keys_out,
    int cap
);

GYROSCOPIC_EXPORT int gyroscopic_horizon_n_cells_u64(uint64_t n);

GYROSCOPIC_EXPORT uint64_t gyroscopic_horizon_key_u64(uint64_t n, uint64_t y);

GYROSCOPIC_EXPORT double gyroscopic_horizon_tensor_mag2_y1_u64(
    uint64_t base,
    uint64_t n,
    uint64_t Q,
    uint64_t k,
    const uint64_t *keys,
    int n_cells
);

GYROSCOPIC_EXPORT int gyroscopic_horizon_tensor_step_drift_u64(
    uint64_t base,
    uint64_t n,
    uint64_t Q,
    uint64_t k,
    const uint64_t *keys,
    int n_cells,
    double *out_exact_mag2,
    double *out_tensor_mag2,
    int out_cap
);

GYROSCOPIC_EXPORT double gyroscopic_dlp_2d_tensor_mag2_u64(
    uint64_t base_g,
    uint64_t base_h,
    uint64_t n,
    uint64_t Q,
    uint64_t k1,
    uint64_t k2,
    const uint64_t *keys,
    int n_cells
);

#ifdef __cplusplus
}
#endif
