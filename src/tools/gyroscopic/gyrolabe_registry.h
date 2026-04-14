#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Block matmul metadata: GENERIC = 4. Not interchangeable with gyrolabe_operator_class_t. */
typedef enum {
    GYRO_CLASS_SHELL_RADIAL = 0,
    GYRO_CLASS_SHELL_X_GAUGE = 1,
    GYRO_CLASS_CHI_INVARIANT = 2,
    GYRO_CLASS_CHI_X_GAUGE = 3,
    GYRO_CLASS_GENERIC = 4,
} gyro_class_id_t;

#if defined(_MSC_VER)
#define GYRO_ALIGN64 __declspec(align(64))
#else
#define GYRO_ALIGN64 __attribute__((aligned(64)))
#endif

typedef struct GYRO_ALIGN64 {
    uint8_t class_id;
    /*
     * dq_lattice_empty (written only in pack_DQ_lattice, model load / register time):
     * 1 if every integer defect entry D[i][j] = B[i][j] - P[i][j] is exactly zero, so the
     * packed K4 lattice representation of D has no bits to apply. Runtime check is a single byte.
     * Hot path: gyrolabe_qubec_matmul_q8_0 may skip k4_gemv64_avx2 when this is 1 (y_d = 0),
     * which is exact for D == 0. Do not repurpose without updating pack_DQ_lattice.
     */
    uint8_t dq_lattice_empty;
    uint8_t _pad[2];
    /*
     * Per-class eigen storage (interpret by class_id):
     * - CHI_INVARIANT: phi_64[] = unnormalized float WHT of kernel row B[0,*] (matches
     *   gyrolabe_wht64_f32_inplace twice + single 1/64 in apply_spectral_tile64).
     * - SHELL_RADIAL: lambda_7[] from is_shell_radial_exact (same scale as K7 inverse /64).
     * SDK gyrolabe_analyze_operator_64 uses a different normalized convention for floats.
     */
    union {
        int32_t lambda_7[7];
        int32_t lambda_28[28];
        float phi_64[64];
        int32_t Phi_256[256];
    } eigenvalues;
    struct {
        uint64_t sign_mask[64];
        uint64_t bitplanes[64][16];
        float scale_w;
    } packed_DQ;
    uint64_t valid_col_mask;
    uint16_t ne_orig;
    uint16_t _pad2;
} gyrolabe_block_info_t;

GGML_API void gyrolabe_registry_register_tensor(const struct ggml_tensor * w);
GGML_API void gyrolabe_registry_register_q8_buffer(
    const void * data,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    size_t row_stride_bytes,
    const char * name
);

GGML_API const gyrolabe_block_info_t * gyrolabe_registry_get_block(
    const struct ggml_tensor * w,
    const void * w_q8_base,
    int row_block,
    int k_block
);

typedef void * gyrolabe_registry_entry_t;

GGML_API gyrolabe_registry_entry_t gyrolabe_registry_find_entry(const struct ggml_tensor * w, const void * data);

GGML_API const gyrolabe_block_info_t * gyrolabe_registry_get_block_from_entry(
    gyrolabe_registry_entry_t entry,
    int row_block,
    int k_block
);

GGML_API float gyrolabe_registry_tensor_max_scr(const struct ggml_tensor * w);

GGML_API int gyrolabe_registry_n_blocks(int n_cols);
GGML_API int gyrolabe_registry_entry_count(void);

GGML_API void gyrolabe_registry_clear(void);

#ifdef __cplusplus
}
#endif
