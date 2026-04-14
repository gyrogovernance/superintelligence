#pragma once

#include <stdint.h>

struct ggml_tensor;
struct gyromatmul_block_q8_0;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gyrolabe_qubec_call_stats {
    int used_radial;
    int used_chi;
    int used_chi_gauge;
    int structured_rows;
    int spectral_sparse_rows;
    int dense_rows;
    int parity_mismatch_rows;
    int structured_attempt_rows;
    int exact_witness_rows;
    float max_abs_row_error;
} gyrolabe_qubec_call_stats;

typedef struct gyrolabe_qubec_dispatch_stats {
    int scanned_blocks;
    int structured_blocks;
    int no_structured_fallback;
    int no_k64_blocks;
    int chi_calls;
    int chi_gauge_calls;
} gyrolabe_qubec_dispatch_stats;

int gyrolabe_qubec_matmul_q8_0(
    int m,
    int n,
    int k,
    const struct gyromatmul_block_q8_0 * a,
    int lda_bytes,
    const struct gyromatmul_block_q8_0 * w,
    int ldb_bytes,
    float * c,
    int ldc,
    int weight_col0,
    const struct ggml_tensor * w_tensor,
    const void * w_q8_lookup_base,
    int row_start,
    int k_block,
    int32_t cell_idx
);

void gyrolabe_qubec_get_last_call_stats(gyrolabe_qubec_call_stats * out_stats);
void gyrolabe_qubec_get_last_dispatch_stats(gyrolabe_qubec_dispatch_stats * out_stats);

const char * gyrolabe_qubec_residual_mode_label(void);

#ifdef __cplusplus
}
#endif
