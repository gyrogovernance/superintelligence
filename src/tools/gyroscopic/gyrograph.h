#ifndef GYROGRAPH_H
#define GYROGRAPH_H

#include <stdint.h>

#include "gyrolabe_registry.h"
#include "gyrograph_types.h"

#if defined(_WIN32) || defined(_WIN64)
#  define GYROGRAPH_EXPORT __declspec(dllexport)
#else
#  define GYROGRAPH_EXPORT __attribute__((visibility("default")))
#endif

#if defined(_MSC_VER)
#  define GYRO_RESTRICT __restrict
#else
#  define GYRO_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

GYROGRAPH_EXPORT void gyrograph_init(void);

GYROGRAPH_EXPORT void gyrograph_trace_word4_batch_indexed(
    const int64_t * GYRO_RESTRICT cell_ids,
    const int32_t * GYRO_RESTRICT omega12_in,
    const uint8_t * GYRO_RESTRICT words4_in,
    int64_t n,
    int32_t * GYRO_RESTRICT omega_trace4_out,
    uint8_t * GYRO_RESTRICT chi_trace4_out
);

GYROGRAPH_EXPORT void gyrograph_apply_trace_word4_batch_indexed(
    const int64_t * GYRO_RESTRICT cell_ids,
    int32_t * GYRO_RESTRICT omega12_io,
    uint64_t * GYRO_RESTRICT step_io,
    uint8_t * GYRO_RESTRICT last_byte_io,
    uint8_t * GYRO_RESTRICT has_closed_word_io,
    uint8_t * GYRO_RESTRICT word4_io,
    uint8_t * GYRO_RESTRICT chi_ring64_io,
    uint8_t * GYRO_RESTRICT chi_ring_pos_io,
    uint8_t * GYRO_RESTRICT chi_valid_len_io,
    uint16_t * GYRO_RESTRICT chi_hist64_io,
    uint16_t * GYRO_RESTRICT shell_hist7_io,
    uint8_t * GYRO_RESTRICT family_ring64_io,
    uint16_t * GYRO_RESTRICT family_hist4_io,
    int32_t * GYRO_RESTRICT omega_sig_io,
    uint16_t * GYRO_RESTRICT parity_O12_io,
    uint16_t * GYRO_RESTRICT parity_E12_io,
    uint8_t * GYRO_RESTRICT parity_bit_io,
    const uint8_t * GYRO_RESTRICT words4_in,
    const int32_t * GYRO_RESTRICT omega_trace4_in,
    const uint8_t * GYRO_RESTRICT chi_trace4_in,
    uint32_t * GYRO_RESTRICT resonance_key_io,
    uint8_t profile,
    int64_t n
);

GYROGRAPH_EXPORT void gyrograph_ingest_word4_batch_indexed(
    const int64_t * GYRO_RESTRICT cell_ids,
    int32_t * GYRO_RESTRICT omega12_io,
    uint64_t * GYRO_RESTRICT step_io,
    uint8_t * GYRO_RESTRICT last_byte_io,
    uint8_t * GYRO_RESTRICT has_closed_word_io,
    uint8_t * GYRO_RESTRICT word4_io,
    uint8_t * GYRO_RESTRICT chi_ring64_io,
    uint8_t * GYRO_RESTRICT chi_ring_pos_io,
    uint8_t * GYRO_RESTRICT chi_valid_len_io,
    uint16_t * GYRO_RESTRICT chi_hist64_io,
    uint16_t * GYRO_RESTRICT shell_hist7_io,
    uint8_t * GYRO_RESTRICT family_ring64_io,
    uint16_t * GYRO_RESTRICT family_hist4_io,
    int32_t * GYRO_RESTRICT omega_sig_io,
    uint16_t * GYRO_RESTRICT parity_O12_io,
    uint16_t * GYRO_RESTRICT parity_E12_io,
    uint8_t * GYRO_RESTRICT parity_bit_io,
    const uint8_t * GYRO_RESTRICT words4_in,
    uint32_t * GYRO_RESTRICT resonance_key_io,
    uint8_t profile,
    int64_t n
);

GYROGRAPH_EXPORT double gyrograph_compute_m2_empirical(
    const uint16_t * GYRO_RESTRICT chi_hist64,
    uint64_t total
);

GYROGRAPH_EXPORT int gyrograph_strict(void);

GYROGRAPH_EXPORT double gyrograph_compute_m2_equilibrium(
    const uint16_t * GYRO_RESTRICT shell_hist7,
    uint64_t total
);

/* cell_id indexes parallel *_io arrays; caller must size buffers past max id. */
GYROGRAPH_EXPORT int gyrograph_pack_moment(
    int64_t cell_id,
    const int32_t *omega12_io,
    const uint64_t *step_io,
    const uint8_t *last_byte_io,
    const uint16_t *parity_O12_io,
    const uint16_t *parity_E12_io,
    const uint8_t *parity_bit_io,
    const int32_t *omega_sig_io,
    gyrograph_moment *out
);

GYROGRAPH_EXPORT int gyrograph_word_signature_from_bytes(
    const uint8_t *bytes,
    int64_t n,
    gyrograph_word_signature *out
);

GYROGRAPH_EXPORT int gyrograph_compose_signatures(
    const gyrograph_word_signature *left,
    const gyrograph_word_signature *right,
    gyrograph_word_signature *out
);

GYROGRAPH_EXPORT int gyrograph_apply_signature(
    uint32_t state24,
    const gyrograph_word_signature *sig,
    uint32_t *out_state24
);

GYROGRAPH_EXPORT int gyrograph_compute_constitutional(
    uint32_t state24,
    gyrograph_constitutional *out
);

GYROGRAPH_EXPORT int gyrograph_emit_slcp_batch(
    int64_t n_cells,
    const int64_t *cell_ids,
    const int32_t *omega12_io,
    const uint64_t *step_io,
    const uint8_t *last_byte_io,
    const uint8_t *word4_io,
    const uint16_t *chi_hist64_io,
    const uint16_t *shell_hist7_io,
    const uint8_t *family_hist4_io,
    const int32_t *omega_sig_io,
    const uint16_t *parity_O12_io,
    const uint16_t *parity_E12_io,
    const uint8_t *parity_bit_io,
    const uint32_t *resonance_key_io,
    gyrograph_slcp_record *outs
);

GYROGRAPH_EXPORT int gyrograph_emit_slcp(
    int64_t cell_id,
    const int32_t *omega12_io,
    const uint64_t *step_io,
    const uint8_t *last_byte_io,
    const uint8_t *word4_io,
    const uint16_t *chi_hist64_io,
    const uint16_t *shell_hist7_io,
    const uint8_t *family_hist4_io,
    const int32_t *omega_sig_io,
    const uint16_t *parity_O12_io,
    const uint16_t *parity_E12_io,
    const uint8_t *parity_bit_io,
    const uint32_t *resonance_key_io,
    gyrograph_slcp_record *out
);

GYROGRAPH_EXPORT int gyrograph_moment_from_ledger(
    const uint8_t *ledger,
    int64_t len,
    gyrograph_moment *out
);

GYROGRAPH_EXPORT int gyrograph_verify_moment(
    const gyrograph_moment *m,
    const uint8_t *ledger,
    int64_t len
);

GYROGRAPH_EXPORT int gyrograph_compare_ledgers(
    const uint8_t *a,
    int64_t alen,
    const uint8_t *b,
    int64_t blen,
    int64_t *common_prefix_out
);

GYROGRAPH_EXPORT uint32_t gyrograph_step_state24_by_byte(uint32_t state24, uint8_t b);

GYROGRAPH_EXPORT uint32_t gyrograph_inverse_step_state24_by_byte(uint32_t state24, uint8_t b);

#ifdef __cplusplus
}
#endif

#endif
