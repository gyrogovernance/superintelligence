#pragma once

/*
 * Gyroscopic kernel C API.
 *
 * Inference (ggml hook) calls gyroscopic_gravity_scale via quants.c TLS only.
 * Other exports are used by tests and offline diagnostics.
 */

#include <stdint.h>
#include "constants.h"

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

#ifndef GYROSCOPIC_WAVEFUNCTION_SIZE
#define GYROSCOPIC_WAVEFUNCTION_SIZE OMEGA_SIZE
#endif

#ifndef GYROSCOPIC_HOLO_DIM
#define GYROSCOPIC_HOLO_DIM HORIZON_SIZE
#endif

#ifndef GYROSCOPIC_DEFAULT_TOTAL_LAYERS
#define GYROSCOPIC_DEFAULT_TOTAL_LAYERS 36
#endif

enum gyroscopic_k4 {
    GYROSCOPIC_K4_ID  = 0,
    GYROSCOPIC_K4_W2  = 1,
    GYROSCOPIC_K4_W2P = 2,
    GYROSCOPIC_K4_F   = 3
};

typedef struct gyro_accum {
    float a;
    float b;
} gyro_accum_t;

typedef struct gyroscopic_q1_meta {
    uint8_t q_class;
    uint8_t shell;
    uint8_t k4_char;
    uint8_t h_zone;
    uint8_t route_path;
    float   gravity_scale;
} gyroscopic_q1_meta;

#define GYROSCOPIC_PATH_ISOTROPIC 0u
#define GYROSCOPIC_PATH_BULK_CS   1u
#define GYROSCOPIC_PATH_BULK_UNA  2u
#define GYROSCOPIC_PATH_BULK_ONA  3u
#define GYROSCOPIC_PATH_BULK_BU   4u
#define GYROSCOPIC_PATH_COUNT     5u

GYROSCOPIC_EXPORT uint32_t gyroscopic_step_omega12(uint32_t state24, uint8_t byte);

GYROSCOPIC_EXPORT void gyroscopic_apply_K4(
    float psi[GYROSCOPIC_WAVEFUNCTION_SIZE],
    int gate
);

GYROSCOPIC_EXPORT void gyroscopic_to_holographic(
    const float psi[GYROSCOPIC_WAVEFUNCTION_SIZE],
    float holo[GYROSCOPIC_HOLO_DIM][GYROSCOPIC_HOLO_DIM]
);
GYROSCOPIC_EXPORT void gyroscopic_from_holographic(
    const float holo[GYROSCOPIC_HOLO_DIM][GYROSCOPIC_HOLO_DIM],
    float psi[GYROSCOPIC_WAVEFUNCTION_SIZE]
);

GYROSCOPIC_EXPORT uint8_t gyroscopic_chirality_from_signs64(uint64_t signs);

GYROSCOPIC_EXPORT void gyroscopic_analyze_q1_group(
    const uint8_t signs[16],
    uint8_t * q_class,
    uint8_t * shell,
    uint8_t * k4_char
);

GYROSCOPIC_EXPORT void gyroscopic_extract_phase_native(
    const uint8_t signs[16],
    uint8_t * k4_char,
    uint8_t * shell_proxy
);

GYROSCOPIC_EXPORT float gyroscopic_k4_compose_gyroacc(
    const gyro_accum_t accum[4],
    float gravity
);

GYROSCOPIC_EXPORT float gyroscopic_sum_gyroacc(
    const gyro_accum_t accum[4],
    float gravity
);

GYROSCOPIC_EXPORT float gyroscopic_depth4_bu_factor(void);

GYROSCOPIC_EXPORT uint8_t gyroscopic_pack_q1_meta(uint8_t shell, uint8_t k4_char, uint8_t h);
GYROSCOPIC_EXPORT void    gyroscopic_unpack_q1_meta(
    uint8_t packed, uint8_t * shell, uint8_t * k4_char, uint8_t * h_zone
);

GYROSCOPIC_EXPORT uint8_t gyroscopic_route_path(uint8_t shell, uint8_t k4_char);

GYROSCOPIC_EXPORT float gyroscopic_gravity_g1(void);

/** Per-layer scale exp(g1 * L/N). k4_char and shell are ignored for magnitude. */
GYROSCOPIC_EXPORT float gyroscopic_gravity_scale(
    int layer,
    int total_layers,
    uint8_t k4_char,
    uint8_t shell
);

GYROSCOPIC_EXPORT void gyroscopic_analyze_q1_group_full(
    const uint8_t signs[16],
    int layer,
    int total_layers,
    gyroscopic_q1_meta * out
);

/* Native cyclic QFT over Z_{2^n_bits}: radix-2 DIT WHT-atom butterflies. */
GYROSCOPIC_EXPORT void gyroscopic_cyclic_qft(
    float * re,
    float * im,
    int n_bits
);

/* Byte-ledger modular multiply / exponentiate / multiplicative order. */
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
GYROSCOPIC_EXPORT uint64_t gyroscopic_multiplicative_period(
    uint64_t a,
    uint64_t n,
    uint64_t max_len
);

/* Sparse period comb + cyclic QFT spectral peak (Shor readout). q_bits <= 20. */
GYROSCOPIC_EXPORT uint32_t gyroscopic_comb_qft_peak(
    uint64_t period,
    int q_bits,
    float * peak_amp_out
);

#ifdef __cplusplus
}
#endif
