#pragma once

/*
 * Gyroscopic kernel C API.
 *
 * Inference (ggml hook) reads per-layer gravity_scale via quants.c TLS only.
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

GYROSCOPIC_EXPORT uint64_t gyroscopic_signs64_from_f32(const float * x);
GYROSCOPIC_EXPORT uint64_t gyroscopic_signs64_from_q8(const int8_t * q, int n);
GYROSCOPIC_EXPORT uint8_t gyroscopic_activation_chirality(const float * x);
GYROSCOPIC_EXPORT uint8_t gyroscopic_activation_chirality_q8(
    const int8_t * q0,
    const int8_t * q1
);

/** Hamming distance on GF(2)^6 chirality words. */
GYROSCOPIC_EXPORT int gyroscopic_chirality_distance(uint8_t chi_a, uint8_t chi_b);

/** Return g_layer when Hamming(chi_act, chi_weight) <= 2, else 0. */
GYROSCOPIC_EXPORT float gyroscopic_route_resonance(
    uint8_t chi_activation,
    uint8_t chi_weight,
    int layer,
    int total_layers,
    uint8_t k4_char,
    uint8_t shell,
    float g_layer
);

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

/** Rényi-2 effective support M̂₂ = W²/Σh² and spectral damping η from chi_hist64 (QuBEC §21.3). */
GYROSCOPIC_EXPORT void gyroscopic_chi_hist_m2_eta(
    const uint32_t hist[64],
    float *          m2_out,
    float *          eta_out);

/** 6-bit chirality register χ(s) from packed state24 (dipole pairs of A^B). */
GYROSCOPIC_EXPORT uint8_t gyroscopic_chirality_word6(uint32_t state24);

/** Map a 64-wide float block to a depth-4 kernel word (bridge serializer). */
GYROSCOPIC_EXPORT void gyroscopic_kv_f32_to_word4(
    const float * x,
    uint8_t       word4[4]);

/** Step word4 on Ω; optional state_inout carries temporal ledger (NULL → from rest). */
GYROSCOPIC_EXPORT uint8_t gyroscopic_word4_chirality(
    const uint8_t word4[4],
    uint32_t *    state_inout);

/** Serialize block → word4 → Ω step; updates *state_inout when non-NULL. */
GYROSCOPIC_EXPORT uint8_t gyroscopic_kv_f32_block_chirality(
    const float * x,
    uint32_t *    state_inout);

/**
 * Percolation-aware Hamming aperture from chi_hist64 and query χ (5_Perlocation).
 * Returns d in [0,3] so candidate fraction meets p_c target derived from M₂.
 */
GYROSCOPIC_EXPORT int gyroscopic_chi_hist_d_eff(
    const uint32_t hist[64],
    uint8_t        chi_q,
    float *        m2_out,
    float *        eta_out);

typedef struct gyro_kv_polar64 {
    uint8_t  boundary; /* 6-bit boundary anchor after first 2 bytes of word4 */
    uint8_t  chi;      /* 6-bit chirality after full word closure */
    uint8_t  shell;    /* popcount(chi), 0..6 */
    uint16_t r_bits;   /* scaled L2 norm (Runtime §19 polar summary) */
} gyro_kv_polar64_t;

/** Polar KV summary per 64-wide block (Runtime §19.1). */
GYROSCOPIC_EXPORT void gyroscopic_kv_polar_encode_block64(
    const float *      x,
    gyro_kv_polar64_t * out);

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

#ifndef GYROSCOPIC_TILE_SIZE
#define GYROSCOPIC_TILE_SIZE 64
#endif

typedef struct gyroscopic_tile_ratios {
    float r_shell;
    float r_chi;
    float r_chi_minus_shell;
    float r_defect;
    float norm;
} gyroscopic_tile_ratios_t;

/** Chirality XOR-circulant coeffs f[d] = mean(W[i, i^d]) for 64x64 row-major W. */
GYROSCOPIC_EXPORT void gyroscopic_project_chi_coeffs(
    const float * W,
    float *       f_out
);

/** Frobenius energy ratios (matches helpers/diagnostics/tiles.py). */
GYROSCOPIC_EXPORT void gyroscopic_tile_decompose_ratios(
    const float *              W,
    gyroscopic_tile_ratios_t * out
);

/** y[i] = sum_j f[i^j] * x[j] (chi-circulant matvec). */
GYROSCOPIC_EXPORT void gyroscopic_chi_circulant_matvec(
    const float * f,
    const float * x,
    float *       y
);

/** Exact hybrid matvec: y = P_chi(W)·x + (W - P_chi(W))·x for 64x64 W. */
GYROSCOPIC_EXPORT void gyroscopic_tile_hybrid_matvec(
    const float * W,
    const float * x,
    float *       y
);

/** One output row of hybrid matvec (row index 0..63). */
GYROSCOPIC_EXPORT float gyroscopic_tile_hybrid_dot_row(
    const float * W,
    int           row,
    const float * x
);

#ifdef __cplusplus
}
#endif
