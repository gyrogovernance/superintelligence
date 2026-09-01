#pragma once

/*
 * Gyroscopic kernel C API (offline / native DLL / tests).
 *
 * Not linked into ggml-cpu inference. Live MatMul displace uses ledger.h.
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

/** Rényi-2 effective support M̂₂ = W²/Σh² and spectral damping η from chi_hist64 (hQVM_QuBEC_Theory.md §21.3). */
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

/** Native 64-point Walsh-Hadamard transform on a float vector (in place). */
GYROSCOPIC_EXPORT void gyroscopic_wht64_float(float data[64]);

/** Dense chirality climate: x <- M^n x with M row-major 64x64. */
GYROSCOPIC_EXPORT void gyroscopic_climate_dense_nstep(
    float *       x64,
    const float * M64x64,
    int           n_steps);

/** Spectral climate: WHT, pointwise phi^n, iWHT (/64). Cost independent of dense M^n. */
GYROSCOPIC_EXPORT void gyroscopic_climate_spectral_nstep(
    float *       x64,
    const float * phi64,
    int           n_steps);

/** Collapse chi[64] -> shell hist[7] by popcount; apply 7 gains; expand uniform on shell. */
GYROSCOPIC_EXPORT void gyroscopic_shell7_apply(
    float *       chi64,
    const float * gains7);

/** Build XOR-circulant M[i,j]=f[i^j] and phi=WHT(f) for spectral/dense agreement smoke. */
GYROSCOPIC_EXPORT void gyroscopic_climate_from_kernel(
    const float * f64,
    float *       M64x64,
    float *       phi64);

/* ---------------------------------------------------------------------------
 * G-equivariant 2080-sector layer (Group Theory sec 10.2/12.3, QuBEC 18.3).
 *
 * G = (GF(2)^6 x GF(2)^6) rt C2 acts on Omega=(u,v) by translations and the
 * coordinate swap. Orbits on Omega x Omega are classified by the unordered
 * pair {u^u', v^v'} of raw 6-bit XOR differences: C(64,2)+64 = 2080 classes,
 * matching the multiplicity-free sector count (64 one-dim + 2016 two-dim).
 * End_G(L^2(Omega)) is the 2080-dim algebra of orbital kernels
 *   K[(u,v),(u',v')] = gains[tri(min(du,dv), max(du,dv))],
 * applied as one structured 4096x4096 matvec using 2080 parameters
 * (vs 16,777,216 dense entries).
 * ------------------------------------------------------------------------- */
#define HQVM_EQUIV2080_GAINS 2080

/** Triangular orbital index for du,dv in [0,63]: a=min, b=max,
 *  idx = a*64 - a*(a-1)/2 + (b-a), range 0..2079. */
GYROSCOPIC_EXPORT int hqvm_equiv2080_sector_index(uint8_t du, uint8_t dv);

/** out[s] = sum_t K[s,t] psi[t] with the 2080 orbital gains (structured matvec). */
GYROSCOPIC_EXPORT void hqvm_equiv2080_apply(
    const float * psi4096,
    float *       out4096,
    const float * gains2080);

/** Plain dense 4096x4096 row-major matvec (baseline for the equivariant apply). */
GYROSCOPIC_EXPORT void hqvm_dense4096_matvec(
    const float * M4096x4096,
    const float * x4096,
    float *       y4096);

/** y[i] = sum_j f[i^j] * x[j] (chi-circulant matvec). */
GYROSCOPIC_EXPORT void gyroscopic_chi_circulant_matvec(
    const float * f,
    const float * x,
    float *       y
);

/** Owned holonomic affinity step (spine). chi_q/chi_k = one chi6 per token;
 *  aff_out[q] = A[chi_q[q]] where A = iWHT(Khat . WHT(H)), H the key
*  chi histogram, Khat the spectral kernel (WHT of a chi-circulant column).
*  Cost O(nq+nk+64 log64), not O(nq*nk*d). Returns 0 / -1. */
GYROSCOPIC_EXPORT int gyroscopic_affinity_step(
    const uint8_t * chi_q, int64_t nq,
    const uint8_t * chi_k, int64_t nk,
    const float   * khat,
    float         * aff_out
);

/** Per-pair chi-coupling entry (owned QK-equivalent). score[i] =
 *  kdir[chi_q[i] ^ chi_k[i]]; 64-LUT, no d-MACs. Returns 0 / -1. */
GYROSCOPIC_EXPORT int gyroscopic_chi_coupling(
    const uint8_t * chi_q, const uint8_t * chi_k, int64_t n,
    const float   * kdir, float * score
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

/* ---------------------------------------------------------------------------
 * Kernel-in-attention (Runtime Spec §6, pen-test L1/L4, analysis §7.2 fork A).
 *
 * The kernel executes a native score term inside flash-attn using Bonsai's Q/K
 * activations. No residual gate, no head replacement, no training.
 *
 * chi6_from_plane64: peak-index chirality (0..63) of one 64-dim head plane.
 *   Input is the first 64 of a 128-dim head channel (head dim = 2x64).
 * gyro_sim: exact GF(2)^6 chirality distance, bounded 0..6.
 * ------------------------------------------------------------------------- */

/* Peak-index chi6 (0..63) of a 64-wide float plane (Q/K head half-plane). */
GYROSCOPIC_EXPORT uint8_t gyroscopic_chi6_from_plane64(const float * plane64);

/* Exact chirality similarity: 6 - popcount(a xor b), bounded 0..6. */
GYROSCOPIC_EXPORT int gyroscopic_gyro_sim(uint8_t chi_a, uint8_t chi_b);

/* ---------------------------------------------------------------------------
 * Carrier projection from the TRUE 4096-dim residual (Omega = U x V, |U|=64).
 *
 * Bonsai hidden dim = 4096 = Omega. A 64-dim SLICE of Q/K is NOT the U factor
 * (it is i.i.d. Rademacher -> thermal M2). The constitutional carrier is
 * computed by grouping the 4096 residual into 64 blocks of 64 and applying the
 * kernel's chirality code to each block: chi_out[b] = peak-index WHT64 of the
 * sign pattern of block b. This yields the 64-element U factor, natively
 * projected from the full Omega space (not a linear slice).
 *
 * x4096 must point at a contiguous 4096-wide residual row. chi_out[64] receives
 * one chi6 per block. Returns 0 on success, -1 on null input.
 * ------------------------------------------------------------------------- */
GYROSCOPIC_EXPORT int gyroscopic_project_to_carrier_64(
    const float * x4096, uint8_t chi_out[64]);


/* ---------------------------------------------------------------------------
 * Trajectory + receipt (carrier bookkeeping). Canonical Omega-12 step via
 * gyroscopic_step_omega12. Single-owner instance lives in attn.c (lift path).
 * ------------------------------------------------------------------------- */
#define HQVM_ARCHETYPE_STATE24 GENE_MAC_REST

typedef struct gyro_trajectory_state {
    uint32_t state24;   /* (A12 << 12) | B12 */
    uint64_t depth;     /* ledger depth (time) = tokens committed */
    uint32_t n_trans;   /* transitions applied this token (36 layers) */
    uint32_t phase_idx; /* emission counter; fam = phase_idx & 3 */
} gyro_trajectory_state_t;

typedef struct hqvm_receipt {
    uint16_t anchor12;
    uint8_t  k4_family;
    uint32_t state24;
    uint64_t depth;
    uint32_t fnv1a;
} hqvm_receipt_t;

GYROSCOPIC_EXPORT void hqvm_traj_reset(gyro_trajectory_state_t *t);
GYROSCOPIC_EXPORT void hqvm_traj_step(gyro_trajectory_state_t *t, uint8_t intron_byte);
GYROSCOPIC_EXPORT uint32_t hqvm_receipt_seal(const hqvm_receipt_t *r);
GYROSCOPIC_EXPORT void hqvm_receipt_print(const hqvm_receipt_t *r);

/* ---------------------------------------------------------------------------
 * Wavefunction sparse-wave, byte-fiber, and Omega sig13 helpers (formerly wave.h).
 * K4 operators, step_omega12, chirality: already above / constants.h.
 * ------------------------------------------------------------------------- */

#ifndef HQVM_W2_BYTE0
#define HQVM_W2_BYTE0  0xAAu
#endif
#ifndef HQVM_W2_BYTE1
#define HQVM_W2_BYTE1  0xABu
#endif
#ifndef HQVM_W2P_BYTE0
#define HQVM_W2P_BYTE0 0x2Au
#endif
#ifndef HQVM_W2P_BYTE1
#define HQVM_W2P_BYTE1 0x2Bu
#endif
#ifndef HQVM_CHI_FLIP_6
#define HQVM_CHI_FLIP_6 0x3Fu
#endif

typedef struct hqvm_byte_fiber {
    uint8_t byte;
    uint8_t intron;
    uint8_t q6;
    uint8_t family;
    uint8_t phase_net;
    uint8_t phase_common;
    uint8_t fold_degree;
    uint8_t is_flat;
} hqvm_byte_fiber;

typedef struct hqvm_wave_term {
    uint16_t omega_index;
    int8_t   sign;
    uint8_t  multiplicity;
} hqvm_wave_term;

typedef struct hqvm_carrier_cell {
    uint32_t state24;
    uint16_t program_id;
    uint16_t memory_id;
} hqvm_carrier_cell;

typedef struct hqvm_wave_grammar_result {
    int w2_sig_ok;
    int w2p_sig_ok;
    int w2_involution_ok;
    int w2p_involution_ok;
    int f_composition_ok;
    int t2_chi_ok;
    int t2_shell_ok;
    int sparse_k4_ok;
    int byte_table_ok;
} hqvm_wave_grammar_result;

GYROSCOPIC_EXPORT void hqvm_byte_table_init(void);
GYROSCOPIC_EXPORT int  hqvm_byte_table_ok(void);
GYROSCOPIC_EXPORT uint8_t hqvm_byte_of_q6_fam(uint8_t q6, uint8_t fam);
GYROSCOPIC_EXPORT uint8_t hqvm_q6_of_byte(uint8_t byte);
GYROSCOPIC_EXPORT uint8_t hqvm_fam_of_byte(uint8_t byte);
GYROSCOPIC_EXPORT void hqvm_decompose_byte(uint8_t byte, hqvm_byte_fiber * out);

GYROSCOPIC_EXPORT void hqvm_state24_to_uv6(uint32_t state24, uint8_t * u6, uint8_t * v6);
GYROSCOPIC_EXPORT uint32_t hqvm_uv6_to_state24(uint8_t u6, uint8_t v6);
GYROSCOPIC_EXPORT uint8_t hqvm_chi6_uv(uint8_t u6, uint8_t v6);
GYROSCOPIC_EXPORT int hqvm_code_shell(uint8_t u6, uint8_t v6);

GYROSCOPIC_EXPORT void hqvm_trace_word_bytes(
    const uint8_t * word,
    int             n_bytes,
    uint8_t         u6_in,
    uint8_t         v6_in,
    uint8_t *       u6_out,
    uint8_t *       v6_out);
GYROSCOPIC_EXPORT uint32_t hqvm_trace_word_state24(const uint8_t * word, int n_bytes, uint32_t state24);

/*
 * Packed Omega chart: state12 = (u6 << 6) | v6.
 * Packed sig13 = (parity << 12) | (tau_u6 << 6) | tau_v6.  (|G| = 8192 = 2^13)
 */
GYROSCOPIC_EXPORT uint16_t hqvm_pack_state12(uint8_t u6, uint8_t v6);
GYROSCOPIC_EXPORT void     hqvm_unpack_state12(uint16_t s12, uint8_t * u6, uint8_t * v6);
GYROSCOPIC_EXPORT uint16_t hqvm_step_state12_by_byte(uint16_t s12, uint8_t byte);
GYROSCOPIC_EXPORT uint16_t hqvm_trace_word_state12(uint16_t s12, const uint8_t * word, int n_bytes);

GYROSCOPIC_EXPORT uint16_t hqvm_sig13_compile(const uint8_t * word, int n_bytes);
GYROSCOPIC_EXPORT uint16_t hqvm_sig13_compose(uint16_t left, uint16_t right);
GYROSCOPIC_EXPORT uint16_t hqvm_sig13_inv(uint16_t sig);
GYROSCOPIC_EXPORT uint16_t hqvm_sig13_apply(uint16_t s12, uint16_t sig);
GYROSCOPIC_EXPORT void     hqvm_sig13_apply_batch(
    const uint16_t * states,
    int              n_states,
    uint16_t         sig,
    uint16_t *       out);

GYROSCOPIC_EXPORT int hqvm_route2_witnesses(
    uint16_t src12,
    uint16_t tgt12,
    uint8_t  out_b1[16],
    uint8_t  out_b2[16]);

/* Closed-form 16 witnesses (flag enumeration); same set as brute route2. */
GYROSCOPIC_EXPORT int hqvm_route2_synthesize(
    uint16_t src12,
    uint16_t tgt12,
    uint8_t  out_b1[16],
    uint8_t  out_b2[16]);

/* 8192-entry ACTION table: cache[sig] = packed apply key (identity fill = sig). */
GYROSCOPIC_EXPORT void hqvm_sig13_cache_build(uint16_t cache[8192]);
GYROSCOPIC_EXPORT void hqvm_sig13_cache_apply_batch(
    const uint16_t * states,
    int              n_states,
    uint16_t         sig,
    const uint16_t * cache,
    uint16_t *       out);
GYROSCOPIC_EXPORT void hqvm_sig13_apply_many_sigs(
    const uint16_t * states,
    int              n_states,
    const uint16_t * sigs,
    int              n_sigs,
    const uint16_t * cache);
GYROSCOPIC_EXPORT void hqvm_sig13_compile_apply_many(
    const uint16_t * states,
    int              n_states,
    const uint8_t *  words_flat,
    const int *      lens,
    int              n_words);

GYROSCOPIC_EXPORT int hqvm_wave_apply_k4(
    hqvm_wave_term * terms,
    int              n_terms,
    int              k4_gate,
    int              max_terms);
GYROSCOPIC_EXPORT int hqvm_wave_merge(hqvm_wave_term * terms, int * n_terms);
GYROSCOPIC_EXPORT int hqvm_wave_grammar_verify(hqvm_wave_grammar_result * receipt);

#ifdef __cplusplus
}
#endif
