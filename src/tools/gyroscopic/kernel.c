/*
 * Gyroscopic kernel implementation (see kernel.h).
 */

#include "kernel.h"
#include <stdio.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* internal alias: existing callers use the short name */
#define wht64_float gyroscopic_wht64_float

/* ------------------------------------------------------------------ */
/* CGM constants used only here. Derived in gyroscopic_gravity_g1().  */
/* ------------------------------------------------------------------ */

#ifndef GYROSCOPIC_C4_REF
#define GYROSCOPIC_C4_REF (-1.75)        /* depth-4 closure coefficient   */
#endif
#ifndef GYROSCOPIC_V_EW_GEV
#define GYROSCOPIC_V_EW_GEV 246.22       /* electroweak scale (GeV)       */
#endif
#ifndef GYROSCOPIC_E_CS_GEV
#define GYROSCOPIC_E_CS_GEV 1.22e19      /* CS / Planck scale (GeV)       */
#endif

/* ------------------------------------------------------------------ */
/* Transition-rule tables (ported bit-exactly from src/constants.py).  */
/* intron = byte ^ 0xAA ; mask12 from 6-bit payload (bits 1..6).      */
/* ------------------------------------------------------------------ */

static uint16_t g_mask12_by_intron[256];
static int      g_tables_ready = 0;

static uint16_t micro_ref_to_mask12(uint8_t micro_ref) {
    uint16_t mask12 = 0;
    int i;
    for (i = 0; i < (int) CHIRALITY_QUBITS_6; ++i) {
        if ((micro_ref >> i) & 1u) {
            mask12 |= (uint16_t) (0x3u << (2 * i));
        }
    }
    return (uint16_t) (mask12 & LAYER_MASK_12);
}

static void ensure_tables(void) {
    int b;
    /* Idempotent init; benign if threads race (same table contents). Not hot path. */
    if (g_tables_ready) {
        return;
    }
    for (b = 0; b < 256; ++b) {
        const uint8_t intron = (uint8_t) (b ^ (int) GENE_MIC_S);
        const uint8_t micro_ref = (uint8_t) ((intron >> 1) & CHIRALITY_MASK_6);
        g_mask12_by_intron[b] = micro_ref_to_mask12(micro_ref);
    }
    g_tables_ready = 1;
}

/* ------------------------------------------------------------------ */
/* Small bit helpers.                                                  */
/* ------------------------------------------------------------------ */

static int popcount64(uint64_t x) {
#if defined(_MSC_VER)
    return (int) __popcnt64(x);
#elif defined(__GNUC__) || defined(__clang__)
    return (int) __builtin_popcountll(x);
#else
    int n = 0;
    while (x) { n += (int) (x & 1u); x >>= 1; }
    return n;
#endif
}

static int popcount32(uint32_t x) {
#if defined(_MSC_VER)
    return (int) __popcnt(x);
#elif defined(__GNUC__) || defined(__clang__)
    return (int) __builtin_popcount(x);
#else
    int n = 0;
    while (x) { n += (int) (x & 1u); x >>= 1; }
    return n;
#endif
}

/* ================================================================== */
/* 1. Verified transition rule.                                        */
/* ================================================================== */

GYROSCOPIC_EXPORT uint32_t gyroscopic_step_omega12(uint32_t state24, uint8_t byte) {
    /* omega12 is packed (A12 << 12 | B12); equals state24 on Omega. */
    uint8_t intron;
    uint16_t m12;
    uint16_t a12;
    uint16_t b12;
    uint16_t a_mut;
    uint16_t invert_a;
    uint16_t invert_b;
    uint16_t a_next;
    uint16_t b_next;

    ensure_tables();

    intron = (uint8_t) (byte ^ (int) GENE_MIC_S);
    m12   = g_mask12_by_intron[byte];
    a12   = (uint16_t) ((state24 >> 12) & LAYER_MASK_12);
    b12   = (uint16_t) (state24 & LAYER_MASK_12);
    a_mut = (uint16_t) ((a12 ^ m12) & LAYER_MASK_12);
    invert_a = (intron & L0_BIT_0) ? (uint16_t) COMPLEMENT_MASK_12 : 0u;
    invert_b = (intron & L0_BIT_7) ? (uint16_t) COMPLEMENT_MASK_12 : 0u;
    a_next = (uint16_t) ((b12 ^ invert_a) & LAYER_MASK_12);
    b_next = (uint16_t) ((a_mut ^ invert_b) & LAYER_MASK_12);

    return ((uint32_t) a_next << 12) | (uint32_t) b_next;
}

/* ================================================================== */
/* 2. Wavefunction K4 operators (permutation only).                   */
/*                                                                    */
/* Psi is a length-4096 array over Omega = U x V.                      */
/* Indices: i = u6 * 64 + v6, with u6, v6 in [0, 63].                 */
/* The depth-4 half-words are order-preserving translations on         */
/* (u6, v6): parity 0 (the per-byte U/V swap cancels over two bytes),  */
/* so the coordinates are complemented, not exchanged. For m = 0       */
/* (Analysis_hQVM_Wavefunction_Corrections sec 3-4):                   */
/*   W2  : (u, v) -> (u ^ 63, v)        signature (0, 63, 0)  chi^63  */
/*   W2' : (u, v) -> (u, v ^ 63)        signature (0, 0, 63)  chi^63  */
/*   F   : (u, v) -> (u ^ 63, v ^ 63)   = W2 o W2'            chi     */
/* Each is an involution; F = W2 o W2' holds by construction.         */
/* ================================================================== */

/* Map (h, q) -> (perm(h), perm(q)); perm is an involution on 0..63.
 * Uses ~16 KB stack buffer; research path only, not matmul inner loop. */
static void apply_pairwise(
    float psi[GYROSCOPIC_WAVEFUNCTION_SIZE],
    const uint8_t perm_h[HORIZON_SIZE],
    const uint8_t perm_q[HORIZON_SIZE])
{
    float tmp[GYROSCOPIC_WAVEFUNCTION_SIZE];
    int h, q;
    for (h = 0; h < (int) HORIZON_SIZE; ++h) {
        for (q = 0; q < (int) HORIZON_SIZE; ++q) {
            const size_t src = (size_t) h * HORIZON_SIZE + (size_t) q;
            const size_t dst = (size_t) perm_h[h] * HORIZON_SIZE + (size_t) perm_q[q];
            tmp[dst] = psi[src];
        }
    }
    memcpy(psi, tmp, sizeof(tmp));
}

GYROSCOPIC_EXPORT void gyroscopic_apply_K4(
    float psi[GYROSCOPIC_WAVEFUNCTION_SIZE],
    int gate)
{
    uint8_t comp[HORIZON_SIZE];
    uint8_t ident[HORIZON_SIZE];
    int i;

    if (psi == NULL) {
        return;
    }

    for (i = 0; i < (int) HORIZON_SIZE; ++i) {
        comp[i]  = (uint8_t) ((~(unsigned) i) & CHIRALITY_MASK_6);
        ident[i] = (uint8_t) i;
    }

    switch (gate) {
        case GYROSCOPIC_K4_ID:
            return;
        case GYROSCOPIC_K4_W2:
            /* (u, v) -> (u ^ 63, v): complement U, preserve V (chi ^ 63). */
            apply_pairwise(psi, comp, ident);
            return;
        case GYROSCOPIC_K4_W2P:
            /* (u, v) -> (u, v ^ 63): preserve U, complement V (chi ^ 63). */
            apply_pairwise(psi, ident, comp);
            return;
        case GYROSCOPIC_K4_F:
            /* (u, v) -> (u ^ 63, v ^ 63) = W2 o W2' (chi preserved). */
            apply_pairwise(psi, comp, comp);
            return;
        default:
            return;
    }
}

/* ================================================================== */
/* 3. Holographic reshape (pure permutation).                         */
/* ================================================================== */

GYROSCOPIC_EXPORT void gyroscopic_to_holographic(
    const float psi[GYROSCOPIC_WAVEFUNCTION_SIZE],
    float holo[GYROSCOPIC_HOLO_DIM][GYROSCOPIC_HOLO_DIM])
{
    int h, q;
    if (psi == NULL || holo == NULL) {
        return;
    }
    for (h = 0; h < (int) HORIZON_SIZE; ++h) {
        for (q = 0; q < (int) HORIZON_SIZE; ++q) {
            holo[h][q] = psi[(size_t) h * HORIZON_SIZE + (size_t) q];
        }
    }
}

GYROSCOPIC_EXPORT void gyroscopic_from_holographic(
    const float holo[GYROSCOPIC_HOLO_DIM][GYROSCOPIC_HOLO_DIM],
    float psi[GYROSCOPIC_WAVEFUNCTION_SIZE])
{
    int h, q;
    if (psi == NULL || holo == NULL) {
        return;
    }
    for (h = 0; h < (int) HORIZON_SIZE; ++h) {
        for (q = 0; q < (int) HORIZON_SIZE; ++q) {
            psi[(size_t) h * HORIZON_SIZE + (size_t) q] = holo[h][q];
        }
    }
}

/* ================================================================== */
/* 4. Per-group analysis (closed-form on 128 sign bits).              */
/* ================================================================== */

/* In-place 64-point integer Walsh-Hadamard transform. */
static void wht64_int32(int32_t data[64]) {
    int stride, i, j;
    for (stride = 32; stride >= 1; stride >>= 1) {
        for (i = 0; i < 64; i += 2 * stride) {
            for (j = 0; j < stride; ++j) {
                const int32_t a = data[i + j];
                const int32_t b = data[i + j + stride];
                data[i + j] = a + b;
                data[i + j + stride] = a - b;
            }
        }
    }
}

/* In-place 64-point Walsh-Hadamard on floats (same butterfly as wht64_int32). */
GYROSCOPIC_EXPORT void gyroscopic_wht64_float(float data[64]);

GYROSCOPIC_EXPORT void gyroscopic_wht64_float(float data[64]) {
    int stride, i, j;
    for (stride = 32; stride >= 1; stride >>= 1) {
        for (i = 0; i < 64; i += 2 * stride) {
            for (j = 0; j < stride; ++j) {
                const float a = data[i + j];
                const float b = data[i + j + stride];
                data[i + j] = a + b;
                data[i + j + stride] = a - b;
            }
        }
    }
}

GYROSCOPIC_EXPORT void gyroscopic_climate_dense_nstep(
    float *       x64,
    const float * M64x64,
    int           n_steps)
{
    float tmp[64];
    int   step, i, j;
    if (!x64 || !M64x64 || n_steps <= 0) {
        return;
    }
    for (step = 0; step < n_steps; ++step) {
        for (i = 0; i < 64; ++i) {
            float s = 0.0f;
            const float * row = M64x64 + (size_t) i * 64u;
            for (j = 0; j < 64; ++j) {
                s += row[j] * x64[j];
            }
            tmp[i] = s;
        }
        memcpy(x64, tmp, sizeof(tmp));
    }
}

GYROSCOPIC_EXPORT void gyroscopic_climate_spectral_nstep(
    float *       x64,
    const float * phi64,
    int           n_steps)
{
    int i;
    if (!x64 || !phi64 || n_steps < 0) {
        return;
    }
    wht64_float(x64);
    if (n_steps == 0) {
        /* iWHT only */
    } else if (n_steps == 1) {
        for (i = 0; i < 64; ++i) {
            x64[i] *= phi64[i];
        }
    } else {
        for (i = 0; i < 64; ++i) {
            x64[i] *= powf(phi64[i], (float) n_steps);
        }
    }
    wht64_float(x64);
    for (i = 0; i < 64; ++i) {
        x64[i] *= (1.0f / 64.0f);
    }
}

GYROSCOPIC_EXPORT void gyroscopic_shell7_apply(
    float *       chi64,
    const float * gains7)
{
    float shells[7];
    int   i, n, pop;
    if (!chi64 || !gains7) {
        return;
    }
    for (n = 0; n < 7; ++n) {
        shells[n] = 0.0f;
    }
    for (i = 0; i < 64; ++i) {
        pop = 0;
        {
            unsigned v = (unsigned) i;
            while (v) {
                pop += (int) (v & 1u);
                v >>= 1;
            }
        }
        shells[pop] += chi64[i];
    }
    for (n = 0; n < 7; ++n) {
        shells[n] *= gains7[n];
    }
    for (i = 0; i < 64; ++i) {
        pop = 0;
        {
            unsigned v = (unsigned) i;
            while (v) {
                pop += (int) (v & 1u);
                v >>= 1;
            }
        }
        /* Uniform expand: each chi on shell gets shells[N] / C(6,N). */
        {
            static const float inv_binom[7] = {
                1.0f / 1.0f,
                1.0f / 6.0f,
                1.0f / 15.0f,
                1.0f / 20.0f,
                1.0f / 15.0f,
                1.0f / 6.0f,
                1.0f / 1.0f,
            };
            chi64[i] = shells[pop] * inv_binom[pop];
        }
    }
}

GYROSCOPIC_EXPORT void gyroscopic_climate_from_kernel(
    const float * f64,
    float *       M64x64,
    float *       phi64)
{
    int i, j;
    if (!f64) {
        return;
    }
    if (phi64) {
        memcpy(phi64, f64, 64 * sizeof(float));
        wht64_float(phi64);
    }
    if (M64x64) {
        for (i = 0; i < 64; ++i) {
            for (j = 0; j < 64; ++j) {
                M64x64[i * 64 + j] = f64[i ^ j];
            }
        }
    }
}

/* ================================================================== */
/* G-equivariant 2080-sector commutant layer (Group Theory 10.2/12.3). */
/* ================================================================== */

GYROSCOPIC_EXPORT int hqvm_equiv2080_sector_index(uint8_t du, uint8_t dv)
{
    const int a = (du <= dv) ? du : dv;
    const int b = (du <= dv) ? dv : du;
    /* idx(a,b) with 0<=a<=b<=63: a*64 - a*(a-1)/2 + (b-a); total C(65,2)=2080. */
    return a * 64 - (a * (a - 1)) / 2 + (b - a);
}

GYROSCOPIC_EXPORT void hqvm_equiv2080_apply(
    const float * psi4096,
    float *       out4096,
    const float * gains2080)
{
    int su, sv, tu, tv;
    if (!psi4096 || !out4096 || !gains2080) {
        return;
    }
    for (su = 0; su < 64; ++su) {
        for (sv = 0; sv < 64; ++sv) {
            const int s_idx = su * 64 + sv;
            double acc = 0.0;
            for (tu = 0; tu < 64; ++tu) {
                const uint8_t du = (uint8_t) (su ^ tu);
                const size_t row = (size_t) tu * 64u;
                for (tv = 0; tv < 64; ++tv) {
                    const uint8_t dv = (uint8_t) (sv ^ tv);
                    acc += (double) gains2080[
                        hqvm_equiv2080_sector_index(du, dv)]
                        * (double) psi4096[row + (size_t) tv];
                }
            }
            out4096[s_idx] = (float) acc;
        }
    }
}

GYROSCOPIC_EXPORT void hqvm_dense4096_matvec(
    const float * M4096x4096,
    const float * x4096,
    float *       y4096)
{
    int i, j;
    if (!M4096x4096 || !x4096 || !y4096) {
        return;
    }
    for (i = 0; i < 4096; ++i) {
        const float * row = M4096x4096 + (size_t) i * 4096u;
        double acc = 0.0;
        for (j = 0; j < 4096; ++j) {
            acc += (double) row[j] * (double) x4096[j];
        }
        y4096[i] = (float) acc;
    }
}

GYROSCOPIC_EXPORT uint8_t gyroscopic_chirality_from_signs64(uint64_t signs) {
    int32_t data[64];
    int k;
    int32_t best_mag = 0;
    int best_k = 0;

    for (k = 0; k < 64; ++k) {
        data[k] = ((signs >> (unsigned) k) & 1u) ? 1 : -1;
    }
    wht64_int32(data);
    for (k = 0; k < 64; ++k) {
        int32_t mag = data[k] < 0 ? -data[k] : data[k];
        if (mag > best_mag) {
            best_mag = mag;
            best_k = k;
        }
    }
    return (uint8_t) (best_k & (int) CHIRALITY_MASK_6);
}

GYROSCOPIC_EXPORT uint64_t gyroscopic_signs64_from_f32(const float * x) {
    uint64_t signs = 0;
    int k;

    if (x == NULL) {
        return 0;
    }
    for (k = 0; k < 64; ++k) {
        if (x[k] >= 0.0f) {
            signs |= (1ULL << (unsigned) k);
        }
    }
    return signs;
}

GYROSCOPIC_EXPORT uint64_t gyroscopic_signs64_from_q8(const int8_t * q, int n) {
    uint64_t signs = 0;
    int k;

    if (q == NULL || n <= 0) {
        return 0;
    }
    if (n > 64) {
        n = 64;
    }
    for (k = 0; k < n; ++k) {
        if (q[k] >= 0) {
            signs |= (1ULL << (unsigned) k);
        }
    }
    return signs;
}

GYROSCOPIC_EXPORT uint8_t gyroscopic_activation_chirality(const float * x) {
    return gyroscopic_chirality_from_signs64(gyroscopic_signs64_from_f32(x));
}

GYROSCOPIC_EXPORT uint8_t gyroscopic_activation_chirality_q8(
    const int8_t * q0,
    const int8_t * q1)
{
    uint64_t signs;

    signs = gyroscopic_signs64_from_q8(q0, 32);
    signs |= (gyroscopic_signs64_from_q8(q1, 32) << 32);
    return gyroscopic_chirality_from_signs64(signs);
}

/* ---------------------------------------------------------------------------
 * Kernel-in-attention (Runtime Spec §6, analysis §7.2 fork A).
 * chi6 is the peak-index chirality of a 64-wide sign pattern (WHT64 argmax).
 * The kernel operates on Bonsai Q/K planes; no training, no residual gate.
 * ------------------------------------------------------------------------- */

GYROSCOPIC_EXPORT uint8_t gyroscopic_chi6_from_plane64(const float * plane64) {
    if (plane64 == NULL) {
        return 0;
    }
    return gyroscopic_chirality_from_signs64(gyroscopic_signs64_from_f32(plane64));
}

GYROSCOPIC_EXPORT int gyroscopic_gyro_sim(uint8_t chi_a, uint8_t chi_b) {
    uint64_t x = (uint64_t) chi_a ^ (uint64_t) chi_b;
    int       n = 0;

    while (x) {
        x &= x - 1u;
        ++n;
    }
    return (int) (CHIRALITY_QUBITS_6 - n);
}

/* Carrier projection from the true 4096-dim residual (Omega = U x V, |U|=64).
 * Group x4096 into 64 blocks of 64; chi_out[b] = peak-index WHT64 of the sign
 * pattern of block b. Reuses the kernel's native chirality code (no new
 * projection invented). A 64-slice of Q/K is NOT the U factor; this is. */
GYROSCOPIC_EXPORT int gyroscopic_project_to_carrier_64(
    const float * x4096, uint8_t chi_out[64]) {
    int b;
    if (x4096 == NULL || chi_out == NULL) {
        return -1;
    }
    for (b = 0; b < 64; ++b) {
        const float * block = x4096 + (size_t) b * 64;
        chi_out[b] = gyroscopic_chirality_from_signs64(
            gyroscopic_signs64_from_f32(block));
    }
    return 0;
}

GYROSCOPIC_EXPORT int gyroscopic_chirality_distance(uint8_t chi_a, uint8_t chi_b) {
    return popcount32((uint32_t) (chi_a ^ chi_b));
}

GYROSCOPIC_EXPORT float gyroscopic_route_resonance(
    uint8_t chi_activation,
    uint8_t chi_weight,
    int layer,
    int total_layers,
    uint8_t k4_char,
    uint8_t shell,
    float g_layer)
{
    (void) layer;
    (void) total_layers;
    (void) k4_char;
    (void) shell;

    if (g_layer <= 0.0f) {
        return 0.0f;
    }
    if (popcount32((uint32_t) (chi_activation ^ chi_weight)) > 2u) {
        return 0.0f;
    }
    return g_layer;
}

GYROSCOPIC_EXPORT void gyroscopic_extract_phase_native(
    const uint8_t signs[16],
    uint8_t * k4_char,
    uint8_t * shell_proxy)
{
    uint64_t signs_a;
    uint64_t signs_b;
    uint8_t parity_a;
    uint8_t parity_b;
    uint8_t k4;
    uint8_t proxy;

    if (signs == NULL) {
        if (k4_char)    *k4_char = GYROSCOPIC_K4_ID;
        if (shell_proxy) *shell_proxy = 0;
        return;
    }

    memcpy(&signs_a, signs, sizeof(uint64_t));
    memcpy(&signs_b, signs + 8, sizeof(uint64_t));

    parity_a = (uint8_t) (popcount64(signs_a) & 1u);
    parity_b = (uint8_t) (popcount64(signs_b) & 1u);
    k4 = (uint8_t) (parity_a | (parity_b << 1));
    proxy = (uint8_t) ((popcount64(signs_a ^ signs_b) >> 4) & 0x7u);

    if (k4_char)     *k4_char = k4;
    if (shell_proxy) *shell_proxy = proxy;
}

GYROSCOPIC_EXPORT float gyroscopic_k4_compose_gyroacc(
    const gyro_accum_t accum[4],
    float gravity)
{
    float cs_a;
    float cs_b;
    float una_a;
    float una_b;
    float ona_a;
    float ona_b;
    float bu_a;
    float bu_b;
    float composed_a;
    float composed_b;

    if (accum == NULL) {
        return 0.0f;
    }

    cs_a  = accum[GYROSCOPIC_K4_ID].a;
    cs_b  = accum[GYROSCOPIC_K4_ID].b;
    una_a = accum[GYROSCOPIC_K4_W2].a;
    una_b = accum[GYROSCOPIC_K4_W2].b;
    ona_a = accum[GYROSCOPIC_K4_W2P].a;
    ona_b = accum[GYROSCOPIC_K4_W2P].b;
    bu_a  = accum[GYROSCOPIC_K4_F].a;
    bu_b  = accum[GYROSCOPIC_K4_F].b;
    composed_a = cs_a + una_b - ona_b - bu_a;
    composed_b = cs_b + una_a - ona_a - bu_b;
    return gravity * (composed_a + composed_b);
}

GYROSCOPIC_EXPORT float gyroscopic_sum_gyroacc(
    const gyro_accum_t accum[4],
    float gravity)
{
    float sum = 0.0f;
    int i;

    if (accum == NULL) {
        return 0.0f;
    }
    for (i = 0; i < 4; ++i) {
        sum += accum[i].a + accum[i].b;
    }
    return gravity * sum;
}


GYROSCOPIC_EXPORT float gyroscopic_depth4_bu_factor(void) {
    const float rho   = (float) RHO;
    const float delta = (float) APERTURE_GAP;
    const float d2    = delta * delta;
    const float d4    = d2 * d2;

    return 1.0f
        - 4.0f * rho * d2
        + (float) GYROSCOPIC_C4_REF * d4;
}

GYROSCOPIC_EXPORT void gyroscopic_analyze_q1_group(
    const uint8_t signs[16],
    uint8_t * q_class,
    uint8_t * shell,
    uint8_t * k4_char)
{
    uint64_t signs_a;
    uint64_t signs_b;
    uint8_t chi_a;
    uint8_t chi_b;
    uint8_t q;
    uint8_t k4;

    if (signs == NULL) {
        if (q_class) *q_class = 0;
        if (shell)   *shell = 0;
        if (k4_char) *k4_char = GYROSCOPIC_K4_ID;
        return;
    }

    memcpy(&signs_a, signs, sizeof(uint64_t));
    memcpy(&signs_b, signs + 8, sizeof(uint64_t));
    chi_a = gyroscopic_chirality_from_signs64(signs_a);
    chi_b = gyroscopic_chirality_from_signs64(signs_b);

    /* q-class is the XOR of the two chiralities; shell its population. */
    q = (uint8_t) (chi_a ^ chi_b);

    /*
     * K4 assignment as a group homomorphism (Z/2 x Z/2):
     *   bit 0 = parity of chi_a, bit 1 = parity of chi_b.
     * This is closed under composition and independent of position.
     */
    k4 = (uint8_t) ((popcount32(chi_a) & 1u) | ((popcount32(chi_b) & 1u) << 1));

    if (q_class) *q_class = q;
    if (shell)   *shell = (uint8_t) popcount32((uint32_t) (q & CHIRALITY_MASK_6));
    if (k4_char) *k4_char = k4;
}

GYROSCOPIC_EXPORT uint8_t gyroscopic_pack_q1_meta(uint8_t shell, uint8_t k4_char, uint8_t h) {
    return (uint8_t) ((shell & 0x7u)
                    | ((k4_char & 0x3u) << 3)
                    | (((h >> 3) & 0x7u) << 5));
}

GYROSCOPIC_EXPORT void gyroscopic_unpack_q1_meta(
    uint8_t packed, uint8_t * shell, uint8_t * k4_char, uint8_t * h_zone)
{
    if (shell)   *shell = (uint8_t) (packed & 0x7u);
    if (k4_char) *k4_char = (uint8_t) ((packed >> 3) & 0x3u);
    if (h_zone)  *h_zone = (uint8_t) ((packed >> 5) & 0x7u);
}

GYROSCOPIC_EXPORT uint8_t gyroscopic_route_path(uint8_t shell, uint8_t k4_char) {
    if (shell == 0u || shell >= 6u) {
        return GYROSCOPIC_PATH_ISOTROPIC;
    }
    switch (k4_char & 0x3u) {
        case GYROSCOPIC_K4_ID:  return GYROSCOPIC_PATH_BULK_CS;
        case GYROSCOPIC_K4_W2:  return GYROSCOPIC_PATH_BULK_UNA;
        case GYROSCOPIC_K4_W2P: return GYROSCOPIC_PATH_BULK_ONA;
        default:                return GYROSCOPIC_PATH_BULK_BU;
    }
}

/* ================================================================== */
/* 5. Gravity scale.                                                  */
/* ================================================================== */

GYROSCOPIC_EXPORT float gyroscopic_gravity_g1(void) {
    const double delta = (double) APERTURE_GAP;       /* 1 - rho            */
    const double rho   = (double) RHO;                /* DELTA_BU / M_A     */
    const double f_ext =
        1.0 - 4.0 * rho * delta * delta
        + (double) GYROSCOPIC_C4_REF * delta * delta * delta * delta;
    const double tau_g =
        (double) OMEGA_SIZE * delta * pow(rho, 5.0) * f_ext;
    const double eta = log((double) GYROSCOPIC_V_EW_GEV / (double) GYROSCOPIC_E_CS_GEV);
    return (float) (tau_g + 2.0 * eta);
}

GYROSCOPIC_EXPORT float gyroscopic_gravity_scale(
    int layer,
    int total_layers,
    uint8_t k4_char,
    uint8_t shell)
{
    float g1;
    float psi;

    /* (k4_char, shell) are metadata only; never magnitude factors. */
    (void) k4_char;
    (void) shell;

    if (total_layers < 1) {
        total_layers = GYROSCOPIC_DEFAULT_TOTAL_LAYERS;
    }
    if (layer < 0) {
        layer = 0;
    }
    if (layer > total_layers) {
        layer = total_layers;
    }

    g1 = gyroscopic_gravity_g1();
    psi = (float) layer / (float) total_layers;
    return expf(g1 * psi);
}

GYROSCOPIC_EXPORT void gyroscopic_chi_hist_m2_eta(
    const uint32_t hist[64],
    float *          m2_out,
    float *          eta_out)
{
    uint64_t W = 0;
    uint64_t sumsq = 0;
    float    fw[64];
    float    e0;
    float    etot;
    int      i;

    if (m2_out != NULL) {
        *m2_out = 64.0f;
    }
    if (eta_out != NULL) {
        *eta_out = 0.0f;
    }
    if (hist == NULL) {
        return;
    }

    for (i = 0; i < 64; ++i) {
        W += (uint64_t) hist[i];
        sumsq += (uint64_t) hist[i] * (uint64_t) hist[i];
    }
    if (W == 0 || sumsq == 0) {
        return;
    }

    if (m2_out != NULL) {
        *m2_out = (float) ((double) W * (double) W / (double) sumsq);
    }

    if (eta_out == NULL) {
        return;
    }

    for (i = 0; i < 64; ++i) {
        fw[i] = (float) hist[i] / (float) W;
    }
    wht64_float(fw);
    e0 = fabsf(fw[0]);
    etot = 0.0f;
    for (i = 0; i < 64; ++i) {
        etot += fw[i] * fw[i];
    }
    if (etot > 0.0f) {
        *eta_out = 1.0f - (e0 * e0) / etot;
    }
}

GYROSCOPIC_EXPORT uint8_t gyroscopic_chirality_word6(uint32_t state24) {
    const uint16_t a12 = (uint16_t) ((state24 >> 12) & LAYER_MASK_12);
    const uint16_t b12 = (uint16_t) (state24 & LAYER_MASK_12);
    const uint16_t diff = (uint16_t) (a12 ^ b12);
    uint8_t          out = 0;
    int              i;

    for (i = 0; i < (int) CHIRALITY_QUBITS_6; ++i) {
        const uint16_t pair = (uint16_t) ((diff >> (2 * i)) & 3u);
        if (pair == 3u) {
            out |= (uint8_t) (1u << i);
        }
    }
    return out;
}

GYROSCOPIC_EXPORT void gyroscopic_kv_f32_to_word4(const float * x, uint8_t word4[4]) {
    uint64_t signs;
    int      i;
    int      j;

    if (word4 == NULL) {
        return;
    }
    if (x == NULL) {
        memset(word4, 0, 4);
        return;
    }

    signs = gyroscopic_signs64_from_f32(x);
    for (i = 0; i < 4; ++i) {
        const uint32_t sig = (uint32_t) ((signs >> (16 * i)) & 0xFFFFu);
        float          norm = 0.0f;
        uint8_t        mag;

        for (j = 0; j < 16; ++j) {
            const float v = x[i * 16 + j];
            norm += v * v;
        }
        mag = (uint8_t) fminf(255.0f, sqrtf(norm) * 16.0f);
        word4[i] = (uint8_t) ((sig ^ (sig >> 8) ^ mag) & 0xFFu);
    }
}

GYROSCOPIC_EXPORT uint8_t gyroscopic_word4_chirality(
    const uint8_t word4[4],
    uint32_t *    state_inout)
{
    uint32_t s;
    int      i;

    if (word4 == NULL) {
        return 0;
    }
    s = state_inout != NULL ? *state_inout : 0u;
    for (i = 0; i < 4; ++i) {
        s = gyroscopic_step_omega12(s, word4[i]);
    }
    if (state_inout != NULL) {
        *state_inout = s;
    }
    return gyroscopic_chirality_word6(s);
}

GYROSCOPIC_EXPORT uint8_t gyroscopic_kv_f32_block_chirality(
    const float * x,
    uint32_t *    state_inout)
{
    uint8_t word4[4];

    gyroscopic_kv_f32_to_word4(x, word4);
    return gyroscopic_word4_chirality(word4, state_inout);
}

GYROSCOPIC_EXPORT int gyroscopic_chi_hist_d_eff(
    const uint32_t hist[64],
    uint8_t        chi_q,
    float *        m2_out,
    float *        eta_out)
{
    float    m2 = 64.0f;
    float    eta = 0.0f;
    uint64_t W = 0;
    float    target;
    int      d;
    int      chi;

    gyroscopic_chi_hist_m2_eta(hist, &m2, &eta);
    if (m2_out != NULL) {
        *m2_out = m2;
    }
    if (eta_out != NULL) {
        *eta_out = eta;
    }
    if (hist == NULL) {
        return 3;
    }

    for (chi = 0; chi < 64; ++chi) {
        W += (uint64_t) hist[chi];
    }
    if (W == 0) {
        return 3;
    }

    /* Condensed (low M₂) → sparser graph still percolates; thermal → widen aperture. */
    target = 0.02f + ((m2 - 1.0f) / 63.0f) * 0.03f;
    if (target < 0.01f) {
        target = 0.01f;
    }
    if (target > 0.05f) {
        target = 0.05f;
    }

    for (d = 0; d <= 3; ++d) {
        uint64_t cum = 0;
        for (chi = 0; chi < 64; ++chi) {
            if (gyroscopic_chirality_distance(chi_q, (uint8_t) chi) <= d) {
                cum += (uint64_t) hist[chi];
            }
        }
        if ((float) cum / (float) W >= target) {
            return d;
        }
    }
    return 3;
}

GYROSCOPIC_EXPORT void gyroscopic_kv_polar_encode_block64(
    const float *      x,
    gyro_kv_polar64_t * out)
{
    uint8_t  word4[4];
    uint32_t s_mid = 0;
    uint32_t s_full = 0;
    float    norm = 0.0f;
    int      i;

    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (x == NULL) {
        return;
    }

    gyroscopic_kv_f32_to_word4(x, word4);
    s_mid = gyroscopic_step_omega12(s_mid, word4[0]);
    s_mid = gyroscopic_step_omega12(s_mid, word4[1]);
    for (i = 0; i < 4; ++i) {
        s_full = gyroscopic_step_omega12(s_full, word4[i]);
    }
    for (i = 0; i < 64; ++i) {
        norm += x[i] * x[i];
    }

    out->boundary = gyroscopic_chirality_word6(s_mid);
    out->chi = gyroscopic_chirality_word6(s_full);
    out->shell = (uint8_t) popcount32((uint32_t) out->chi);
    out->r_bits = (uint16_t) fminf(65535.0f, sqrtf(norm) * 256.0f);
}

GYROSCOPIC_EXPORT void gyroscopic_analyze_q1_group_full(
    const uint8_t signs[16],
    int layer,
    int total_layers,
    gyroscopic_q1_meta * out)
{
    if (out == NULL) {
        return;
    }
    gyroscopic_analyze_q1_group(signs, &out->q_class, &out->shell, &out->k4_char);
    {
        uint64_t signs_a;
        uint8_t chi_a;
        memcpy(&signs_a, signs, sizeof(uint64_t));
        chi_a = gyroscopic_chirality_from_signs64(signs_a);
        out->h_zone = (uint8_t) ((chi_a >> 3) & 0x7u);
    }
    out->route_path = gyroscopic_route_path(out->shell, out->k4_char);
    out->gravity_scale = gyroscopic_gravity_scale(
        layer, total_layers, out->k4_char, out->shell);
}

/* ================================================================== */
/* 6. Native cyclic QFT (radix-2 DIT, WHT-atom butterflies).          */
/* Matches Python cyclic_qft_butterfly_executed convention:           */
/* bit-reversed input, +i twiddles, 1/sqrt(2) per stage.              */
/* ================================================================== */

static uint32_t gyro_bit_reverse(uint32_t i, int n_bits) {
    uint32_t r = 0u;
    int b;
    for (b = 0; b < n_bits; ++b) {
        r = (r << 1) | (i & 1u);
        i >>= 1;
    }
    return r;
}

GYROSCOPIC_EXPORT void gyroscopic_cyclic_qft(
    float * re,
    float * im,
    int n_bits)
{
    uint32_t n = 1u << n_bits;
    float inv_sqrt2 = (float) (1.0 / sqrt(2.0));
    uint32_t i;
    int stage;

    /* Bit-reversal reorder in place via a scratch pass. */
    for (i = 0; i < n; ++i) {
        uint32_t j = gyro_bit_reverse(i, n_bits);
        if (j > i) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }

    for (stage = 0; stage < n_bits; ++stage) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half = m >> 1;
        double base_angle = 6.283185307179586476925286766559 / (double) m;
        uint32_t base;
        for (base = 0; base < n; base += m) {
            uint32_t j;
            for (j = 0; j < half; ++j) {
                double ang = base_angle * (double) j;
                float wr = (float) cos(ang);
                float wi = (float) sin(ang);
                uint32_t i0 = base + j;
                uint32_t i1 = base + j + half;
                float br = re[i1] * wr - im[i1] * wi;
                float bi = re[i1] * wi + im[i1] * wr;
                float ar = re[i0];
                float ai = im[i0];
                re[i0] = (ar + br) * inv_sqrt2;
                im[i0] = (ai + bi) * inv_sqrt2;
                re[i1] = (ar - br) * inv_sqrt2;
                im[i1] = (ai - bi) * inv_sqrt2;
            }
        }
    }
}

/* ================================================================== */
/* 7. Byte-ledger modular arithmetic (shift-add ladder).              */
/* ================================================================== */

static uint64_t gyro_mul_mod_ladder(uint64_t y, uint64_t multiplier, uint64_t n) {
    uint64_t acc = 0;
    uint64_t addend;
    if (n <= 1u) {
        return 0;
    }
    addend = multiplier % n;
    while (y) {
        if (y & 1u) {
            acc = (acc + addend) % n;
        }
        addend = (addend + addend) % n;
        y >>= 1;
    }
    return acc;
}

GYROSCOPIC_EXPORT uint64_t gyroscopic_mul_mod_ladder(
    uint64_t y,
    uint64_t multiplier,
    uint64_t n)
{
    return gyro_mul_mod_ladder(y, multiplier, n);
}

GYROSCOPIC_EXPORT uint64_t gyroscopic_exp_mod_ladder(
    uint64_t a,
    uint64_t x,
    uint64_t n)
{
    uint64_t acc = 1u;
    uint64_t base;
    if (n <= 1u) {
        return 0;
    }
    base = a % n;
    while (x) {
        if (x & 1u) {
            acc = gyro_mul_mod_ladder(acc, base, n);
        }
        base = gyro_mul_mod_ladder(base, base, n);
        x >>= 1;
    }
    return acc;
}

GYROSCOPIC_EXPORT uint64_t gyroscopic_multiplicative_period(
    uint64_t a,
    uint64_t n,
    uint64_t max_len)
{
    uint64_t cur = 1u;
    uint64_t base;
    uint64_t i;
    if (n <= 1u || max_len <= 1u) {
        return 0;
    }
    base = a % n;
    for (i = 1; i < max_len; ++i) {
        cur = gyro_mul_mod_ladder(cur, base, n);
        if (cur == 1u) {
            return i;
        }
    }
    return 0;
}

/* Period comb on Z_{2^q_bits}, cyclic QFT, return spectral peak index (>=1). */
GYROSCOPIC_EXPORT uint32_t gyroscopic_comb_qft_peak(
    uint64_t period,
    int q_bits,
    float * peak_amp_out)
{
    uint32_t n;
    uint32_t k;
    uint32_t peak = 0;
    float peak_amp = 0.f;
    float * re;
    float * im;
    uint64_t spikes;
    float amp;

    if (q_bits < 1 || q_bits > 20 || period == 0) {
        return 0;
    }
    n = 1u << q_bits;
    if (period >= (uint64_t) n) {
        return 0;
    }

    spikes = 0;
    for (k = 0; k < n; k += (uint32_t) period) {
        spikes++;
    }
    if (spikes == 0) {
        return 0;
    }
    amp = 1.f / sqrtf((float) spikes);

    re = (float *) calloc((size_t) n, sizeof(float));
    im = (float *) calloc((size_t) n, sizeof(float));
    if (re == NULL || im == NULL) {
        free(re);
        free(im);
        return 0;
    }

    for (k = 0; k < n; k += (uint32_t) period) {
        re[k] = amp;
    }

    gyroscopic_cyclic_qft(re, im, q_bits);

    for (k = 1; k < n; ++k) {
        float mag = re[k] * re[k] + im[k] * im[k];
        if (mag > peak_amp) {
            peak_amp = mag;
            peak = k;
        }
    }

    free(re);
    free(im);

    if (peak_amp < 1e-18f) {
        return 0;
    }
    if (peak_amp_out != NULL) {
        *peak_amp_out = sqrtf(peak_amp);
    }
    return peak;
}

/* ------------------------------------------------------------------ */
/* 64x64 tile projection / hybrid GEMV (Pi_basis, tiles.py parity). */
/* ------------------------------------------------------------------ */

#define GYRO_TILE GYROSCOPIC_TILE_SIZE

static float gyro_tile_frob_norm(const float * W) {
    double acc = 0.0;
    int i;
    for (i = 0; i < GYRO_TILE * GYRO_TILE; ++i) {
        const double v = (double) W[i];
        acc += v * v;
    }
    return (float) sqrt(acc);
}

static int gyro_tile_popcount8(uint8_t v) {
#if defined(_MSC_VER)
    return (int) __popcnt((unsigned) v);
#elif defined(__GNUC__) || defined(__clang__)
    return (int) __builtin_popcount((unsigned) v);
#else
    int n = 0;
    while (v) { n += (int) (v & 1u); v >>= 1; }
    return n;
#endif
}

GYROSCOPIC_EXPORT void gyroscopic_project_chi_coeffs(
    const float * W,
    float *       f_out)
{
    uint8_t idx[GYRO_TILE];
    int d;
    int i;
    for (i = 0; i < GYRO_TILE; ++i) {
        idx[i] = (uint8_t) i;
    }
    for (d = 0; d < GYRO_TILE; ++d) {
        double acc = 0.0;
        for (i = 0; i < GYRO_TILE; ++i) {
            const int j = (int) (idx[i] ^ (uint8_t) d);
            acc += (double) W[i * GYRO_TILE + j];
        }
        f_out[d] = (float) (acc / (double) GYRO_TILE);
    }
}

static void gyro_tile_project_shell(const float * W, float * P) {
    int i;
    int j;
    for (i = 0; i < GYRO_TILE; ++i) {
        for (j = 0; j < GYRO_TILE; ++j) {
            const uint8_t d = (uint8_t) (i ^ j);
            const int shell = gyro_tile_popcount8(d);
            double acc = 0.0;
            int cnt = 0;
            int ii;
            int jj;
            for (ii = 0; ii < GYRO_TILE; ++ii) {
                for (jj = 0; jj < GYRO_TILE; ++jj) {
                    if (gyro_tile_popcount8((uint8_t) (ii ^ jj)) == shell) {
                        acc += (double) W[ii * GYRO_TILE + jj];
                        ++cnt;
                    }
                }
            }
            P[i * GYRO_TILE + j] = (cnt > 0) ? (float) (acc / (double) cnt) : 0.0f;
        }
    }
}

GYROSCOPIC_EXPORT void gyroscopic_tile_decompose_ratios(
    const float *              W,
    gyroscopic_tile_ratios_t * out)
{
    float P_shell[GYRO_TILE * GYRO_TILE];
    float P_chi[GYRO_TILE * GYRO_TILE];
    float f[GYRO_TILE];
    float norm_w;
    int i;
    int j;

    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof(*out));

    norm_w = gyro_tile_frob_norm(W);
    out->norm = norm_w;
    if (norm_w <= 0.0f) {
        return;
    }

    gyro_tile_project_shell(W, P_shell);
    gyroscopic_project_chi_coeffs(W, f);
    for (i = 0; i < GYRO_TILE; ++i) {
        for (j = 0; j < GYRO_TILE; ++j) {
            P_chi[i * GYRO_TILE + j] = f[i ^ j];
        }
    }

    {
        double s_shell = 0.0;
        double s_chi = 0.0;
        double s_cms = 0.0;
        double s_def = 0.0;
        for (i = 0; i < GYRO_TILE * GYRO_TILE; ++i) {
            const double w = (double) W[i];
            const double pc = (double) P_chi[i];
            const double ps = (double) P_shell[i];
            const double d = w - pc;
            const double co = pc - ps;
            s_shell += ps * ps;
            s_chi   += pc * pc;
            s_cms   += co * co;
            s_def   += d * d;
        }
        out->r_shell            = (float) (sqrt(s_shell) / (double) norm_w);
        out->r_chi              = (float) (sqrt(s_chi)   / (double) norm_w);
        out->r_chi_minus_shell  = (float) (sqrt(s_cms)  / (double) norm_w);
        out->r_defect           = (float) (sqrt(s_def)  / (double) norm_w);
    }
}

GYROSCOPIC_EXPORT void gyroscopic_chi_circulant_matvec(
    const float * f,
    const float * x,
    float *       y)
{
    float fw[GYRO_TILE];
    float xw[GYRO_TILE];
    int   i;

    if (f == NULL || x == NULL || y == NULL) {
        return;
    }

    memcpy(xw, x, (size_t) GYRO_TILE * sizeof(float));
    wht64_float(xw);
    memcpy(fw, f, (size_t) GYRO_TILE * sizeof(float));
    wht64_float(fw);
    for (i = 0; i < GYRO_TILE; ++i) {
        y[i] = fw[i] * xw[i] * (1.0f / (float) GYRO_TILE);
    }
    wht64_float(y);
}

/* ---------------------------------------------------------------------------
 * Owned holonomic affinity step (the spine, Runtime Spec §19-§21 + SDK §11.10).
 *
 * This IS the attention-like job, computed as exact GF(2)^6 circulant
 * algebra instead of float QK^T + softmax. Tokens carry their carrier
 * egress chi (the chart), not a foreign residual to be scored.
 *
 *   H[64]  = gather key mass at chi_k positions (density over the 64-bin
 *            chirality register; O(nk))
 *   Hw     = WHT(H)                      (O(64 log 64))
 *   Aw[r]  = Hw[r] * Khat[r]             (pointwise spectral kernel, O(64))
 *   A      = iWHT(Aw)                    (O(64 log 64))
 *   aff[q] = A[ chi_q[q] ]                 (emit, O(nq))
 *
 * Total O(nq + nk + 64 log 64), independent of d. Brute
 * XOR-distance attention is O(nq * nk * d). The kernel owns the
 * structured affinity step; the continuous head consumes A as a chart.
 *
 * Khat is the spectral kernel (WHT of the native chi-circulant column
 * f, i.e. gyroscopic_project_chi_coeffs(W, Khat)). Self-affinity
 * = pass the same chi array for q and k.
 *
 * chi_q/chi_k: one chi6 per token. aff_out: nq scores.
 * Returns 0 on success, -1 on null input.
 * ------------------------------------------------------------------------- */
GYROSCOPIC_EXPORT int gyroscopic_affinity_step(
        const uint8_t * chi_q, int64_t nq,
        const uint8_t * chi_k, int64_t nk,
        const float   * khat,
        float         * aff_out) {
    int64_t i;
    int     r;
    float   H[GYRO_TILE];
    float   Hw[GYRO_TILE];
    float   Aw[GYRO_TILE];

    if (chi_q == NULL || chi_k == NULL || khat == NULL || aff_out == NULL) {
        return -1;
    }
    if (nq <= 0 || nk <= 0) {
        return -1;
    }

    /* 1. gather key mass into the 64-bin chi histogram */
    for (r = 0; r < GYRO_TILE; ++r) {
        H[r] = 0.0f;
    }
    for (i = 0; i < nk; ++i) {
        const int b = (int) chi_k[i] & 0x3F;
        H[b] += 1.0f;
    }

    /* 2. WHT(H) */
    memcpy(Hw, H, (size_t) GYRO_TILE * sizeof(float));
    wht64_float(Hw);

    /* 3. pointwise spectral kernel */
    for (r = 0; r < GYRO_TILE; ++r) {
        Aw[r] = Hw[r] * khat[r];
    }

    /* 4. iWHT -> A in chi basis */
    wht64_float(Aw);
    {
        const float inv = 1.0f / (float) GYRO_TILE;
        for (r = 0; r < GYRO_TILE; ++r) {
            Aw[r] *= inv;
        }
    }

    /* 5. emit: score[q] = A[ chi_q[q] ] */
    for (i = 0; i < nq; ++i) {
        const int b = (int) chi_q[i] & 0x3F;
        aff_out[i] = Aw[b];
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Per-pair chi-coupling entry (the genuine attention score, Runtime Spec
 * sec 19-21). score[i] = K_direct[ chi_q[i] ^ chi_k[i] ], where K_direct is
 * the circulant column. Constant-time via a 64-entry LUT; no d-MACs, no
 * float GEMM. chi_q/chi_k: one chi6 per token (length n). score: length n.
 * This is the kernel-owned QK-equivalent: a class function of the XOR, which
 * is why the whole channel diagonalizes under WHT (see
 * gyroscopic_affinity_step). Returns 0 / -1.
 * ------------------------------------------------------------------------- */
GYROSCOPIC_EXPORT int gyroscopic_chi_coupling(
        const uint8_t * chi_q, const uint8_t * chi_k, int64_t n,
        const float   * kdir, float * score) {
    int64_t i;
    if (chi_q == NULL || chi_k == NULL || kdir == NULL || score == NULL) {
        return -1;
    }
    if (n <= 0) {
        return -1;
    }
    for (i = 0; i < n; ++i) {
        const int xor = ((int) chi_q[i] ^ (int) chi_k[i]) & 0x3F;
        score[i] = kdir[xor];
    }
    return 0;
}

GYROSCOPIC_EXPORT void gyroscopic_tile_hybrid_matvec(
    const float * W,
    const float * x,
    float *       y)
{
    float f[GYRO_TILE];
    int i;
    int j;

    gyroscopic_project_chi_coeffs(W, f);
    gyroscopic_chi_circulant_matvec(f, x, y);

    for (i = 0; i < GYRO_TILE; ++i) {
        for (j = 0; j < GYRO_TILE; ++j) {
            const float w = W[i * GYRO_TILE + j];
            const float p = f[i ^ j];
            y[i] += (w - p) * x[j];
        }
    }
}

GYROSCOPIC_EXPORT float gyroscopic_tile_hybrid_dot_row(
    const float * W,
    int           row,
    const float * x)
{
    float f[GYRO_TILE];
    float y = 0.0f;
    int j;

    if (row < 0 || row >= GYRO_TILE) {
        return 0.0f;
    }

    gyroscopic_project_chi_coeffs(W, f);
    for (j = 0; j < GYRO_TILE; ++j) {
        y += f[row ^ j] * x[j];
    }
    for (j = 0; j < GYRO_TILE; ++j) {
        const float w = W[row * GYRO_TILE + j];
        const float p = f[row ^ j];
        y += (w - p) * x[j];
    }
    return y;
}

/* Trajectory + receipt types / steppers (single-owner instance lives in attn). */
GYROSCOPIC_EXPORT void hqvm_traj_reset(gyro_trajectory_state_t *t) {
    if (!t) return;
    t->state24 = (uint32_t)GENE_MAC_REST;
    t->depth = 0;
    t->n_trans = 0;
    t->phase_idx = 0;
}

GYROSCOPIC_EXPORT void hqvm_traj_step(gyro_trajectory_state_t *t, uint8_t intron_byte) {
    if (!t) return;
    t->state24 = gyroscopic_step_omega12(t->state24, intron_byte);
    t->n_trans++;
}

GYROSCOPIC_EXPORT uint32_t hqvm_receipt_seal(const hqvm_receipt_t *r) {
    uint32_t h = 2166136261u;
    const unsigned char *p;
    size_t n, i;
    if (!r) return 0;
    p = (const unsigned char *)&r->anchor12; n = sizeof(r->anchor12);
    for (i = 0; i < n; ++i) { h ^= p[i]; h *= 16777619u; }
    h ^= r->k4_family; h *= 16777619u;
    p = (const unsigned char *)&r->state24; n = sizeof(r->state24);
    for (i = 0; i < n; ++i) { h ^= p[i]; h *= 16777619u; }
    p = (const unsigned char *)&r->depth; n = sizeof(r->depth);
    for (i = 0; i < n; ++i) { h ^= p[i]; h *= 16777619u; }
    return h;
}

GYROSCOPIC_EXPORT void hqvm_receipt_print(const hqvm_receipt_t *r) {
    if (!r) return;
    fprintf(stderr,
            "[hqvm-receipt] anchor=%03x k4=%u state24=%06x depth=%llu fnv=%08x\n",
            (unsigned)(r->anchor12 & 0xFFF), (unsigned)(r->k4_family & 0x3),
            (unsigned)(r->state24 & 0xFFFFFFu), (unsigned long long)r->depth,
            (unsigned)r->fnv1a);
    fflush(stderr);
}


/* Wavefunction grammar (formerly wave.c). */
static uint8_t s_byte_of_q6_fam[64][4];
static int     s_byte_table_ready = 0;
static int     s_byte_table_ok    = 0;

static int popcount8(uint8_t x) {
#if defined(_MSC_VER)
    return (int) __popcnt((unsigned) x);
#elif defined(__GNUC__) || defined(__clang__)
    return (int) __builtin_popcount((unsigned) x);
#else
    int n = 0;
    while (x) { n += (int) (x & 1u); x >>= 1; }
    return n;
#endif
}

static uint8_t fam_of_intron(uint8_t intron) {
    return (uint8_t) ((intron & 1u) | ((intron >> 6) & 2u));
}

static uint8_t q6_of_intron(uint8_t intron) {
    return (uint8_t) ((intron >> 1) & CHIRALITY_MASK_6);
}

void hqvm_byte_table_init(void) {
    int b, ok = 1;
    int q, f;
    if (s_byte_table_ready) {
        return;
    }
    memset(s_byte_of_q6_fam, 0, sizeof(s_byte_of_q6_fam));
    for (b = 0; b < 256; ++b) {
        const uint8_t intron = (uint8_t) (b ^ (int) GENE_MIC_S);
        const uint8_t q6     = q6_of_intron(intron);
        const uint8_t fam    = fam_of_intron(intron);
        s_byte_of_q6_fam[q6][fam] = (uint8_t) b;
    }
    for (b = 0; b < 256; ++b) {
        const uint8_t intron = (uint8_t) (b ^ (int) GENE_MIC_S);
        const uint8_t q6     = q6_of_intron(intron);
        const uint8_t fam    = fam_of_intron(intron);
        if (s_byte_of_q6_fam[q6][fam] != (uint8_t) b) {
            ok = 0;
        }
    }
    for (q = 0; q < 64; ++q) {
        for (f = 0; f < 4; ++f) {
            const uint8_t byte   = s_byte_of_q6_fam[q][f];
            const uint8_t intron = (uint8_t) (byte ^ (int) GENE_MIC_S);
            if (q6_of_intron(intron) != (uint8_t) q) {
                ok = 0;
            }
            if (fam_of_intron(intron) != (uint8_t) f) {
                ok = 0;
            }
        }
    }
    s_byte_table_ok    = ok;
    s_byte_table_ready = 1;
}

int hqvm_byte_table_ok(void) {
    if (!s_byte_table_ready) {
        hqvm_byte_table_init();
    }
    return s_byte_table_ok;
}

uint8_t hqvm_byte_of_q6_fam(uint8_t q6, uint8_t fam) {
    if (!s_byte_table_ready) {
        hqvm_byte_table_init();
    }
    return s_byte_of_q6_fam[q6 & 63][fam & 3];
}

uint8_t hqvm_q6_of_byte(uint8_t byte) {
    return q6_of_intron((uint8_t) (byte ^ (int) GENE_MIC_S));
}

uint8_t hqvm_fam_of_byte(uint8_t byte) {
    return fam_of_intron((uint8_t) (byte ^ (int) GENE_MIC_S));
}

void hqvm_decompose_byte(uint8_t byte, hqvm_byte_fiber * out) {
    uint8_t intron;
    uint8_t fwd;
    uint8_t rev;
    uint8_t b[8];
    int     i;
    if (!out) {
        return;
    }
    intron = (uint8_t) (byte ^ (int) GENE_MIC_S);
    fwd    = (uint8_t) (intron & 0x0Fu);
    rev    = (uint8_t) ((intron >> 4) & 0x0Fu);
    for (i = 0; i < 8; ++i) {
        b[i] = (uint8_t) ((intron >> i) & 1u);
    }
    out->byte         = byte;
    out->intron       = intron;
    out->q6           = hqvm_q6_of_byte(byte);
    out->family       = hqvm_fam_of_byte(byte);
    out->phase_net    = (uint8_t) ((b[0] ^ b[7]) | ((b[1] ^ b[6]) << 1) | ((b[2] ^ b[5]) << 2) | ((b[3] ^ b[4]) << 3));
    out->phase_common = (uint8_t) ((b[0] & b[7]) | ((b[1] & b[6]) << 1) | ((b[2] & b[5]) << 2) | ((b[3] & b[4]) << 3));
    out->fold_degree  = (uint8_t) popcount8(out->phase_net);
    out->is_flat      = (uint8_t) (fwd == rev);
}

void hqvm_state24_to_uv6(uint32_t state24, uint8_t * u6, uint8_t * v6) {
    const uint16_t a12 = (uint16_t) ((state24 >> 12) & LAYER_MASK_12);
    const uint16_t b12 = (uint16_t) (state24 & LAYER_MASK_12);
    const uint16_t ua12 = (uint16_t) (a12 ^ (uint16_t) GENE_MAC_A12);
    const uint16_t vb12 = (uint16_t) (b12 ^ (uint16_t) GENE_MAC_A12);
    uint8_t        u   = 0;
    uint8_t        v   = 0;
    int            j;
    if (!u6 || !v6) {
        return;
    }
    for (j = 0; j < 6; ++j) {
        if ((ua12 >> (2 * j)) & 3u) {
            u |= (uint8_t) (1u << j);
        }
        if ((vb12 >> (2 * j)) & 3u) {
            v |= (uint8_t) (1u << j);
        }
    }
    *u6 = u;
    *v6 = v;
}

uint32_t hqvm_uv6_to_state24(uint8_t u6, uint8_t v6) {
    uint16_t a12 = (uint16_t) GENE_MAC_A12;
    uint16_t b12 = (uint16_t) GENE_MAC_A12;
    int      j;
    for (j = 0; j < 6; ++j) {
        if ((u6 >> j) & 1u) {
            a12 ^= (uint16_t) (3u << (2 * j));
        }
        if ((v6 >> j) & 1u) {
            b12 ^= (uint16_t) (3u << (2 * j));
        }
    }
    return ((uint32_t) (a12 & LAYER_MASK_12) << 12) | (uint32_t) (b12 & LAYER_MASK_12);
}

uint8_t hqvm_chi6_uv(uint8_t u6, uint8_t v6) {
    return (uint8_t) (u6 ^ v6);
}

int hqvm_code_shell(uint8_t u6, uint8_t v6) {
    return popcount8((uint8_t) (u6 ^ v6));
}

static void trace_word_state24(
    const uint8_t * word,
    int             n_bytes,
    uint32_t        state24,
    uint8_t *       u6_out,
    uint8_t *       v6_out)
{
    int i;
    for (i = 0; i < n_bytes; ++i) {
        state24 = gyroscopic_step_omega12(state24, word[i]);
    }
    hqvm_state24_to_uv6(state24, u6_out, v6_out);
}

void hqvm_trace_word_bytes(
    const uint8_t * word,
    int             n_bytes,
    uint8_t         u6_in,
    uint8_t         v6_in,
    uint8_t *       u6_out,
    uint8_t *       v6_out)
{
    trace_word_state24(word, n_bytes, hqvm_uv6_to_state24(u6_in, v6_in), u6_out, v6_out);
}

uint32_t hqvm_trace_word_state24(const uint8_t * word, int n_bytes, uint32_t state24) {
    int i;
    for (i = 0; i < n_bytes; ++i) {
        state24 = gyroscopic_step_omega12(state24, word[i]);
    }
    return state24;
}

static int omega_perm_index(int gate, int u6, int v6) {
    const uint8_t comp = (uint8_t) CHIRALITY_MASK_6;
    switch (gate) {
        case GYROSCOPIC_K4_ID:
            return u6 * 64 + v6;
        case GYROSCOPIC_K4_W2:
            return ((int) (u6 ^ comp)) * 64 + v6;
        case GYROSCOPIC_K4_W2P:
            return u6 * 64 + (int) (v6 ^ comp);
        case GYROSCOPIC_K4_F:
            return ((int) (u6 ^ comp)) * 64 + (int) (v6 ^ comp);
        default:
            return u6 * 64 + v6;
    }
}

int hqvm_wave_merge(hqvm_wave_term * terms, int * n_terms) {
    int i = 0;
    int w = 0;
    if (!terms || !n_terms || *n_terms <= 0) {
        return -1;
    }
    while (i < *n_terms) {
        int j = i + 1;
        int acc = (int) terms[i].sign * (int) terms[i].multiplicity;
        while (j < *n_terms && terms[j].omega_index == terms[i].omega_index) {
            acc += (int) terms[j].sign * (int) terms[j].multiplicity;
            ++j;
        }
        if (acc != 0) {
            terms[w].omega_index  = terms[i].omega_index;
            terms[w].sign         = (int8_t) (acc > 0 ? 1 : -1);
            terms[w].multiplicity = (uint8_t) (acc > 0 ? acc : -acc);
            ++w;
        }
        i = j;
    }
    *n_terms = w;
    return 0;
}

int hqvm_wave_apply_k4(
    hqvm_wave_term * terms,
    int              n_terms,
    int              k4_gate,
    int              max_terms)
{
    hqvm_wave_term tmp[4096];
    int            i;
    if (!terms || n_terms <= 0 || n_terms > 4096 || max_terms <= 0) {
        return -1;
    }
    if (k4_gate == GYROSCOPIC_K4_ID) {
        return n_terms;
    }
    if (n_terms > max_terms) {
        return -1;
    }
    for (i = 0; i < n_terms; ++i) {
        const int u6 = (int) (terms[i].omega_index / 64);
        const int v6 = (int) (terms[i].omega_index % 64);
        tmp[i]         = terms[i];
        tmp[i].omega_index = (uint16_t) omega_perm_index(k4_gate, u6, v6);
    }
    memcpy(terms, tmp, (size_t) n_terms * sizeof(hqvm_wave_term));
    return hqvm_wave_merge(terms, &n_terms) == 0 ? n_terms : -1;
}

static int check_w2_signature(const uint8_t * word, int expect_tau_u, int expect_tau_v) {
    uint8_t uo, vo;
    hqvm_trace_word_bytes(word, 2, 0, 0, &uo, &vo);
    return uo == (uint8_t) expect_tau_u && vo == (uint8_t) expect_tau_v;
}

static int check_w2_t2(const uint8_t * word) {
    int u, v;
    int chi_bad = 0;
    int shell_bad = 0;
    for (u = 0; u < 64; ++u) {
        for (v = 0; v < 64; ++v) {
            const uint8_t chi = hqvm_chi6_uv((uint8_t) u, (uint8_t) v);
            uint8_t       uo, vo;
            hqvm_trace_word_bytes(word, 2, (uint8_t) u, (uint8_t) v, &uo, &vo);
            if (hqvm_chi6_uv(uo, vo) != (uint8_t) (chi ^ HQVM_CHI_FLIP_6)) {
                ++chi_bad;
            }
            if (hqvm_code_shell(uo, vo) != (6 - hqvm_code_shell((uint8_t) u, (uint8_t) v))) {
                ++shell_bad;
            }
        }
    }
    return chi_bad == 0 && shell_bad == 0;
}

static int check_sparse_k4(void) {
    hqvm_wave_term terms[4];
    int            n = 3;
    terms[0].omega_index  = (uint16_t) (10 * 64 + 20);
    terms[0].sign         = 1;
    terms[0].multiplicity = 1;
    terms[1].omega_index  = (uint16_t) (10 * 64 + 20);
    terms[1].sign         = -1;
    terms[1].multiplicity = 1;
    terms[2].omega_index  = (uint16_t) (5 * 64 + 7);
    terms[2].sign         = 1;
    terms[2].multiplicity = 2;
        if (hqvm_wave_merge(terms, &n) != 0 || n != 1) {
        return 0;
    }
    if (terms[0].omega_index != (uint16_t) (5 * 64 + 7)) {
        return 0;
    }
    n = 1;
    if (hqvm_wave_apply_k4(terms, n, GYROSCOPIC_K4_W2, 8) != 1) {
        return 0;
    }
    if (terms[0].omega_index != (uint16_t) omega_perm_index(GYROSCOPIC_K4_W2, 5, 7)) {
        return 0;
    }
    return 1;
}

uint16_t hqvm_pack_state12(uint8_t u6, uint8_t v6) {
    return (uint16_t) (((u6 & CHIRALITY_MASK_6) << 6) | (v6 & CHIRALITY_MASK_6));
}

void hqvm_unpack_state12(uint16_t s12, uint8_t * u6, uint8_t * v6) {
    if (u6) {
        *u6 = (uint8_t) ((s12 >> 6) & CHIRALITY_MASK_6);
    }
    if (v6) {
        *v6 = (uint8_t) (s12 & CHIRALITY_MASK_6);
    }
}

/* Exact Omega affine step on (u,v): same map as src.api.step_omega12_by_byte. */
static void step_uv6(uint8_t * u6, uint8_t * v6, uint8_t byte) {
    const uint8_t intron = (uint8_t) (byte ^ (uint8_t) GENE_MIC_S);
    const uint8_t mr     = (uint8_t) ((intron >> 1) & CHIRALITY_MASK_6);
    const uint8_t ea     = (uint8_t) ((intron & 1u) ? CHIRALITY_MASK_6 : 0u);
    const uint8_t eb     = (uint8_t) ((intron & 0x80u) ? CHIRALITY_MASK_6 : 0u);
    const uint8_t u_next = (uint8_t) ((*v6) ^ ea);
    const uint8_t v_next = (uint8_t) ((*u6) ^ mr ^ eb);
    *u6 = u_next;
    *v6 = v_next;
}

uint16_t hqvm_step_state12_by_byte(uint16_t s12, uint8_t byte) {
    uint8_t u = (uint8_t) ((s12 >> 6) & CHIRALITY_MASK_6);
    uint8_t v = (uint8_t) (s12 & CHIRALITY_MASK_6);
    step_uv6(&u, &v, byte);
    return hqvm_pack_state12(u, v);
}

uint16_t hqvm_trace_word_state12(uint16_t s12, const uint8_t * word, int n_bytes) {
    int i;
    uint8_t u = (uint8_t) ((s12 >> 6) & CHIRALITY_MASK_6);
    uint8_t v = (uint8_t) (s12 & CHIRALITY_MASK_6);
    if (!word || n_bytes < 0) {
        return s12;
    }
    for (i = 0; i < n_bytes; ++i) {
        step_uv6(&u, &v, word[i]);
    }
    return hqvm_pack_state12(u, v);
}

uint16_t hqvm_sig13_compile(const uint8_t * word, int n_bytes) {
    uint8_t u = 0;
    uint8_t v = 0;
    int     i;
    if (!word || n_bytes <= 0) {
        return 0;
    }
    for (i = 0; i < n_bytes; ++i) {
        step_uv6(&u, &v, word[i]);
    }
    return (uint16_t) (((n_bytes & 1) << 12) | ((u & CHIRALITY_MASK_6) << 6) | (v & CHIRALITY_MASK_6));
}

uint16_t hqvm_sig13_compose(uint16_t left, uint16_t right) {
    const uint8_t lp = (uint8_t) ((left >> 12) & 1u);
    const uint8_t lu = (uint8_t) ((left >> 6) & CHIRALITY_MASK_6);
    const uint8_t lv = (uint8_t) (left & CHIRALITY_MASK_6);
    const uint8_t rp = (uint8_t) ((right >> 12) & 1u);
    const uint8_t ru0 = (uint8_t) ((right >> 6) & CHIRALITY_MASK_6);
    const uint8_t rv0 = (uint8_t) (right & CHIRALITY_MASK_6);
    const uint8_t ru = lp ? rv0 : ru0;
    const uint8_t rv = lp ? ru0 : rv0;
    return (uint16_t) (((lp ^ rp) << 12) | (((ru ^ lu) & CHIRALITY_MASK_6) << 6) | ((rv ^ lv) & CHIRALITY_MASK_6));
}

uint16_t hqvm_sig13_inv(uint16_t sig) {
    const uint8_t p  = (uint8_t) ((sig >> 12) & 1u);
    const uint8_t tu = (uint8_t) ((sig >> 6) & CHIRALITY_MASK_6);
    const uint8_t tv = (uint8_t) (sig & CHIRALITY_MASK_6);
    if (p == 0) {
        return sig;
    }
    return (uint16_t) ((1u << 12) | ((tv & CHIRALITY_MASK_6) << 6) | (tu & CHIRALITY_MASK_6));
}

uint16_t hqvm_sig13_apply(uint16_t s12, uint16_t sig) {
    const uint8_t p  = (uint8_t) ((sig >> 12) & 1u);
    const uint8_t tu = (uint8_t) ((sig >> 6) & CHIRALITY_MASK_6);
    const uint8_t tv = (uint8_t) (sig & CHIRALITY_MASK_6);
    const uint8_t u  = (uint8_t) ((s12 >> 6) & CHIRALITY_MASK_6);
    const uint8_t v  = (uint8_t) (s12 & CHIRALITY_MASK_6);
    if (p == 0) {
        return hqvm_pack_state12((uint8_t) (u ^ tu), (uint8_t) (v ^ tv));
    }
    return hqvm_pack_state12((uint8_t) (v ^ tu), (uint8_t) (u ^ tv));
}

void hqvm_sig13_apply_batch(
    const uint16_t * states,
    int              n_states,
    uint16_t         sig,
    uint16_t *       out)
{
    int i;
    if (!states || !out || n_states <= 0) {
        return;
    }
    for (i = 0; i < n_states; ++i) {
        out[i] = hqvm_sig13_apply(states[i], sig);
    }
}

int hqvm_route2_witnesses(
    uint16_t src12,
    uint16_t tgt12,
    uint8_t  out_b1[16],
    uint8_t  out_b2[16])
{
    int b1;
    int b2;
    int n = 0;
    for (b1 = 0; b1 < 256; ++b1) {
        const uint16_t mid = hqvm_step_state12_by_byte(src12, (uint8_t) b1);
        for (b2 = 0; b2 < 256; ++b2) {
            if (hqvm_step_state12_by_byte(mid, (uint8_t) b2) == tgt12) {
                if (n < 16) {
                    if (out_b1) {
                        out_b1[n] = (uint8_t) b1;
                    }
                    if (out_b2) {
                        out_b2[n] = (uint8_t) b2;
                    }
                }
                ++n;
            }
        }
    }
    return n;
}

int hqvm_route2_synthesize(
    uint16_t src12,
    uint16_t tgt12,
    uint8_t  out_b1[16],
    uint8_t  out_b2[16])
{
    const uint8_t u0 = (uint8_t) ((src12 >> 6) & CHIRALITY_MASK_6);
    const uint8_t v0 = (uint8_t) (src12 & CHIRALITY_MASK_6);
    const uint8_t ut = (uint8_t) ((tgt12 >> 6) & CHIRALITY_MASK_6);
    const uint8_t vt = (uint8_t) (tgt12 & CHIRALITY_MASK_6);
    const uint8_t du = (uint8_t) (ut ^ u0);
    const uint8_t dv = (uint8_t) (vt ^ v0);
    const uint8_t mic = (uint8_t) GENE_MIC_S;
    int flags;

    for (flags = 0; flags < 16; ++flags) {
        const uint8_t ea1 = (uint8_t) ((flags & 1) ? CHIRALITY_MASK_6 : 0u);
        const uint8_t eb1 = (uint8_t) ((flags & 2) ? CHIRALITY_MASK_6 : 0u);
        const uint8_t ea2 = (uint8_t) ((flags & 4) ? CHIRALITY_MASK_6 : 0u);
        const uint8_t eb2 = (uint8_t) ((flags & 8) ? CHIRALITY_MASK_6 : 0u);
        const uint8_t mr1 = (uint8_t) (du ^ eb1 ^ ea2);
        const uint8_t mr2 = (uint8_t) (dv ^ ea1 ^ eb2);
        const uint8_t in1 = (uint8_t) (((flags & 2) ? 0x80u : 0u) | ((mr1 & CHIRALITY_MASK_6) << 1) | ((flags & 1) ? 1u : 0u));
        const uint8_t in2 = (uint8_t) (((flags & 8) ? 0x80u : 0u) | ((mr2 & CHIRALITY_MASK_6) << 1) | ((flags & 4) ? 1u : 0u));
        if (out_b1) {
            out_b1[flags] = (uint8_t) (in1 ^ mic);
        }
        if (out_b2) {
            out_b2[flags] = (uint8_t) (in2 ^ mic);
        }
    }
    return 16;
}

void hqvm_sig13_cache_build(uint16_t cache[8192]) {
    int i;
    if (!cache) {
        return;
    }
    for (i = 0; i < 8192; ++i) {
        cache[i] = (uint16_t) i;
    }
}

void hqvm_sig13_cache_apply_batch(
    const uint16_t * states,
    int              n_states,
    uint16_t         sig,
    const uint16_t * cache,
    uint16_t *       out)
{
    uint16_t key;
    if (!states || !out || !cache || n_states <= 0) {
        return;
    }
    key = cache[sig & 0x1FFF];
    hqvm_sig13_apply_batch(states, n_states, key, out);
}

/* Apply many known sigs (cache optional; NULL => use sigs directly). */
void hqvm_sig13_apply_many_sigs(
    const uint16_t * states,
    int              n_states,
    const uint16_t * sigs,
    int              n_sigs,
    const uint16_t * cache)
{
    int i;
    uint16_t out_stack[4096];
    uint16_t * out = out_stack;
    uint16_t * heap = NULL;
    if (!states || !sigs || n_states <= 0 || n_sigs <= 0) {
        return;
    }
    if (n_states > 4096) {
        heap = (uint16_t *) malloc((size_t) n_states * sizeof(uint16_t));
        if (!heap) {
            return;
        }
        out = heap;
    }
    for (i = 0; i < n_sigs; ++i) {
        const uint16_t key = cache ? cache[sigs[i] & 0x1FFF] : (uint16_t) (sigs[i] & 0x1FFF);
        hqvm_sig13_apply_batch(states, n_states, key, out);
    }
    if (heap) {
        free(heap);
    }
}

/* Compile each word then apply to all states (LEDGER-style compile tax). */
void hqvm_sig13_compile_apply_many(
    const uint16_t * states,
    int              n_states,
    const uint8_t *  words_flat,
    const int *      lens,
    int              n_words)
{
    int i;
    int off = 0;
    uint16_t out_stack[4096];
    uint16_t * out = out_stack;
    uint16_t * heap = NULL;
    if (!states || !words_flat || !lens || n_states <= 0 || n_words <= 0) {
        return;
    }
    if (n_states > 4096) {
        heap = (uint16_t *) malloc((size_t) n_states * sizeof(uint16_t));
        if (!heap) {
            return;
        }
        out = heap;
    }
    for (i = 0; i < n_words; ++i) {
        const int n = lens[i];
        const uint16_t sig = hqvm_sig13_compile(words_flat + off, n);
        hqvm_sig13_apply_batch(states, n_states, sig, out);
        off += n;
    }
    if (heap) {
        free(heap);
    }
}

int hqvm_wave_grammar_verify(hqvm_wave_grammar_result * receipt) {
    static const uint8_t w2[2]  = { HQVM_W2_BYTE0, HQVM_W2_BYTE1 };
    static const uint8_t w2p[2] = { HQVM_W2P_BYTE0, HQVM_W2P_BYTE1 };
    uint8_t              ru, rv, rback_u, rback_v;
    uint8_t              u_rest, v_rest;
    uint32_t             s_rest, s_w2;
    hqvm_wave_grammar_result local;

    if (!receipt) {
        receipt = &local;
    }
    memset(receipt, 0, sizeof(*receipt));

    hqvm_byte_table_init();
    receipt->byte_table_ok = s_byte_table_ok;

    receipt->w2_sig_ok  = check_w2_signature(w2, 63, 0);
    receipt->w2p_sig_ok = check_w2_signature(w2p, 0, 63);

    hqvm_state24_to_uv6(GENE_MAC_REST, &u_rest, &v_rest);
    hqvm_trace_word_bytes(w2, 2, u_rest, v_rest, &ru, &rv);
    receipt->w2_involution_ok = (ru == 63 && rv == 63);
    hqvm_trace_word_bytes(w2, 2, ru, rv, &rback_u, &rback_v);
    receipt->w2_involution_ok = receipt->w2_involution_ok && (rback_u == u_rest && rback_v == v_rest);

    hqvm_trace_word_bytes(w2p, 2, u_rest, v_rest, &ru, &rv);
    receipt->w2p_involution_ok = (ru == 0 && rv == 0);
    hqvm_trace_word_bytes(w2p, 2, ru, rv, &rback_u, &rback_v);
    receipt->w2p_involution_ok = receipt->w2p_involution_ok && (rback_u == u_rest && rback_v == v_rest);

    s_rest = GENE_MAC_REST;
    s_w2   = hqvm_trace_word_state24(w2, 2, s_rest);
    {
        uint8_t uo, vo;
        hqvm_trace_word_bytes(w2, 2, u_rest, v_rest, &uo, &vo);
        receipt->f_composition_ok = (hqvm_uv6_to_state24(uo, vo) == s_w2);
    }

    receipt->t2_chi_ok   = check_w2_t2(w2);
    receipt->t2_shell_ok = receipt->t2_chi_ok;
    receipt->sparse_k4_ok = check_sparse_k4();

    return (receipt->w2_sig_ok && receipt->w2p_sig_ok && receipt->w2_involution_ok &&
            receipt->w2p_involution_ok && receipt->f_composition_ok && receipt->t2_chi_ok &&
            receipt->t2_shell_ok && receipt->sparse_k4_ok && receipt->byte_table_ok)
               ? 0
               : -1;
}
