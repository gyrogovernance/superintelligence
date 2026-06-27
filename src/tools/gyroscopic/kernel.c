/*
 * Gyroscopic kernel implementation (see kernel.h).
 */

#include "kernel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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
/* Transition-law tables (ported bit-exactly from src/constants.py).  */
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
/* 1. Verified transition law.                                        */
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
/* Psi is a length-4096 array over Omega = HORIZON x q-class.         */
/* Indices: i = h * 64 + q, with h, q in [0, 63].                     */
/*   W2  : swap (h, q) -> (q, h)         [horizon/q-class swap]       */
/*   W2' : (h,q)->(~q,~h) = swap then complement both (6-bit)        */
/*   F   : global complement (h,q)->(~h,~q) = W2 o W2'                */
/* Each is an involution; F = W2 o W2' holds by construction.         */
/* ================================================================== */

static void apply_swap(float psi[GYROSCOPIC_WAVEFUNCTION_SIZE]) {
    int h, q;
    for (h = 0; h < (int) HORIZON_SIZE; ++h) {
        for (q = h + 1; q < (int) HORIZON_SIZE; ++q) {
            const size_t i = (size_t) h * HORIZON_SIZE + (size_t) q;
            const size_t j = (size_t) q * HORIZON_SIZE + (size_t) h;
            const float t = psi[i];
            psi[i] = psi[j];
            psi[j] = t;
        }
    }
}

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

    switch (gate) {
        case GYROSCOPIC_K4_ID:
            return;
        case GYROSCOPIC_K4_W2:
            apply_swap(psi);
            return;
        case GYROSCOPIC_K4_W2P:
            for (i = 0; i < (int) HORIZON_SIZE; ++i) {
                comp[i]  = (uint8_t) ((~(unsigned) i) & CHIRALITY_MASK_6);
                ident[i] = (uint8_t) i;
            }
            /* complement-swap: (h, q) -> (~q, ~h) = swap then complement both */
            apply_swap(psi);
            apply_pairwise(psi, comp, comp);
            (void) ident;
            return;
        case GYROSCOPIC_K4_F:
            for (i = 0; i < (int) HORIZON_SIZE; ++i) {
                comp[i] = (uint8_t) ((~(unsigned) i) & CHIRALITY_MASK_6);
            }
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
static void wht64_float(float data[64]) {
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
