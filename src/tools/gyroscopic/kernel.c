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
