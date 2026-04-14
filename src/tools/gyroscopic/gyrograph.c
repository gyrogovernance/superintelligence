// src/tools/gyroscopic/gyrograph.c
//
// Native batched helpers for GyroGraph.
//
// This file does not redefine kernel physics.
// It implements the existing exact Omega law and local-memory update rules
// in a multicellular batched form.

#include "gyrograph.h"
#include "gyrograph_policy.h"
#include "gyrolabe_wht.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <pthread.h>
#endif

#if defined(_OPENMP)
#  define GYROGRAPH_OMP_MIN_N 2048
#  define GYROGRAPH_PAR_FOR _Pragma("omp parallel for schedule(static) if(n >= GYROGRAPH_OMP_MIN_N)")
#else
#  define GYROGRAPH_PAR_FOR
#endif

static inline uint32_t pack_omega_sig(uint8_t parity, uint8_t tau_u6, uint8_t tau_v6);
static inline void unpack_omega_sig(uint32_t sig, uint8_t* parity, uint8_t* tau_u6, uint8_t* tau_v6);
static inline uint32_t compose_omega_signatures(uint32_t left, uint32_t right);

#define GENE_MIC_S   0xAAu
#define EPSILON_6    0x3Fu
#define LAYER_MASK_12 0x0FFFu

uint16_t MASK12_BY_BYTE[256];
uint8_t  MICRO_BY_BYTE[256];
static uint8_t  EPS_A6_BY_BYTE[256];
static uint8_t  EPS_B6_BY_BYTE[256];
uint32_t OMEGA_SIG_BY_BYTE[256];
uint8_t  FAMILY_BY_BYTE[256];
uint8_t  Q6_BY_BYTE[256];
static uint8_t  SHELL_BY_CHI6[64];

int gyrograph_strict(void) {
    const gyro_policy * policy = gyro_policy_get();
    return (policy != NULL && policy->strict != 0) ? 1 : 0;
}

static inline uint8_t gyrograph_popcnt8(uint8_t x) {
#if defined(_MSC_VER)
    uint32_t v = (uint32_t)x;
    return (uint8_t)__popcnt(v);
#else
    return (uint8_t)__builtin_popcount((unsigned int)x);
#endif
}

static inline uint8_t intron_of_byte(uint8_t b) {
    return (uint8_t)(b ^ GENE_MIC_S);
}

static inline uint16_t mask12_from_micro_ref(uint8_t micro_ref) {
    uint16_t mask12 = 0u;
    for (uint8_t i = 0u; i < 6u; ++i) {
        if ((micro_ref >> i) & 1u) {
            mask12 |= (uint16_t)(0x3u << (2u * i));
        }
    }
    return (uint16_t)(mask12 & LAYER_MASK_12);
}

static inline uint8_t collapse_pairdiag12_to_word6(uint16_t x) {
    uint8_t out = 0u;
    for (uint8_t i = 0u; i < 6u; ++i) {
        uint16_t pair = (uint16_t)((x >> (2u * i)) & 0x3u);
        if (pair == 0x3u) {
            out |= (uint8_t)(1u << i);
        }
    }
    return out;
}

static void gyrograph_init_tables_impl(void) {
    for (uint32_t b = 0u; b < 256u; ++b) {
        uint8_t byte = (uint8_t)b;
        uint8_t intron = intron_of_byte(byte);
        uint8_t micro = (uint8_t)((intron >> 1u) & 0x3Fu);

        MICRO_BY_BYTE[b] = micro;
        MASK12_BY_BYTE[b] = mask12_from_micro_ref(micro);
        {
            uint16_t pair = (uint16_t)(MASK12_BY_BYTE[b] ^
                (((uint8_t)((intron & 1u) ^ ((intron >> 7u) & 1u)))
                    ? LAYER_MASK_12
                    : 0u));
            Q6_BY_BYTE[b] = collapse_pairdiag12_to_word6(pair);
        }
        EPS_A6_BY_BYTE[b] = (uint8_t)((intron & 0x01u) ? EPSILON_6 : 0u);
        EPS_B6_BY_BYTE[b] = (uint8_t)((intron & 0x80u) ? EPSILON_6 : 0u);
        FAMILY_BY_BYTE[b] = (uint8_t)((((intron >> 7u) & 1u) << 1u) | (intron & 1u));
        OMEGA_SIG_BY_BYTE[b] = pack_omega_sig(
            1u,
            EPS_A6_BY_BYTE[b],
            (uint8_t)(MICRO_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b])
        );
    }

    for (uint32_t x = 0u; x < 64u; ++x) {
        SHELL_BY_CHI6[x] = gyrograph_popcnt8((uint8_t)x);
    }
}

#if defined(_WIN32)
static INIT_ONCE g_gyrograph_tables_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK gyrograph_init_tables_once_cb(PINIT_ONCE po, PVOID pv, PVOID * ctx) {
    (void) po;
    (void) pv;
    (void) ctx;
    gyrograph_init_tables_impl();
    return TRUE;
}

static void gyrograph_init_tables(void) {
    PVOID ctx = NULL;
    InitOnceExecuteOnce(
        &g_gyrograph_tables_once,
        gyrograph_init_tables_once_cb,
        NULL,
        &ctx
    );
}
#else
static pthread_once_t g_gyrograph_tables_once = PTHREAD_ONCE_INIT;

static void gyrograph_init_tables(void) {
    pthread_once(&g_gyrograph_tables_once, gyrograph_init_tables_impl);
}
#endif

GYROGRAPH_EXPORT void gyrograph_init(void) {
    gyrograph_init_tables();
    gyrolabe_wht64_verify_self_inverse();
}

static inline uint32_t step_omega12_by_byte_packed(uint32_t omega12, uint8_t b) {
    uint8_t u6 = (uint8_t)((omega12 >> 6u) & 0x3Fu);
    uint8_t v6 = (uint8_t)(omega12 & 0x3Fu);

    uint8_t u_next = (uint8_t)((v6 ^ EPS_A6_BY_BYTE[b]) & 0x3Fu);
    uint8_t v_next = (uint8_t)((u6 ^ MICRO_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b]) & 0x3Fu);

    return (((uint32_t)u_next) << 6u) | ((uint32_t)v_next);
}

static inline uint8_t chi6_from_omega12(uint32_t omega12) {
    return (uint8_t)((((omega12 >> 6u) & 0x3Fu) ^ (omega12 & 0x3Fu)) & 0x3Fu);
}

static inline uint32_t pack_omega_sig(uint8_t parity, uint8_t tau_u6, uint8_t tau_v6) {
    return (((uint32_t)(parity & 1u)) << 12u)
         | (((uint32_t)(tau_u6 & 0x3Fu)) << 6u)
         | ((uint32_t)(tau_v6 & 0x3Fu));
}

static inline void unpack_omega_sig(uint32_t sig, uint8_t* parity, uint8_t* tau_u6, uint8_t* tau_v6) {
    *parity = (uint8_t)((sig >> 12u) & 1u);
    *tau_u6 = (uint8_t)((sig >> 6u) & 0x3Fu);
    *tau_v6 = (uint8_t)(sig & 0x3Fu);
}

static inline uint32_t compose_omega_signatures(uint32_t left, uint32_t right) {
    uint8_t lp, rp, ltu, ltv, rtu, rtv, ru, rv;

    unpack_omega_sig(left, &lp, &ltu, &ltv);
    unpack_omega_sig(right, &rp, &rtu, &rtv);

    if (lp == 0u) {
        ru = rtu;
        rv = rtv;
    } else {
        ru = rtv;
        rv = rtu;
    }

    return pack_omega_sig(
        (uint8_t)(lp ^ rp),
        (uint8_t)(ru ^ ltu),
        (uint8_t)(rv ^ ltv)
    );
}

static inline uint32_t compute_resonance_key(
    uint8_t profile,
    uint32_t omega12,
    const uint8_t* w4,
    uint32_t omega_sig
) {
    uint8_t chi6 = chi6_from_omega12(omega12);
    switch (profile) {
    case 1:
        return (uint32_t)chi6;
    case 2:
        return (uint32_t)gyrograph_popcnt8(chi6);
    case 3:
        if (chi6 == 0u) {
            return 0u;
        }
        if (chi6 == 0x3Fu) {
            return 1u;
        }
        return 2u;
    case 4:
        return omega12 & 0x0FFFu;
    case 5:
        return omega_sig & 0x1FFFu;
    case 6:
        return (uint32_t)(
            (Q6_BY_BYTE[w4[0]] ^ Q6_BY_BYTE[w4[1]] ^ Q6_BY_BYTE[w4[2]] ^ Q6_BY_BYTE[w4[3]])
            & 0x3Fu
        );
    default:
        return 0u;
    }
}

/* Per-cell chi ring, histograms, and shell counts are stored as SoA buffers on the
 * batch APIs below; that matches the logical fields of a per-request gyro_req_state_t. */

static inline void push_state_row(
    uint8_t* chi_ring_row,
    uint8_t* family_ring_row,
    uint8_t* ring_pos,
    uint8_t* valid_len,
    uint16_t* chi_hist_row,
    uint16_t* shell_row,
    uint16_t* family_hist_row,
    uint8_t chi6,
    uint8_t family
) {
    uint8_t pos = *ring_pos;
    uint8_t valid = *valid_len;
    uint8_t shell = SHELL_BY_CHI6[chi6];

    if (valid < 64u) {
        chi_ring_row[pos] = chi6;
        family_ring_row[pos] = family;
        chi_hist_row[chi6] += 1u;
        family_hist_row[family] += 1u;
        shell_row[shell] += 1u;
        *ring_pos = (uint8_t)((pos + 1u) & 63u);
        *valid_len = (uint8_t)(valid + 1u);
        return;
    }

    uint8_t chi_old = chi_ring_row[pos];
    uint8_t family_old = family_ring_row[pos];
    uint8_t shell_old = SHELL_BY_CHI6[chi_old];

    chi_hist_row[chi_old] -= 1u;
    shell_row[shell_old] -= 1u;
    family_hist_row[family_old] -= 1u;

    chi_ring_row[pos] = chi6;
    family_ring_row[pos] = family;
    chi_hist_row[chi6] += 1u;
    family_hist_row[family] += 1u;
    shell_row[shell] += 1u;
    *ring_pos = (uint8_t)((pos + 1u) & 63u);
}

GYROGRAPH_EXPORT void gyrograph_trace_word4_batch_indexed(
    const int64_t* GYRO_RESTRICT cell_ids,
    const int32_t* GYRO_RESTRICT omega12_in,
    const uint8_t* GYRO_RESTRICT words4_in,
    int64_t n,
    int32_t* GYRO_RESTRICT omega_trace4_out,
    uint8_t* GYRO_RESTRICT chi_trace4_out
) {
    if (
        cell_ids == NULL || omega12_in == NULL || words4_in == NULL
        || omega_trace4_out == NULL || chi_trace4_out == NULL || n < 0
    ) {
        return;
    }

    gyrograph_init_tables();

    {
        int64_t i;
        GYROGRAPH_PAR_FOR
        for (i = 0; i < n; ++i) {
        int64_t cid = cell_ids[i];
        uint32_t s = (uint32_t)omega12_in[cid] & 0x0FFFu;
        const uint8_t* w = words4_in + (size_t)(4u * (uint64_t)i);

        for (int k = 0; k < 4; ++k) {
            s = step_omega12_by_byte_packed(s, w[k]);
            omega_trace4_out[(4 * i) + k] = (int32_t)s;
            chi_trace4_out[(4 * i) + k] = chi6_from_omega12(s);
        }
        }
    }
}

GYROGRAPH_EXPORT void gyrograph_apply_trace_word4_batch_indexed(
    const int64_t* GYRO_RESTRICT cell_ids,
    int32_t* GYRO_RESTRICT omega12_io,
    uint64_t* GYRO_RESTRICT step_io,
    uint8_t* GYRO_RESTRICT last_byte_io,
    uint8_t* GYRO_RESTRICT has_closed_word_io,
    uint8_t* GYRO_RESTRICT word4_io,
    uint8_t* GYRO_RESTRICT chi_ring64_io,
    uint8_t* GYRO_RESTRICT chi_ring_pos_io,
    uint8_t* GYRO_RESTRICT chi_valid_len_io,
    uint16_t* GYRO_RESTRICT chi_hist64_io,
    uint16_t* GYRO_RESTRICT shell_hist7_io,
    uint8_t* GYRO_RESTRICT family_ring64_io,
    uint16_t* GYRO_RESTRICT family_hist4_io,
    int32_t* GYRO_RESTRICT omega_sig_io,
    uint16_t* GYRO_RESTRICT parity_O12_io,
    uint16_t* GYRO_RESTRICT parity_E12_io,
    uint8_t* GYRO_RESTRICT parity_bit_io,
    const uint8_t* GYRO_RESTRICT words4_in,
    const int32_t* GYRO_RESTRICT omega_trace4_in,
    const uint8_t* GYRO_RESTRICT chi_trace4_in,
    uint32_t* GYRO_RESTRICT resonance_key_io,
    uint8_t profile,
    int64_t n
) {
    if (
        cell_ids == NULL ||
        omega12_io == NULL || step_io == NULL || last_byte_io == NULL ||
        has_closed_word_io == NULL || word4_io == NULL || chi_ring64_io == NULL ||
        chi_ring_pos_io == NULL || chi_valid_len_io == NULL || chi_hist64_io == NULL ||
        shell_hist7_io == NULL || family_ring64_io == NULL || family_hist4_io == NULL ||
        omega_sig_io == NULL || parity_O12_io == NULL ||
        parity_E12_io == NULL || parity_bit_io == NULL || words4_in == NULL ||
        omega_trace4_in == NULL || chi_trace4_in == NULL || resonance_key_io == NULL
        || n < 0
    ) {
        return;
    }

    gyrograph_init_tables();

    {
        int64_t i;
        GYROGRAPH_PAR_FOR
        for (i = 0; i < n; ++i) {
        int64_t cid = cell_ids[i];
        size_t base_in = (size_t)(4u * (uint64_t)i);
        size_t base_row = (size_t)(4u * (uint64_t)cid);
        const uint8_t* chi_t = chi_trace4_in + base_in;
        const uint8_t* w = words4_in + base_in;
        uint8_t* ring_row = chi_ring64_io + (64u * (uint64_t)cid);
        uint16_t* hist_row = chi_hist64_io + (64u * (uint64_t)cid);
        uint16_t* shell_row = shell_hist7_io + (7u * (uint64_t)cid);
        uint8_t* family_row = family_ring64_io + (64u * (uint64_t)cid);
        uint16_t* family_hist = family_hist4_io + (4u * (uint64_t)cid);
        uint8_t* out_word = word4_io + base_row;
        uint32_t sig = 0u;
        uint16_t O12 = 0u;
        uint16_t E12 = 0u;
        uint8_t ring_pos = chi_ring_pos_io[cid];
        uint8_t valid_len = chi_valid_len_io[cid];

        for (int k = 0; k < 4; ++k) {
            uint8_t b = w[k];
            step_io[cid] += 1u;
            last_byte_io[cid] = b;
            out_word[k] = b;
            push_state_row(
                ring_row,
                family_row,
                &ring_pos,
                &valid_len,
                hist_row,
                shell_row,
                family_hist,
                chi_t[k],
                FAMILY_BY_BYTE[b]
            );

            sig = compose_omega_signatures(OMEGA_SIG_BY_BYTE[b], sig);
            if ((k & 1) == 0) {
                O12 ^= MASK12_BY_BYTE[b];
            } else {
                E12 ^= MASK12_BY_BYTE[b];
            }
        }

        chi_ring_pos_io[cid] = ring_pos;
        chi_valid_len_io[cid] = valid_len;
        omega12_io[cid] = omega_trace4_in[base_in + 3u];
        has_closed_word_io[cid] = 1u;
        omega_sig_io[cid] = (int32_t)sig;
        parity_O12_io[cid] = (uint16_t)(O12 & 0x0FFFu);
        parity_E12_io[cid] = (uint16_t)(E12 & 0x0FFFu);
        resonance_key_io[cid] = compute_resonance_key(
            profile,
            (uint32_t)omega_trace4_in[base_in + 3u],
            out_word,
            sig
        );
        parity_bit_io[cid] = 0u;
        }
    }
}

GYROGRAPH_EXPORT void gyrograph_ingest_word4_batch_indexed(
    const int64_t* GYRO_RESTRICT cell_ids,
    int32_t* GYRO_RESTRICT omega12_io,
    uint64_t* GYRO_RESTRICT step_io,
    uint8_t* GYRO_RESTRICT last_byte_io,
    uint8_t* GYRO_RESTRICT has_closed_word_io,
    uint8_t* GYRO_RESTRICT word4_io,
    uint8_t* GYRO_RESTRICT chi_ring64_io,
    uint8_t* GYRO_RESTRICT chi_ring_pos_io,
    uint8_t* GYRO_RESTRICT chi_valid_len_io,
    uint16_t* GYRO_RESTRICT chi_hist64_io,
    uint16_t* GYRO_RESTRICT shell_hist7_io,
    uint8_t* GYRO_RESTRICT family_ring64_io,
    uint16_t* GYRO_RESTRICT family_hist4_io,
    int32_t* GYRO_RESTRICT omega_sig_io,
    uint16_t* GYRO_RESTRICT parity_O12_io,
    uint16_t* GYRO_RESTRICT parity_E12_io,
    uint8_t* GYRO_RESTRICT parity_bit_io,
    const uint8_t* GYRO_RESTRICT words4_in,
    uint32_t* GYRO_RESTRICT resonance_key_io,
    uint8_t profile,
    int64_t n
) {
    if (
        cell_ids == NULL ||
        omega12_io == NULL || step_io == NULL || last_byte_io == NULL ||
        has_closed_word_io == NULL || word4_io == NULL || chi_ring64_io == NULL ||
        chi_ring_pos_io == NULL || chi_valid_len_io == NULL || chi_hist64_io == NULL ||
        shell_hist7_io == NULL || family_ring64_io == NULL || family_hist4_io == NULL ||
        omega_sig_io == NULL || parity_O12_io == NULL ||
        parity_E12_io == NULL || parity_bit_io == NULL || words4_in == NULL ||
        resonance_key_io == NULL || n < 0
    ) {
        return;
    }

    gyrograph_init_tables();

    {
        int64_t i;
        GYROGRAPH_PAR_FOR
        for (i = 0; i < n; ++i) {
        int64_t cid = cell_ids[i];
        size_t base_in = (size_t)(4u * (uint64_t)i);
        size_t base_row = (size_t)(4u * (uint64_t)cid);
        const uint8_t* w = words4_in + base_in;
        uint8_t* ring_row = chi_ring64_io + (64u * (uint64_t)cid);
        uint16_t* hist_row = chi_hist64_io + (64u * (uint64_t)cid);
        uint16_t* shell_row = shell_hist7_io + (7u * (uint64_t)cid);
        uint8_t* family_row = family_ring64_io + (64u * (uint64_t)cid);
        uint16_t* family_hist = family_hist4_io + (4u * (uint64_t)cid);
        uint8_t* out_word = word4_io + base_row;
        uint8_t ring_pos = chi_ring_pos_io[cid];
        uint8_t valid_len = chi_valid_len_io[cid];
        uint32_t s = (uint32_t)omega12_io[cid] & 0x0FFFu;
        uint32_t sig = 0u;
        uint16_t O12 = 0u;
        uint16_t E12 = 0u;

        for (int k = 0; k < 4; ++k) {
            uint8_t b = w[k];
            s = step_omega12_by_byte_packed(s, b);
            step_io[cid] += 1u;
            last_byte_io[cid] = b;
            out_word[k] = b;

            push_state_row(
                ring_row,
                family_row,
                &ring_pos,
                &valid_len,
                hist_row,
                shell_row,
                family_hist,
                chi6_from_omega12(s),
                FAMILY_BY_BYTE[b]
            );

            sig = compose_omega_signatures(OMEGA_SIG_BY_BYTE[b], sig);
            if ((k & 1) == 0) {
                O12 ^= MASK12_BY_BYTE[b];
            } else {
                E12 ^= MASK12_BY_BYTE[b];
            }
        }

        chi_ring_pos_io[cid] = ring_pos;
        chi_valid_len_io[cid] = valid_len;
        omega12_io[cid] = (int32_t)s;
        has_closed_word_io[cid] = 1u;
        omega_sig_io[cid] = (int32_t)sig;
        parity_O12_io[cid] = (uint16_t)(O12 & 0x0FFFu);
        parity_E12_io[cid] = (uint16_t)(E12 & 0x0FFFu);
        parity_bit_io[cid] = 0u;
        resonance_key_io[cid] = compute_resonance_key(
            profile,
            s,
            out_word,
            sig
        );
        }
    }
}

GYROGRAPH_EXPORT double gyrograph_compute_m2_empirical(
    const uint16_t* GYRO_RESTRICT chi_hist64,
    uint64_t total
) {
    if (chi_hist64 == NULL || total == 0u) {
        return 64.0;
    }

    uint64_t sumsq = 0u;
    for (int i = 0; i < 64; ++i) {
        uint64_t h = (uint64_t)chi_hist64[i];
        sumsq += h * h;
    }

    if (sumsq == 0u) {
        return 64.0;
    }

    double t = (double)total;
    return (t * t) / (double)sumsq;
}

GYROGRAPH_EXPORT double gyrograph_compute_m2_equilibrium(
    const uint16_t* GYRO_RESTRICT shell_hist7,
    uint64_t total
) {
    if (shell_hist7 == NULL || total == 0u) {
        return 4096.0;
    }

    uint64_t sum_wx = 0u;
    for (int w = 0; w < 7; ++w) {
        sum_wx += (uint64_t)w * (uint64_t)shell_hist7[w];
    }

    double mean_N = (double)sum_wx / (double)total;
    double rho = mean_N / 6.0;
    double eta = 1.0 - 2.0 * rho;
    double eta_sq = eta * eta;
    double denom = 1.0 + eta_sq;
    double denom6 = denom * denom * denom * denom * denom * denom;
    return 4096.0 / denom6;
}