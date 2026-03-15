// src/tools/gyrograph/gyrograph.c
//
// Native batched helpers for GyroGraph.
//
// This file does not redefine kernel physics.
// It implements the existing exact Ω law and local-memory update rules
// in a multicellular batched form.
//
// Exports:
//   - gyrograph_init
//   - gyrograph_trace_word4_batch
//   - gyrograph_apply_trace_word4_batch
//   - gyrograph_ingest_word4_batch

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
  #define GYROGRAPH_EXPORT __declspec(dllexport)
#else
  #define GYROGRAPH_EXPORT __attribute__((visibility("default")))
#endif

#define GENE_MIC_S   0xAAu
#define EPSILON_6    0x3Fu
#define LAYER_MASK_12 0x0FFFu

static uint16_t MASK12_BY_BYTE[256];
static uint8_t  MICRO_BY_BYTE[256];
static uint8_t  EPS_A6_BY_BYTE[256];
static uint8_t  EPS_B6_BY_BYTE[256];
static int      TABLES_READY = 0;

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

static void gyrograph_init_tables(void) {
    if (TABLES_READY) {
        return;
    }

    for (uint32_t b = 0u; b < 256u; ++b) {
        uint8_t byte = (uint8_t)b;
        uint8_t intron = intron_of_byte(byte);
        uint8_t micro = (uint8_t)((intron >> 1u) & 0x3Fu);

        MICRO_BY_BYTE[b] = micro;
        MASK12_BY_BYTE[b] = mask12_from_micro_ref(micro);
        EPS_A6_BY_BYTE[b] = (uint8_t)((intron & 0x01u) ? EPSILON_6 : 0u);
        EPS_B6_BY_BYTE[b] = (uint8_t)((intron & 0x80u) ? EPSILON_6 : 0u);
    }

    TABLES_READY = 1;
}

GYROGRAPH_EXPORT void gyrograph_init(void) {
    gyrograph_init_tables();
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

static inline uint32_t omega_byte_signature(uint8_t b) {
    return pack_omega_sig(
        1u,
        EPS_A6_BY_BYTE[b],
        (uint8_t)(MICRO_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b])
    );
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

static inline void push_chi_row(
    uint8_t* ring_row,
    uint8_t* ring_pos,
    uint8_t* valid_len,
    uint16_t* hist_row,
    uint16_t* shell_row,
    uint8_t chi6
) {
    uint8_t pos = *ring_pos;
    uint8_t valid = *valid_len;
    uint8_t shell = gyrograph_popcnt8(chi6);

    if (valid < 64u) {
        ring_row[pos] = chi6;
        hist_row[chi6] += 1u;
        shell_row[shell] += 1u;
        *ring_pos = (uint8_t)((pos + 1u) & 63u);
        *valid_len = (uint8_t)(valid + 1u);
        return;
    }

    uint8_t chi_old = ring_row[pos];
    uint8_t shell_old = gyrograph_popcnt8(chi_old);

    hist_row[chi_old] -= 1u;
    shell_row[shell_old] -= 1u;

    ring_row[pos] = chi6;
    hist_row[chi6] += 1u;
    shell_row[shell] += 1u;
    *ring_pos = (uint8_t)((pos + 1u) & 63u);
}

GYROGRAPH_EXPORT void gyrograph_trace_word4_batch(
    const int32_t* omega12_in,
    const uint8_t* words4_in,
    int64_t n,
    int32_t* omega_trace4_out,
    uint8_t* chi_trace4_out
) {
    if (omega12_in == NULL || words4_in == NULL || omega_trace4_out == NULL || chi_trace4_out == NULL || n < 0) {
        return;
    }

    gyrograph_init_tables();

    for (int64_t i = 0; i < n; ++i) {
        uint32_t s = (uint32_t)omega12_in[i] & 0x0FFFu;
        const uint8_t* w = words4_in + (4 * i);

        for (int k = 0; k < 4; ++k) {
            s = step_omega12_by_byte_packed(s, w[k]);
            omega_trace4_out[(4 * i) + k] = (int32_t)s;
            chi_trace4_out[(4 * i) + k] = chi6_from_omega12(s);
        }
    }
}

GYROGRAPH_EXPORT void gyrograph_apply_trace_word4_batch(
    int32_t* omega12_io,
    uint64_t* step_io,
    uint8_t* last_byte_io,
    uint8_t* has_closed_word_io,
    uint8_t* word4_io,
    uint8_t* chi_ring64_io,
    uint8_t* chi_ring_pos_io,
    uint8_t* chi_valid_len_io,
    uint16_t* chi_hist64_io,
    uint16_t* shell_hist7_io,
    int32_t* omega_sig_io,
    uint16_t* parity_O12_io,
    uint16_t* parity_E12_io,
    uint8_t* parity_bit_io,
    const uint8_t* words4_in,
    const int32_t* omega_trace4_in,
    const uint8_t* chi_trace4_in,
    int64_t n
) {
    if (
        omega12_io == NULL || step_io == NULL || last_byte_io == NULL ||
        has_closed_word_io == NULL || word4_io == NULL || chi_ring64_io == NULL ||
        chi_ring_pos_io == NULL || chi_valid_len_io == NULL || chi_hist64_io == NULL ||
        shell_hist7_io == NULL || omega_sig_io == NULL || parity_O12_io == NULL ||
        parity_E12_io == NULL || parity_bit_io == NULL || words4_in == NULL ||
        omega_trace4_in == NULL || chi_trace4_in == NULL || n < 0
    ) {
        return;
    }

    gyrograph_init_tables();

    for (int64_t i = 0; i < n; ++i) {
        const uint8_t* w = words4_in + (4 * i);
        const int32_t* omega_t = omega_trace4_in + (4 * i);
        const uint8_t* chi_t = chi_trace4_in + (4 * i);

        uint8_t* ring_row = chi_ring64_io + (64 * i);
        uint16_t* hist_row = chi_hist64_io + (64 * i);
        uint16_t* shell_row = shell_hist7_io + (7 * i);

        uint32_t sig = 0u;
        uint16_t O12 = 0u;
        uint16_t E12 = 0u;

        for (int k = 0; k < 4; ++k) {
            uint8_t b = w[k];
            step_io[i] += 1u;
            last_byte_io[i] = b;
            word4_io[(4 * i) + k] = b;

            push_chi_row(
                ring_row,
                &chi_ring_pos_io[i],
                &chi_valid_len_io[i],
                hist_row,
                shell_row,
                chi_t[k]
            );

            sig = compose_omega_signatures(omega_byte_signature(b), sig);
            if ((k & 1) == 0) {
                O12 ^= MASK12_BY_BYTE[b];
            } else {
                E12 ^= MASK12_BY_BYTE[b];
            }
        }

        omega12_io[i] = omega_t[3];
        has_closed_word_io[i] = 1u;
        omega_sig_io[i] = (int32_t)sig;
        parity_O12_io[i] = (uint16_t)(O12 & 0x0FFFu);
        parity_E12_io[i] = (uint16_t)(E12 & 0x0FFFu);
        parity_bit_io[i] = 0u;
    }
}

GYROGRAPH_EXPORT void gyrograph_ingest_word4_batch(
    int32_t* omega12_io,
    uint64_t* step_io,
    uint8_t* last_byte_io,
    uint8_t* has_closed_word_io,
    uint8_t* word4_io,
    uint8_t* chi_ring64_io,
    uint8_t* chi_ring_pos_io,
    uint8_t* chi_valid_len_io,
    uint16_t* chi_hist64_io,
    uint16_t* shell_hist7_io,
    int32_t* omega_sig_io,
    uint16_t* parity_O12_io,
    uint16_t* parity_E12_io,
    uint8_t* parity_bit_io,
    const uint8_t* words4_in,
    int64_t n
) {
    if (
        omega12_io == NULL || step_io == NULL || last_byte_io == NULL ||
        has_closed_word_io == NULL || word4_io == NULL || chi_ring64_io == NULL ||
        chi_ring_pos_io == NULL || chi_valid_len_io == NULL || chi_hist64_io == NULL ||
        shell_hist7_io == NULL || omega_sig_io == NULL || parity_O12_io == NULL ||
        parity_E12_io == NULL || parity_bit_io == NULL || words4_in == NULL || n < 0
    ) {
        return;
    }

    gyrograph_init_tables();

    for (int64_t i = 0; i < n; ++i) {
        const uint8_t* w = words4_in + (4 * i);
        uint8_t* ring_row = chi_ring64_io + (64 * i);
        uint16_t* hist_row = chi_hist64_io + (64 * i);
        uint16_t* shell_row = shell_hist7_io + (7 * i);

        uint32_t s = (uint32_t)omega12_io[i] & 0x0FFFu;
        uint32_t sig = 0u;
        uint16_t O12 = 0u;
        uint16_t E12 = 0u;

        for (int k = 0; k < 4; ++k) {
            uint8_t b = w[k];

            s = step_omega12_by_byte_packed(s, b);
            step_io[i] += 1u;
            last_byte_io[i] = b;
            word4_io[(4 * i) + k] = b;

            push_chi_row(
                ring_row,
                &chi_ring_pos_io[i],
                &chi_valid_len_io[i],
                hist_row,
                shell_row,
                chi6_from_omega12(s)
            );

            sig = compose_omega_signatures(omega_byte_signature(b), sig);
            if ((k & 1) == 0) {
                O12 ^= MASK12_BY_BYTE[b];
            } else {
                E12 ^= MASK12_BY_BYTE[b];
            }
        }

        omega12_io[i] = (int32_t)s;
        has_closed_word_io[i] = 1u;
        omega_sig_io[i] = (int32_t)sig;
        parity_O12_io[i] = (uint16_t)(O12 & 0x0FFFu);
        parity_E12_io[i] = (uint16_t)(E12 & 0x0FFFu);
        parity_bit_io[i] = 0u;
    }
}