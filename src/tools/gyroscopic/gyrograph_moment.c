#include "gyrograph.h"
#include "gyrograph_types.h"
#include "gyrolabe_transforms.h"
#include <stddef.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
static inline uint32_t gyro_popcnt32(uint32_t x) {
    return (uint32_t)__popcnt(x);
}
#else
static inline uint32_t gyro_popcnt32(uint32_t x) {
    return (uint32_t)__builtin_popcount((int)x);
}
#endif

extern uint8_t FAMILY_BY_BYTE[256];
extern uint8_t MICRO_BY_BYTE[256];
extern uint8_t Q6_BY_BYTE[256];
extern uint16_t MASK12_BY_BYTE[256];
static uint32_t pack_omega_sig_local(uint8_t parity, uint8_t tau_u6, uint8_t tau_v6) {
    return (((uint32_t)(parity & 1u)) << 12u)
         | (((uint32_t)(tau_u6 & 0x3Fu)) << 6u)
         | ((uint32_t)(tau_v6 & 0x3Fu));
}

static int pairdiag12_to_word6(uint16_t x, uint8_t *out6) {
    uint8_t o = 0u;
    int i;
    for (i = 0; i < 6; ++i) {
        uint16_t pair = (uint16_t)((x >> (2 * i)) & 3u);
        if (pair == 3u) {
            o |= (uint8_t)(1u << i);
        } else if (pair != 0u) {
            return -1;
        }
    }
    *out6 = (uint8_t)(o & 0x3Fu);
    return 0;
}

static int omega_sig_from_word_sig(const gyrograph_word_signature *ws, uint32_t *out) {
    uint8_t u6 = 0u;
    uint8_t v6 = 0u;
    if (pairdiag12_to_word6(ws->tau_a12, &u6) != 0) {
        return -1;
    }
    if (pairdiag12_to_word6(ws->tau_b12, &v6) != 0) {
        return -1;
    }
    *out = pack_omega_sig_local((uint8_t)(ws->parity & 1u), u6, v6);
    return 0;
}

static uint32_t step_state24_by_byte(uint32_t state24, uint8_t b) {
    uint8_t intron = (uint8_t)((unsigned int)b ^ 0xAAu);
    uint16_t m12 = MASK12_BY_BYTE[b];
    uint32_t a12 = (state24 >> 12) & 0xFFFu;
    uint32_t b12 = state24 & 0xFFFu;
    uint32_t a_mut = (a12 ^ (uint32_t)m12) & 0xFFFu;
    uint32_t invert_a = (intron & 1u) ? 0xFFFu : 0u;
    uint32_t invert_b = (intron & 0x80u) ? 0xFFFu : 0u;
    uint32_t a_next = (b12 ^ invert_a) & 0xFFFu;
    uint32_t b_next = (a_mut ^ invert_b) & 0xFFFu;
    return (a_next << 12) | b_next;
}

static uint32_t inverse_step_state24_by_byte(uint32_t state24, uint8_t b) {
    uint8_t intron = (uint8_t)((unsigned int)b ^ 0xAAu);
    uint16_t m12 = MASK12_BY_BYTE[b];
    uint32_t a_next = (state24 >> 12) & 0xFFFu;
    uint32_t b_next = state24 & 0xFFFu;
    uint32_t invert_a = (intron & 1u) ? 0xFFFu : 0u;
    uint32_t invert_b = (intron & 0x80u) ? 0xFFFu : 0u;
    uint32_t b_pred = (a_next ^ invert_a) & 0xFFFu;
    uint32_t a_pred = ((b_next ^ invert_b) ^ (uint32_t)m12) & 0xFFFu;
    return (a_pred << 12) | b_pred;
}

GYROGRAPH_EXPORT uint32_t gyrograph_step_state24_by_byte(uint32_t state24, uint8_t b) {
    gyrograph_init();
    return step_state24_by_byte(state24 & 0xFFFFFFu, b) & 0xFFFFFFu;
}

GYROGRAPH_EXPORT uint32_t gyrograph_inverse_step_state24_by_byte(uint32_t state24, uint8_t b) {
    gyrograph_init();
    return inverse_step_state24_by_byte(state24 & 0xFFFFFFu, b) & 0xFFFFFFu;
}

static inline uint8_t popcnt6(uint8_t x) {
    return (uint8_t)gyro_popcnt32((uint32_t)(x & 0x3F));
}

static inline uint32_t popcnt12(uint16_t x) {
    return gyro_popcnt32((uint32_t)(x & 0xFFF));
}

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
) {
    if (!omega12_io ||!step_io ||!last_byte_io ||!out) return -1;
    if (cell_id < 0) return -1;

    memset(out, 0, sizeof(*out));
    out->step = step_io[cell_id];
    out->state24 = ((uint32_t)(omega12_io[cell_id] & 0xFFF) << 12) |
                   ((uint32_t)(omega12_io[cell_id] & 0xFFF));
    out->last_byte = last_byte_io[cell_id];
    out->parity_O12 = parity_O12_io? parity_O12_io[cell_id] : 0;
    out->parity_E12 = parity_E12_io? parity_E12_io[cell_id] : 0;
    out->parity_bit = parity_bit_io? parity_bit_io[cell_id] : 0;
    out->omega_sig = omega_sig_io? (uint32_t)omega_sig_io[cell_id] : 0;
    out->q_transport6 = Q6_BY_BYTE[out->last_byte];
    return 0;
}

GYROGRAPH_EXPORT int gyrograph_word_signature_from_bytes(
    const uint8_t *bytes,
    int64_t n,
    gyrograph_word_signature *out
) {
    if (!bytes ||!out || n < 0) return -1;

    uint32_t s = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint8_t b = bytes[i];
        uint8_t intron = b ^ 0xAA;
        uint16_t mask12 = MASK12_BY_BYTE[b];
        uint32_t a12 = (s >> 12) & 0xFFF;
        uint32_t b12 = s & 0xFFF;
        uint32_t a_mut = a12 ^ mask12;
        uint32_t invert_a = (intron & 0x01)? 0xFFF : 0;
        uint32_t invert_b = (intron & 0x80)? 0xFFF : 0;
        uint32_t a_next = (b12 ^ invert_a) & 0xFFF;
        uint32_t b_next = (a_mut ^ invert_b) & 0xFFF;
        s = (a_next << 12) | b_next;
    }

    out->parity = n & 1;
    out->tau_a12 = (s >> 12) & 0xFFF;
    out->tau_b12 = s & 0xFFF;
    return 0;
}

GYROGRAPH_EXPORT int gyrograph_compose_signatures(
    const gyrograph_word_signature *left,
    const gyrograph_word_signature *right,
    gyrograph_word_signature *out
) {
    if (!left ||!right ||!out) return -1;

    uint16_t ra = right->tau_a12;
    uint16_t rb = right->tau_b12;
    if (left->parity) {
        uint16_t tmp = ra; ra = rb; rb = tmp;
    }

    out->parity = left->parity ^ right->parity;
    out->tau_a12 = (ra ^ left->tau_a12) & 0xFFF;
    out->tau_b12 = (rb ^ left->tau_b12) & 0xFFF;
    return 0;
}

GYROGRAPH_EXPORT int gyrograph_apply_signature(
    uint32_t state24,
    const gyrograph_word_signature *sig,
    uint32_t *out_state24
) {
    if (!sig ||!out_state24) return -1;

    uint32_t a12 = (state24 >> 12) & 0xFFF;
    uint32_t b12 = state24 & 0xFFF;

    if (sig->parity == 0) {
        *out_state24 = (((a12 ^ sig->tau_a12) & 0xFFF) << 12) |
                       ((b12 ^ sig->tau_b12) & 0xFFF);
    } else {
        *out_state24 = (((b12 ^ sig->tau_a12) & 0xFFF) << 12) |
                       ((a12 ^ sig->tau_b12) & 0xFFF);
    }
    return 0;
}

GYROGRAPH_EXPORT int gyrograph_compute_constitutional(
    uint32_t state24,
    gyrograph_constitutional *out
) {
    if (!out) return -1;

    uint32_t a12 = (state24 >> 12) & 0xFFF;
    uint32_t b12 = state24 & 0xFFF;
    const uint32_t GENE_MAC_REST = 0xAAA555;

    out->rest_distance = gyro_popcnt32(state24 ^ GENE_MAC_REST);
    out->horizon_distance = popcnt12(a12 ^ (b12 ^ 0xFFF));
    out->ab_distance = popcnt12(a12 ^ b12);
    out->on_complement_horizon = (a12 == (b12 ^ 0xFFF))? 1 : 0;
    out->on_equality_horizon = (a12 == b12)? 1 : 0;
    out->a_density = (float)gyro_popcnt32(a12) / 12.0f;
    out->b_density = (float)gyro_popcnt32(b12) / 12.0f;
    out->complementarity_sum = out->horizon_distance + out->ab_distance;
    return 0;
}

static int emit_slcp_fill_record(
    int64_t cell_id,
    const int32_t *omega12_io,
    const uint64_t *step_io,
    const uint8_t *last_byte_io,
    const uint8_t *word4_io,
    const uint16_t *chi_hist_row,
    const uint16_t *shell_hist7_io,
    const uint8_t *family_hist4_io,
    const int32_t *omega_sig_io,
    const uint16_t *parity_O12_io,
    const uint16_t *parity_E12_io,
    const uint8_t *parity_bit_io,
    const uint32_t *resonance_key_io,
    const int32_t *chi_fwht64,
    gyrograph_slcp_record *out
) {
    uint32_t total;
    int i;
    int ci;
    uint8_t u6;
    uint8_t v6;
    uint32_t a12;
    uint32_t b12;

    (void)word4_io;

    if (!out || cell_id < 0) {
        return -1;
    }
    if (
        !omega12_io || !step_io || !last_byte_io || !chi_hist_row || !chi_fwht64
        || !omega_sig_io || !parity_O12_io || !parity_E12_io || !parity_bit_io
        || !resonance_key_io
    ) {
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->cell_id = cell_id;
    out->step = step_io[cell_id];
    out->omega12 = omega12_io[cell_id];
    out->state24 = ((uint32_t)(out->omega12 & 0xFFF) << 12) | (out->omega12 & 0xFFF);
    out->last_byte = last_byte_io[cell_id];
    out->family = FAMILY_BY_BYTE[out->last_byte];
    out->micro_ref = MICRO_BY_BYTE[out->last_byte];
    out->q6 = Q6_BY_BYTE[out->last_byte];

    u6 = (uint8_t)((out->omega12 >> 6) & 0x3F);
    v6 = (uint8_t)(out->omega12 & 0x3F);
    out->chi6 = (uint8_t)(u6 ^ v6);
    out->shell = popcnt6(out->chi6);

    a12 = (out->state24 >> 12) & 0xFFF;
    b12 = out->state24 & 0xFFF;
    out->horizon_distance = popcnt12((uint16_t)(a12 ^ (b12 ^ 0xFFF)));
    out->ab_distance = popcnt12((uint16_t)(a12 ^ b12));

    out->omega_sig = omega_sig_io[cell_id];
    out->parity_O12 = parity_O12_io[cell_id];
    out->parity_E12 = parity_E12_io[cell_id];
    out->parity_bit = parity_bit_io[cell_id];
    out->resonance_key = resonance_key_io[cell_id];

    total = 0;
    for (i = 0; i < 64; ++i) {
        total += chi_hist_row[i];
    }
    out->current_resonance = total > 0u ? chi_hist_row[out->chi6] : 0;

    for (ci = 0; ci < 64; ++ci) {
        out->spectral64[ci] = (float)chi_fwht64[ci] / 64.0f;
    }

    if (family_hist4_io) {
        int32_t fh32[4];
        float fh[4];
        const uint8_t *pf = family_hist4_io + (size_t)cell_id * 4u;
        int gi;
        for (gi = 0; gi < 4; ++gi) {
            fh32[gi] = (int32_t)pf[gi];
        }
        for (gi = 0; gi < 4; ++gi) {
            fh[gi] = (float)fh32[gi];
        }
        gyrolabe_k4char4_float(fh, out->gauge_spectral);
    }

    if (shell_hist7_io) {
        const uint16_t *sh = shell_hist7_io + (size_t)cell_id * 7u;
        float shf[7];
        int si;
        for (si = 0; si < 7; ++si) {
            shf[si] = (float)sh[si];
        }
        gyrolabe_krawtchouk7_float(shf, out->shell_spectral);
    }

    return 0;
}

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
) {
    int64_t base;
    if (!cell_ids || !outs || n_cells <= 0) {
        return -1;
    }
    if (
        !omega12_io || !step_io || !last_byte_io || !chi_hist64_io
        || !omega_sig_io || !parity_O12_io || !parity_E12_io || !parity_bit_io
        || !resonance_key_io
    ) {
        return -1;
    }

    for (base = 0; base < n_cells; base += 8) {
        int32_t chi_buf[8][64];
        int32_t *chi_ptrs[8];
        int chunk;
        int i;
        int ci;
        int64_t cid;

        chunk = (int)(n_cells - base);
        if (chunk > 8) {
            chunk = 8;
        }

        for (i = 0; i < chunk; ++i) {
            const uint16_t *row;
            cid = cell_ids[base + i];
            if (cid < 0) {
                return -1;
            }
            row = chi_hist64_io + (size_t)cid * 64u;
            for (ci = 0; ci < 64; ++ci) {
                chi_buf[i][ci] = (int32_t)row[ci];
            }
            chi_ptrs[i] = chi_buf[i];
        }
        gyrolabe_wht64_batch_int32(chi_ptrs, chunk);

        for (i = 0; i < chunk; ++i) {
            const uint16_t *row;
            cid = cell_ids[base + i];
            row = chi_hist64_io + (size_t)cid * 64u;
            if (
                emit_slcp_fill_record(
                    cid,
                    omega12_io,
                    step_io,
                    last_byte_io,
                    word4_io,
                    row,
                    shell_hist7_io,
                    family_hist4_io,
                    omega_sig_io,
                    parity_O12_io,
                    parity_E12_io,
                    parity_bit_io,
                    resonance_key_io,
                    chi_buf[i],
                    outs + (size_t)(base + (int64_t)i)
                ) != 0
            ) {
                return -1;
            }
        }
    }
    return 0;
}

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
) {
    return gyrograph_emit_slcp_batch(
        1,
        &cell_id,
        omega12_io,
        step_io,
        last_byte_io,
        word4_io,
        chi_hist64_io,
        shell_hist7_io,
        family_hist4_io,
        omega_sig_io,
        parity_O12_io,
        parity_E12_io,
        parity_bit_io,
        resonance_key_io,
        out
    );
}

GYROGRAPH_EXPORT int gyrograph_moment_from_ledger(
    const uint8_t *ledger,
    int64_t len,
    gyrograph_moment *out
) {
    int64_t i;
    uint32_t state;
    uint8_t q_acc;
    uint16_t O12;
    uint16_t E12;
    int idx;
    gyrograph_word_signature wsig;

    if (!out) {
        return -1;
    }
    if (len < 0) {
        return -1;
    }
    if (len > 0 && !ledger) {
        return -1;
    }

    gyrograph_init();

    memset(out, 0, sizeof(*out));
    state = 0xAAA555u;
    q_acc = 0u;
    O12 = 0u;
    E12 = 0u;
    idx = 0;

    for (i = 0; i < len; ++i) {
        uint8_t b = ledger[i];
        state = step_state24_by_byte(state, b);
        q_acc = (uint8_t)(q_acc ^ Q6_BY_BYTE[b]);
        {
            uint16_t m = MASK12_BY_BYTE[b];
            if ((idx & 1) == 0) {
                O12 = (uint16_t)(O12 ^ m);
            } else {
                E12 = (uint16_t)(E12 ^ m);
            }
            idx++;
        }
    }

    out->step = (uint64_t)len;
    out->state24 = state & 0xFFFFFFu;
    if (len > 0) {
        out->last_byte = ledger[len - 1];
    } else {
        out->last_byte = 0xAAu;
    }
    out->ledger_len = (uint8_t)(len > 255 ? 255 : len);
    if (out->ledger_len > 0 && ledger) {
        memcpy(out->ledger, ledger, (size_t)out->ledger_len);
    }
    out->parity_O12 = (uint16_t)(O12 & 0xFFFu);
    out->parity_E12 = (uint16_t)(E12 & 0xFFFu);
    out->parity_bit = (uint8_t)(idx & 1);
    out->omega_sig = pack_omega_sig_local(0u, 0u, 0u);
    if (gyrograph_word_signature_from_bytes(ledger, len, &wsig) == 0) {
        uint32_t os = 0u;
        if (omega_sig_from_word_sig(&wsig, &os) == 0) {
            out->omega_sig = os;
        }
    }
    out->q_transport6 = (uint8_t)(q_acc & 0x3Fu);
    return 0;
}

GYROGRAPH_EXPORT int gyrograph_verify_moment(
    const gyrograph_moment *m,
    const uint8_t *ledger,
    int64_t len
) {
    gyrograph_moment cmp;

    if (!m) {
        return -1;
    }
    if (len > 0 && !ledger) {
        return -1;
    }
    if (gyrograph_moment_from_ledger(ledger, len, &cmp) != 0) {
        return -1;
    }
    if (cmp.state24 != m->state24 || cmp.step != m->step) {
        return 0;
    }
    if (cmp.last_byte != m->last_byte) {
        return 0;
    }
    if (cmp.parity_O12 != m->parity_O12 || cmp.parity_E12 != m->parity_E12) {
        return 0;
    }
    if (cmp.parity_bit != m->parity_bit) {
        return 0;
    }
    if (cmp.omega_sig != m->omega_sig || cmp.q_transport6 != m->q_transport6) {
        return 0;
    }
    return 1;
}

GYROGRAPH_EXPORT int gyrograph_compare_ledgers(
    const uint8_t *a,
    int64_t alen,
    const uint8_t *b,
    int64_t blen,
    int64_t *common_prefix_out
) {
    int64_t n;
    int64_t i;

    if (common_prefix_out) {
        *common_prefix_out = 0;
    }
    if (!a || !b || alen < 0 || blen < 0) {
        return -1;
    }
    n = alen < blen ? alen : blen;
    for (i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            break;
        }
    }
    if (common_prefix_out) {
        *common_prefix_out = i;
    }
    if (i == alen && i == blen) {
        return 0;
    }
    if (i == alen) {
        return -1;
    }
    return 1;
}