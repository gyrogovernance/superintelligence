// csrc/gyrolabe.c
//
// Plain-C ALU for GyroLabe.
// Cross-platform, ctypes-friendly, CPU-first.
//
// Exports:
//   - gyro_signature_scan
//   - gyro_chirality_distance
//   - gyro_chirality_distance_adjacent
//   - gyro_extract_scan
//   - gyro_wht64_float
//   - gyro_qmap_extract
//
// Notes:
//   - All integer kernel math is exact.
//   - WHT is normalized by 1/8, so it is orthonormal on length-64 vectors.
//   - The signature packing is:
//       bits 24     : parity (0/1)
//       bits 23..12 : tau_a12
//       bits 11..0  : tau_b12

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
  #define GYRO_EXPORT __declspec(dllexport)
#else
  #define GYRO_EXPORT __attribute__((visibility("default")))
#endif

#if defined(_MSC_VER)
  #include <intrin.h>
  static __forceinline uint32_t gyro_popcnt32(uint32_t x) { return (uint32_t)__popcnt(x); }
  #define POPCNT64(x) ((int64_t)__popcnt64((uint64_t)(x)))
#else
  static inline uint32_t gyro_popcnt32(uint32_t x) { return (uint32_t)__builtin_popcount(x); }
  #define POPCNT64(x) ((int64_t)__builtin_popcountll((uint64_t)(x)))
#endif

#include <math.h>
#include <stdlib.h>

#define GENE_MIC_S      0xAAu
#define LAYER_MASK_12   0x0FFFu
#define GENE_MAC_A12    0x0AAAu
#define GENE_MAC_B12    0x0555u
#define EPSILON_6       0x3Fu

static uint16_t MASK12_BY_BYTE[256];
static uint8_t  FAMILY_BY_BYTE[256];
static uint8_t  MICRO_BY_BYTE[256];
static uint8_t  Q6_BY_BYTE[256];
static uint16_t INVERT_A_BY_BYTE[256];
static uint16_t INVERT_B_BY_BYTE[256];
static uint8_t  EPS_A6_BY_BYTE[256];
static uint8_t  EPS_B6_BY_BYTE[256];
static int      TABLES_READY = 0;

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

static void gyro_init_tables(void) {
    if (TABLES_READY) {
        return;
    }

    for (uint32_t b = 0u; b < 256u; ++b) {
        uint8_t byte = (uint8_t)b;
        uint8_t intron = intron_of_byte(byte);
        uint8_t family = (uint8_t)((((intron >> 7u) & 1u) << 1u) | (intron & 1u));
        uint8_t micro = (uint8_t)((intron >> 1u) & 0x3Fu);
        uint8_t l0_parity = (uint8_t)((intron & 1u) ^ ((intron >> 7u) & 1u));
        uint16_t mask12 = mask12_from_micro_ref(micro);
        uint16_t invert_a = (uint16_t)((intron & 0x01u) ? LAYER_MASK_12 : 0u);
        uint16_t invert_b = (uint16_t)((intron & 0x80u) ? LAYER_MASK_12 : 0u);
        uint16_t q12 = (uint16_t)(mask12 ^ (l0_parity ? LAYER_MASK_12 : 0u));
        uint8_t q6 = collapse_pairdiag12_to_word6(q12);

        FAMILY_BY_BYTE[b] = family;
        MICRO_BY_BYTE[b] = micro;
        MASK12_BY_BYTE[b] = mask12;
        INVERT_A_BY_BYTE[b] = invert_a;
        INVERT_B_BY_BYTE[b] = invert_b;
        Q6_BY_BYTE[b] = q6;
        EPS_A6_BY_BYTE[b] = (uint8_t)((intron & 0x01u) ? EPSILON_6 : 0u);
        EPS_B6_BY_BYTE[b] = (uint8_t)((intron & 0x80u) ? EPSILON_6 : 0u);
    }

    TABLES_READY = 1;
}

GYRO_EXPORT void gyro_init(void) {
    gyro_init_tables();
}

static inline uint32_t pack_signature(uint8_t parity, uint16_t tau_a12, uint16_t tau_b12) {
    uint32_t out = 0u;
    out |= ((uint32_t)(parity & 1u) << 24u);
    out |= ((uint32_t)(tau_a12 & LAYER_MASK_12) << 12u);
    out |= ((uint32_t)(tau_b12 & LAYER_MASK_12));
    return out;
}

static inline void unpack_signature(uint32_t sig, uint8_t* parity, uint16_t* tau_a12, uint16_t* tau_b12) {
    *parity = (uint8_t)((sig >> 24u) & 1u);
    *tau_a12 = (uint16_t)((sig >> 12u) & LAYER_MASK_12);
    *tau_b12 = (uint16_t)(sig & LAYER_MASK_12);
}

static inline uint32_t byte_signature(uint8_t b) {
    // Single-byte signature:
    // parity = 1 (swap)
    // tau_a = invert_a
    // tau_b = mask12 ^ invert_b
    uint16_t tau_a = INVERT_A_BY_BYTE[b];
    uint16_t tau_b = (uint16_t)(MASK12_BY_BYTE[b] ^ INVERT_B_BY_BYTE[b]);
    return pack_signature(1u, tau_a, tau_b);
}

static inline uint32_t compose_signatures(uint32_t left, uint32_t right) {
    // Composition law:
    // f_(p1,t1) o f_(p2,t2) = f_(p1 xor p2, L^p1(t2) xor t1)
    // where left = (p1,t1), right = (p2,t2)

    uint8_t lp, rp;
    uint16_t lta, ltb, rta, rtb;
    uint16_t ra, rb;

    unpack_signature(left, &lp, &lta, &ltb);
    unpack_signature(right, &rp, &rta, &rtb);

    if (lp == 0u) {
        ra = rta;
        rb = rtb;
    } else {
        ra = rtb;
        rb = rta;
    }

    return pack_signature(
        (uint8_t)(lp ^ rp),
        (uint16_t)((ra ^ lta) & LAYER_MASK_12),
        (uint16_t)((rb ^ ltb) & LAYER_MASK_12)
    );
}

static inline uint32_t state24_from_signature_on_rest(uint32_t sig) {
    uint8_t parity;
    uint16_t tau_a12, tau_b12;
    uint16_t a12, b12;

    unpack_signature(sig, &parity, &tau_a12, &tau_b12);

    if (parity == 0u) {
        a12 = (uint16_t)(GENE_MAC_A12 ^ tau_a12);
        b12 = (uint16_t)(GENE_MAC_B12 ^ tau_b12);
    } else {
        a12 = (uint16_t)(GENE_MAC_B12 ^ tau_a12);
        b12 = (uint16_t)(GENE_MAC_A12 ^ tau_b12);
    }

    return (((uint32_t)a12) << 12u) | ((uint32_t)b12);
}

static inline uint8_t chirality_word6_from_state24(uint32_t state24) {
    uint16_t a12 = (uint16_t)((state24 >> 12u) & LAYER_MASK_12);
    uint16_t b12 = (uint16_t)(state24 & LAYER_MASK_12);
    uint16_t diff = (uint16_t)((a12 ^ b12) & LAYER_MASK_12);
    return collapse_pairdiag12_to_word6(diff);
}

GYRO_EXPORT void gyro_extract_scan(
    const uint8_t* bytes,
    int64_t len,
    uint8_t* q_class_out,
    uint8_t* family_out,
    uint8_t* micro_ref_out,
    int32_t* signatures_out,
    int32_t* states_out
) {
    if (bytes == NULL || q_class_out == NULL || family_out == NULL || micro_ref_out == NULL
        || signatures_out == NULL || states_out == NULL || len < 0) {
        return;
    }

    gyro_init_tables();

    uint32_t accum = 0u;
    for (int64_t i = 0; i < len; ++i) {
        uint8_t b = bytes[i];
        q_class_out[i] = Q6_BY_BYTE[b];
        family_out[i] = FAMILY_BY_BYTE[b];
        micro_ref_out[i] = MICRO_BY_BYTE[b];
        uint32_t sig_b = byte_signature(b);
        accum = compose_signatures(sig_b, accum);
        signatures_out[i] = (int32_t)accum;
        states_out[i] = (int32_t)state24_from_signature_on_rest(accum);
    }
}

GYRO_EXPORT void gyro_chirality_distance_adjacent(
    const int32_t* states,
    int64_t len,
    int32_t lookahead,
    uint8_t* distances_out
) {
    if (states == NULL || distances_out == NULL || len < 0 || lookahead < 0) {
        return;
    }
    for (int64_t i = 0; i < len; ++i) {
        int64_t j = i + (int64_t)lookahead;
        if (j >= len) {
            distances_out[i] = 0u;
            continue;
        }
        uint32_t sa = (uint32_t)states[i];
        uint32_t sb = (uint32_t)states[j];
        uint8_t ca = chirality_word6_from_state24(sa);
        uint8_t cb = chirality_word6_from_state24(sb);
        distances_out[i] = (uint8_t)gyro_popcnt32((uint32_t)(ca ^ cb));
    }
}

GYRO_EXPORT void gyro_qmap_extract(
    const uint8_t* bytes,
    int64_t n,
    uint8_t* q_class_out,
    uint8_t* family_out,
    uint8_t* micro_ref_out
) {
    if (bytes == NULL || q_class_out == NULL || family_out == NULL || micro_ref_out == NULL || n < 0) {
        return;
    }

    gyro_init_tables();

    for (int64_t i = 0; i < n; ++i) {
        uint8_t b = bytes[i];
        q_class_out[i] = Q6_BY_BYTE[b];
        family_out[i] = FAMILY_BY_BYTE[b];
        micro_ref_out[i] = MICRO_BY_BYTE[b];
    }
}

GYRO_EXPORT void gyro_signature_scan(
    const uint8_t* bytes,
    int64_t len,
    int32_t* signatures_out,
    int num_threads
) {
    (void)num_threads; // Reserved for future native parallel implementation.

    if (bytes == NULL || signatures_out == NULL || len < 0) {
        return;
    }

    gyro_init_tables();

    uint32_t accum = 0u; // identity signature

    for (int64_t i = 0; i < len; ++i) {
        uint32_t sig_b = byte_signature(bytes[i]);
        accum = compose_signatures(sig_b, accum);
        signatures_out[i] = (int32_t)accum;
    }
}

GYRO_EXPORT void gyro_chirality_distance(
    const int32_t* states_a,
    const int32_t* states_b,
    int64_t n,
    uint8_t* distances_out
) {
    if (states_a == NULL || states_b == NULL || distances_out == NULL || n < 0) {
        return;
    }

    for (int64_t i = 0; i < n; ++i) {
        uint32_t sa = (uint32_t)states_a[i];
        uint32_t sb = (uint32_t)states_b[i];
        uint8_t ca = chirality_word6_from_state24(sa);
        uint8_t cb = chirality_word6_from_state24(sb);
        distances_out[i] = (uint8_t)gyro_popcnt32((uint32_t)(ca ^ cb));
    }
}

GYRO_EXPORT void gyro_wht64_float(
    const float* input,
    float* output,
    int64_t batch
) {
    if (input == NULL || output == NULL || batch < 0) {
        return;
    }

    for (int64_t b = 0; b < batch; ++b) {
        float tmp[64];
        const float* src = input + (b * 64);
        float* dst = output + (b * 64);

        memcpy(tmp, src, 64 * sizeof(float));

        for (int stride = 1; stride < 64; stride <<= 1) {
            int jump = stride << 1;
            for (int base = 0; base < 64; base += jump) {
                for (int j = 0; j < stride; ++j) {
                    float u = tmp[base + j];
                    float v = tmp[base + j + stride];
                    tmp[base + j] = u + v;
                    tmp[base + j + stride] = u - v;
                }
            }
        }

        for (int i = 0; i < 64; ++i) {
            dst[i] = tmp[i] * 0.125f; // 1 / sqrt(64)
        }
    }
}

GYRO_EXPORT int32_t gyro_apply_signature_to_rest(int32_t signature) {
    return (int32_t)state24_from_signature_on_rest((uint32_t)signature);
}

static inline uint32_t gyro_apply_signature_to_state24(
    uint32_t state24,
    uint32_t sig
) {
    uint8_t parity;
    uint16_t tau_a12, tau_b12;
    uint16_t a12, b12, a_in;

    unpack_signature(sig, &parity, &tau_a12, &tau_b12);

    a12 = (uint16_t)((state24 >> 12u) & LAYER_MASK_12);
    b12 = (uint16_t)(state24 & LAYER_MASK_12);

    if (parity == 0u) {
        a12 = (uint16_t)((a12 ^ tau_a12) & LAYER_MASK_12);
        b12 = (uint16_t)((b12 ^ tau_b12) & LAYER_MASK_12);
    } else {
        a_in = a12;
        a12 = (uint16_t)((b12 ^ tau_a12) & LAYER_MASK_12);
        b12 = (uint16_t)((a_in ^ tau_b12) & LAYER_MASK_12);
    }

    return (((uint32_t)a12) << 12u) | ((uint32_t)b12);
}

static inline uint32_t gyro_step_state24_by_byte(
    uint32_t state24,
    uint8_t b
) {
    uint16_t a12 = (uint16_t)((state24 >> 12u) & LAYER_MASK_12);
    uint16_t b12 = (uint16_t)(state24 & LAYER_MASK_12);
    uint16_t a_mut = (uint16_t)((a12 ^ MASK12_BY_BYTE[b]) & LAYER_MASK_12);
    uint16_t a_next = (uint16_t)((b12 ^ INVERT_A_BY_BYTE[b]) & LAYER_MASK_12);
    uint16_t b_next = (uint16_t)((a_mut ^ INVERT_B_BY_BYTE[b]) & LAYER_MASK_12);
    return (((uint32_t)a_next) << 12u) | ((uint32_t)b_next);
}

GYRO_EXPORT int32_t gyro_apply_signature_to_state(
    int32_t state24,
    int32_t signature
) {
    return (int32_t)gyro_apply_signature_to_state24(
        (uint32_t)state24,
        (uint32_t)signature
    );
}

GYRO_EXPORT void gyro_apply_signature_batch(
    const int32_t* states_in,
    const int32_t* signatures,
    int64_t n,
    int32_t* states_out
) {
    if (states_in == NULL || signatures == NULL || states_out == NULL || n < 0) {
        return;
    }

    for (int64_t i = 0; i < n; ++i) {
        states_out[i] = (int32_t)gyro_apply_signature_to_state24(
            (uint32_t)states_in[i],
            (uint32_t)signatures[i]
        );
    }
}

GYRO_EXPORT void gyro_step_byte_batch(
    const int32_t* states_in,
    int64_t n,
    uint8_t byte,
    int32_t* states_out
) {
    if (states_in == NULL || states_out == NULL || n < 0) {
        return;
    }

    gyro_init_tables();

    for (int64_t i = 0; i < n; ++i) {
        states_out[i] = (int32_t)gyro_step_state24_by_byte(
            (uint32_t)states_in[i],
            byte
        );
    }
}

/* -------------------------------------------------------------------------
   Omega (Omega) packed 12-bit chart - batch conversions and stepping
   ------------------------------------------------------------------------- */

static inline uint32_t gyro_pack_omega12(uint8_t u6, uint8_t v6) {
    return (((uint32_t)(u6 & 0x3Fu)) << 6u) | ((uint32_t)(v6 & 0x3Fu));
}

static inline void gyro_unpack_omega12(uint32_t omega12, uint8_t* u6, uint8_t* v6) {
    *u6 = (uint8_t)((omega12 >> 6u) & 0x3Fu);
    *v6 = (uint8_t)(omega12 & 0x3Fu);
}

static inline uint32_t gyro_pack_omega_sig12(uint8_t parity, uint8_t tau_u6, uint8_t tau_v6) {
    return (((uint32_t)(parity & 1u)) << 12u)
         | (((uint32_t)(tau_u6 & 0x3Fu)) << 6u)
         | ((uint32_t)(tau_v6 & 0x3Fu));
}

static inline void gyro_unpack_omega_sig12(uint32_t sig, uint8_t* parity, uint8_t* tau_u6, uint8_t* tau_v6) {
    *parity = (uint8_t)((sig >> 12u) & 1u);
    *tau_u6 = (uint8_t)((sig >> 6u) & 0x3Fu);
    *tau_v6 = (uint8_t)(sig & 0x3Fu);
}

static inline uint32_t omega_sig_pack(uint8_t parity, uint8_t tau_u6, uint8_t tau_v6) {
    return gyro_pack_omega_sig12(parity, tau_u6, tau_v6);
}

static inline uint32_t omega_byte_signature(uint8_t b) {
    return omega_sig_pack(
        1u,
        EPS_A6_BY_BYTE[b],
        (uint8_t)(MICRO_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b])
    );
}

static inline uint32_t compose_omega_signatures(uint32_t left, uint32_t right) {
    uint8_t lp, rp, ltu, ltv, rtu, rtv, ru, rv;

    gyro_unpack_omega_sig12(left, &lp, &ltu, &ltv);
    gyro_unpack_omega_sig12(right, &rp, &rtu, &rtv);

    if (lp == 0u) {
        ru = rtu;
        rv = rtv;
    } else {
        ru = rtv;
        rv = rtu;
    }

    return omega_sig_pack(
        (uint8_t)(lp ^ rp),
        (uint8_t)(ru ^ ltu),
        (uint8_t)(rv ^ ltv)
    );
}

static inline int is_pair_diagonal12(uint16_t x) {
    for (uint8_t i = 0u; i < 6u; ++i) {
        uint16_t pair = (uint16_t)((x >> (2u * i)) & 0x3u);
        if (pair != 0u && pair != 0x3u) {
            return 0;
        }
    }
    return 1;
}

static inline int gyro_try_state24_to_omega12(uint32_t state24, uint32_t* out_omega12) {
    uint16_t a12 = (uint16_t)((state24 >> 12u) & LAYER_MASK_12);
    uint16_t b12 = (uint16_t)(state24 & LAYER_MASK_12);

    uint16_t ua = (uint16_t)(a12 ^ GENE_MAC_A12);
    uint16_t vb = (uint16_t)(b12 ^ GENE_MAC_A12);

    if (!is_pair_diagonal12(ua) || !is_pair_diagonal12(vb)) {
        return 0;
    }

    uint8_t u6 = collapse_pairdiag12_to_word6(ua);
    uint8_t v6 = collapse_pairdiag12_to_word6(vb);
    *out_omega12 = gyro_pack_omega12(u6, v6);
    return 1;
}

GYRO_EXPORT void gyro_state24_to_omega12_batch(
    const int32_t* states_in,
    int64_t n,
    int32_t* omega12_out,
    uint8_t* valid_out
) {
    if (states_in == NULL || omega12_out == NULL || valid_out == NULL || n < 0) {
        return;
    }
    gyro_init_tables();
    for (int64_t i = 0; i < n; ++i) {
        uint32_t packed = 0u;
        int ok = gyro_try_state24_to_omega12((uint32_t)states_in[i], &packed);
        omega12_out[i] = (int32_t)packed;
        valid_out[i] = (uint8_t)(ok ? 1u : 0u);
    }
}

GYRO_EXPORT void gyro_omega12_to_state24_batch(
    const int32_t* omega12_in,
    int64_t n,
    int32_t* states_out
) {
    if (omega12_in == NULL || states_out == NULL || n < 0) {
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        uint8_t u6, v6;
        gyro_unpack_omega12((uint32_t)omega12_in[i], &u6, &v6);
        uint16_t a12 = (uint16_t)(GENE_MAC_A12 ^ mask12_from_micro_ref(u6));
        uint16_t b12 = (uint16_t)(GENE_MAC_A12 ^ mask12_from_micro_ref(v6));
        states_out[i] = (int32_t)((((uint32_t)a12) << 12u) | (uint32_t)b12);
    }
}

static inline uint32_t gyro_step_omega12_by_byte(uint32_t omega12, uint8_t b) {
    uint8_t u6, v6;
    gyro_unpack_omega12(omega12, &u6, &v6);
    uint8_t u_next = (uint8_t)((v6 ^ EPS_A6_BY_BYTE[b]) & 0x3Fu);
    uint8_t v_next = (uint8_t)((u6 ^ MICRO_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b]) & 0x3Fu);
    return gyro_pack_omega12(u_next, v_next);
}

GYRO_EXPORT void gyro_step_omega12_batch(
    const int32_t* omega12_in,
    int64_t n,
    uint8_t byte,
    int32_t* omega12_out
) {
    if (omega12_in == NULL || omega12_out == NULL || n < 0) {
        return;
    }
    gyro_init_tables();
    for (int64_t i = 0; i < n; ++i) {
        omega12_out[i] = (int32_t)gyro_step_omega12_by_byte((uint32_t)omega12_in[i], byte);
    }
}

GYRO_EXPORT void gyro_apply_omega_signature_batch(
    const int32_t* omega12_in,
    const int32_t* omega_sig_in,
    int64_t n,
    int32_t* omega12_out
) {
    if (omega12_in == NULL || omega_sig_in == NULL || omega12_out == NULL || n < 0) {
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        uint8_t u6, v6, parity, tau_u6, tau_v6;
        gyro_unpack_omega12((uint32_t)omega12_in[i], &u6, &v6);
        gyro_unpack_omega_sig12((uint32_t)omega_sig_in[i], &parity, &tau_u6, &tau_v6);

        if (parity == 0u) {
            omega12_out[i] = (int32_t)gyro_pack_omega12(
                (uint8_t)(u6 ^ tau_u6),
                (uint8_t)(v6 ^ tau_v6)
            );
        } else {
            omega12_out[i] = (int32_t)gyro_pack_omega12(
                (uint8_t)(v6 ^ tau_u6),
                (uint8_t)(u6 ^ tau_v6)
            );
        }
    }
}

GYRO_EXPORT void gyro_shell_histogram_state24(
    const int32_t* states,
    int64_t n,
    int32_t* hist_out
) {
    if (states == NULL || hist_out == NULL || n < 0) {
        return;
    }
    for (int i = 0; i < 7; ++i) hist_out[i] = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint8_t chi = chirality_word6_from_state24((uint32_t)states[i]);
        uint8_t w = (uint8_t)gyro_popcnt32((uint32_t)chi);
        if (w < 7u) hist_out[w] += 1;
    }
}

GYRO_EXPORT void gyro_shell_histogram_omega12(
    const int32_t* omega12_in,
    int64_t n,
    int32_t* hist_out
) {
    if (omega12_in == NULL || hist_out == NULL || n < 0) {
        return;
    }
    for (int i = 0; i < 7; ++i) hist_out[i] = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint8_t u6, v6;
        gyro_unpack_omega12((uint32_t)omega12_in[i], &u6, &v6);
        uint8_t w = (uint8_t)gyro_popcnt32((uint32_t)(u6 ^ v6));
        if (w < 7u) hist_out[w] += 1;
    }
}

GYRO_EXPORT void gyro_omega_signature_scan(
    const uint8_t* bytes,
    int64_t len,
    int32_t* omega_signatures_out
) {
    if (bytes == NULL || omega_signatures_out == NULL || len < 0) {
        return;
    }

    gyro_init_tables();

    uint32_t accum = 0u;

    for (int64_t i = 0; i < len; ++i) {
        uint32_t sig_b = omega_byte_signature(bytes[i]);
        accum = compose_omega_signatures(sig_b, accum);
        omega_signatures_out[i] = (int32_t)accum;
    }
}

GYRO_EXPORT void gyro_omega12_scan_from_omega12(
    const uint8_t* bytes,
    int64_t len,
    int32_t start_omega12,
    int32_t* omega12_out
) {
    if (bytes == NULL || omega12_out == NULL || len < 0) {
        return;
    }

    gyro_init_tables();

    uint32_t s = (uint32_t)start_omega12 & 0x0FFFu;
    for (int64_t i = 0; i < len; ++i) {
        s = gyro_step_omega12_by_byte(s, bytes[i]);
        omega12_out[i] = (int32_t)s;
    }
}

GYRO_EXPORT int64_t gyro_shell_histogram_state24_checked(
    const int32_t* states,
    int64_t n,
    int32_t* hist_out
) {
    if (states == NULL || hist_out == NULL || n < 0) {
        return -1;
    }

    for (int i = 0; i < 7; ++i) hist_out[i] = 0;

    int64_t invalid = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t packed = 0u;
        if (!gyro_try_state24_to_omega12((uint32_t)states[i], &packed)) {
            invalid++;
            continue;
        }

        uint8_t u6, v6;
        gyro_unpack_omega12(packed, &u6, &v6);
        uint8_t w = (uint8_t)gyro_popcnt32((uint32_t)(u6 ^ v6));
        if (w < 7u) hist_out[w] += 1;
    }

    return invalid;
}

GYRO_EXPORT void gyro_apply_omega_gate_batch(
    const int32_t* omega12_in,
    int64_t n,
    uint8_t gate_code,
    int32_t* omega12_out
) {
    if (omega12_in == NULL || omega12_out == NULL || n < 0) {
        return;
    }

    for (int64_t i = 0; i < n; ++i) {
        uint8_t u6, v6;
        gyro_unpack_omega12((uint32_t)omega12_in[i], &u6, &v6);

        switch (gate_code) {
            case 0:
                omega12_out[i] = (int32_t)gyro_pack_omega12(u6, v6);
                break;
            case 1:
                omega12_out[i] = (int32_t)gyro_pack_omega12(v6, u6);
                break;
            case 2:
                omega12_out[i] = (int32_t)gyro_pack_omega12(
                    (uint8_t)(v6 ^ EPSILON_6),
                    (uint8_t)(u6 ^ EPSILON_6)
                );
                break;
            case 3:
                omega12_out[i] = (int32_t)gyro_pack_omega12(
                    (uint8_t)(u6 ^ EPSILON_6),
                    (uint8_t)(v6 ^ EPSILON_6)
                );
                break;
            default:
                omega12_out[i] = 0;
                break;
        }
    }
}

GYRO_EXPORT void gyro_state_scan_from_state(
    const uint8_t* bytes,
    int64_t len,
    int32_t start_state24,
    int32_t* states_out
) {
    if (bytes == NULL || states_out == NULL || len < 0) {
        return;
    }

    gyro_init_tables();

    uint32_t s = (uint32_t)start_state24;
    for (int64_t i = 0; i < len; ++i) {
        s = gyro_step_state24_by_byte(s, bytes[i]);
        states_out[i] = (int32_t)s;
    }
}

/* -------------------------------------------------------------------------
   aQPU Bitplane GEMV Engine
   ------------------------------------------------------------------------- */

GYRO_EXPORT void gyro_float_to_fixed(
    const float* data,
    int64_t n,
    float scale,
    int32_t* out
) {
    if (data == NULL || out == NULL || n < 0) {
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        float v = data[i] * scale;
        out[i] = (int32_t)nearbyintf(v);
    }
}

GYRO_EXPORT void gyro_bitplane_gemv(
    const int32_t* W_int,
    const int32_t* x_int,
    int64_t rows,
    int64_t cols,
    int32_t n_bits,
    int64_t* y_out
) {
    if (W_int == NULL || x_int == NULL || y_out == NULL || rows < 0 || cols < 0 || n_bits < 1 || n_bits > 30) {
        return;
    }
    if (cols > 64) {
        return;
    }

    uint64_t x_bp[30];
    for (int32_t k = 0; k < n_bits; ++k) {
        uint64_t acc = 0;
        for (int64_t j = 0; j < cols; ++j) {
            int32_t v = x_int[j];
            uint64_t mag = (uint64_t)(v < 0 ? -v : v);
            uint64_t bit = (mag >> k) & 1u;
            acc |= (bit << j);
        }
        x_bp[k] = acc;
    }

    for (int64_t i = 0; i < rows; ++i) {
        uint64_t pos_mask = 0;
        uint64_t neg_mask = 0;
        for (int64_t j = 0; j < cols; ++j) {
            int32_t w = W_int[i * cols + j];
            int32_t xv = x_int[j];
            int sign_w = (w < 0) ? 1 : 0;
            int sign_x = (xv < 0) ? 1 : 0;
            if ((sign_w ^ sign_x) == 0) {
                pos_mask |= (uint64_t)(1ull << j);
            } else {
                neg_mask |= (uint64_t)(1ull << j);
            }
        }

        uint64_t W_bp[30];
        for (int32_t m = 0; m < n_bits; ++m) {
            uint64_t acc = 0;
            for (int64_t j = 0; j < cols; ++j) {
                int32_t w = W_int[i * cols + j];
                uint64_t mag = (uint64_t)(w < 0 ? -w : w);
                uint64_t bit = (mag >> m) & 1u;
                acc |= (bit << j);
            }
            W_bp[m] = acc;
        }

        int64_t pos_dot = 0;
        int64_t neg_dot = 0;
        for (int32_t m = 0; m < n_bits; ++m) {
            for (int32_t k = 0; k < n_bits; ++k) {
                uint64_t pm = W_bp[m] & x_bp[k];
                int64_t partial_pos = POPCNT64(pm & pos_mask);
                int64_t partial_neg = POPCNT64(pm & neg_mask);
                int shift = m + k;
                int64_t mul = (int64_t)(1ull << shift);
                pos_dot += partial_pos * mul;
                neg_dot += partial_neg * mul;
            }
        }
        y_out[i] = pos_dot - neg_dot;
    }
}

GYRO_EXPORT void gyro_bitplane_gemv_f32(
    const float* W,
    const float* x,
    int64_t rows,
    int64_t cols,
    int32_t n_bits,
    float* y_out
) {
    if (W == NULL || x == NULL || y_out == NULL || rows < 0 || cols < 0 || n_bits < 1 || n_bits > 30) {
        return;
    }

    float max_abs = 1e-12f;
    for (int64_t i = 0; i < rows * cols; ++i) {
        float a = (float)fabsf(W[i]);
        if (a > max_abs) max_abs = a;
    }
    for (int64_t j = 0; j < cols; ++j) {
        float a = (float)fabsf(x[j]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs < 1e-12f) {
        for (int64_t i = 0; i < rows; ++i) {
            y_out[i] = 0.0f;
        }
        return;
    }

    int64_t scale_max = (int64_t)(1 << (n_bits - 1)) - 1;
    float scale = (float)scale_max / max_abs;

    int32_t* W_int = (int32_t*)malloc((size_t)(rows * cols) * sizeof(int32_t));
    int32_t* x_int = (int32_t*)malloc((size_t)cols * sizeof(int32_t));
    int64_t* y_int = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (W_int == NULL || x_int == NULL || y_int == NULL) {
        if (W_int) free(W_int);
        if (x_int) free(x_int);
        if (y_int) free(y_int);
        return;
    }

    gyro_float_to_fixed(W, rows * cols, scale, W_int);
    gyro_float_to_fixed(x, cols, scale, x_int);
    gyro_bitplane_gemv(W_int, x_int, rows, cols, n_bits, y_int);

    float scale_sq = scale * scale;
    for (int64_t i = 0; i < rows; ++i) {
        y_out[i] = (float)y_int[i] / scale_sq;
    }

    free(W_int);
    free(x_int);
    free(y_int);
}

/* Packed-weight path: pack W once, gemv many times */

GYRO_EXPORT void gyro_pack_bitplane_matrix_f32(
    const float* W,
    int64_t rows,
    int64_t cols,
    int32_t n_bits,
    float* scale_w_out,
    uint64_t* W_sign_out,
    uint64_t* W_bp_out
) {
    if (W == NULL || scale_w_out == NULL || W_sign_out == NULL || W_bp_out == NULL
        || rows < 0 || cols < 0 || n_bits < 1 || n_bits > 30 || cols > 64) {
        return;
    }

    float max_abs_W = 1e-12f;
    for (int64_t i = 0; i < rows * cols; ++i) {
        float a = (float)fabsf(W[i]);
        if (a > max_abs_W) max_abs_W = a;
    }

    if (max_abs_W < 1e-12f) {
        *scale_w_out = 1.0f;
        for (int64_t i = 0; i < rows; ++i) W_sign_out[i] = 0;
        for (int64_t i = 0; i < rows * (int64_t)n_bits; ++i) W_bp_out[i] = 0;
        return;
    }

    int64_t scale_max = (int64_t)(1 << (n_bits - 1)) - 1;
    float scale_w = (float)scale_max / max_abs_W;
    *scale_w_out = scale_w;

    int32_t w_int_row[64];
    for (int64_t i = 0; i < rows; ++i) {
        uint64_t sign_mask = 0;
        for (int64_t j = 0; j < cols; ++j) {
            float v = W[i * cols + j] * scale_w;
            w_int_row[j] = (int32_t)nearbyintf(v);
            if (w_int_row[j] < 0) {
                sign_mask |= (uint64_t)(1ull << j);
            }
        }
        W_sign_out[i] = sign_mask;

        for (int32_t m = 0; m < n_bits; ++m) {
            uint64_t acc = 0;
            for (int64_t j = 0; j < cols; ++j) {
                int32_t w = w_int_row[j];
                uint64_t mag = (uint64_t)(w < 0 ? -w : w);
                uint64_t bit = (mag >> m) & 1u;
                acc |= (bit << j);
            }
            W_bp_out[i * (int64_t)n_bits + m] = acc;
        }
    }
}

GYRO_EXPORT void gyro_bitplane_gemv_packed_f32(
    const uint64_t* W_sign,
    const uint64_t* W_bp,
    float scale_w,
    const float* x,
    int64_t rows,
    int64_t cols,
    int32_t n_bits,
    float* y_out
) {
    if (W_sign == NULL || W_bp == NULL || x == NULL || y_out == NULL
        || rows < 0 || cols < 0 || n_bits < 1 || n_bits > 30 || cols > 64) {
        return;
    }

    float max_abs_x = 1e-12f;
    for (int64_t j = 0; j < cols; ++j) {
        float a = (float)fabsf(x[j]);
        if (a > max_abs_x) max_abs_x = a;
    }

    if (max_abs_x < 1e-12f) {
        for (int64_t i = 0; i < rows; ++i) y_out[i] = 0.0f;
        return;
    }

    int64_t scale_max = (int64_t)(1 << (n_bits - 1)) - 1;
    float scale_x = (float)scale_max / max_abs_x;

    int32_t x_int[64];
    for (int64_t j = 0; j < cols; ++j) {
        float v = x[j] * scale_x;
        x_int[j] = (int32_t)nearbyintf(v);
    }

    uint64_t x_sign_mask = 0;
    for (int64_t j = 0; j < cols; ++j) {
        if (x_int[j] < 0) x_sign_mask |= (uint64_t)(1ull << j);
    }

    uint64_t x_bp[30];
    for (int32_t k = 0; k < n_bits; ++k) {
        uint64_t acc = 0;
        for (int64_t j = 0; j < cols; ++j) {
            int32_t v = x_int[j];
            uint64_t mag = (uint64_t)(v < 0 ? -v : v);
            uint64_t bit = (mag >> k) & 1u;
            acc |= (bit << j);
        }
        x_bp[k] = acc;
    }

    uint64_t col_mask = (cols == 64) ? ~(uint64_t)0 : ((uint64_t)(1ull << cols) - 1);

    for (int64_t i = 0; i < rows; ++i) {
        uint64_t neg_mask = W_sign[i] ^ x_sign_mask;
        uint64_t pos_mask = (~neg_mask) & col_mask;

        int64_t pos_dot = 0;
        int64_t neg_dot = 0;
        for (int32_t m = 0; m < n_bits; ++m) {
            for (int32_t k = 0; k < n_bits; ++k) {
                uint64_t pm = W_bp[i * (int64_t)n_bits + m] & x_bp[k];
                int64_t partial_pos = POPCNT64(pm & pos_mask);
                int64_t partial_neg = POPCNT64(pm & neg_mask);
                int shift = m + k;
                int64_t mul = (int64_t)(1ull << shift);
                pos_dot += partial_pos * mul;
                neg_dot += partial_neg * mul;
            }
        }
        float scale_prod = scale_w * scale_x;
        y_out[i] = (float)(pos_dot - neg_dot) / scale_prod;
    }
}

GYRO_EXPORT void gyro_pack_bitplane_vector_f32(
    const float* x,
    int64_t cols,
    int32_t n_bits,
    float* scale_x_out,
    uint64_t* x_sign_out,
    uint64_t* x_bp_out
) {
    if (x == NULL || scale_x_out == NULL || x_sign_out == NULL || x_bp_out == NULL
        || cols < 0 || cols > 64 || n_bits < 1 || n_bits > 30) {
        return;
    }

    float max_abs_x = 1e-12f;
    for (int64_t j = 0; j < cols; ++j) {
        float a = (float)fabsf(x[j]);
        if (a > max_abs_x) max_abs_x = a;
    }

    if (max_abs_x < 1e-12f) {
        *scale_x_out = 1.0f;
        *x_sign_out = 0;
        for (int32_t k = 0; k < n_bits; ++k) {
            x_bp_out[k] = 0;
        }
        return;
    }

    int64_t scale_max = (int64_t)(1 << (n_bits - 1)) - 1;
    float scale_x = (float)scale_max / max_abs_x;
    *scale_x_out = scale_x;

    int32_t x_int[64];
    uint64_t x_sign = 0;

    for (int64_t j = 0; j < cols; ++j) {
        float v = x[j] * scale_x;
        x_int[j] = (int32_t)nearbyintf(v);
        if (x_int[j] < 0) {
            x_sign |= (uint64_t)(1ull << j);
        }
    }
    *x_sign_out = x_sign;

    for (int32_t k = 0; k < n_bits; ++k) {
        uint64_t acc = 0;
        for (int64_t j = 0; j < cols; ++j) {
            int32_t v = x_int[j];
            uint64_t mag = (uint64_t)(v < 0 ? -v : v);
            uint64_t bit = (mag >> k) & 1u;
            acc |= (bit << j);
        }
        x_bp_out[k] = acc;
    }
}

#define GEMV_PACKED_X_F32_IMPL(N_BITS) \
static void gyro_bitplane_gemv_packed_x_f32_##N_BITS( \
    const uint64_t* W_sign, \
    const uint64_t* W_bp, \
    float scale_w, \
    uint64_t x_sign, \
    const uint64_t* x_bp, \
    float scale_x, \
    int64_t rows, \
    int64_t cols, \
    float* y_out \
) { \
    uint64_t col_mask = (cols == 64) ? ~(uint64_t)0 : ((uint64_t)(1ull << cols) - 1); \
    float scale_prod = scale_w * scale_x; \
    for (int64_t i = 0; i < rows; ++i) { \
        uint64_t neg_mask = W_sign[i] ^ x_sign; \
        uint64_t pos_mask = (~neg_mask) & col_mask; \
        int64_t pos_dot = 0; \
        int64_t neg_dot = 0; \
        for (int32_t m = 0; m < (N_BITS); ++m) { \
            for (int32_t k = 0; k < (N_BITS); ++k) { \
                uint64_t pm = W_bp[i * (int64_t)(N_BITS) + m] & x_bp[k]; \
                int64_t partial_pos = POPCNT64(pm & pos_mask); \
                int64_t partial_neg = POPCNT64(pm & neg_mask); \
                int shift = m + k; \
                int64_t mul = (int64_t)(1ull << shift); \
                pos_dot += partial_pos * mul; \
                neg_dot += partial_neg * mul; \
            } \
        } \
        y_out[i] = (float)(pos_dot - neg_dot) / scale_prod; \
    } \
}

GEMV_PACKED_X_F32_IMPL(8)
GEMV_PACKED_X_F32_IMPL(12)
GEMV_PACKED_X_F32_IMPL(16)

GYRO_EXPORT void gyro_bitplane_gemv_packed_x_f32(
    const uint64_t* W_sign,
    const uint64_t* W_bp,
    float scale_w,
    uint64_t x_sign,
    const uint64_t* x_bp,
    float scale_x,
    int64_t rows,
    int64_t cols,
    int32_t n_bits,
    float* y_out
) {
    if (W_sign == NULL || W_bp == NULL || x_bp == NULL || y_out == NULL
        || rows < 0 || cols < 0 || cols > 64 || n_bits < 1 || n_bits > 30) {
        return;
    }

    if (n_bits == 8) {
        gyro_bitplane_gemv_packed_x_f32_8(W_sign, W_bp, scale_w, x_sign, x_bp, scale_x, rows, cols, y_out);
        return;
    }
    if (n_bits == 12) {
        gyro_bitplane_gemv_packed_x_f32_12(W_sign, W_bp, scale_w, x_sign, x_bp, scale_x, rows, cols, y_out);
        return;
    }
    if (n_bits == 16) {
        gyro_bitplane_gemv_packed_x_f32_16(W_sign, W_bp, scale_w, x_sign, x_bp, scale_x, rows, cols, y_out);
        return;
    }

    uint64_t col_mask = (cols == 64) ? ~(uint64_t)0 : ((uint64_t)(1ull << cols) - 1);
    float scale_prod = scale_w * scale_x;

    for (int64_t i = 0; i < rows; ++i) {
        uint64_t neg_mask = W_sign[i] ^ x_sign;
        uint64_t pos_mask = (~neg_mask) & col_mask;

        int64_t pos_dot = 0;
        int64_t neg_dot = 0;

        for (int32_t m = 0; m < n_bits; ++m) {
            for (int32_t k = 0; k < n_bits; ++k) {
                uint64_t pm = W_bp[i * (int64_t)n_bits + m] & x_bp[k];
                int64_t partial_pos = POPCNT64(pm & pos_mask);
                int64_t partial_neg = POPCNT64(pm & neg_mask);
                int shift = m + k;
                int64_t mul = (int64_t)(1ull << shift);
                pos_dot += partial_pos * mul;
                neg_dot += partial_neg * mul;
            }
        }

        y_out[i] = (float)(pos_dot - neg_dot) / scale_prod;
    }
}

/* Integer-native path: no quantization, no scale. Exact internal multiplication. */

GYRO_EXPORT void gyro_pack_bitplane_matrix_i32(
    const int32_t* W_int,
    int64_t rows,
    int64_t cols,
    int32_t n_bits,
    uint64_t* W_sign_out,
    uint64_t* W_bp_out
) {
    if (W_int == NULL || W_sign_out == NULL || W_bp_out == NULL
        || rows < 0 || cols < 0 || cols > 64 || n_bits < 1 || n_bits > 30) {
        return;
    }

    for (int64_t i = 0; i < rows; ++i) {
        uint64_t sign_mask = 0;
        for (int64_t j = 0; j < cols; ++j) {
            int32_t w = W_int[i * cols + j];
            if (w < 0) {
                sign_mask |= (uint64_t)(1ull << j);
            }
        }
        W_sign_out[i] = sign_mask;

        for (int32_t m = 0; m < n_bits; ++m) {
            uint64_t acc = 0;
            for (int64_t j = 0; j < cols; ++j) {
                int32_t w = W_int[i * cols + j];
                uint64_t mag = (uint64_t)(w < 0 ? -w : w);
                uint64_t bit = (mag >> m) & 1u;
                acc |= (bit << j);
            }
            W_bp_out[i * (int64_t)n_bits + m] = acc;
        }
    }
}

GYRO_EXPORT void gyro_pack_bitplane_vector_i32(
    const int32_t* x_int,
    int64_t cols,
    int32_t n_bits,
    uint64_t* x_sign_out,
    uint64_t* x_bp_out
) {
    if (x_int == NULL || x_sign_out == NULL || x_bp_out == NULL
        || cols < 0 || cols > 64 || n_bits < 1 || n_bits > 30) {
        return;
    }

    uint64_t x_sign = 0;
    for (int64_t j = 0; j < cols; ++j) {
        if (x_int[j] < 0) {
            x_sign |= (uint64_t)(1ull << j);
        }
    }
    *x_sign_out = x_sign;

    for (int32_t k = 0; k < n_bits; ++k) {
        uint64_t acc = 0;
        for (int64_t j = 0; j < cols; ++j) {
            int32_t v = x_int[j];
            uint64_t mag = (uint64_t)(v < 0 ? -v : v);
            uint64_t bit = (mag >> k) & 1u;
            acc |= (bit << j);
        }
        x_bp_out[k] = acc;
    }
}

GYRO_EXPORT void gyro_bitplane_gemv_packed_i32(
    const uint64_t* W_sign,
    const uint64_t* W_bp,
    uint64_t x_sign,
    const uint64_t* x_bp,
    int64_t rows,
    int64_t cols,
    int32_t n_bits,
    int64_t* y_out
) {
    if (W_sign == NULL || W_bp == NULL || x_bp == NULL || y_out == NULL
        || rows < 0 || cols < 0 || cols > 64 || n_bits < 1 || n_bits > 30) {
        return;
    }

    uint64_t col_mask = (cols == 64) ? ~(uint64_t)0 : ((uint64_t)(1ull << cols) - 1);

    for (int64_t i = 0; i < rows; ++i) {
        uint64_t neg_mask = W_sign[i] ^ x_sign;
        uint64_t pos_mask = (~neg_mask) & col_mask;

        int64_t pos_dot = 0;
        int64_t neg_dot = 0;

        for (int32_t m = 0; m < n_bits; ++m) {
            for (int32_t k = 0; k < n_bits; ++k) {
                uint64_t pm = W_bp[i * (int64_t)n_bits + m] & x_bp[k];
                int64_t partial_pos = POPCNT64(pm & pos_mask);
                int64_t partial_neg = POPCNT64(pm & neg_mask);
                int shift = m + k;
                int64_t mul = (int64_t)(1ull << shift);
                pos_dot += partial_pos * mul;
                neg_dot += partial_neg * mul;
            }
        }

        y_out[i] = pos_dot - neg_dot;
    }
}

/* Batched packed GEMM: one packed matrix, many packed vectors. */

GYRO_EXPORT void gyro_pack_bitplane_vector_batch_f32(
    const float* X,
    int64_t batch,
    int64_t cols,
    int32_t n_bits,
    float* scale_x_out,
    uint64_t* X_sign_out,
    uint64_t* X_bp_out
) {
    if (X == NULL || scale_x_out == NULL || X_sign_out == NULL || X_bp_out == NULL
        || batch < 0 || cols < 0 || cols > 64 || n_bits < 1 || n_bits > 30) {
        return;
    }

    for (int64_t b = 0; b < batch; ++b) {
        const float* x = X + b * cols;
        float max_abs_x = 1e-12f;
        for (int64_t j = 0; j < cols; ++j) {
            float a = (float)fabsf(x[j]);
            if (a > max_abs_x) max_abs_x = a;
        }

        if (max_abs_x < 1e-12f) {
            scale_x_out[b] = 1.0f;
            X_sign_out[b] = 0;
            for (int32_t k = 0; k < n_bits; ++k) {
                X_bp_out[b * (int64_t)n_bits + k] = 0;
            }
            continue;
        }

        int64_t scale_max = (int64_t)(1 << (n_bits - 1)) - 1;
        float scale_x = (float)scale_max / max_abs_x;
        scale_x_out[b] = scale_x;

        int32_t x_int[64];
        uint64_t x_sign = 0;
        for (int64_t j = 0; j < cols; ++j) {
            float v = x[j] * scale_x;
            x_int[j] = (int32_t)nearbyintf(v);
            if (x_int[j] < 0) {
                x_sign |= (uint64_t)(1ull << j);
            }
        }
        X_sign_out[b] = x_sign;

        for (int32_t k = 0; k < n_bits; ++k) {
            uint64_t acc = 0;
            for (int64_t j = 0; j < cols; ++j) {
                int32_t v = x_int[j];
                uint64_t mag = (uint64_t)(v < 0 ? -v : v);
                uint64_t bit = (mag >> k) & 1u;
                acc |= (bit << j);
            }
            X_bp_out[b * (int64_t)n_bits + k] = acc;
        }
    }
}

GYRO_EXPORT void gyro_bitplane_gemm_packed_x_batch_f32(
    const uint64_t* W_sign,
    const uint64_t* W_bp,
    float scale_w,
    const float* scale_x,
    const uint64_t* X_sign,
    const uint64_t* X_bp,
    int64_t rows,
    int64_t cols,
    int64_t batch,
    int32_t n_bits,
    float* Y_out
) {
    if (W_sign == NULL || W_bp == NULL || scale_x == NULL || X_sign == NULL || X_bp == NULL || Y_out == NULL
        || rows < 0 || cols < 0 || cols > 64 || batch < 0 || n_bits < 1 || n_bits > 30) {
        return;
    }

    uint64_t col_mask = (cols == 64) ? ~(uint64_t)0 : ((uint64_t)(1ull << cols) - 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        float scale_prod = scale_w * scale_x[b];
        uint64_t x_sign = X_sign[b];
        const uint64_t* x_bp = X_bp + b * (int64_t)n_bits;
        float* y_batch = Y_out + b * rows;

        for (int64_t i = 0; i < rows; ++i) {
            uint64_t neg_mask = W_sign[i] ^ x_sign;
            uint64_t pos_mask = (~neg_mask) & col_mask;

            int64_t pos_dot = 0;
            int64_t neg_dot = 0;

            for (int32_t m = 0; m < n_bits; ++m) {
                for (int32_t k = 0; k < n_bits; ++k) {
                    uint64_t pm = W_bp[i * (int64_t)n_bits + m] & x_bp[k];
                    int64_t partial_pos = POPCNT64(pm & pos_mask);
                    int64_t partial_neg = POPCNT64(pm & neg_mask);
                    int shift = m + k;
                    int64_t mul = (int64_t)(1ull << shift);
                    pos_dot += partial_pos * mul;
                    neg_dot += partial_neg * mul;
                }
            }

            y_batch[i] = (float)(pos_dot - neg_dot) / scale_prod;
        }
    }
}