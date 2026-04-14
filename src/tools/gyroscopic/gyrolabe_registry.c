/*
 * GyroLabe weight-tensor registry (Q8_0 blocks, structured / residual metadata).
 *
 * Threading: gyrolabe_registry_register_tensor, register_q8_buffer, and
 * gyrolabe_registry_clear are not synchronized. Call them only from a single
 * thread during model load before worker threads read registered tensors.
 */

#include "gyrolabe_registry.h"

#include "gyrolabe.h"
#include "gyrolabe_wht.h"

#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const struct ggml_tensor * w;
    gyrolabe_block_info_t * blocks;
    int n_row_blocks;
    int n_k_blocks;
    float max_scr;
    int owns_tensor;
} gyrolabe_reg_entry;

static gyrolabe_reg_entry * g_entries;
static int g_n;
static int g_cap;
static int64_t g_class_counts[5];

static int gyrolabe_registry_trace_enabled(void) {
    const char * v = getenv("GGML_GYROSCOPIC_TRACE");
    return v != NULL && v[0] != '\0' && v[0] != '0';
}

static int ensure_cap(void) {
    if (g_n < g_cap) {
        return 0;
    }
    {
        int ncap = g_cap ? g_cap * 2 : 64;
        gyrolabe_reg_entry * next = (gyrolabe_reg_entry *) realloc(g_entries, (size_t) ncap * sizeof(gyrolabe_reg_entry));
        if (next == NULL) {
            return -1;
        }
        g_entries = next;
        g_cap = ncap;
    }
    return 0;
}

static int64_t gyrolabe_norm_tensor_dim(int64_t n) {
    return (n < 1) ? 1 : n;
}

static int tensor_matches_registered_weight(const struct ggml_tensor * w, const struct ggml_tensor * reg_w) {
    const size_t exp_q8_row = ggml_row_size(GGML_TYPE_Q8_0, w->ne[0]);
    if (w == NULL || reg_w == NULL) {
        return 0;
    }
    if (w->ne[0] != reg_w->ne[0] || w->ne[1] != reg_w->ne[1]) {
        return 0;
    }
    if (gyrolabe_norm_tensor_dim(w->ne[2]) != gyrolabe_norm_tensor_dim(reg_w->ne[2]) ||
        gyrolabe_norm_tensor_dim(w->ne[3]) != gyrolabe_norm_tensor_dim(reg_w->ne[3])) {
        return 0;
    }
    if (reg_w->nb[1] != exp_q8_row) {
        return 0;
    }
    return 1;
}

static int tensor_signature_matches(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    if (a->type != b->type) {
        return 0;
    }
    if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1]) {
        return 0;
    }
    if (gyrolabe_norm_tensor_dim(a->ne[2]) != gyrolabe_norm_tensor_dim(b->ne[2]) ||
        gyrolabe_norm_tensor_dim(a->ne[3]) != gyrolabe_norm_tensor_dim(b->ne[3])) {
        return 0;
    }
    if (a->nb[1] != b->nb[1]) {
        return 0;
    }
    return 1;
}

static int find_index_by_data(const void * data) {
    int i;
    if (data == NULL) {
        return -1;
    }

    for (i = 0; i < g_n; ++i) {
        if (g_entries[i].w == NULL) {
            continue;
        }
        if (g_entries[i].w->data == data) {
            return i;
        }
    }
    return -1;
}

static void free_block_info_allocs(gyrolabe_block_info_t * info);

static void free_registry_entry(int i) {
    if (i < 0 || i >= g_n) {
        return;
    }

    {
        gyrolabe_block_info_t * bl = g_entries[i].blocks;
        int j;
        if (bl != NULL) {
            const int n_total = g_entries[i].n_row_blocks * g_entries[i].n_k_blocks;
            for (j = 0; j < n_total; ++j) {
                free_block_info_allocs(&bl[j]);
            }
            free(bl);
        }
        if (g_entries[i].owns_tensor && g_entries[i].w != NULL) {
            free((void *) g_entries[i].w);
        }
    }

    if (i + 1 < g_n) {
        g_entries[i] = g_entries[g_n - 1];
    }
    g_entries[g_n - 1].w = NULL;
    g_entries[g_n - 1].blocks = NULL;
    g_entries[g_n - 1].n_row_blocks = 0;
    g_entries[g_n - 1].n_k_blocks = 0;
    g_entries[g_n - 1].max_scr = 0.0f;
    g_entries[g_n - 1].owns_tensor = 0;
    g_n--;
}

static void gyrolabe_registry_register_tensor_impl(const struct ggml_tensor * w, int owns_tensor);

static int find_index(const struct ggml_tensor * w) {
    int i;

    for (i = 0; i < g_n; ++i) {
        if (g_entries[i].w == w) {
            return i;
        }
    }
    return -1;
}

static void set_generic(gyrolabe_block_info_t * info) {
    memset(info, 0, sizeof(*info));
    info->class_id = (uint8_t) GYRO_CLASS_GENERIC;
    info->valid_col_mask = ~0ULL;
    info->ne_orig = 64;
}

static void dequantize_tile_64x64_q8_0(
    const struct ggml_tensor * w,
    int row_block,
    int k_block,
    float * out_64x64
);

/*
 * XOR chi-invariant fit: k[d] = (1/64) sum_i T[i][i^d], P[i][j] = k[i^j].
 * *rel_out = ||T-P||_F / ||T||_F. k_out[64] receives the spatial kernel.
 */
static int popcount6(int x) {
#if defined(_MSC_VER)
    return (int) __popcnt((unsigned int) x);
#else
    return __builtin_popcount((unsigned int) x);
#endif
}

static int is_chi_invariant_f32_exact(const float B[64][64], float eps, float k_out[64]) {
    int i;
    int j;
    for (i = 1; i < 64; ++i) {
        for (j = 0; j < 64; ++j) {
            if (fabsf(B[i][j] - B[0][j ^ i]) > eps) {
                return 0;
            }
        }
    }
    for (j = 0; j < 64; ++j) {
        k_out[j] = B[0][j];
    }
    return 1;
}

static int is_shell_radial_f32_exact(const float k[64], float eps, float shell_k[7]) {
    float sum[7] = {0.0f};
    int cnt[7] = {0};
    int d;
    int r;

    for (d = 0; d < 64; ++d) {
        r = popcount6(d);
        sum[r] += k[d];
        cnt[r] += 1;
    }
    for (r = 0; r < 7; ++r) {
        if (cnt[r] == 0) {
            return 0;
        }
        shell_k[r] = sum[r] / (float) cnt[r];
    }
    for (d = 0; d < 64; ++d) {
        r = popcount6(d);
        if (fabsf(k[d] - shell_k[r]) > eps) {
            return 0;
        }
    }
    return 1;
}

static void reconstruct_chi_invariant_f32(float outP[64][64], const float k[64]) {
    int i;
    int j;
    for (i = 0; i < 64; ++i) {
        for (j = 0; j < 64; ++j) {
            outP[i][j] = k[i ^ j];
        }
    }
}

static void reconstruct_shell_radial_f32(float outP[64][64], const float shell_k[7]) {
    int i;
    int j;
    for (i = 0; i < 64; ++i) {
        for (j = 0; j < 64; ++j) {
            outP[i][j] = shell_k[popcount6(i ^ j)];
        }
    }
}

/*
 * Pack residual D = B - P into packed_DQ for k4_gemv64_avx2. Sets dq_lattice_empty when every
 * exact int32 D[i][j] is zero (see gyrolabe_registry.h).
 */
static int pack_DQ_lattice_f32(const float B[64][64], const float P[64][64], gyrolabe_block_info_t * m, float eps) {
    float dmax = 0.0f;
    int i;
    int j;
    int k;

    for (i = 0; i < 64; ++i) {
        for (j = 0; j < 64; ++j) {
            const float dv = B[i][j] - P[i][j];
            const float av = fabsf(dv);
            if (av > dmax) {
                dmax = av;
            }
        }
    }

    for (i = 0; i < 64; ++i) {
        m->packed_DQ.sign_mask[i] = 0ULL;
        for (k = 0; k < 16; ++k) {
            m->packed_DQ.bitplanes[i][k] = 0ULL;
        }
    }

    if (dmax <= eps) {
        m->packed_DQ.scale_w = 1.0f;
        m->dq_lattice_empty = (uint8_t) 1;
        return 1;
    }

    m->packed_DQ.scale_w = dmax / 32767.0f;
    m->dq_lattice_empty = (uint8_t) 0;

    for (i = 0; i < 64; ++i) {
        for (j = 0; j < 64; ++j) {
            const float dv = B[i][j] - P[i][j];
            float qf;
            int32_t q;
            int32_t mag;
            float reconstructed;

            if (m->packed_DQ.scale_w <= 0.0f) {
                return 0;
            }
            qf = dv / m->packed_DQ.scale_w;
            if (qf > 32767.0f || qf < -32767.0f) {
                return 0;
            }
            q = (int32_t) lrintf(qf);
            reconstructed = (float) q * m->packed_DQ.scale_w;
            if (fabsf(reconstructed - dv) > eps) {
                return 0;
            }

            if (q < 0) {
                m->packed_DQ.sign_mask[i] |= (1ULL << j);
                mag = -q;
            } else {
                mag = q;
            }
            for (k = 0; k < 16; ++k) {
                if (((mag >> k) & 1) != 0) {
                    m->packed_DQ.bitplanes[i][k] |= (1ULL << j);
                }
            }
        }
    }
    return 1;
}

static float unpack_dq_value_f32(const gyrolabe_block_info_t * info, int row, int col) {
    int32_t mag = 0;
    int k;
    for (k = 0; k < 16; ++k) {
        if ((info->packed_DQ.bitplanes[row][k] & (1ULL << col)) != 0ULL) {
            mag |= (1 << k);
        }
    }
    if ((info->packed_DQ.sign_mask[row] & (1ULL << col)) != 0ULL) {
        mag = -mag;
    }
    return (float) mag * info->packed_DQ.scale_w;
}

static void reconstruct_P_from_eigenvalues(float P[64][64], const gyrolabe_block_info_t * info) {
    int i;
    int j;
    if (info->class_id == (uint8_t) GYRO_CLASS_CHI_INVARIANT) {
        float k[64];
        for (i = 0; i < 64; ++i) {
            k[i] = info->eigenvalues.phi_64[i];
        }
        gyrolabe_wht64_f32_inplace(k);
        for (i = 0; i < 64; ++i) {
            k[i] *= (1.0f / 64.0f);
        }
        for (i = 0; i < 64; ++i) {
            for (j = 0; j < 64; ++j) {
                P[i][j] = k[i ^ j];
            }
        }
    } else if (info->class_id == (uint8_t) GYRO_CLASS_SHELL_RADIAL) {
        const int32_t * lam = info->eigenvalues.lambda_7;
        for (i = 0; i < 64; ++i) {
            for (j = 0; j < 64; ++j) {
                P[i][j] = (float) lam[popcount6(i ^ j)];
            }
        }
    } else {
        for (i = 0; i < 64; ++i) {
            for (j = 0; j < 64; ++j) {
                P[i][j] = 0.0f;
            }
        }
    }
}

static void fill_generic_blocks(gyrolabe_block_info_t * blocks, int n_blocks) {
    int b;
    for (b = 0; b < n_blocks; ++b) {
        set_generic(&blocks[b]);
    }
}

static int tensor_total_row_count_4d(const struct ggml_tensor * w) {
    int64_t n = w->ne[1];
    if (w->ne[2] > 1) {
        if (n > LLONG_MAX / w->ne[2]) {
            return INT_MAX;
        }
        n *= w->ne[2];
    }
    if (w->ne[3] > 1) {
        if (n > LLONG_MAX / w->ne[3]) {
            return INT_MAX;
        }
        n *= w->ne[3];
    }
    if (n > INT_MAX) {
        return INT_MAX;
    }
    return (int) n;
}

static void dequantize_tile_64x64_q8_0(
    const struct ggml_tensor * w,
    int row_block,
    int k_block,
    float * out_64x64
) {
    const int row0 = row_block * 64;
    const size_t q8_col_offset = (size_t) k_block * 2u;
    const int total_rows = tensor_total_row_count_4d(w);
    int r;

    for (r = 0; r < 64; ++r) {
        const int global_row = row0 + r;
        int sub;
        int i;
        if (global_row < 0 || global_row >= total_rows) {
            for (i = 0; i < 64; ++i) {
                out_64x64[(size_t) r * 64u + (size_t) i] = 0.0f;
            }
            continue;
        }
        {
            const char * row_ptr = (const char *) w->data + (size_t) global_row * w->nb[1];
            const gyromatmul_block_q8_0 * row = (const gyromatmul_block_q8_0 *) row_ptr + q8_col_offset;
            const int64_t ne0 = w->ne[0];
            for (sub = 0; sub < 2; ++sub) {
                for (i = 0; i < 32; ++i) {
                    const int k_idx = k_block * 64 + sub * 32 + i;
                    if (k_idx < 0 || (int64_t) k_idx >= ne0) {
                        out_64x64[(size_t) r * 64u + (size_t) sub * 32u + (size_t) i] = 0.0f;
                    } else {
                        const float d = ggml_fp16_to_fp32((ggml_fp16_t) row[sub].d);
                        out_64x64[(size_t) r * 64u + (size_t) sub * 32u + (size_t) i] =
                            (float) row[sub].qs[i] * d;
                    }
                }
            }
        }
    }
}

static void free_block_info_allocs(gyrolabe_block_info_t * info) {
    if (info == NULL) return;
    set_generic(info);
}

static void gyrolabe_registry_register_tensor_impl(const struct ggml_tensor * w, int owns_tensor) {
    gyrolabe_block_info_t * blocks;
    int n_k_blocks;
    int n_row_blocks;
    int b;
    int rb;
    int n_rows;

    int existing_idx;

    if (w == NULL) {
        return;
    }
    if (w->type != GGML_TYPE_Q8_0 || w->data == NULL) {
        return;
    }
    /* All Q8_0 tensors register. Class 5 is not a skip, it is D_Q != 0. */
    const size_t expected_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, w->ne[0]);
    const size_t actual_row_bytes = (size_t) (w->nb[1] < 0 ? 0 : w->nb[1]);
    if (actual_row_bytes != expected_row_bytes) {
        const char * name = w->name == NULL ? "(unnamed)" : w->name;
        fprintf(
            stderr,
            "ALARM: INVARIANT VIOLATED. TENSOR '%s' HAS UNEXPECTED MEMORY LAYOUT.\n"
            "  - This is the bug. The registry assumes contiguous rows for dequantization.\n"
            "  - Dimensions: [%lld, %lld, %lld, %lld]\n"
            "  - Strides:    [%lld, %lld, %lld, %lld]\n"
            "  - Expected nb[1]: %zu\n"
            "  - Actual   nb[1]: %zu\n",
            name,
            (long long) w->ne[0],
            (long long) w->ne[1],
            (long long) w->ne[2],
            (long long) w->ne[3],
            (long long) w->nb[0],
            (long long) w->nb[1],
            (long long) w->nb[2],
            (long long) w->nb[3],
            expected_row_bytes,
            actual_row_bytes
        );
        fflush(stderr);
        abort();
    }
    if (w->ne[0] <= 0) {
        return;
    }
    while ((existing_idx = find_index_by_data(w->data)) >= 0) {
        const gyrolabe_reg_entry * existing = &g_entries[existing_idx];
        if (tensor_signature_matches(existing->w, w)) {
            return;
        }
        free_registry_entry(existing_idx);
    }
    if (find_index(w) >= 0) {
        return;
    }

    n_k_blocks = (int) ((w->ne[0] + 63) / 64);
    n_rows = tensor_total_row_count_4d(w);
    if (n_rows <= 0) {
        return;
    }
    n_row_blocks = (n_rows + 63) / 64;
    if (n_row_blocks <= 0) {
        return;
    }

    if (ensure_cap() != 0) {
        return;
    }
    blocks = (gyrolabe_block_info_t *) calloc((size_t) n_row_blocks * (size_t) n_k_blocks, sizeof(gyrolabe_block_info_t));
    if (blocks == NULL) {
        return;
    }
    fill_generic_blocks(blocks, n_row_blocks * n_k_blocks);
    {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        for (rb = 0; rb < n_row_blocks; ++rb) {
            for (b = 0; b < n_k_blocks; ++b) {
                gyrolabe_block_info_t * info = &blocks[(size_t) rb * (size_t) n_k_blocks + (size_t) b];
                float tile[64u * 64u];
                float B[64][64];
                float P[64][64];
                float k[64];
                float shell_k[7];
                float eps = 1.0e-4f;
                int i;
                int j;

                dequantize_tile_64x64_q8_0(w, rb, b, tile);
                for (i = 0; i < 64; ++i) {
                    for (j = 0; j < 64; ++j) {
                        B[i][j] = tile[(size_t) i * 64u + (size_t) j];
                        P[i][j] = 0.0f;
                    }
                }

                if (is_chi_invariant_f32_exact(B, eps, k)) {
                    if (is_shell_radial_f32_exact(k, eps, shell_k)) {
                        info->class_id = (uint8_t) GYRO_CLASS_SHELL_RADIAL;
                        reconstruct_shell_radial_f32(P, shell_k);
                        for (i = 0; i < 7; ++i) {
                            info->eigenvalues.lambda_7[i] = (int32_t) lrintf(shell_k[i]);
                        }
                    } else {
                        float tmp[64];
                        info->class_id = (uint8_t) GYRO_CLASS_CHI_INVARIANT;
                        reconstruct_chi_invariant_f32(P, k);
                        for (j = 0; j < 64; ++j) {
                            tmp[j] = k[j];
                        }
                        gyrolabe_wht64_f32_inplace(tmp);
                        for (j = 0; j < 64; ++j) {
                            info->eigenvalues.phi_64[j] = tmp[j];
                        }
                    }
                } else {
                    info->class_id = (uint8_t) GYRO_CLASS_GENERIC;
                }
                {
                    float P_exact[64][64];
                    int packed_ok;
                    if (info->class_id == (uint8_t) GYRO_CLASS_SHELL_RADIAL ||
                        info->class_id == (uint8_t) GYRO_CLASS_CHI_INVARIANT) {
                        reconstruct_P_from_eigenvalues(P_exact, info);
                    } else {
                        memset(P_exact, 0, sizeof(P_exact));
                    }
                    packed_ok = pack_DQ_lattice_f32(B, P_exact, info, eps);
                    if (!packed_ok && info->class_id != (uint8_t) GYRO_CLASS_GENERIC) {
                        set_generic(info);
                        memset(P_exact, 0, sizeof(P_exact));
                        packed_ok = pack_DQ_lattice_f32(B, P_exact, info, eps);
                    }
                    if (!packed_ok) {
                        memset(&info->packed_DQ, 0, sizeof(info->packed_DQ));
                        info->packed_DQ.scale_w = 1.0f;
                        info->dq_lattice_empty = (uint8_t) 1;
                    }
                }
                info->valid_col_mask = ~0ULL;
                info->ne_orig = 64;
                info->_pad2 = 0;
            }
        }
    }
    g_entries[g_n].w = w;
    g_entries[g_n].blocks = blocks;
    g_entries[g_n].n_row_blocks = n_row_blocks;
    g_entries[g_n].n_k_blocks = n_k_blocks;
    {
        g_entries[g_n].max_scr = 0.0f;
    }
    g_entries[g_n].owns_tensor = owns_tensor;
    {
        int64_t class_counts[5] = {0, 0, 0, 0, 0};
        const int n_total = n_row_blocks * n_k_blocks;
        int idx;
        for (idx = 0; idx < n_total; ++idx) {
            const int cid = (int) blocks[idx].class_id;
            if (cid >= 0 && cid < 5) {
                class_counts[cid] += 1;
                g_class_counts[cid] += 1;
            }
        }
        if (gyrolabe_registry_trace_enabled()) {
            fprintf(
                stderr,
                "REG_STATS: tensor='%s' class0=%lld class1=%lld class2=%lld class3=%lld class4=%lld total=%d agg0=%lld agg1=%lld agg2=%lld agg3=%lld agg4=%lld\n",
                (w->name && w->name[0]) ? w->name : "(unnamed)",
                (long long) class_counts[0],
                (long long) class_counts[1],
                (long long) class_counts[2],
                (long long) class_counts[3],
                (long long) class_counts[4],
                n_total,
                (long long) g_class_counts[0],
                (long long) g_class_counts[1],
                (long long) g_class_counts[2],
                (long long) g_class_counts[3],
                (long long) g_class_counts[4]
            );
        }
    }
    g_n++;
    return;
}

GGML_API void gyrolabe_registry_register_tensor(const struct ggml_tensor * w) {
    gyrolabe_registry_register_tensor_impl(w, 0);
}

GGML_API void gyrolabe_registry_register_q8_buffer(
    const void * data,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    size_t row_stride_bytes,
    const char * name
) {
#ifdef GYROLABE_DEBUG
    static int debug_q8_buffer_calls = 8;
    if (debug_q8_buffer_calls > 0) {
        --debug_q8_buffer_calls;
        fprintf(
            stderr,
            "gyrolabe_registry_REGISTER_Q8_BUFFER: data=%p ne=[%lld,%lld,%lld,%lld] row_stride=%zu\n",
            data,
            (long long) ne0,
            (long long) ne1,
            (long long) ne2,
            (long long) ne3,
            row_stride_bytes
        );
    }
#endif

    struct ggml_tensor * wq;
    const char * tensor_name = "src1_q8data";
    size_t name_len;
    size_t nb2;
    size_t nb3;
    const int64_t ne2n = ne2 < 1 ? 1 : ne2;
    const int64_t ne3n = ne3 < 1 ? 1 : ne3;

    if (data == NULL) {
        return;
    }
    if (ne0 <= 0 || ne1 <= 0 || ne2 < 0 || ne3 < 0 || row_stride_bytes == 0) {
        return;
    }

    {
        const int ix = find_index_by_data(data);
        if (ix >= 0) {
            const struct ggml_tensor * ew = g_entries[ix].w;
            if (ew != NULL && ew->ne[0] == ne0 && ew->ne[1] == ne1 &&
                gyrolabe_norm_tensor_dim(ew->ne[2]) == gyrolabe_norm_tensor_dim(ne2n) &&
                gyrolabe_norm_tensor_dim(ew->ne[3]) == gyrolabe_norm_tensor_dim(ne3n) &&
                (size_t) ew->nb[1] == row_stride_bytes) {
                return;
            }
        }
    }

    if (name != NULL && name[0] != '\0') {
        tensor_name = name;
    }
    if (row_stride_bytes > 0 && ne1 > 0 && ne1 > (int64_t) (SIZE_MAX / row_stride_bytes)) {
        return;
    }
    nb2 = row_stride_bytes * (size_t) ne1;
    if (ne2 > 0 && nb2 > 0 && ne2 > (int64_t) (SIZE_MAX / nb2)) {
        return;
    }
    nb3 = nb2 * (size_t) ne2n;
    if (ne3n > 0 && nb3 > 0 && ne3n > (int64_t) (SIZE_MAX / nb3)) {
        return;
    }

    wq = (struct ggml_tensor *) malloc(sizeof(*wq));
    if (wq == NULL) {
        return;
    }
    memset(wq, 0, sizeof(*wq));
    wq->type = GGML_TYPE_Q8_0;
    wq->buffer = NULL;
    wq->ne[0] = ne0;
    wq->ne[1] = ne1;
    wq->ne[2] = ne2n;
    wq->ne[3] = ne3n;
    wq->nb[0] = ggml_type_size(GGML_TYPE_Q8_0);
    wq->nb[1] = row_stride_bytes;
    wq->nb[2] = nb2;
    wq->nb[3] = nb3;
    wq->data = (void *) data;
    name_len = strlen(tensor_name);
    if (name_len > sizeof(wq->name) - 1) {
        name_len = sizeof(wq->name) - 1;
    }
    memcpy(wq->name, tensor_name, name_len);
    wq->name[name_len] = '\0';

    gyrolabe_registry_register_tensor_impl(wq, 1);
    if (find_index(wq) < 0) {
        free(wq);
    }
}

GGML_API const gyrolabe_block_info_t * gyrolabe_registry_get_block(
    const struct ggml_tensor * w,
    const void * w_q8_base,
    int row_block,
    int k_block
) {
    const struct ggml_tensor * root = w;
    size_t abs_off = 0;
    int ix;
    const void * data_for_lookup;

    while (root != NULL && root->view_src != NULL) {
        abs_off += root->view_offs;
        root = root->view_src;
    }

    data_for_lookup = (w_q8_base != NULL) ? w_q8_base : ((w != NULL) ? (const void *) w->data : NULL);

    ix = find_index(w);
    if (ix < 0) {
        ix = find_index(root);
    }
    if (ix < 0 && w != NULL && data_for_lookup != NULL) {
        const int data_ix = find_index_by_data(data_for_lookup);
        if (data_ix >= 0) {
            if (tensor_matches_registered_weight(w, g_entries[data_ix].w)) {
                ix = data_ix;
            } else {
#ifdef GYROLABE_DEBUG
                static int shape_dbg = 6;
                if (shape_dbg > 0) {
                    --shape_dbg;
                    fprintf(
                        stderr,
                        "gyrolabe_get_block shape_mismatch: reg_nb1=%zu need=%zu ne=[%lld,%lld,%lld,%lld]\n",
                        (size_t) g_entries[data_ix].w->nb[1],
                        (size_t) ggml_row_size(GGML_TYPE_Q8_0, w->ne[0]),
                        (long long) w->ne[0],
                        (long long) w->ne[1],
                        (long long) w->ne[2],
                        (long long) w->ne[3]
                    );
                }
#endif
            }
        }
    }
    if (ix < 0) {
#ifdef GYROLABE_DEBUG
        static int miss_dbg = 6;
        if (miss_dbg > 0 && w != NULL) {
            --miss_dbg;
            fprintf(
                stderr,
                "gyrolabe_get_block miss: g_n=%d w_data=%p lookup=%p ne0=%lld nb1=%lld\n",
                g_n,
                (void *) w->data,
                data_for_lookup,
                (long long) w->ne[0],
                (long long) w->nb[1]
            );
            if (g_n > 0 && g_entries[0].w != NULL) {
                fprintf(
                    stderr,
                    "  entry0 data=%p ne0=%lld nb1=%lld\n",
                    (void *) g_entries[0].w->data,
                    (long long) g_entries[0].w->ne[0],
                    (long long) g_entries[0].w->nb[1]
                );
            }
        }
#endif
        return NULL;
    }

    if (root != NULL && w != NULL && root != w) {
        const struct ggml_tensor * reg_w = g_entries[ix].w;
        if (reg_w == root && root->nb[1] > 0) {
            if (abs_off % (size_t) root->nb[1] != 0) {
                return NULL;
            }
            {
                const int64_t view_row0 = (int64_t)(abs_off / (size_t) root->nb[1]);
                if ((view_row0 & 63) != 0) {
                    return NULL;
                }
                {
                    const int row_shift = (int)(view_row0 / 64);
                    const int64_t rb = (int64_t) row_shift + (int64_t) row_block;
                    if (rb < 0 || rb > INT_MAX) {
                        return NULL;
                    }
                    row_block = (int) rb;
                }
            }
        }
    }

    if (row_block < 0 || row_block >= g_entries[ix].n_row_blocks) {
#ifdef GYROLABE_DEBUG
        static int rb_dbg = 8;
        if (rb_dbg > 0) {
            --rb_dbg;
            fprintf(
                stderr,
                "gyrolabe_get_block row_oob: row_block=%d n_row_blocks=%d k_block=%d n_k_blocks=%d ix=%d\n",
                row_block,
                g_entries[ix].n_row_blocks,
                k_block,
                g_entries[ix].n_k_blocks,
                ix
            );
        }
#endif
        return NULL;
    }
    if (k_block < 0 || k_block >= g_entries[ix].n_k_blocks) {
#ifdef GYROLABE_DEBUG
        static int kb_dbg = 8;
        if (kb_dbg > 0) {
            --kb_dbg;
            fprintf(
                stderr,
                "gyrolabe_get_block k_oob: row_block=%d k_block=%d n_k_blocks=%d\n",
                row_block,
                k_block,
                g_entries[ix].n_k_blocks
            );
        }
#endif
    }
    return &g_entries[ix].blocks[
        (size_t) row_block * (size_t) g_entries[ix].n_k_blocks +
        (size_t) k_block
    ];
}

GGML_API gyrolabe_registry_entry_t gyrolabe_registry_find_entry(const struct ggml_tensor * w, const void * data) {
    int ix;
    const struct ggml_tensor * root = w;
    while (root != NULL && root->view_src != NULL) {
        root = root->view_src;
    }
    ix = find_index(w);
    if (ix < 0) ix = find_index(root);
    if (ix < 0 && data != NULL) ix = find_index_by_data(data);
    if (ix < 0) return NULL;
    return (gyrolabe_registry_entry_t) &g_entries[ix];
}

GGML_API const gyrolabe_block_info_t * gyrolabe_registry_get_block_from_entry(
    gyrolabe_registry_entry_t entry,
    int row_block,
    int k_block
) {
    gyrolabe_reg_entry * ent = (gyrolabe_reg_entry *) entry;
    if (ent == NULL) return NULL;
    if (row_block < 0 || row_block >= ent->n_row_blocks) return NULL;
    if (k_block < 0 || k_block >= ent->n_k_blocks) return NULL;
    return &ent->blocks[(size_t) row_block * (size_t) ent->n_k_blocks + (size_t) k_block];
}

GGML_API float gyrolabe_registry_tensor_max_scr(const struct ggml_tensor * w) {
    const int ix = find_index(w);
    if (ix < 0) {
        return 0.0f;
    }
    return g_entries[ix].max_scr;
}

GGML_API int gyrolabe_registry_n_blocks(int n_cols) {
    if (n_cols <= 0 || (n_cols % 64) != 0) {
        return 0;
    }
    return n_cols / 64;
}

GGML_API int gyrolabe_registry_entry_count(void) {
    return g_n;
}

GGML_API void gyrolabe_registry_clear(void) {
    while (g_n > 0) {
        free_registry_entry(0);
    }
    free(g_entries);
    g_entries = NULL;
    g_n = 0;
    g_cap = 0;
}

