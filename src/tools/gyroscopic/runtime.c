/*
 * Native Gyroscopic Inference Loop: request cells + genealogy (runtime.c).
 *
 * Byte cadence and word closure follow Runtime Specs 12.3; the O(1) ring and
 * histogram updates follow 8.5 with one shared ring position. The ingest log
 * is the append-only (cell_id, word4) ledger of 16.4 with request-boundary
 * markers. Pure primitives never touch the log; chassis paths do.
 */

#include "runtime.h"
#include "kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
static int rt_popcount8(uint8_t x) { return (int) __popcnt((unsigned) x); }
#elif defined(__GNUC__) || defined(__clang__)
static int rt_popcount8(uint8_t x) { return (int) __builtin_popcount((unsigned) x); }
#else
static int rt_popcount8(uint8_t x) {
    int n = 0;
    while (x) { n += (int) (x & 1u); x >>= 1; }
    return n;
}
#endif

/* ------------------------------------------------------------------ */
/* Rule hash                                                           */
/* ------------------------------------------------------------------ */

static uint64_t rt_fnv1a_step(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    size_t i;
    for (i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= (uint64_t) 0x100000001b3ull;
    }
    return h;
}

uint64_t hqvm_rt_rule_hash(void) {
    /* FNV-1a over rule surfaces + probes of the live kernel tables. */
    const uint32_t magic   = 0x52544C50u; /* "RTLP" */
    const uint32_t version = 1u;
    const uint8_t  closure_word[4] = {
        HQVM_W2_BYTE0, HQVM_W2_BYTE1, HQVM_W2P_BYTE0, HQVM_W2P_BYTE1
    };
    uint64_t h = (uint64_t) 0xcbf29ce484222325ull;
    uint16_t probe_s12 = 0;
    int b;

    h = rt_fnv1a_step(h, &magic, sizeof(magic));
    h = rt_fnv1a_step(h, &version, sizeof(version));

    /* omega12 chart: sweep all byte values from rest. */
    probe_s12 = 0;
    for (b = 0; b < 256; ++b) {
        probe_s12 = hqvm_step_state12_by_byte(probe_s12, (uint8_t) b);
        h = rt_fnv1a_step(h, &probe_s12, sizeof(probe_s12));
    }

    /* sig13 surface: compile the canonical depth-4 closure word. */
    {
        const uint16_t sig = hqvm_sig13_compile(closure_word, 4);
        h = rt_fnv1a_step(h, &sig, sizeof(sig));
    }

    /* state24 <-> (u,v) projection probes. */
    for (b = 0; b < 8; ++b) {
        const uint32_t s24 =
            hqvm_uv6_to_state24((uint8_t) (b * 7u), (uint8_t) (63u - b * 7u));
        h = rt_fnv1a_step(h, &s24, sizeof(s24));
    }

    /* Byte decomposition table fingerprint. */
    if (!hqvm_byte_table_ok()) {
        hqvm_byte_table_init();
    }
    for (b = 0; b < 256; ++b) {
        hqvm_byte_fiber f;
        hqvm_decompose_byte((uint8_t) b, &f);
        h = rt_fnv1a_step(h, &f.q6, sizeof(f.q6));
        h = rt_fnv1a_step(h, &f.family, sizeof(f.family));
    }
    return h;
}

/* ------------------------------------------------------------------ */
/* Cell lifecycle + O(1) local structural memory                       */
/* ------------------------------------------------------------------ */

void hqvm_rt_cell_init(hqvm_rt_cell * c, int seed_mode, uint8_t a, uint8_t b) {
    if (!c) return;
    memset(c, 0, sizeof(*c));

    switch (seed_mode) {
    case HQVM_RT_SEED_EQUALITY_HORIZON:
        c->omega12 = hqvm_pack_state12(0, 0);
        break;
    case HQVM_RT_SEED_SHELL: {
        /* Lowest-value representative at shell a on the diagonal (u=v=chi). */
        uint8_t chi = 0;
        int k, set = 0;
        if ((int) a > SHELL_MAX) a = (uint8_t) SHELL_MAX;
        for (k = 0; k < 6 && set < (int) a; ++k) {
            chi |= (uint8_t) (1u << k);
            ++set;
        }
        c->omega12 = hqvm_pack_state12(chi, chi);
        break;
    }
    case HQVM_RT_SEED_OMEGA:
        c->omega12 = hqvm_pack_state12(a & CHIRALITY_MASK_6, b & CHIRALITY_MASK_6);
        break;
    case HQVM_RT_SEED_REST:
    default: {
        /* Rest = complement-horizon representative: every dipole aligned,
         * i.e. chi = 63 (u = 63, v = 0). Matches GENE_MAC_REST projection. */
        c->omega12 = hqvm_pack_state12(CHIRALITY_MASK_6, 0);
        break;
    }
    }
}

/* One shared ring position for both rings; valid length counts bytes.
 * Chi values feed the shell histogram, family values the 4-bucket histogram
 * (Spec 8.5: both memories update together in O(1)). */
static void rt_memory_step(
    hqvm_rt_cell * c,
    uint8_t        chi,
    uint8_t        fam)
{
    const uint8_t full = (c->ring_valid_len >= HORIZON_SIZE) ? 1u : 0u;
    const uint8_t pos = c->ring_pos;

    if (full) {
        const uint8_t old_chi = c->chi_ring64[pos];
        const uint8_t old_fam = c->family_ring64[pos];

        c->chi_hist64[old_chi] = (uint16_t) (c->chi_hist64[old_chi] - 1u);
        c->family_hist4[old_fam] = (uint16_t) (c->family_hist4[old_fam] - 1u);
        c->shell_hist7[rt_popcount8(old_chi)] =
            (uint16_t) (c->shell_hist7[rt_popcount8(old_chi)] - 1u);
    } else {
        c->ring_valid_len = (uint8_t) (c->ring_valid_len + 1u);
    }

    c->chi_ring64[pos] = chi;
    c->family_ring64[pos] = fam;
    c->chi_hist64[chi] = (uint16_t) (c->chi_hist64[chi] + 1u);
    c->family_hist4[fam] = (uint16_t) (c->family_hist4[fam] + 1u);
    c->shell_hist7[rt_popcount8(chi)] =
        (uint16_t) (c->shell_hist7[rt_popcount8(chi)] + 1u);
    c->ring_pos = (uint8_t) ((pos + 1u) & 63u);
}

static void rt_byte_cadence(hqvm_rt_cell * c, uint8_t byte, uint8_t position_in_word) {
    uint8_t u;
    uint8_t v;
    uint8_t chi;
    uint8_t intron;
    uint8_t fam;
    uint16_t mask12;
    const uint8_t q6 = hqvm_q6_of_byte(byte);
    int j;

    c->omega12 = hqvm_step_state12_by_byte(c->omega12, byte);
    c->step += 1u;
    c->last_byte = byte;

    u = (uint8_t) ((c->omega12 >> 6) & CHIRALITY_MASK_6);
    v = (uint8_t) (c->omega12 & CHIRALITY_MASK_6);
    chi = hqvm_chi6_uv(u, v);

    /* Family ring advances at the same shared ring position (Spec 8.5). */
    intron = (uint8_t) (byte ^ (int) GENE_MIC_S);
    fam = (uint8_t) ((intron & 1u) | ((intron >> 6) & 2u));
    rt_memory_step(c, chi, fam);

    /* Parity commitment: XOR of the byte's mutation mask by position parity.
     * mask12 = micro_ref expanded to bit pairs, identical to kernel m12. */
    mask12 = 0;
    for (j = 0; j < 6; ++j) {
        if ((q6 >> j) & 1u) {
            mask12 |= (uint16_t) (0x3u << (2 * j));
        }
    }
    if (position_in_word & 1u) {
        c->parity_O12 ^= mask12;
    } else {
        c->parity_E12 ^= mask12;
    }
    c->parity_bit = (uint8_t) (rt_popcount8((uint8_t) ((c->parity_O12 ^ c->parity_E12) & 0xFFu))
        + rt_popcount8((uint8_t) (((c->parity_O12 ^ c->parity_E12) >> 8) & 0xFFu))) & 1u;
}

static void rt_word_closure(hqvm_rt_cell * c, const uint8_t word4[4]) {
    if (!c || !word4) return;
    memcpy(c->word4, word4, 4);
    c->has_closed_word = 1;
    c->word_len = 0;
    c->omega_sig = (int32_t) hqvm_sig13_compile(word4, 4);
    c->resonance_key = hqvm_rt_resonance_key_of(c, HQVM_RT_PROFILE_CHIRALITY);
}

void hqvm_rt_ingest_word(hqvm_rt_cell * c, const uint8_t word4[4]) {
    int k;

    if (!c || !word4) return;

    for (k = 0; k < 4; ++k) {
        rt_byte_cadence(c, word4[k], (uint8_t) k);
    }
    rt_word_closure(c, word4);
}

void hqvm_rt_ingest_bytes(hqvm_rt_cell * c, const uint8_t * bytes, int n) {
    int i;

    if (!c || !bytes || n <= 0) return;

    for (i = 0; i < n; ++i) {
        const uint8_t pos = (uint8_t) (c->word_len & 3u);
        rt_byte_cadence(c, bytes[i], pos);
        c->open_word[c->word_len++] = bytes[i];
        if (c->word_len == 4) {
            rt_word_closure(c, c->open_word);
        }
    }
}

uint32_t hqvm_rt_resonance_key_of(const hqvm_rt_cell * c, uint16_t profile_id) {
    uint8_t u;
    uint8_t v;

    if (!c) return 0;
    u = (uint8_t) ((c->omega12 >> 6) & CHIRALITY_MASK_6);
    v = (uint8_t) (c->omega12 & CHIRALITY_MASK_6);

    if (profile_id == HQVM_RT_PROFILE_SHELL) {
        return (uint32_t) rt_popcount8(hqvm_chi6_uv(u, v));
    }
    return (uint32_t) hqvm_chi6_uv(u, v);
}

int hqvm_rt_chi_distance(const hqvm_rt_cell * a, const hqvm_rt_cell * b) {
    uint8_t chi_a;
    uint8_t chi_b;

    if (!a || !b) return 7;
    chi_a = hqvm_chi6_uv((uint8_t) ((a->omega12 >> 6) & CHIRALITY_MASK_6),
        (uint8_t) (a->omega12 & CHIRALITY_MASK_6));
    chi_b = hqvm_chi6_uv((uint8_t) ((b->omega12 >> 6) & CHIRALITY_MASK_6),
        (uint8_t) (b->omega12 & CHIRALITY_MASK_6));
    return rt_popcount8((uint8_t) ((chi_a ^ chi_b) & CHIRALITY_MASK_6));
}

/* Decode batch grouping per Runtime 20.2. Insertion sort keeps the small-n
 * decode path branchy-but-cheap and stable; group head = first member. */
typedef struct rt_group_sort_ent {
    uint32_t key;
    uint8_t  chi;
    uint16_t cell_idx;
} rt_group_sort_ent;

int hqvm_rt_group_cells(
    const hqvm_rt_cell * const * cells,
    uint32_t                     n,
    uint32_t                     max_batch,
    uint16_t                   * out_group_ids)
{
    rt_group_sort_ent * ents = NULL;
    uint16_t          * head_idx = NULL;
    uint32_t            n_groups = 0;
    uint32_t            i;

    if (!cells || !out_group_ids || n == 0) return -1;
    if (max_batch == 0) max_batch = 1;

    ents = (rt_group_sort_ent *) malloc((size_t) n * sizeof(*ents));
    head_idx = (uint16_t *) malloc((size_t) n * sizeof(*head_idx));
    if (!ents || !head_idx) {
        free(ents);
        free(head_idx);
        return -1;
    }

    for (i = 0; i < n; ++i) {
        uint8_t u, v;
        if (!cells[i]) {
            free(ents);
            free(head_idx);
            return -1;
        }
        u = (uint8_t) ((cells[i]->omega12 >> 6) & CHIRALITY_MASK_6);
        v = (uint8_t) (cells[i]->omega12 & CHIRALITY_MASK_6);
        ents[i].key = cells[i]->resonance_key;
        ents[i].chi = hqvm_chi6_uv(u, v);
        ents[i].cell_idx = (uint16_t) i;
    }

    /* Stable insertion sort by (key, chi). n is a decode batch (small). */
    for (i = 1; i < n; ++i) {
        rt_group_sort_ent e = ents[i];
        uint32_t j = i;
        while (j > 0 && (ents[j - 1].key > e.key ||
                         (ents[j - 1].key == e.key && ents[j - 1].chi > e.chi))) {
            ents[j] = ents[j - 1];
            --j;
        }
        ents[j] = e;
    }

    for (i = 0; i < n; ++i) {
        uint32_t g;
        int placed = 0;
        for (g = 0; g < n_groups && !placed; ++g) {
            /* size of group g: recomputed cheaply from ids already written */
            uint32_t size = 0;
            uint32_t k;
            const rt_group_sort_ent * head;
            for (k = 0; k < i; ++k) {
                if (out_group_ids[ents[k].cell_idx] == g) ++size;
            }
            if (size >= max_batch) continue;
            head = &ents[head_idx[g]];
            if (ents[i].key == head->key ||
                rt_popcount8((uint8_t) ((ents[i].chi ^ head->chi) & CHIRALITY_MASK_6)) <= 2) {
                out_group_ids[ents[i].cell_idx] = (uint16_t) g;
                placed = 1;
            }
        }
        if (!placed) {
            out_group_ids[ents[i].cell_idx] = (uint16_t) n_groups;
            head_idx[n_groups] = (uint16_t) i;
            ++n_groups;
        }
    }

    free(ents);
    free(head_idx);
    return (int) n_groups;
}

/* ------------------------------------------------------------------ */
/* Cell pool                                                           */
/* ------------------------------------------------------------------ */

hqvm_rt_pool * hqvm_rt_pool_create(uint32_t capacity, uint16_t profile_id) {
    hqvm_rt_pool * pool;
    uint32_t i;

    if (capacity == 0) return NULL;
    pool = (hqvm_rt_pool *) malloc(sizeof(*pool));
    if (!pool) return NULL;
    pool->cells = (hqvm_rt_cell *) calloc((size_t) capacity, sizeof(hqvm_rt_cell));
    if (!pool->cells) {
        free(pool);
        return NULL;
    }
    pool->capacity = capacity;
    pool->profile_id = profile_id;
    for (i = 0; i < capacity; ++i) {
        hqvm_rt_cell_init(&pool->cells[i], HQVM_RT_SEED_REST, 0, 0);
    }
    return pool;
}

void hqvm_rt_pool_free(hqvm_rt_pool * pool) {
    if (!pool) return;
    free(pool->cells);
    free(pool);
}

hqvm_rt_cell * hqvm_rt_pool_cell(hqvm_rt_pool * pool, uint32_t cell_id) {
    if (!pool || cell_id >= pool->capacity) return NULL;
    return &pool->cells[cell_id];
}

/* ------------------------------------------------------------------ */
/* Append-only genealogy log                                           */
/* ------------------------------------------------------------------ */

static FILE   * g_log_file       = NULL;
static uint64_t g_log_events     = 0;
static uint64_t g_log_requests   = 0;
static int      g_enabled_cached = -1;

int hqvm_rt_enabled(void) {
    if (g_enabled_cached < 0) {
        const char * e = getenv("GYRO_NATIVE_GENEALOGY");
        g_enabled_cached = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return g_enabled_cached;
}

static void rt_log_write(uint32_t cell_id, const uint8_t * word4) {
    uint8_t rec[8];

    if (!g_log_file) return;
    rec[0] = (uint8_t) (cell_id & 0xFFu);
    rec[1] = (uint8_t) ((cell_id >> 8) & 0xFFu);
    rec[2] = (uint8_t) ((cell_id >> 16) & 0xFFu);
    rec[3] = (uint8_t) ((cell_id >> 24) & 0xFFu);
    rec[4] = word4[0];
    rec[5] = word4[1];
    rec[6] = word4[2];
    rec[7] = word4[3];
    if (fwrite(rec, 1, sizeof(rec), g_log_file) == sizeof(rec)) {
        g_log_events++;
        fflush(g_log_file);
    }
}

int hqvm_rt_log_configure(const char * path) {
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
    g_log_events = 0;
    g_log_requests = 0;
    if (!path || !path[0]) {
        return 0;
    }
    g_log_file = fopen(path, "ab");
    if (!g_log_file) {
        return -1;
    }
    setvbuf(g_log_file, NULL, _IOFBF, 4096);
    return 0;
}

int hqvm_rt_log_begin_session(uint32_t seed_mode) {
    hqvm_rt_snapshot_header hdr;
    uint32_t magic = 0;
    if (!g_log_file) return 0;
    if (fseek(g_log_file, 0, SEEK_SET) != 0) return -1;
    if (fread(&magic, sizeof(magic), 1, g_log_file) == 1 &&
        magic == HQVM_RT_SNAPSHOT_MAGIC) {
        fseek(g_log_file, 0, SEEK_END);
        return 0;
    }
    rewind(g_log_file);
    hqvm_rt_snapshot_header_fill(&hdr, seed_mode);
    if (fwrite(&hdr, sizeof(hdr), 1, g_log_file) != 1) return -1;
    fflush(g_log_file);
    return 0;
}

void hqvm_rt_log_close(void) {
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

uint64_t hqvm_rt_log_events(void) { return g_log_events; }
uint64_t hqvm_rt_log_requests(void) { return g_log_requests; }

/* ------------------------------------------------------------------ */
/* Snapshot header + cell checkpoint                                   */
/* ------------------------------------------------------------------ */

void hqvm_rt_snapshot_header_fill(hqvm_rt_snapshot_header * hdr, uint32_t seed_mode) {
    if (!hdr) return;
    hdr->magic = HQVM_RT_SNAPSHOT_MAGIC;
    hdr->version = HQVM_RT_SNAPSHOT_VERSION;
    hdr->seed_mode = seed_mode;
    hdr->reserved = 0u;
    hdr->n_events = g_log_events;
    hdr->n_requests = g_log_requests;
    hdr->rule_hash = hqvm_rt_rule_hash();
}

/* 16-byte exact checkpoint: omega12, parity pair, step (mod 2^16), and the
 * packed histogram sums. Replay must reproduce every byte of this. */
void hqvm_rt_cell_checkpoint(const hqvm_rt_cell * c, uint8_t out[16]) {
    uint16_t chi_sum = 0;
    uint16_t shell_sum = 0;
    uint16_t family_sum = 0;
    int i;

    if (!c || !out) return;
    for (i = 0; i < 64; ++i) {
        chi_sum = (uint16_t) (chi_sum + c->chi_hist64[i]);
    }
    for (i = 0; i < 7; ++i) {
        shell_sum = (uint16_t) (shell_sum + c->shell_hist7[i]);
    }
    for (i = 0; i < 4; ++i) {
        family_sum = (uint16_t) (family_sum + c->family_hist4[i]);
    }

    memset(out, 0, 16);
    out[0] = (uint8_t) (c->omega12 & 0xFFu);
    out[1] = (uint8_t) ((c->omega12 >> 8) & 0xFFu);
    out[2] = (uint8_t) (c->parity_O12 & 0xFFu);
    out[3] = (uint8_t) ((c->parity_O12 >> 8) & 0xFFu);
    out[4] = (uint8_t) (c->parity_E12 & 0xFFu);
    out[5] = (uint8_t) ((c->parity_E12 >> 8) & 0xFFu);
    out[6] = (uint8_t) ((c->step >> 8) & 0xFFu);   /* step mod 2^48 >> ... */
    out[7] = (uint8_t) ((c->step >> 16) & 0xFFu);
    out[8] = c->last_byte;
    out[9] = c->has_closed_word ? 1u : 0u;
    out[10] = (uint8_t) (chi_sum & 0xFFu);
    out[11] = (uint8_t) ((chi_sum >> 8) & 0xFFu);
    out[12] = (uint8_t) (shell_sum & 0xFFu);
    out[13] = (uint8_t) (family_sum & 0xFFu);
    out[14] = c->ring_pos;
    out[15] = c->ring_valid_len;
}

/* ------------------------------------------------------------------ */
/* SLCP emission (Runtime 13) + graph queries (Runtime 14)             */
/* ------------------------------------------------------------------ */

static int rt_popcount12(uint16_t x) {
    return rt_popcount8((uint8_t) (x & 0xFFu)) + rt_popcount8((uint8_t) ((x >> 8) & 0xFFu));
}

static int32_t rt_bucket_weight(const hqvm_rt_pool * pool, uint32_t key) {
    uint32_t i, n = 0;
    if (!pool) return 0;
    for (i = 0; i < pool->capacity; ++i) {
        if (pool->cells[i].step > 0 && pool->cells[i].resonance_key == key) ++n;
    }
    return (int32_t) n;
}

void hqvm_rt_slcp_fill(
    const hqvm_rt_cell * c,
    uint32_t             cell_id,
    const hqvm_rt_pool * pool,
    hqvm_rt_slcp_t *     out)
{
    uint8_t u, v, chi;
    uint16_t a12, b12;
    hqvm_byte_fiber fiber;
    float spec[64];
    double sum = 0.0;
    int i;

    if (!c || !out) return;
    memset(out, 0, sizeof(*out));

    u = (uint8_t) ((c->omega12 >> 6) & CHIRALITY_MASK_6);
    v = (uint8_t) (c->omega12 & CHIRALITY_MASK_6);
    chi = hqvm_chi6_uv(u, v);
    a12 = (uint16_t) ((c->omega12 >> 6) & LAYER_MASK_12);
    b12 = (uint16_t) (c->omega12 & LAYER_MASK_12);

    if (!hqvm_byte_table_ok()) hqvm_byte_table_init();
    hqvm_decompose_byte(c->last_byte, &fiber);

    out->cell_id = cell_id;
    out->step = c->step;
    out->omega12 = (int32_t) c->omega12;
    out->state24 = (int32_t) c->omega12;
    out->last_byte = c->last_byte;
    out->family = (int32_t) fiber.family;
    out->micro_ref = (int32_t) ((fiber.intron >> 1) & 0x3Fu);
    out->q6 = (int32_t) fiber.q6;
    out->chi6 = (int32_t) chi;
    out->shell = rt_popcount8(chi);
    out->horizon_distance = rt_popcount12((uint16_t) (a12 ^ (b12 ^ COMPLEMENT_MASK_12)));
    out->ab_distance = rt_popcount12((uint16_t) (a12 ^ b12));
    out->resonance_key = c->resonance_key;
    out->current_resonance = rt_bucket_weight(pool, c->resonance_key);

    if (c->has_closed_word) {
        out->omega_sig = c->omega_sig;
        out->parity_O12 = c->parity_O12;
        out->parity_E12 = c->parity_E12;
        out->parity_bit = c->parity_bit;
    }

    for (i = 0; i < 64; ++i) {
        spec[i] = (float) c->chi_hist64[i];
        sum += (double) spec[i];
    }
    if (sum > 0.0) {
        for (i = 0; i < 64; ++i) spec[i] = (float) ((double) spec[i] / sum);
    }
    gyroscopic_wht64_float(spec);
    memcpy(out->spectral64, spec, sizeof(out->spectral64));
}

int32_t hqvm_rt_bucket_population(const hqvm_rt_pool * pool, uint32_t key) {
    return rt_bucket_weight(pool, key);
}

static int rt_pool_scan(
    const hqvm_rt_pool * pool,
    int (*match)(const hqvm_rt_cell *, const void *),
    const void * ctx,
    uint32_t * out_ids,
    int max_out)
{
    int n = 0;
    uint32_t i;
    if (!pool || max_out <= 0) return 0;
    for (i = 0; i < pool->capacity && n < max_out; ++i) {
        if (pool->cells[i].step == 0) continue;
        if (match(&pool->cells[i], ctx)) {
            if (out_ids) out_ids[n] = i;
            ++n;
        }
    }
    return n;
}

static int rt_match_key(const hqvm_rt_cell * c, const void * ctx) {
    const uint32_t * key = (const uint32_t *) ctx;
    return c->resonance_key == *key;
}

static int rt_match_shell(const hqvm_rt_cell * c, const void * ctx) {
    const int * shell = (const int *) ctx;
    uint8_t u = (uint8_t) ((c->omega12 >> 6) & CHIRALITY_MASK_6);
    uint8_t v = (uint8_t) (c->omega12 & CHIRALITY_MASK_6);
    return rt_popcount8(hqvm_chi6_uv(u, v)) == *shell;
}

static int rt_match_chi6(const hqvm_rt_cell * c, const void * ctx) {
    const uint8_t * chi = (const uint8_t *) ctx;
    uint8_t u = (uint8_t) ((c->omega12 >> 6) & CHIRALITY_MASK_6);
    uint8_t v = (uint8_t) (c->omega12 & CHIRALITY_MASK_6);
    return hqvm_chi6_uv(u, v) == *chi;
}

static int rt_match_sig(const hqvm_rt_cell * c, const void * ctx) {
    const int32_t * sig = (const int32_t *) ctx;
    return c->has_closed_word && c->omega_sig == *sig;
}

int hqvm_rt_bucket_cells(
    const hqvm_rt_pool * pool, uint32_t key, uint32_t * out_ids, int max_out)
{
    return rt_pool_scan(pool, rt_match_key, &key, out_ids, max_out);
}

int hqvm_rt_co_resonant_count(const hqvm_rt_pool * pool, uint32_t cell_id) {
    const hqvm_rt_cell * c;
    if (!pool || cell_id >= pool->capacity) return 0;
    c = &pool->cells[cell_id];
    if (c->step == 0) return 0;
    return (int) rt_bucket_weight(pool, c->resonance_key);
}

int hqvm_rt_cells_on_shell(
    const hqvm_rt_pool * pool, int shell, uint32_t * out_ids, int max_out)
{
    return rt_pool_scan(pool, rt_match_shell, &shell, out_ids, max_out);
}

int hqvm_rt_cells_with_chi6(
    const hqvm_rt_pool * pool, uint8_t chi6, uint32_t * out_ids, int max_out)
{
    return rt_pool_scan(pool, rt_match_chi6, &chi6, out_ids, max_out);
}

int hqvm_rt_cells_with_signature(
    const hqvm_rt_pool * pool, int32_t omega_sig, uint32_t * out_ids, int max_out)
{
    return rt_pool_scan(pool, rt_match_sig, &omega_sig, out_ids, max_out);
}

/* ------------------------------------------------------------------ */
/* Polar prefilter + decode-consultation counters                      */
/* ------------------------------------------------------------------ */

static uint64_t rt_popcount64(uint64_t x) {
#if defined(_MSC_VER)
    return (uint64_t) __popcnt64(x);
#elif defined(__GNUC__) || defined(__clang__)
    return (uint64_t) __builtin_popcountll(x);
#else
    uint64_t n = 0;
    while (x) { n += (x & 1u); x >>= 1; }
    return n;
#endif
}

float hqvm_rt_polar_score(const hqvm_rt_polar_summary * q, const hqvm_rt_polar_summary * k) {
    int chi_dist;
    float shell_sim;
    float anchor_align;

    if (!q || !k || q->radius <= 0.0f || k->radius <= 0.0f) return 0.0f;
    chi_dist = rt_popcount8((uint8_t) ((q->chi6 ^ k->chi6) & CHIRALITY_MASK_6));
    shell_sim = (float) (SHELL_MAX - chi_dist) / (float) SHELL_MAX;
    anchor_align = 1.0f - (float) rt_popcount64(q->anchor64 ^ k->anchor64) / 64.0f;
    return q->radius * k->radius * shell_sim * anchor_align;
}

/* Counters for claimed-native decode-consultation sites. stock_ops_total is
 * the sum of every counted stock operation this request; a site claiming
 * nativeness must show zero stock work there (receipts gate).
 * Grouping counters (Runtime 20.2) sit in the same request-scoped block. */
static int      s_group_env = -1;
static uint64_t s_group_calls = 0;
static uint64_t s_group_rows = 0;
static uint64_t s_group_groups = 0;
static uint64_t s_rt_stock_ops_total = 0;
static uint64_t s_rt_prefilter_calls = 0;
static uint64_t s_rt_prefilter_skipped = 0;

void hqvm_rt_stock_ops_add(uint32_t n) { s_rt_stock_ops_total += n; }
uint64_t hqvm_rt_stock_ops_total(void) { return s_rt_stock_ops_total; }

void hqvm_rt_prefilter_inc(void) { s_rt_prefilter_calls++; }

void hqvm_rt_prefilter_report(int64_t n_candidates, int64_t n_kept) {
    s_rt_prefilter_calls++;
    if (n_candidates > n_kept && n_kept >= 0) {
        s_rt_prefilter_skipped += (uint64_t) (n_candidates - n_kept);
    }
}

uint64_t hqvm_rt_prefilter_skipped(void) { return s_rt_prefilter_skipped; }
uint64_t hqvm_rt_prefilter_calls(void) { return s_rt_prefilter_calls; }

void hqvm_rt_counters_request_reset(void) {
    s_rt_stock_ops_total = 0;
    s_rt_prefilter_skipped = 0;
    s_group_calls = 0;
    s_group_rows = 0;
    s_group_groups = 0;
}

/* ------------------------------------------------------------------ */
/* Chassis decode grouping (Runtime 20.2), GYRO_NATIVE_GROUP-gated     */
/* ------------------------------------------------------------------ */

int hqvm_rt_group_enabled(void) {
    if (s_group_env < 0) {
        const char * e = getenv("GYRO_NATIVE_GROUP");
        s_group_env = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s_group_env;
}

void hqvm_rt_group_report(int64_t rows, int64_t groups) {
    if (rows < 0 || groups < 0 || groups > rows) return;
    s_group_calls++;
    s_group_rows += (uint64_t) rows;
    s_group_groups += (uint64_t) groups;
}

uint64_t hqvm_rt_group_calls(void) { return s_group_calls; }
uint64_t hqvm_rt_group_rows(void) { return s_group_rows; }
uint64_t hqvm_rt_group_groups(void) { return s_group_groups; }

/* ------------------------------------------------------------------ */
/* Pool ingestion with logging                                         */
/* ------------------------------------------------------------------ */

int hqvm_rt_pool_ingest_word(hqvm_rt_pool * pool, uint32_t cell_id, const uint8_t word4[4]) {
    hqvm_rt_cell * cell = hqvm_rt_pool_cell(pool, cell_id);
    if (!cell || !word4) return -1;
    hqvm_rt_ingest_word(cell, word4);
    rt_log_write(cell_id, word4);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Chassis request cell                                                */
/* ------------------------------------------------------------------ */

static hqvm_rt_cell g_request_cell;
static int          g_request_live = 0;
static int          g_request_seed_mode = HQVM_RT_SEED_REST;

void hqvm_rt_request_reset(int seed_mode) {
    const uint8_t marker[4] = { (uint8_t) seed_mode, 0, 0, 0 };

    if (seed_mode < HQVM_RT_SEED_REST || seed_mode > HQVM_RT_SEED_OMEGA) {
        seed_mode = HQVM_RT_SEED_REST;
    }
    (void) hqvm_rt_log_begin_session((uint32_t) seed_mode);
    rt_log_write(HQVM_RT_LOG_CELL_RESET, marker);
    g_log_requests++;
    hqvm_rt_cell_init(&g_request_cell, seed_mode, 0, 0);
    g_request_seed_mode = seed_mode;
    g_request_live = 1;
}

void hqvm_rt_request_ingest_bytes(const uint8_t * bytes, int n) {
    int i;

    if (!g_request_live || !bytes || n <= 0) return;

    for (i = 0; i < n; ++i) {
        const uint8_t pos = (uint8_t) (g_request_cell.word_len & 3u);
        rt_byte_cadence(&g_request_cell, bytes[i], pos);
        g_request_cell.open_word[g_request_cell.word_len++] = bytes[i];
        if (g_request_cell.word_len == 4) {
            rt_word_closure(&g_request_cell, g_request_cell.open_word);
            rt_log_write(HQVM_RT_REQUEST_CELL_ID, g_request_cell.word4);
        }
    }
}

const hqvm_rt_cell * hqvm_rt_request_cell(void) {
    if (!g_request_live) return NULL;
    /* Expose seed mode through open_word[0] while nothing has been ingested,
     * so receipts can verify the seeding contract without extra state. */
    if (g_request_cell.step == 0 && g_request_cell.word_len == 0) {
        g_request_cell.open_word[0] = (uint8_t) g_request_seed_mode;
    }
    return &g_request_cell;
}

/* ------------------------------------------------------------------ */
/* Standalone medium session (Runtime Part II product)                 */
/* ------------------------------------------------------------------ */

static hqvm_rt_pool *  g_medium_pool = NULL;
static hqvm_rt_slcp_t g_medium_last_slcp;
static int            g_medium_open = 0;
static int            g_medium_seed = HQVM_RT_SEED_REST;

static void rt_medium_ingest_bytes(const uint8_t * bytes, int n, int emit_slcp) {
    hqvm_rt_cell * cell;
    int i;

    if (!g_medium_pool || !bytes || n <= 0) return;
    cell = hqvm_rt_pool_cell(g_medium_pool, 0);
    if (!cell) return;

    for (i = 0; i < n; ++i) {
        const uint8_t pos = (uint8_t) (cell->word_len & 3u);
        rt_byte_cadence(cell, bytes[i], pos);
        cell->open_word[cell->word_len++] = bytes[i];
        if (cell->word_len == 4) {
            rt_word_closure(cell, cell->open_word);
            rt_log_write(0, cell->word4);
            if (emit_slcp) {
                hqvm_rt_slcp_fill(cell, 0, g_medium_pool, &g_medium_last_slcp);
            }
        }
    }
}

int hqvm_rt_medium_open(const char * log_path, int seed_mode, uint32_t pool_capacity) {
    const uint8_t marker[4] = { (uint8_t) seed_mode, 0, 0, 0 };

    if (seed_mode < HQVM_RT_SEED_REST || seed_mode > HQVM_RT_SEED_OMEGA) {
        seed_mode = HQVM_RT_SEED_REST;
    }
    if (pool_capacity == 0) pool_capacity = 64;
    if (pool_capacity > 4096) pool_capacity = 4096;

    hqvm_rt_medium_close();

    g_medium_pool = hqvm_rt_pool_create(pool_capacity, HQVM_RT_PROFILE_CHIRALITY);
    if (!g_medium_pool) return -1;

    if (log_path && log_path[0]) {
        if (hqvm_rt_log_configure(log_path) != 0) {
            hqvm_rt_pool_free(g_medium_pool);
            g_medium_pool = NULL;
            return -2;
        }
        if (hqvm_rt_log_begin_session((uint32_t) seed_mode) != 0) {
            hqvm_rt_log_close();
            hqvm_rt_pool_free(g_medium_pool);
            g_medium_pool = NULL;
            return -2;
        }
        rt_log_write(HQVM_RT_LOG_CELL_RESET, marker);
        g_log_requests++;
    }

    hqvm_rt_cell_init(hqvm_rt_pool_cell(g_medium_pool, 0), seed_mode, 0, 0);
    memset(&g_medium_last_slcp, 0, sizeof(g_medium_last_slcp));
    g_medium_seed = seed_mode;
    g_medium_open = 1;
    return 0;
}

int hqvm_rt_medium_ingest(const uint8_t * bytes, int n, int emit_slcp) {
    if (!g_medium_open || !g_medium_pool) return -1;
    rt_medium_ingest_bytes(bytes, n, emit_slcp);
    return 0;
}

int hqvm_rt_medium_close(void) {
    if (g_medium_pool) {
        hqvm_rt_pool_free(g_medium_pool);
        g_medium_pool = NULL;
    }
    hqvm_rt_log_close();
    g_medium_open = 0;
    memset(&g_medium_last_slcp, 0, sizeof(g_medium_last_slcp));
    return 0;
}

const hqvm_rt_slcp_t * hqvm_rt_medium_last_slcp(void) {
    if (!g_medium_open) return NULL;
    return &g_medium_last_slcp;
}

const hqvm_rt_cell * hqvm_rt_medium_cell(void) {
    if (!g_medium_open || !g_medium_pool) return NULL;
    return hqvm_rt_pool_cell(g_medium_pool, 0);
}
