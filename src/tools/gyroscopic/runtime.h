#pragma once

/*
 * Native Gyroscopic Inference Loop: request cells + genealogy (runtime.h).
 *
 * State model per Gyroscopic_ASI_Runtime_Specs.md:
 *   7.2   per-cell stored state (primary omega12, O(1) rings + histograms)
 *   8.5   constant-time ring/histogram updates (warmup vs full ring);
 *         chirality and family rings advance at the same ring position
 *   9     compiled action omega_sig = sig13(word4) at word closure
 *   11.2  seeding modes (rest / equality horizon / shell / omega)
 *   12    ingestion protocol: byte cadence + word closure
 *   16.4  ingest log: append-only (uint32 cell_id, 4 x uint8 word4)
 *
 * Parity commitments follow hQVM_SDK_Quantum_Computing 5.2: XOR accumulation
 * of each byte's 12-bit mutation mask at even / odd positions of the
 * trajectory; parity_bit = popcount(parity_O12 ^ parity_E12) & 1.
 *
 * Logging policy: the pure primitives (cell_init / ingest_word /
 * ingest_bytes) never touch the log. The append-only genealogy log is
 * written by the chassis paths: the request cell logs every closed word
 * under cell_id 0 plus a boundary record per request reset; explicit
 * pool ingestion logs under the given cell id. Records are 8 bytes:
 * word4[0] of a boundary record carries the fresh seed mode, and
 * cell_id HQVM_RT_LOG_CELL_RESET (0xFFFFFFFF) marks the boundary.
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

#define HQVM_RT_SEED_REST              0
#define HQVM_RT_SEED_EQUALITY_HORIZON  1
#define HQVM_RT_SEED_SHELL             2
#define HQVM_RT_SEED_OMEGA             3

#define HQVM_RT_LOG_CELL_RESET 0xFFFFFFFFu
#define HQVM_RT_REQUEST_CELL_ID 0u

#define HQVM_RT_PROFILE_CHIRALITY 0u
#define HQVM_RT_PROFILE_SHELL     1u

/* Field order avoids interior padding; mirrored by ops.RtCell for receipts.
 * word4 holds the most recent CLOSED word; open_word buffers up to 3 bytes
 * of the next word between ingest_bytes calls (word_len = fill count). */
typedef struct hqvm_rt_cell {
    uint64_t step;                 /* total bytes consumed */
    uint32_t resonance_key;
    int32_t  omega_sig;            /* sig13 of most recent closed word */
    uint16_t omega12;              /* packed (u6 << 6) | v6 */
    uint16_t chi_hist64[64];
    uint16_t shell_hist7[7];
    uint16_t family_hist4[4];
    uint16_t parity_O12;
    uint16_t parity_E12;
    uint8_t  last_byte;
    uint8_t  word4[4];
    uint8_t  open_word[4];
    uint8_t  word_len;             /* bytes buffered toward the next closure */
    uint8_t  has_closed_word;
    uint8_t  chi_ring64[64];
    uint8_t  family_ring64[64];
    uint8_t  ring_pos;             /* shared by both rings (Spec 8.5) */
    uint8_t  ring_valid_len;       /* 0..64 */
    uint8_t  parity_bit;
} hqvm_rt_cell;

/* Fingerprint of kernel rule surfaces + behavioral probes; snapshots and
 * receipts recorded under a different hash are rejected (Spec 16.2). */
GYROSCOPIC_EXPORT uint64_t hqvm_rt_rule_hash(void);

/* GYRO_NATIVE_GENEALOGY=1 enables the chassis entry/logging hooks;
 * off => zero hot-path cost and no log writes. */
GYROSCOPIC_EXPORT int hqvm_rt_enabled(void);

/* Seed a cell; (a, b) are (shell, 0) for SEED_SHELL and (u6, v6) for SEED_OMEGA. */
GYROSCOPIC_EXPORT void hqvm_rt_cell_init(
    hqvm_rt_cell * c,
    int            seed_mode,
    uint8_t        a,
    uint8_t        b);

/* One closed 4-byte word through the byte cadence (pure state update). */
GYROSCOPIC_EXPORT void hqvm_rt_ingest_word(hqvm_rt_cell * c, const uint8_t word4[4]);

/* Split arbitrary bytes into depth-4 words; short tails buffer until closed. */
GYROSCOPIC_EXPORT void hqvm_rt_ingest_bytes(hqvm_rt_cell * c, const uint8_t * bytes, int n);

/* Resonance key of the current position: profile 0 -> chi6, profile 1 -> shell. */
GYROSCOPIC_EXPORT uint32_t hqvm_rt_resonance_key_of(const hqvm_rt_cell * c, uint16_t profile_id);

/* Hamming distance between two cells' chi6 (Runtime 20.2 chirality_distance). */
GYROSCOPIC_EXPORT int hqvm_rt_chi_distance(const hqvm_rt_cell * a, const hqvm_rt_cell * b);

/* Decode batch grouping (Runtime 20.2): sort by (resonance_key, chi6), greedily
 * pack cells while they share the group head's resonance key OR sit within
 * chi-distance <= 2 of it, capped at max_batch_size. Writes a group id per
 * input cell and returns the number of groups; ids are dense from 0 and
 * follow group formation order. out_group_ids may alias scratch of length n. */
GYROSCOPIC_EXPORT int hqvm_rt_group_cells(
    const hqvm_rt_cell * const * cells,
    uint32_t                    n,
    uint32_t                    max_batch,
    uint16_t                  * out_group_ids);

/* Finite cell pool (Runtime 11.1). */
typedef struct hqvm_rt_pool {
    uint32_t      capacity;
    uint16_t      profile_id;
    hqvm_rt_cell *cells;
} hqvm_rt_pool;

GYROSCOPIC_EXPORT hqvm_rt_pool * hqvm_rt_pool_create(uint32_t capacity, uint16_t profile_id);
GYROSCOPIC_EXPORT void           hqvm_rt_pool_free(hqvm_rt_pool * pool);
GYROSCOPIC_EXPORT hqvm_rt_cell * hqvm_rt_pool_cell(hqvm_rt_pool * pool, uint32_t cell_id);

/* Apply one word to a pool cell and log it under that cell id (when the log
 * is configured). Returns 0, or -1 on bad ids. */
GYROSCOPIC_EXPORT int hqvm_rt_pool_ingest_word(
    hqvm_rt_pool * pool,
    uint32_t       cell_id,
    const uint8_t  word4[4]);

/* Chassis request cell (single-owner, alongside the attn trajectory owner). */
GYROSCOPIC_EXPORT void                 hqvm_rt_request_reset(int seed_mode);
GYROSCOPIC_EXPORT void                 hqvm_rt_request_ingest_bytes(const uint8_t * bytes, int n);
GYROSCOPIC_EXPORT const hqvm_rt_cell * hqvm_rt_request_cell(void);

/* Append-only genealogy log (Runtime 16.4). NULL path disables writing.
 * Chassis default path comes from GYRO_GENEALOGY_PATH; explicit configure
 * always wins. Returns 0 on success, -1 when fopen fails. */
GYROSCOPIC_EXPORT int      hqvm_rt_log_configure(const char * path);
GYROSCOPIC_EXPORT void     hqvm_rt_log_close(void);
GYROSCOPIC_EXPORT uint64_t hqvm_rt_log_events(void);
GYROSCOPIC_EXPORT uint64_t hqvm_rt_log_requests(void);

/* Snapshot header for receipts (Runtime 16.1/16.2): magic "GLOG", version,
 * seed mode, event/request counts, rule hash. Written once by the chassis
 * when logging is configured; readers must reject on hash mismatch. */
#define HQVM_RT_SNAPSHOT_MAGIC 0x474F4C47u /* "GLOG" little-endian */
#define HQVM_RT_SNAPSHOT_VERSION 1u

typedef struct hqvm_rt_snapshot_header {
    uint32_t magic;
    uint32_t version;
    uint32_t seed_mode;
    uint32_t reserved;
    uint64_t n_events;
    uint64_t n_requests;
    uint64_t rule_hash;
} hqvm_rt_snapshot_header;

GYROSCOPIC_EXPORT void hqvm_rt_snapshot_header_fill(hqvm_rt_snapshot_header * hdr, uint32_t seed_mode);
GYROSCOPIC_EXPORT void hqvm_rt_cell_checkpoint(const hqvm_rt_cell * c, uint8_t out[16]);

/* Write the snapshot header when the log file is empty (Runtime 16.1/16.2). */
GYROSCOPIC_EXPORT int hqvm_rt_log_begin_session(uint32_t seed_mode);

/* SLCP record (Runtime 13.2): structured BU-Ingress output per word closure. */
typedef struct hqvm_rt_slcp {
    uint32_t cell_id;
    uint64_t step;
    int32_t  omega12;
    int32_t  state24;
    uint8_t  last_byte;
    uint8_t  _pad0[3];
    int32_t  family;
    int32_t  micro_ref;
    int32_t  q6;
    int32_t  chi6;
    int32_t  shell;
    int32_t  horizon_distance;
    int32_t  ab_distance;
    int32_t  omega_sig;
    uint16_t parity_O12;
    uint16_t parity_E12;
    uint8_t  parity_bit;
    uint8_t  _pad1[3];
    uint32_t resonance_key;
    int32_t  current_resonance;
    float    spectral64[64];
} hqvm_rt_slcp_t;

/* Fill SLCP from a live cell. current_resonance is the pool bucket weight
 * (count of cells sharing resonance_key) when pool is non-NULL. */
GYROSCOPIC_EXPORT void hqvm_rt_slcp_fill(
    const hqvm_rt_cell * c,
    uint32_t             cell_id,
    const hqvm_rt_pool * pool,
    hqvm_rt_slcp_t *     out);

/* Graph query surface (Runtime 14). Scans the pool; returns counts or ids
 * written (capped at max_out). */
GYROSCOPIC_EXPORT int32_t hqvm_rt_bucket_population(const hqvm_rt_pool * pool, uint32_t key);
GYROSCOPIC_EXPORT int hqvm_rt_bucket_cells(
    const hqvm_rt_pool * pool, uint32_t key, uint32_t * out_ids, int max_out);
GYROSCOPIC_EXPORT int hqvm_rt_co_resonant_count(const hqvm_rt_pool * pool, uint32_t cell_id);
GYROSCOPIC_EXPORT int hqvm_rt_cells_on_shell(
    const hqvm_rt_pool * pool, int shell, uint32_t * out_ids, int max_out);
GYROSCOPIC_EXPORT int hqvm_rt_cells_with_chi6(
    const hqvm_rt_pool * pool, uint8_t chi6, uint32_t * out_ids, int max_out);
GYROSCOPIC_EXPORT int hqvm_rt_cells_with_signature(
    const hqvm_rt_pool * pool, int32_t omega_sig, uint32_t * out_ids, int max_out);

/* Standalone medium session (Runtime Part II product; no llama chassis). */
GYROSCOPIC_EXPORT int hqvm_rt_medium_open(
    const char * log_path, int seed_mode, uint32_t pool_capacity);
GYROSCOPIC_EXPORT int hqvm_rt_medium_ingest(const uint8_t * bytes, int n, int emit_slcp);
GYROSCOPIC_EXPORT int hqvm_rt_medium_close(void);
GYROSCOPIC_EXPORT const hqvm_rt_slcp_t * hqvm_rt_medium_last_slcp(void);
GYROSCOPIC_EXPORT const hqvm_rt_cell * hqvm_rt_medium_cell(void);

/* Polar attention prefilter score (Runtime 21.1): structural heuristic over
 * encoded summaries only — popcount + multiplies, no embeddings.
 *   chi_dist     = popcount(chi_q ^ chi_k)
 *   shell_sim    = (6 - chi_dist) / 6
 *   anchor_align = 1 - popcount(c_q ^ c_k) / 64
 *   score        = r_q * r_k * shell_sim * anchor_align
 */
typedef struct hqvm_rt_polar_summary {
    uint8_t  chi6;       /* chirality word of the head summary */
    uint64_t anchor64;   /* sign anchor (bit i set iff dim i >= 0) */
    float    radius;     /* summary magnitude r >= 0 */
} hqvm_rt_polar_summary;

GYROSCOPIC_EXPORT float hqvm_rt_polar_score(
    const hqvm_rt_polar_summary * q,
    const hqvm_rt_polar_summary * k);

/* Counters for claimed-native decode-consultation sites. stock_ops_total is
 * the sum of every counted stock operation this request; a site claiming
 * nativeness must show zero stock work there (receipts gate). */
GYROSCOPIC_EXPORT void     hqvm_rt_stock_ops_add(uint32_t n);
GYROSCOPIC_EXPORT uint64_t hqvm_rt_stock_ops_total(void);
GYROSCOPIC_EXPORT void     hqvm_rt_prefilter_inc(void);
GYROSCOPIC_EXPORT uint64_t hqvm_rt_prefilter_calls(void);
GYROSCOPIC_EXPORT uint64_t hqvm_rt_prefilter_skipped(void);
/* Report one prefilter decision: n_candidates in, n_kept after the gate. */
GYROSCOPIC_EXPORT void     hqvm_rt_prefilter_report(int64_t n_candidates, int64_t n_kept);
GYROSCOPIC_EXPORT void     hqvm_rt_counters_request_reset(void);

/* Decode batch grouping in the chassis (Runtime 20.2), gated by
 * GYRO_NATIVE_GROUP. Each decode ubatch builds one ephemeral cell per row
 * from that row's token piece bytes; the committed grouping law partitions
 * them. The receipt reports calls, total rows, total formed groups; the
 * reduction is rows - groups (printed plainly, zero included). */
GYROSCOPIC_EXPORT int      hqvm_rt_group_enabled(void);
GYROSCOPIC_EXPORT void     hqvm_rt_group_report(int64_t rows, int64_t groups);
GYROSCOPIC_EXPORT uint64_t hqvm_rt_group_calls(void);
GYROSCOPIC_EXPORT uint64_t hqvm_rt_group_rows(void);
GYROSCOPIC_EXPORT uint64_t hqvm_rt_group_groups(void);

#ifdef __cplusplus
}
#endif
