#ifndef GYROSCOPIC_CONSTANTS_H
#define GYROSCOPIC_CONSTANTS_H

/*
 * Gyroscopic C constants.
 * Keep in sync with src/tools/gyroscopic/constants.py.
 *
 * Live ledger env (read by ggml-gyroscopic hooks / ledger.c / attn.c / codec.c):
 *   GYRO_LEDGER_PATH      thin HQVMLEDS (or fat HQVMLEDG) file
 *   GYRO_LEDGER_STRICT    abort if allowlisted site cannot displace
 *   GYRO_LEDGER_ALLOW     comma allowlist override
 *   GYRO_LEDGER_VERBOSE   extra displace logs
 *   GYRO_LEDGER_TENSOR    fat/default name substring
 *   GYRO_KV_LEDGER=1      capture KV coordinates on RoPE (WHT-peak)
 *   GYRO_KV_KQ8=1         displace float K cache with ggml Q8_0 (no-alloc)
 *   GYRO_KV_V=1           displace float V cache with ggml Q8_0 (no-alloc)
 *   GYRO_HOLONOMIC_ATTN=1 flash_attn score + Attn@V from Q8_0 K/V
 *   GYRO_HOLONOMIC_ATTN_MODE  unset|zero_scores|random_scores (dot = default)
 *   GYRO_COORD_PERTURB=zero_kq8  force zero K scores (score-loop proof)
 *   GYRO_V_PERTURB=1      zero V contribution (Attn@V proof)
 *   GYRO_PERCOLATION_SOFTMAX=1  shadow percolation θ vs stock softmax
 *   GYRO_SHELL_NORM=1     shadow shell-equilibration vs RMSNorm
 *   GYRO_RESIDUAL_SHADOW=1  shadow residual RMS / depth
 *   GYRO_RECEIPTS=1       emit per-token GENE_Mac receipts (kernel step)
 *   GYRO_APERTURE_SOFTMAX=1  aperture-constrained softmax (rank deficit * Delta)
 *   GYRO_ROPE_CODEC=1     RoPE via T_256^(turn) LUT (live)
 *   GYRO_SILU_CODEC=1     SwiGLU gate LUT apply (owned)
 *   GYRO_CGM_LIFT=1       lift attn argmax → (q6,fam) phase byte; chi6 at KV write
 *   GYRO_RESIDUAL_LAW=1   residual-stream law: add gain = 1+Delta*m from lift traj
 *   GYRO_RESIDUAL_HYBRID=1  deprecated alias of GYRO_RESIDUAL_LAW
 *   GYRO_NORM_CODEC / GYRO_NORM_COMMIT  signed Delta-ruler Norm (COMMIT applies)
 *   GYRO_NORM_G0          Norm gain reference (default 1.0; export geomean TODO)
 *   GYRO_CGM_LIFT_PERTURB=1  flip q6/fam (causal proof; not production)
 */

/* Bonsai-8B layer count — GyroClock depth = token_pos * HQVM_N_LAYER + layer_idx */
#ifndef HQVM_N_LAYER
#define HQVM_N_LAYER 36
#endif

/*
 * ABI / header contract (mirrored in Python _C_HEADER_MAP; used by kernel.c + ggml):
 *   LAYER_MASK_12, MASK_STATE24, GENE_*, OMEGA_SIZE, HORIZON_SIZE,
 *   BOUNDARY_SIZE, BULK_SIZE, DEPTH_CLOSURE, CHIRALITY_*, SHELL_*, etc.
 */

#ifndef LAYER_MASK_12
#define LAYER_MASK_12 0x0FFFu
#endif

#ifndef MASK_STATE24
#define MASK_STATE24 0x00FFFFFFu
#endif

#ifndef L0_MASK
#define L0_MASK 0x81u
#endif

#ifndef LI_MASK
#define LI_MASK 0x42u
#endif

#ifndef FG_MASK
#define FG_MASK 0x24u
#endif

#ifndef BG_MASK
#define BG_MASK 0x18u
#endif

#ifndef GENE_MIC_S
#define GENE_MIC_S 0xAAu
#endif

#ifndef GENE_MAC_A12
#define GENE_MAC_A12 0x0AAAu
#endif

#ifndef GENE_MAC_B12
#define GENE_MAC_B12 0x0555u
#endif

#ifndef GENE_MAC_REST
#define GENE_MAC_REST 0xAAA555u
#endif

#ifndef CHIRALITY_MASK_6
#define CHIRALITY_MASK_6 0x3Fu
#endif

#ifndef CHIRALITY_QUBITS_6
#define CHIRALITY_QUBITS_6 6u
#endif

#ifndef EPSILON_6
#define EPSILON_6 0x3Fu
#endif

#ifndef OMEGA_SIZE
#define OMEGA_SIZE 4096u
#endif

#ifndef HORIZON_SIZE
#define HORIZON_SIZE 64u
#endif

#ifndef BOUNDARY_SIZE
#define BOUNDARY_SIZE 128u
#endif

#ifndef BULK_SIZE
#define BULK_SIZE (OMEGA_SIZE - BOUNDARY_SIZE)
#endif

#ifndef DEPTH_CLOSURE
#define DEPTH_CLOSURE 4u
#endif

#ifndef MASK_CODE_SIZE
#define MASK_CODE_SIZE 64u
#endif

#ifndef LAYER_BITS
#define LAYER_BITS 12u
#endif

#ifndef L0_BIT_0
#define L0_BIT_0 0x01u
#endif

#ifndef L0_BIT_7
#define L0_BIT_7 0x80u
#endif

#ifndef FAMILY_MASK
#define FAMILY_MASK 0x03u
#endif

#ifndef SHADOW_PARTNER_MASK
#define SHADOW_PARTNER_MASK 0xFEu
#endif

#ifndef UINT8_MASK
#define UINT8_MASK 0xFFu
#endif

#ifndef UINT16_MASK
#define UINT16_MASK 0xFFFFu
#endif

#ifndef UINT32_MASK
#define UINT32_MASK 0xFFFFFFFFu
#endif

#ifndef UINT64_MASK
#define UINT64_MASK 0xFFFFFFFFFFFFFFFFu
#endif

#ifndef COMPLEMENT_MASK_12
#define COMPLEMENT_MASK_12 LAYER_MASK_12
#endif

#ifndef SHELL_MIDPOINT
#define SHELL_MIDPOINT 3u
#endif

#ifndef SHELL_MAX_POPULATION
#define SHELL_MAX_POPULATION 1280u
#endif

#ifndef COMPLEMENTARITY_SUM
#define COMPLEMENTARITY_SUM LAYER_BITS
#endif

#ifndef BYTE_COUNT
#define BYTE_COUNT 256u
#endif

#ifndef SHELL_COUNT
#define SHELL_COUNT 7u
#endif

#ifndef SHELL_MAX
#define SHELL_MAX CHIRALITY_QUBITS_6
#endif

#ifndef GAUGE_COUNT
#define GAUGE_COUNT 4u
#endif

#ifndef SHADOW_STATES
#define SHADOW_STATES 128u
#endif

#ifndef GYRO_M_PI
#define GYRO_M_PI 3.14159265358979323846
#endif

/*
 * Internal physics numerics (gravity / aperture closure; kernel-only math):
 *   Q_G, M_A, DELTA_BU, RHO, APERTURE_GAP, APERTURE_GAP_Q256
 */

#ifndef Q_G
#define Q_G (4.0 * GYRO_M_PI)
#endif

#ifndef M_A
#define M_A 0.19947114020071635
#endif

#ifndef DELTA_BU
#define DELTA_BU 0.195342176580
#endif

#ifndef RHO
#define RHO (DELTA_BU / M_A)
#endif

#ifndef APERTURE_GAP
#define APERTURE_GAP (1.0 - RHO)
#endif

#ifndef APERTURE_GAP_Q256
#define APERTURE_GAP_Q256 5u
#endif

#endif /* GYROSCOPIC_CONSTANTS_H */