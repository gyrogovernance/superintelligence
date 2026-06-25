#ifndef GYROSCOPIC_CONSTANTS_H
#define GYROSCOPIC_CONSTANTS_H

/*
 * Gyroscopic C constants.
 * Keep in sync with src/tools/gyroscopic/constants.py.
 *
 * Environment variables read by the gyroscopic backend:
 *   GGML_GYROSCOPIC=1          enable per-layer gravity scale on Q1_0 matmul
 *   GYROSCOPIC_TOTAL_LAYERS    model depth for gravity scale (default 36)
 */

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