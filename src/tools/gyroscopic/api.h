#pragma once

/*
 * Gyroscopic public API.
 *
 * This is the single header the llama backend hook and the Python ctypes
 * bindings include. It re-exports the kernel surface (kernel.h) and documents
 * the cache-line-aligned hand-off the llama hook relies on. There is no logic
 * here and no second matmul: the backend only reads a per-group scale.
 *
 * ---------------------------------------------------------------------------
 * Cache-line hand-off (x86-64, 64-byte line).
 *
 *   block_q1_0 in llama is: { ggml_half d; uint8_t qs[16]; } = 18 bytes,
 *   i.e. one FP16 scale + 128 sign bits.
 *
 *   The kernel's per-group metadata is ONE byte (gyroscopic_pack_q1_meta:
 *   shell[2:0] | k4[4:3] | h_zone[7:5] where h_zone = h>>3).
 *   shell in bits 0..2, k4_char in bits 3..4). For a contiguous row of groups
 *   the metadata buffer therefore packs 64 groups per 64-byte cache line, so
 *   reading the metadata for the two groups whose signs share a sign-buffer
 *   cache line costs no extra miss. The "6-bit offset = horizon, 2-bit family
 *   = K4" correspondence is exact: the offset within a line indexes the group,
 *   the packed family bits are the K4 character.
 *
 * The hand-off is one-way and completes before the matmul inner loop:
 * the kernel authors scales/metadata at load; the backend reads them. No
 * callbacks, no shared mutable state during inference.
 * ---------------------------------------------------------------------------
 */

#include "kernel.h"
