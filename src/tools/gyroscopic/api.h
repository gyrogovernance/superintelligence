#pragma once

/*
 * Gyroscopic public API for the ggml-gyroscopic backend and Python ctypes.
 *
 * Live (linked into ggml-cpu):
 *   - Q1_0 weight MatMul displace via thin HQVMLEDS (ledger.h)
 *   - Attention KV / holonomic / lift (attn.h)
 *   - Continuous codecs (codec.h)
 *   - Kernel trajectory/receipts + Omega-12 (kernel.h)
 *
 * Env — closed MatMul+KV:
 *   GYRO_LEDGER_PATH, GYRO_LEDGER_STRICT, GYRO_LEDGER_ALLOW,
 *   GYRO_LEDGER_VERBOSE, GYRO_LEDGER_TENSOR,
 *   GYRO_KV_KQ8, GYRO_KV_V, GYRO_HOLONOMIC_ATTN
 *
 * Env — unfinished forward sites (stress/debug; not a product mode):
 *   GYRO_APERTURE_SOFTMAX, GYRO_ROPE_CODEC, GYRO_SILU_CODEC,
 *   GYRO_CGM_LIFT, GYRO_RESIDUAL_HYBRID
 *
 * Env — scaffolding / proofs:
 *   GYRO_NORM_CODEC, GYRO_NORM_COMMIT (Norm live apply broken),
 *   GYRO_CGM_LIFT_PERTURB, GYRO_V_PERTURB, GYRO_COORD_PERTURB,
 *   GYRO_HOLONOMIC_ATTN_MODE, GYRO_KV_LEDGER, GYRO_RECEIPTS
 *
 * Prefer production_gyroscopic_env(...) over hand-rolled flag sets.
 */

#include "constants.h"
#include "kernel.h"
#include "ledger.h"
#include "attn.h"
#include "codec.h"
