/*
 * kernel/native.c — gyrocrypt native scaling: modexp, sparse CQFT, Shor period.
 */

#include "native.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shor path audit tag (set before any gyro_shor_path_set call sites). */
static const char *g_shor_last_path = "NONE";

static void gyro_shor_path_set(const char *tag)
{
 if (tag != NULL) {
  g_shor_last_path = tag;
 }
}

/* ── uint64 path ─────────────────────────────────────────────────────────── */

static uint64_t gyro_safe_add_mod(uint64_t a, uint64_t b, uint64_t n) {
 a %= n;
 b %= n;
 if (a >= n - b) {
  return a + b - n;
 }
 return a + b;
}

uint64_t gyro_mul_mod_ladder(uint64_t y, uint64_t multiplier, uint64_t n) {
 uint64_t acc = 0;
 uint64_t addend;
 if (n <= 1u) {
  return 0;
 }
 addend = multiplier % n;
 while (y) {
  if (y & 1u) {
   acc = gyro_safe_add_mod(acc, addend, n);
  }
  addend = gyro_safe_add_mod(addend, addend, n);
  y >>= 1;
 }
 return acc;
}

GYROSCOPIC_EXPORT uint64_t gyroscopic_mul_mod_ladder(
 uint64_t y,
 uint64_t multiplier,
 uint64_t n)
{
 return gyro_mul_mod_ladder(y, multiplier, n);
}

GYROSCOPIC_EXPORT uint64_t gyroscopic_exp_mod_ladder(
 uint64_t a,
 uint64_t x,
 uint64_t n)
{
 uint64_t acc = 1u;
 uint64_t base;
 if (n <= 1u) {
  return 0;
 }
 base = a % n;
 while (x) {
  if (x & 1u) {
   acc = gyro_mul_mod_ladder(acc, base, n);
  }
  base = gyro_mul_mod_ladder(base, base, n);
  x >>= 1;
 }
 return acc;
}

int gyro_mod_inv_u64(uint64_t a, uint64_t n, uint64_t *out) {
 int64_t t0 = 0;
 int64_t t1 = 1;
 int64_t r0;
 int64_t r1;
 if (out == NULL || n <= 1u) {
  return 0;
 }
 r0 = (int64_t) n;
 r1 = (int64_t) (a % n);
 while (r1 != 0) {
  int64_t q = r0 / r1;
  int64_t tmp = t0 - q * t1;
  t0 = t1;
  t1 = tmp;
  tmp = r0 - q * r1;
  r0 = r1;
  r1 = tmp;
 }
 if (r0 != 1) {
  return 0;
 }
 while (t0 < 0) {
  t0 += (int64_t) n;
 }
 *out = (uint64_t) t0;
 return 1;
}

double gyro_unit_frac_pow2(uint64_t num, int exp2) {
 if (exp2 <= 0) {
  return 0.0;
 }
 if (exp2 <= 53) {
  uint64_t mask;
  if (exp2 >= 64) {
   mask = ~0ULL;
  } else {
   mask = (1ULL << (unsigned) exp2) - 1ULL;
  }
  return (double) (num & mask) / (double) (1ULL << (unsigned) exp2);
 }
 {
  int shift = exp2 - 53;
  uint64_t window = (num >> (unsigned) shift) & ((1ULL << 53) - 1ULL);
  return (double) window / (double) (1ULL << 53);
 }
}

/* ── limb path (N > 2^64) ────────────────────────────────────────────────── */

#define GYRO_LIMB_MAX 136

static int gyro_limb_len(const uint32_t *x, int n) {
 while (n > 0 && x[n - 1] == 0u) {
  --n;
 }
 return n;
}

static int gyro_limb_cmp(const uint32_t *a, int an, const uint32_t* b, int bn) {
 an = gyro_limb_len(a, an);
 bn = gyro_limb_len(b, bn);
 if (an != bn) {
  return an > bn ? 1 : -1;
 }
 while (an > 0) {
  --an;
  if (a[an] > b[an]) {
   return 1;
  }
  if (a[an] < b[an]) {
   return -1;
  }
 }
 return 0;
}

static void gyro_limb_sub_n(uint32_t *a, const uint32_t* b, int n) {
 uint64_t borrow = 0u;
 int i;
 for (i = 0; i < n; ++i) {
  uint64_t t = (uint64_t) a[i] - (uint64_t) b[i] - borrow;
  a[i] = (uint32_t) t;
  borrow = (t >> 63) & 1u;
 }
}

static uint32_t gyro_n0inv(uint32_t n0) {
 uint32_t d = 1u;
 int i;
 for (i = 0; i < 32; ++i) {
  d *= 2u - n0 * d;
 }
 return (uint32_t)(0u - d);
}

static void gyro_limb_mul(
 uint32_t *prod,
 int *pn,
 const uint32_t *a,
 int an,
 const uint32_t *b,
 int bn)
{
 int i;
 int j;
 int cap = an + bn;
 if (cap > GYRO_LIMB_MAX * 2) {
  *pn = 0;
  return;
 }
 memset(prod, 0, (size_t) cap * sizeof(uint32_t));
 for (i = 0; i < an; ++i) {
  uint64_t carry = 0u;
  int k;
  for (j = 0; j < bn; ++j) {
   uint64_t t = (uint64_t) prod[i + j] + (uint64_t) a[i] * (uint64_t) b[j] + carry;
   prod[i + j] = (uint32_t) t;
   carry = t >> 32;
  }
  k = i + bn;
  while (carry && k < cap) {
   uint64_t t = (uint64_t) prod[k] + carry;
   prod[k] = (uint32_t) t;
   carry = t >> 32;
   ++k;
  }
 }
 *pn = gyro_limb_len(prod, cap);
}

static void gyro_redc(
 uint32_t *r,
 uint32_t *T,
 const uint32_t *n,
 int nn,
 uint32_t n0inv)
{
 int i;
 int j;
 int cap = 2 * nn + 2;
 for (i = 0; i < nn; ++i) {
  uint32_t m = (uint32_t) ((uint64_t) T[i] * (uint64_t) n0inv);
  uint64_t carry = 0u;
  for (j = 0; j < nn; ++j) {
   carry = (uint64_t) T[i + j] + (uint64_t) m * (uint64_t) n[j] + carry;
   T[i + j] = (uint32_t) carry;
   carry >>= 32;
  }
  j = i + nn;
  while (carry && j < cap) {
   carry = (uint64_t) T[j] + carry;
   T[j] = (uint32_t) carry;
   carry >>= 32;
   ++j;
  }
 }
 memcpy(r, T + nn, (size_t) nn * sizeof(uint32_t));
 if (gyro_limb_cmp(r, nn, n, nn) >= 0) {
  gyro_limb_sub_n(r, n, nn);
 }
}

static void gyro_mont_mul_mont(
 uint32_t *r,
 const uint32_t *a,
 const uint32_t *b,
 const uint32_t *n,
 int nn,
 uint32_t n0inv)
{
 uint32_t T[GYRO_LIMB_MAX * 2 + 2];
 int tn = 0;
 memset(T, 0, sizeof(T));
 gyro_limb_mul(T, &tn, a, nn, b, nn);
 if (tn < 2 * nn) {
  tn = 2 * nn;
 }
 gyro_redc(r, T, n, nn, n0inv);
}

static void gyro_limb_sub_at(
 uint32_t *a,
 int an,
 const uint32_t *b,
 int bn,
 int shift)
{
 uint64_t borrow = 0u;
 int i;
 if (shift < 0 || bn <= 0 || an < shift + bn) {
  return;
 }
 for (i = 0; i < bn; ++i) {
  int idx = i + shift;
  uint64_t t = (uint64_t) a[idx] - (uint64_t) b[i] - borrow;
  a[idx] = (uint32_t) t;
  borrow = (t >> 63) & 1u;
 }
 for (i = shift + bn; i < an && borrow; ++i) {
  uint64_t t = (uint64_t) a[i] - borrow;
  a[i] = (uint32_t) t;
  borrow = (t >> 63) & 1u;
 }
}

static void gyro_reduce_wide(
 uint32_t *out,
 uint32_t *wide,
 int wn,
 const uint32_t *n,
 int nn)
{
 int guard = 0;
 int guard_max = 2 * nn + 4;
 while (guard++ < guard_max) {
  wn = gyro_limb_len(wide, wn);
  if (wn < nn) {
   break;
  }
  if (wn == nn && gyro_limb_cmp(wide, nn, n, nn) < 0) {
   break;
  }
  if (wn > nn) {
   int shift = wn - nn;
   while (shift > 0) {
    if (gyro_limb_cmp(wide + shift, nn, n, nn) >= 0) {
     break;
    }
    --shift;
   }
   gyro_limb_sub_at(wide, wn, n, nn, shift);
  } else {
   gyro_limb_sub_n(wide, n, nn);
  }
 }
 memset(out, 0, (size_t) nn * sizeof(uint32_t));
 memcpy(out, wide, (size_t) nn * sizeof(uint32_t));
}

static void gyro_double_mod_n(uint32_t *a, const uint32_t* n, int nn) {
 uint32_t wide[GYRO_LIMB_MAX * 2];
 int wn;
 int i;
 uint64_t carry = 0u;
 memset(wide, 0, sizeof(wide));
 memcpy(wide, a, (size_t) nn * sizeof(uint32_t));
 for (i = 0; i < nn; ++i) {
  uint64_t v = ((uint64_t) wide[i] << 1) | carry;
  wide[i] = (uint32_t) v;
  carry = v >> 32;
 }
 if (carry) {
  wide[nn] = (uint32_t) carry;
 }
 wn = nn + (carry ? 1 : 0);
 gyro_reduce_wide(a, wide, wn, n, nn);
}

static void gyro_compute_R_mod_n(uint32_t *Rmod, const uint32_t* n, int nn) {
 int i;
 memset(Rmod, 0, (size_t) nn * sizeof(uint32_t));
 Rmod[0] = 1u;
 for (i = 0; i < 32 * nn; ++i) {
  gyro_double_mod_n(Rmod, n, nn);
 }
}

static void gyro_compute_R2_mod_n(
 uint32_t *R2,
 const uint32_t *Rmod,
 const uint32_t *n,
 int nn)
{
 int i;
 memcpy(R2, Rmod, (size_t) nn * sizeof(uint32_t));
 for (i = 0; i < 32 * nn; ++i) {
  gyro_double_mod_n(R2, n, nn);
 }
}

static void gyro_to_mont(
 uint32_t *out,
 const uint32_t *a,
 const uint32_t *Rmod,
 const uint32_t *n,
 int nn,
 uint32_t n0inv)
{
 uint32_t R2[GYRO_LIMB_MAX];
 uint32_t T[GYRO_LIMB_MAX * 2 + 2];
 int tn = 0;
 gyro_compute_R2_mod_n(R2, Rmod, n, nn);
 memset(T, 0, sizeof(T));
 gyro_limb_mul(T, &tn, a, nn, R2, nn);
 if (tn < 2 * nn) {
  tn = 2 * nn;
 }
 gyro_redc(out, T, n, nn, n0inv);
}

static void gyro_from_mont(
 uint32_t *out,
 const uint32_t *am,
 const uint32_t *n,
 int nn,
 uint32_t n0inv)
{
 uint32_t T[GYRO_LIMB_MAX * 2 + 2];
 memset(T, 0, sizeof(T));
 memcpy(T, am, (size_t) nn * sizeof(uint32_t));
 gyro_redc(out, T, n, nn, n0inv);
}

static int gyro_limb_is_one(const uint32_t *x, int n) {
 return gyro_limb_len(x, n) == 1 && x[0] == 1u;
}

static int gyro_limb_exp_mod(
 uint32_t *out,
 const uint32_t *a,
 int a_n,
 const uint32_t *x,
 int x_n,
 const uint32_t *n,
 int n_n)
{
 uint32_t n_buf[GYRO_LIMB_MAX];
 uint32_t Rmod[GYRO_LIMB_MAX];
 uint32_t base_m[GYRO_LIMB_MAX];
 uint32_t acc_m[GYRO_LIMB_MAX];
 uint32_t a_buf[GYRO_LIMB_MAX];
 uint32_t n0inv;
 int nn;
 int top;
 int bit;
 int i;
 nn = gyro_limb_len(n, n_n);
 if (nn <= 0 || nn > GYRO_LIMB_MAX - 1) {
  return 0;
 }
 if ((n[0] & 1u) == 0u) {
  return 0;
 }
 memset(n_buf, 0, sizeof(n_buf));
 memcpy(n_buf, n, (size_t) nn * sizeof(uint32_t));
 n0inv = gyro_n0inv(n_buf[0]);
 memset(a_buf, 0, sizeof(a_buf));
 if (a_n > nn) {
  a_n = nn;
 }
 memcpy(a_buf, a, (size_t) a_n * sizeof(uint32_t));
 if (gyro_limb_cmp(a_buf, nn, n_buf, nn) >= 0) {
  gyro_limb_sub_n(a_buf, n_buf, nn);
 }
 gyro_compute_R_mod_n(Rmod, n_buf, nn);
 if (gyro_limb_is_one(Rmod, nn)) {
  n_buf[nn] = 0u;
  ++nn;
  gyro_compute_R_mod_n(Rmod, n_buf, nn);
 }
 gyro_to_mont(base_m, a_buf, Rmod, n_buf, nn, n0inv);
 memcpy(acc_m, Rmod, (size_t) nn * sizeof(uint32_t));
 x_n = gyro_limb_len(x, x_n);
 if (x_n <= 0) {
  gyro_from_mont(out, acc_m, n_buf, nn, n0inv);
  return 1;
 }
 top = x_n - 1;
 bit = 31;
 while (bit >= 0 && ((x[top] >> (uint32_t) bit) & 1u) == 0u) {
  --bit;
 }
 if (bit < 0) {
  for (top = x_n - 2; top >= 0; --top) {
   bit = 31;
   while (bit >= 0 && ((x[top] >> (uint32_t) bit) & 1u) == 0u) {
    --bit;
   }
   if (bit >= 0) {
    break;
   }
  }
 }
 if (top < 0) {
  gyro_from_mont(out, acc_m, n_buf, nn, n0inv);
  return 1;
 }
 for (i = top; i >= 0; --i) {
  int start_bit = (i == top) ? bit : 31;
  for (; start_bit >= 0; --start_bit) {
   gyro_mont_mul_mont(acc_m, acc_m, acc_m, n_buf, nn, n0inv);
   if ((x[i] >> (uint32_t) start_bit) & 1u) {
    gyro_mont_mul_mont(acc_m, acc_m, base_m, n_buf, nn, n0inv);
   }
  }
 }
 gyro_from_mont(out, acc_m, n_buf, nn, n0inv);
 return 1;
}

GYROSCOPIC_EXPORT int gyroscopic_exp_mod_ladder_limbs(
 const uint32_t *a_limbs,
 int a_n,
 const uint32_t *x_limbs,
 int x_n,
 const uint32_t *n_limbs,
 int n_n,
 uint32_t *out_limbs,
 int out_n)
{
 if (
  a_limbs == NULL || x_limbs == NULL || n_limbs == NULL || out_limbs == NULL
  || a_n < 0 || x_n < 0 || n_n <= 0 || out_n < n_n || n_n > GYRO_LIMB_MAX
 ) {
  return 0;
 }
 memset(out_limbs, 0, (size_t) out_n * sizeof(uint32_t));
 return gyro_limb_exp_mod(out_limbs, a_limbs, a_n, x_limbs, x_n, n_limbs, n_n);
}
/* ── sparse cyclic QFT peaks (Z_Q character) ─────────────────────────────── */

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

typedef struct {
 uint64_t k;
 double mag2;
} gyro_peak_t;

static int gyro_peak_cmp(const void *a, const void *b) {
 const gyro_peak_t *pa = (const gyro_peak_t *) a;
 const gyro_peak_t *pb = (const gyro_peak_t *) b;
 if (pa->mag2 > pb->mag2) {
  return -1;
 }
 if (pa->mag2 < pb->mag2) {
  return 1;
 }
 return 0;
}

GYROSCOPIC_EXPORT int gyroscopic_sparse_cqft_peaks(
 const uint32_t *support,
 int n_support,
 uint32_t Q,
 int k_top,
 uint32_t *out_k,
 double *out_mag2,
 int out_cap)
{
 gyro_peak_t *peaks;
 double inv_norm;
 int n_peaks;
 int i;
 int j;
 int top;
 if (
  support == NULL || out_k == NULL || out_mag2 == NULL
  || n_support <= 0 || Q <= 1u || k_top <= 0 || out_cap <= 0
 ) {
  return 0;
 }
 n_peaks = (int) Q - 1;
 if (n_peaks <= 0) {
  return 0;
 }
 peaks = (gyro_peak_t *) malloc((size_t) n_peaks * sizeof(gyro_peak_t));
 if (peaks == NULL) {
  return 0;
 }
 inv_norm = 1.0 / sqrt((double) n_support);
 for (i = 1; i <= n_peaks; ++i) {
  double re = 0.0;
  double im = 0.0;
  uint32_t k = (uint32_t) i;
  for (j = 0; j < n_support; ++j) {
   double angle = (2.0 * M_PI * (double) ((uint64_t) k * (uint64_t) support[j])) / (double) Q;
   re += cos(angle);
   im += sin(angle);
  }
  peaks[i - 1].k = k;
  peaks[i - 1].mag2 = (re * re + im * im) * inv_norm * inv_norm;
 }
 qsort(peaks, (size_t) n_peaks, sizeof(gyro_peak_t), gyro_peak_cmp);
 top = k_top < out_cap ? k_top : out_cap;
 if (top > n_peaks) {
  top = n_peaks;
 }
 for (i = 0; i < top; ++i) {
  out_k[i] = peaks[i].k;
  out_mag2[i] = peaks[i].mag2;
 }
 free(peaks);
 return top;
}

/* ── Shor period (radix-64 transfer DP on Z_N) — spectral path ──────────── */

#define GYRO_SHOR_DP_MAX_DIGITS 24
#define GYRO_HORIZON_DIM 128
#define GYRO_HORIZON_KEY_BITS 7
#define GYRO_HORIZON_MAX_CELLS 8

typedef enum {
 GYRO_DP_EXACT = 0,
 GYRO_DP_TENSOR = 1
} GyroDpMode;

typedef struct {
 GyroDpMode mode;
 size_t state_len;
 const int32_t *y_map;
 const uint64_t *keys;
 int n_cells;
 int g_scratch[GYRO_HORIZON_MAX_CELLS];
} GyroDpState;

static size_t gyro_horizon_tensor_idx(const int *g, int n_cells) {
 size_t idx = 0u;
 int c;
 for (c = 0; c < n_cells; ++c) {
  idx = idx * (size_t) GYRO_HORIZON_DIM + (size_t) g[c];
 }
 return idx;
}

static void gyro_horizon_tensor_cells_of_key(uint64_t packed, int n_cells, int *g) {
 int c;
 for (c = 0; c < n_cells; ++c) {
  g[c] = (int) ((packed >> (GYRO_HORIZON_KEY_BITS * (unsigned) c)) & 0x7Fu);
 }
}

static size_t gyro_dp_project(uint64_t y_next, const GyroDpState *st) {
 if (st->mode == GYRO_DP_EXACT) {
  return (size_t) (y_next % st->state_len);
 }
 gyro_horizon_tensor_cells_of_key(st->keys[y_next], st->n_cells, st->g_scratch);
 return gyro_horizon_tensor_idx(st->g_scratch, st->n_cells);
}

static void gyro_dp_phase_wj(
 uint64_t k,
 uint64_t Q,
 int digit_j,
 double *wj_re,
 double *wj_im)
{
 uint64_t shift = 1u;
 uint32_t s;
 uint64_t phase_idx;
 double angle;

 for (s = 0; s < (uint32_t) digit_j; ++s) {
  shift *= 64u;
 }
 phase_idx = ((uint64_t) k * shift) % Q;
 angle = (2.0 * M_PI * (double) phase_idx) / (double) Q;
 *wj_re = cos(angle);
 *wj_im = sin(angle);
}

static void gyro_dp_buf_swap(
 double **cur_re,
 double **cur_im,
 double **nxt_re,
 double **nxt_im)
{
 double *tmp;

 tmp = *cur_re;
 *cur_re = *nxt_re;
 *nxt_re = tmp;
 tmp = *cur_im;
 *cur_im = *nxt_im;
 *nxt_im = tmp;
}

static void gyro_dp_step_digit(
 uint64_t n,
 const uint64_t powers_row[64],
 double wj_re,
 double wj_im,
 const GyroDpState *st,
 const double *cur_re,
 const double *cur_im,
 double *nxt_re,
 double *nxt_im)
{
 size_t state_len = st->state_len;
 int d;

 memset(nxt_re, 0, state_len * sizeof(double));
 memset(nxt_im, 0, state_len * sizeof(double));

 if (st->mode == GYRO_DP_EXACT) {
  uint64_t y;
  for (y = 0u; y < n; ++y) {
   double cr = cur_re[y];
   double ci = cur_im[y];
   double tw_re;
   double tw_im;

   if (fabs(cr) < 1e-15 && fabs(ci) < 1e-15) {
    continue;
   }
   tw_re = 1.0;
   tw_im = 0.0;
   for (d = 0; d < 64; ++d) {
    uint64_t y_next = gyro_mul_mod_ladder(y, powers_row[d], n);
    size_t out_idx = gyro_dp_project(y_next, st);
    double ntw_re;
    double ntw_im;

    nxt_re[out_idx] += cr * tw_re - ci * tw_im;
    nxt_im[out_idx] += cr * tw_im + ci * tw_re;
    ntw_re = tw_re * wj_re - tw_im * wj_im;
    ntw_im = tw_re * wj_im + tw_im * wj_re;
    tw_re = ntw_re;
    tw_im = ntw_im;
   }
  }
  return;
 }

 {
  size_t idx;
  for (idx = 0u; idx < state_len; ++idx) {
   int32_t yy = st->y_map[idx];
   uint64_t y;
   double cr;
   double ci;
   double tw_re;
   double tw_im;

   if (yy < 0) {
    continue;
   }
   y = (uint64_t) yy;
   cr = cur_re[idx];
   ci = cur_im[idx];
   if (fabs(cr) < 1e-15 && fabs(ci) < 1e-15) {
    continue;
   }
   tw_re = 1.0;
   tw_im = 0.0;
   for (d = 0; d < 64; ++d) {
    uint64_t y_next = gyro_mul_mod_ladder(y, powers_row[d], n);
    size_t out_idx = gyro_dp_project(y_next, st);
    double ntw_re;
    double ntw_im;

    nxt_re[out_idx] += cr * tw_re - ci * tw_im;
    nxt_im[out_idx] += cr * tw_im + ci * tw_re;
    ntw_re = tw_re * wj_re - tw_im * wj_im;
    ntw_im = tw_re * wj_im + tw_im * wj_re;
    tw_re = ntw_re;
    tw_im = ntw_im;
   }
  }
 }
}

static double gyro_dp_mag2_at(
 const double *re,
 const double *im,
 size_t idx)
{
 return re[idx] * re[idx] + im[idx] * im[idx];
}

/* Per-branch twiddles tw[d] = exp(2πi·k·d·64^j/Q) for radix digit j. */
static void gyro_dp_radix_branch_twiddles(
 uint64_t k,
 uint64_t Q,
 int digit_j,
 double tw_re[64],
 double tw_im[64])
{
 uint64_t shift = 1u;
 uint32_t s;
 int d;

 for (s = 0; s < (uint32_t) digit_j; ++s) {
  shift *= 64u;
 }
 for (d = 0; d < 64; ++d) {
  uint64_t phase_idx = ((uint64_t) k * (uint64_t) d * shift) % Q;
  double angle = (2.0 * M_PI * (double) phase_idx) / (double) Q;
  tw_re[d] = cos(angle);
  tw_im[d] = sin(angle);
 }
}

/* 2D radix step: y → y·g^{d1}·h^{d2} with χ_{k1,k2}(d1,d2) = tw1[d1]·tw2[d2]. */
static void gyro_dp_step_digit_2d(
 uint64_t n,
 const uint64_t powers_g[64],
 const uint64_t powers_h[64],
 const double tw1_re[64],
 const double tw1_im[64],
 const double tw2_re[64],
 const double tw2_im[64],
 const GyroDpState *st,
 const double *cur_re,
 const double *cur_im,
 double *nxt_re,
 double *nxt_im)
{
 size_t state_len = st->state_len;

 memset(nxt_re, 0, state_len * sizeof(double));
 memset(nxt_im, 0, state_len * sizeof(double));

 if (st->mode == GYRO_DP_EXACT) {
  uint64_t y;
  for (y = 0u; y < n; ++y) {
   double cr = cur_re[y];
   double ci = cur_im[y];
   int d1i;
   int d2i;

   if (fabs(cr) < 1e-15 && fabs(ci) < 1e-15) {
    continue;
   }
   for (d1i = 0; d1i < 64; ++d1i) {
    double p1_re;
    double p1_im;
    uint64_t y1;

    p1_re = cr * tw1_re[d1i] - ci * tw1_im[d1i];
    p1_im = cr * tw1_im[d1i] + ci * tw1_re[d1i];
    y1 = gyro_mul_mod_ladder(y, powers_g[d1i], n);
    for (d2i = 0; d2i < 64; ++d2i) {
     uint64_t y_next;
     size_t out_idx;
     double t_re;
     double t_im;

     t_re = p1_re * tw2_re[d2i] - p1_im * tw2_im[d2i];
     t_im = p1_im * tw2_re[d2i] + p1_re * tw2_im[d2i];
     y_next = gyro_mul_mod_ladder(y1, powers_h[d2i], n);
     out_idx = gyro_dp_project(y_next, st);
     nxt_re[out_idx] += t_re;
     nxt_im[out_idx] += t_im;
    }
   }
  }
  return;
 }

 {
  size_t idx;
  for (idx = 0u; idx < state_len; ++idx) {
   int32_t yy = st->y_map[idx];
   uint64_t y;
   double cr;
   double ci;
   int d1i;
   int d2i;

   if (yy < 0) {
    continue;
   }
   y = (uint64_t) yy;
   cr = cur_re[idx];
   ci = cur_im[idx];
   if (fabs(cr) < 1e-15 && fabs(ci) < 1e-15) {
    continue;
   }
   for (d1i = 0; d1i < 64; ++d1i) {
    double p1_re;
    double p1_im;
    uint64_t y1;

    p1_re = cr * tw1_re[d1i] - ci * tw1_im[d1i];
    p1_im = cr * tw1_im[d1i] + ci * tw1_re[d1i];
    y1 = gyro_mul_mod_ladder(y, powers_g[d1i], n);
    for (d2i = 0; d2i < 64; ++d2i) {
     uint64_t y_next;
     size_t out_idx;
     double t_re;
     double t_im;

     t_re = p1_re * tw2_re[d2i] - p1_im * tw2_im[d2i];
     t_im = p1_im * tw2_re[d2i] + p1_re * tw2_im[d2i];
     y_next = gyro_mul_mod_ladder(y1, powers_h[d2i], n);
     out_idx = gyro_dp_project(y_next, st);
     nxt_re[out_idx] += t_re;
     nxt_im[out_idx] += t_im;
    }
   }
  }
 }
}

static int gyro_dp_suffix_run(
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 int B,
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64],
 const GyroDpState *st,
 size_t init_idx,
 double *out_re,
 double *out_im)
{
 double *cur_re;
 double *cur_im;
 double *nxt_re;
 double *nxt_im;
 int j;

 cur_re = (double *) calloc(st->state_len, sizeof(double));
 cur_im = (double *) calloc(st->state_len, sizeof(double));
 nxt_re = (double *) calloc(st->state_len, sizeof(double));
 nxt_im = (double *) calloc(st->state_len, sizeof(double));
 if (out_re == NULL || out_im == NULL || cur_re == NULL || cur_im == NULL
  || nxt_re == NULL || nxt_im == NULL) {
  free(cur_re);
  free(cur_im);
  free(nxt_re);
  free(nxt_im);
  return 0;
 }

 cur_re[init_idx] = 1.0;

 for (j = 0; j < B; ++j) {
  double wj_re;
  double wj_im;

  gyro_dp_phase_wj(k, Q, j, &wj_re, &wj_im);
  gyro_dp_step_digit(
   n, powers[j], wj_re, wj_im, st, cur_re, cur_im, nxt_re, nxt_im);
  gyro_dp_buf_swap(&cur_re, &cur_im, &nxt_re, &nxt_im);
 }

 memcpy(out_re, cur_re, st->state_len * sizeof(double));
 memcpy(out_im, cur_im, st->state_len * sizeof(double));
 free(cur_re);
 free(cur_im);
 free(nxt_re);
 free(nxt_im);
 return 1;
}

static uint64_t gyro_pow64_mod(uint64_t base, uint64_t n) {
 uint64_t x = base % n;
 x = gyro_mul_mod_ladder(x, x, n);
 x = gyro_mul_mod_ladder(x, x, n);
 x = gyro_mul_mod_ladder(x, x, n);
 x = gyro_mul_mod_ladder(x, x, n);
 x = gyro_mul_mod_ladder(x, x, n);
 return gyro_mul_mod_ladder(x, x, n);
}

static int gyro_shor_digit_count(uint64_t Q) {
 uint64_t cur = 1u;
 int B = 0;
 while (cur < Q && B < GYRO_SHOR_DP_MAX_DIGITS) {
  cur *= 64u;
  ++B;
 }
 return B > 0 ? B : 1;
}

static void gyro_shor_build_powers(
 uint64_t a,
 uint64_t n,
 int B,
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64])
{
 int j;
 uint64_t base_j = a % n;
 for (j = 0; j < B; ++j) {
  int d;
  powers[j][0] = 1u % n;
  for (d = 1; d < 64; ++d) {
   powers[j][d] = gyro_mul_mod_ladder(powers[j][d - 1], base_j, n);
  }
  base_j = gyro_pow64_mod(base_j, n);
 }
}

/* Run radix-64 transfer DP; final ψ(y) written to out_re/out_im (size n). */
static int gyro_shor_dp_run(
 uint64_t a,
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 int B,
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64],
 double *out_re,
 double *out_im)
{
 GyroDpState st;

 (void) a;
 memset(&st, 0, sizeof(st));
 st.mode = GYRO_DP_EXACT;
 st.state_len = (size_t) n;
 return gyro_dp_suffix_run(n, Q, k, B, powers, &st, (size_t) (1u % n), out_re, out_im);
}

/* Exact |ψ_k(y=1)|² with pre-allocated buffers (beam inner loop). */
static double gyro_shor_dp_mag2_y1_bufs(
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 int B,
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64],
 const GyroDpState *st,
 double *re,
 double *im,
 double *nxt_re,
 double *nxt_im)
{
 int j;

 memset(re, 0, (size_t) n * sizeof(double));
 memset(im, 0, (size_t) n * sizeof(double));
 re[1u % n] = 1.0;
 for (j = 0; j < B; ++j) {
  double wj_re;
  double wj_im;

  gyro_dp_phase_wj(k, Q, j, &wj_re, &wj_im);
  gyro_dp_step_digit(
   n, powers[j], wj_re, wj_im, st, re, im, nxt_re, nxt_im);
  gyro_dp_buf_swap(&re, &im, &nxt_re, &nxt_im);
 }
 return gyro_dp_mag2_at(re, im, (size_t) (1u % n));
}

/* Quick k ≈ j·Q/n probe + Q/k refinement (exact path, ~200 evals). */
static uint32_t gyro_shor_beam_rational_scan(
 uint64_t a,
 uint64_t n,
 uint64_t Q,
 int B,
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64],
 const GyroDpState *st,
 double *re,
 double *im,
 double *nxt_re,
 double *nxt_im)
{
 uint32_t j;
 uint64_t denom;
 uint64_t best_k;
 double best_mag2;
 int dr;

 if (n <= 1u) {
  return 0u;
 }
 best_k = 0u;
 best_mag2 = 0.0;
 denom = n;
 for (j = 1u; j <= 96u; ++j) {
  uint64_t k = (j * Q) / denom;
  double mag2;
  uint32_t r;

  if (k <= 0u || k >= Q) {
   continue;
  }
  mag2 = gyro_shor_dp_mag2_y1_bufs(
   n, Q, k, B, powers, st, re, im, nxt_re, nxt_im);
  r = gyro_shor_try_k(a, n, Q, k, mag2);
  if (r != 0u) {
   gyro_shor_path_set("DP_EXACT_DOUBLE");
   return r;
  }
  if (mag2 > best_mag2) {
   best_mag2 = mag2;
   best_k = k;
  }
 }

 if (best_k <= 0u || best_mag2 < 1e-6) {
  return 0u;
 }

 {
  uint64_t r_est = (Q + best_k / 2u) / best_k;
  for (dr = -96; dr <= 96; ++dr) {
   int64_t rt = (int64_t) r_est + (int64_t) dr;
   uint64_t k_ref;
   double mag2;
   uint32_t r;

   if (rt < 2) {
    continue;
   }
   k_ref = (Q + (uint64_t) rt / 2u) / (uint64_t) rt;
   if (k_ref <= 0u || k_ref >= Q) {
    continue;
   }
   mag2 = gyro_shor_dp_mag2_y1_bufs(
    n, Q, k_ref, B, powers, st, re, im, nxt_re, nxt_im);
   r = gyro_shor_try_k(a, n, Q, k_ref, mag2);
   if (r != 0u) {
    gyro_shor_path_set("DP_EXACT_DOUBLE");
    return r;
   }
  }
 }
 return 0u;
}

/* Horizon tensor suffix readout via unified GyroDpState (n_cells 2..4). */
static double gyro_horizon_tensor_mag2_y1_core(
 uint64_t base,
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 const uint64_t *keys,
 int n_cells);

typedef struct {
 int active;
 int n_cells;
 uint64_t *keys;
} GyroHorizonBeamCtx;

static void gyro_horizon_beam_ctx_init(GyroHorizonBeamCtx *ctx, uint64_t n) {
 ctx->active = 0;
 ctx->n_cells = 0;
 ctx->keys = NULL;
 if (n == 0u || n > (uint64_t) SIZE_MAX / sizeof(uint64_t)) {
  return;
 }
 ctx->keys = (uint64_t *) malloc((size_t) n * sizeof(uint64_t));
 if (ctx->keys == NULL) {
  return;
 }
 ctx->n_cells = gyroscopic_horizon_pack_keys_u64(n, ctx->keys, (int) n);
 if (ctx->n_cells >= 2 && ctx->n_cells <= 4) {
  ctx->active = 1;
 } else {
  free(ctx->keys);
  ctx->keys = NULL;
  ctx->n_cells = 0;
 }
}

static void gyro_horizon_beam_ctx_free(GyroHorizonBeamCtx *ctx) {
 free(ctx->keys);
 ctx->keys = NULL;
 ctx->active = 0;
 ctx->n_cells = 0;
}

/* Dense radix-64 cyclic QFT readout on G_X=Z_Q (K1/K10/K12). No linear k-scan. */
#define GYRO_SHOR_DENSE_CQFT_MAX 262144u

static int gyro_shor_is_radix64_Q(uint64_t Q, int *out_B) {
 uint64_t q = Q;
 int B = 0;
 if (Q < 64u) {
  return 0;
 }
 while (q > 1u) {
  if ((q % 64u) != 0u) {
   return 0;
  }
  q /= 64u;
  ++B;
 }
 if (out_B != NULL) {
  *out_B = B;
 }
 return B > 0;
}

static int gyro_rev64_index(int i, int B) {
 int y = 0;
 int x = i;
 int t;
 for (t = 0; t < B; ++t) {
  y = y * 64 + (x % 64);
  x /= 64;
 }
 return y;
}

static void gyro_radix64_cqft(double *re, double *im, int B) {
 int Q = 1;
 int stage;
 int m;
 int k0;
 int j;
 int r;
 int k;
 double *tmp_re;
 double *tmp_im;
 int d;

 for (d = 0; d < B; ++d) {
  Q *= 64;
 }
 tmp_re = (double *) calloc((size_t) Q, sizeof(double));
 tmp_im = (double *) calloc((size_t) Q, sizeof(double));
 if (tmp_re == NULL || tmp_im == NULL) {
  free(tmp_re);
  free(tmp_im);
  return;
 }
 for (d = 0; d < Q; ++d) {
  int ri = gyro_rev64_index(d, B);
  tmp_re[d] = re[ri];
  tmp_im[d] = im[ri];
 }
 memcpy(re, tmp_re, (size_t) Q * sizeof(double));
 memcpy(im, tmp_im, (size_t) Q * sizeof(double));

 for (stage = 0; stage < B; ++stage) {
  m = 1;
  for (d = 0; d < stage; ++d) {
   m *= 64;
  }
  for (k0 = 0; k0 < Q; k0 += 64 * m) {
   for (j = 0; j < m; ++j) {
    double t_re[64];
    double t_im[64];
    double out_re[64];
    double out_im[64];
    for (r = 0; r < 64; ++r) {
     int idx = k0 + j + r * m;
     t_re[r] = re[idx];
     t_im[r] = im[idx];
    }
    for (k = 0; k < 64; ++k) {
     double sr = 0.0;
     double si = 0.0;
     for (r = 0; r < 64; ++r) {
      double ang_tw = (2.0 * M_PI * (double) (j * k * m)) / (double) Q;
      double ang_dft = (2.0 * M_PI * (double) (k * r)) / 64.0;
      double twr = cos(ang_tw);
      double twi = sin(ang_tw);
      double dfr = cos(ang_dft);
      double dfi = sin(ang_dft);
      double vr = t_re[r];
      double vi = t_im[r];
      double tr = vr * twr - vi * twi;
      double ti = vr * twi + vi * twr;
      sr += tr * dfr - ti * dfi;
      si += tr * dfi + ti * dfr;
     }
     out_re[k] = sr;
     out_im[k] = si;
    }
    for (r = 0; r < 64; ++r) {
     int idx = k0 + j + r * m;
     re[idx] = out_re[r];
     im[idx] = out_im[r];
    }
   }
  }
 }
 free(tmp_re);
 free(tmp_im);
}

static int gyro_shor_build_coset(
 uint64_t a,
 uint64_t n,
 uint64_t Q,
 double *re,
 double *im)
{
 uint64_t x;
 int hits = 0;
 double scale;

 if (re == NULL || im == NULL) {
  return 0;
 }
 for (x = 0u; x < Q; ++x) {
  re[x] = 0.0;
  im[x] = 0.0;
  if (gyroscopic_exp_mod_ladder(a, x, n) == 1u) {
   re[x] = 1.0;
   ++hits;
  }
 }
 if (hits <= 0) {
  return 0;
 }
 scale = 1.0 / sqrt((double) hits);
 for (x = 0u; x < Q; ++x) {
  re[x] *= scale;
 }
 return hits;
}

static int gyro_is_minimal_order_u64(uint64_t base, uint32_t r, uint64_t n) {
 uint32_t d;
 if (r <= 1u || gyroscopic_exp_mod_ladder(base, (uint64_t) r, n) != 1u) {
  return 0;
 }
 for (d = 2u; (uint64_t) d * (uint64_t) d <= (uint64_t) r; ++d) {
  if (r % d == 0u) {
   if (gyroscopic_exp_mod_ladder(base, (uint64_t) d, n) == 1u) {
    return 0;
   }
   {
    uint32_t other = r / d;
    if (other != d && gyroscopic_exp_mod_ladder(base, (uint64_t) other, n) == 1u) {
     return 0;
    }
   }
  }
 }
 return 1;
}

static void gyro_cf_collect(
 double x,
 int max_denom,
 uint32_t *out,
 int *n_out,
 int cap)
{
 int64_t h0 = 0;
 int64_t h1 = 1;
 int64_t k0 = 1;
 int64_t k1 = 0;
 int64_t a;
 int i;
 if (cap <= 0 || max_denom < 2) {
  return;
 }
 if (x <= 0.0 || x >= 1.0) {
  return;
 }
 for (i = 0; i < 64; ++i) {
  int64_t denom;
  a = (int64_t) x;
  if (a < 0) {
   a = 0;
  }
  denom = k0 + a * k1;
  if (denom > 1 && denom <= (int64_t) max_denom) {
   int j;
   int dup = 0;
   for (j = 0; j < *n_out; ++j) {
    if (out[j] == (uint32_t) denom) {
     dup = 1;
     break;
    }
   }
   if (!dup && *n_out < cap) {
    out[(*n_out)++] = (uint32_t) denom;
   }
  }
  {
   int64_t nh = h0 + a * h1;
   int64_t nk = k0 + a * k1;
   double frac = x - (double) a;
   if (nk == 0) {
    break;
   }
   h0 = h1;
   h1 = nh;
   k0 = k1;
   k1 = nk;
   x = 1.0 / frac;
  }
 }
}

static uint32_t gyro_shor_try_k(
 uint64_t a,
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 double mag2)
{
 uint32_t candidates[256];
 int n_cand = 0;
 double x;
 int lim;
 int c;
 uint32_t r;

 if (k == 0u || k >= Q || mag2 < 1e-12) {
  return 0u;
 }
 x = (double) k / (double) Q;
 lim = (int) n;
 if (lim > 2000000000) {
  lim = 2000000000;
 }
 gyro_cf_collect(x, lim, candidates, &n_cand, 256);
 gyro_cf_collect(x, lim / 2, candidates, &n_cand, 256);
 gyro_cf_collect(x, lim / 8, candidates, &n_cand, 256);
 gyro_cf_collect(x, lim / 32, candidates, &n_cand, 256);
 gyro_cf_collect(x, lim / 128, candidates, &n_cand, 256);
 for (c = 0; c < n_cand; ++c) {
  r = candidates[c];
  if (gyro_is_minimal_order_u64(a, r, n)) {
   return r;
  }
 }
 return 0u;
}

static uint32_t gyro_shor_recover_period_cqft(
 uint64_t a,
 uint64_t n,
 uint64_t Q,
 int B)
{
 double *cre;
 double *cim;
 gyro_peak_t *peaks;
 uint64_t k;
 int n_peaks;
 int top;
 int i;
 uint32_t r;

 (void) B;
 if (Q > (uint64_t) GYRO_SHOR_DENSE_CQFT_MAX) {
  return 0u;
 }
 cre = (double *) calloc((size_t) Q, sizeof(double));
 cim = (double *) calloc((size_t) Q, sizeof(double));
 if (cre == NULL || cim == NULL) {
  free(cre);
  free(cim);
  return 0u;
 }
 if (gyro_shor_build_coset(a, n, Q, cre, cim) <= 0) {
  free(cre);
  free(cim);
  return 0u;
 }
 gyro_radix64_cqft(cre, cim, B);

 n_peaks = (int) Q - 1;
 peaks = (gyro_peak_t *) malloc((size_t) n_peaks * sizeof(gyro_peak_t));
 if (peaks == NULL) {
  free(cre);
  free(cim);
  return 0u;
 }
 for (k = 1u; k < Q; ++k) {
  double rv = cre[k];
  double iv = cim[k];
  peaks[k - 1u].k = k;
  peaks[k - 1u].mag2 = rv * rv + iv * iv;
 }
 qsort(peaks, (size_t) n_peaks, sizeof(gyro_peak_t), gyro_peak_cmp);
 top = n_peaks < 32 ? n_peaks : 32;
 for (i = 0; i < top; ++i) {
  r = gyro_shor_try_k(a, n, Q, (uint64_t) peaks[i].k, peaks[i].mag2);
  if (r != 0u) {
   free(peaks);
   free(cre);
   free(cim);
   gyro_shor_path_set("CQFT_DENSE");
   return r;
  }
 }
 free(peaks);
 free(cre);
 free(cim);
 return 0u;
}

static void gyro_shor_beam_params(uint64_t n, int B, int *ms_depth, int *beam_keep) {
 int bits = 0;
 uint64_t x = n;

 while (x) {
  ++bits;
  x >>= 1;
 }
 (void) bits;
 *ms_depth = B;
 if (*ms_depth < 3) {
  *ms_depth = 3;
 }
 if (bits <= 20 && *ms_depth > 4) {
  *ms_depth = 4;
 }
 if (*ms_depth > 8) {
  *ms_depth = 8;
 }
 if (bits <= 12) {
  *beam_keep = 48;
 } else if (bits <= 20) {
  *beam_keep = 24;
 } else {
  *beam_keep = 64;
 }
}

/* Float exact ℤ_N digit step (4×N×4 bytes buffers). */
static void gyro_dp_step_digit_exact_float(
 uint64_t n,
 const uint64_t powers_row[64],
 float wj_re,
 float wj_im,
 const float *cur_re,
 const float *cur_im,
 float *nxt_re,
 float *nxt_im)
{
 size_t state_len = (size_t) n;
 int d;
 uint64_t y;

 memset(nxt_re, 0, state_len * sizeof(float));
 memset(nxt_im, 0, state_len * sizeof(float));
 for (y = 0u; y < n; ++y) {
  float cr = cur_re[y];
  float ci = cur_im[y];
  float tw_re = 1.0f;
  float tw_im = 0.0f;
  int di;

  if (fabsf(cr) < 1e-7f && fabsf(ci) < 1e-7f) {
   continue;
  }
  for (di = 0; di < 64; ++di) {
   uint64_t y_next = gyro_mul_mod_ladder(y, powers_row[di], n);
   float ntw_re;
   float ntw_im;

   nxt_re[y_next] += cr * tw_re - ci * tw_im;
   nxt_im[y_next] += cr * tw_im + ci * tw_re;
   ntw_re = tw_re * wj_re - tw_im * wj_im;
   ntw_im = tw_re * wj_im + tw_im * wj_re;
   tw_re = ntw_re;
   tw_im = ntw_im;
  }
 }
}

static double gyro_shor_dp_mag2_y1_float(
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 int B,
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64],
 float *buf_re1,
 float *buf_im1,
 float *buf_re2,
 float *buf_im2)
{
 float *cur_re = buf_re1;
 float *cur_im = buf_im1;
 float *nxt_re = buf_re2;
 float *nxt_im = buf_im2;
 size_t one_idx = (size_t) (1u % n);
 int j;

 memset(cur_re, 0, (size_t) n * sizeof(float));
 memset(cur_im, 0, (size_t) n * sizeof(float));
 cur_re[one_idx] = 1.0f;
 for (j = 0; j < B; ++j) {
  double wj_re_d;
  double wj_im_d;
  float wj_re;
  float wj_im;
  float *tmp;

  gyro_dp_phase_wj(k, Q, j, &wj_re_d, &wj_im_d);
  wj_re = (float) wj_re_d;
  wj_im = (float) wj_im_d;
  gyro_dp_step_digit_exact_float(
   n, powers[j], wj_re, wj_im, cur_re, cur_im, nxt_re, nxt_im);
  tmp = cur_re;
  cur_re = nxt_re;
  nxt_re = tmp;
  tmp = cur_im;
  cur_im = nxt_im;
  nxt_im = tmp;
 }
 return (double) (cur_re[one_idx] * cur_re[one_idx]
  + cur_im[one_idx] * cur_im[one_idx]);
}

/* Beam: double exact (N≤2M), float exact (N≤1.5B), or tensor. No classical fallback. */
static uint32_t gyro_shor_recover_period_beam(
 uint64_t a,
 uint64_t n,
 uint64_t Q,
 int B,
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64])
{
 int ms_depth;
 int beam_keep;
 int depth;
 int t;
 int use_exact;
 int use_float;
 GyroHorizonBeamCtx hctx;
 GyroDpState exact_st;
 double *exact_re;
 double *exact_im;
 double *exact_nxt_re;
 double *exact_nxt_im;
 float *f_re1;
 float *f_im1;
 float *f_re2;
 float *f_im2;
 uint64_t prefixes[64];
 int n_prefix;
 gyro_peak_t scored[4096];
 int n_scored;
 int i;
 const char *path_tag;

 use_exact = (n <= 2000000u);
 use_float = (!use_exact && n <= 1500000000u);
 path_tag = "TENSOR_BEAM";
 if (use_exact) {
  path_tag = "DP_EXACT_DOUBLE";
 } else if (use_float) {
  path_tag = "DP_EXACT_FLOAT";
 }
 exact_re = NULL;
 exact_im = NULL;
 exact_nxt_re = NULL;
 exact_nxt_im = NULL;
 f_re1 = NULL;
 f_im1 = NULL;
 f_re2 = NULL;
 f_im2 = NULL;
 memset(&hctx, 0, sizeof(hctx));

 if (use_exact) {
  exact_re = (double *) calloc((size_t) n, sizeof(double));
  exact_im = (double *) calloc((size_t) n, sizeof(double));
  exact_nxt_re = (double *) calloc((size_t) n, sizeof(double));
  exact_nxt_im = (double *) calloc((size_t) n, sizeof(double));
  if (exact_re == NULL || exact_im == NULL
   || exact_nxt_re == NULL || exact_nxt_im == NULL) {
   free(exact_re);
   free(exact_im);
   free(exact_nxt_re);
   free(exact_nxt_im);
   return 0u;
  }
  memset(&exact_st, 0, sizeof(exact_st));
  exact_st.mode = GYRO_DP_EXACT;
  exact_st.state_len = (size_t) n;
  {
   uint32_t r_seed = gyro_shor_beam_rational_scan(
    a, n, Q, B, powers, &exact_st,
    exact_re, exact_im, exact_nxt_re, exact_nxt_im);
   if (r_seed != 0u) {
    free(exact_re);
    free(exact_im);
    free(exact_nxt_re);
    free(exact_nxt_im);
    gyro_shor_path_set("DP_EXACT_DOUBLE");
    return r_seed;
   }
  }
 } else if (use_float) {
  f_re1 = (float *) calloc((size_t) n, sizeof(float));
  f_im1 = (float *) calloc((size_t) n, sizeof(float));
  f_re2 = (float *) calloc((size_t) n, sizeof(float));
  f_im2 = (float *) calloc((size_t) n, sizeof(float));
  if (f_re1 == NULL || f_im1 == NULL || f_re2 == NULL || f_im2 == NULL) {
   free(f_re1);
   free(f_im1);
   free(f_re2);
   free(f_im2);
   return 0u;
  }
 } else {
  gyro_horizon_beam_ctx_init(&hctx, n);
  if (!hctx.active) {
   gyro_horizon_beam_ctx_free(&hctx);
   return 0u;
  }
 }

 gyro_shor_beam_params(n, B, &ms_depth, &beam_keep);
 if (use_exact && n > 200000u) {
  if (ms_depth > 2) {
   ms_depth = 2;
  }
  if (beam_keep > 12) {
   beam_keep = 12;
  }
 }
 if (use_float) {
  if (ms_depth > 2) {
   ms_depth = 2;
  }
  if (beam_keep > 12) {
   beam_keep = 12;
  }
 }
 depth = ms_depth < B ? ms_depth : B;
 if (depth <= 0 || beam_keep <= 0) {
  if (!use_exact && !use_float) {
   gyro_horizon_beam_ctx_free(&hctx);
  }
  free(exact_re);
  free(exact_im);
  free(exact_nxt_re);
  free(exact_nxt_im);
  free(f_re1);
  free(f_im1);
  free(f_re2);
  free(f_im2);
  return 0u;
 }

 n_prefix = 1;
 prefixes[0] = 0u;

 for (t = 0; t < depth; ++t) {
  uint64_t place = 1u;
  int digit_pos = B - 1 - t;
  uint32_t s;
  int pi;
  int d;
  int keep;

  if (digit_pos < 0) {
   break;
  }
  for (s = 0; s < (uint32_t) digit_pos; ++s) {
   place *= 64u;
  }

  n_scored = 0;
  for (pi = 0; pi < n_prefix; ++pi) {
   for (d = 0; d < 64; ++d) {
    uint64_t kv = prefixes[pi] + (uint64_t) d * place;
    double mag2;
    uint32_t r;

    if (kv <= 0u || kv >= Q || n_scored >= 4096) {
     continue;
    }
    if (use_exact) {
     mag2 = gyro_shor_dp_mag2_y1_bufs(
      n, Q, kv, B, powers, &exact_st,
      exact_re, exact_im, exact_nxt_re, exact_nxt_im);
    } else if (use_float) {
     mag2 = gyro_shor_dp_mag2_y1_float(
      n, Q, kv, B, powers, f_re1, f_im1, f_re2, f_im2);
    } else {
     mag2 = gyro_horizon_tensor_mag2_y1_core(
      a, n, Q, kv, hctx.keys, hctx.n_cells);
    }
    scored[n_scored].k = kv;
    scored[n_scored].mag2 = mag2;
    n_scored++;
    r = gyro_shor_try_k(a, n, Q, kv, mag2);
    if (r != 0u) {
     if (!use_exact && !use_float) {
      gyro_horizon_beam_ctx_free(&hctx);
     }
     free(exact_re);
     free(exact_im);
     free(exact_nxt_re);
     free(exact_nxt_im);
     free(f_re1);
     free(f_im1);
     free(f_re2);
     free(f_im2);
     gyro_shor_path_set(path_tag);
     return r;
    }
   }
  }
  if (n_scored <= 0) {
   break;
  }
  qsort(scored, (size_t) n_scored, sizeof(gyro_peak_t), gyro_peak_cmp);
  keep = beam_keep < n_scored ? beam_keep : n_scored;
  n_prefix = keep;
  for (i = 0; i < keep; ++i) {
   prefixes[i] = scored[i].k;
  }
 }

 if (!use_exact && !use_float) {
  gyro_horizon_beam_ctx_free(&hctx);
 }
 free(exact_re);
 free(exact_im);
 free(exact_nxt_re);
 free(exact_nxt_im);
 free(f_re1);
 free(f_im1);
 free(f_re2);
 free(f_im2);
 return 0u;
}

GYROSCOPIC_EXPORT double gyroscopic_shor_dp_mag2_y1_u64(
 uint64_t base,
 uint64_t n,
 uint64_t Q,
 uint64_t k)
{
 uint64_t a;
 int B;
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64];
 double *re;
 double *im;
 double mag2;

 if (n <= 1u || Q <= 1u || k >= Q) {
  return 0.0;
 }
 a = base % n;
 if (a == 0u) {
  return 0.0;
 }
 B = gyro_shor_digit_count(Q);
 gyro_shor_build_powers(a, n, B, powers);
 re = (double *) calloc((size_t) n, sizeof(double));
 im = (double *) calloc((size_t) n, sizeof(double));
 if (re == NULL || im == NULL) {
  free(re);
  free(im);
  return 0.0;
 }
 if (!gyro_shor_dp_run(a, n, Q, k, B, powers, re, im)) {
  free(re);
  free(im);
  return 0.0;
 }
 mag2 = re[1] * re[1] + im[1] * im[1];
 free(re);
 free(im);
 return mag2;
}

static uint32_t gyro_shor_recover_period(
 uint64_t a,
 uint64_t n,
 uint64_t Q,
 int B,
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64])
{
 int B_chk;

 if (!gyro_shor_is_radix64_Q(Q, &B_chk) || B_chk != B) {
  return 0u;
 }
 if (Q <= (uint64_t) GYRO_SHOR_DENSE_CQFT_MAX) {
  return gyro_shor_recover_period_cqft(a, n, Q, B);
 }
 return gyro_shor_recover_period_beam(a, n, Q, B, powers);
}

GYROSCOPIC_EXPORT uint32_t gyroscopic_shor_period_u64(
 uint64_t base,
 uint64_t n,
 uint64_t Q)
{
 uint64_t a;
 int B;
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64];

 if (n <= 1u || Q <= 1u) {
  return 0u;
 }
 if (n > (uint64_t) SIZE_MAX / sizeof(double)) {
  return 0u;
 }
 a = base % n;
 if (a == 0u) {
  return 0u;
 }

 B = gyro_shor_digit_count(Q);
 gyro_shor_build_powers(a, n, B, powers);
 gyro_shor_path_set("NONE");
 {
  uint32_t r = gyro_shor_recover_period(a, n, Q, B, powers);
  if (r == 0u) {
   gyro_shor_path_set("FAIL_CLOSED");
  }
  return r;
 }
}

GYROSCOPIC_EXPORT const char *gyroscopic_shor_last_path_tag(void)
{
 return g_shor_last_path;
}

/* ── Kernel horizon key compile + exact 2D tensor DP (K15) ───────────────── */

#define GYRO_KERN_GENE_MIC_S 0xAAu
#define GYRO_KERN_GENE_MAC_REST 0xAAA555u
#define GYRO_KERN_GENE_MAC_SWAP 0x555AAAu
#define GYRO_KERN_LAYER_MASK 0xFFFu
#define GYRO_KERN_COMPLEMENT 0xFFFu

static uint32_t gyro_kern_popcount32(uint32_t x) {
 x = x - ((x >> 1) & 0x55555555u);
 x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
 x = (x + (x >> 4)) & 0x0F0F0F0Fu;
 return (uint32_t) ((x * 0x01010101u) >> 24);
}

static uint32_t gyro_kern_micro_ref_to_mask12(uint32_t micro_ref) {
 uint32_t m = micro_ref & 0x3Fu;
 uint32_t mask12 = 0u;
 int i;
 for (i = 0; i < 6; ++i) {
  if ((m >> (unsigned) i) & 1u) {
   mask12 |= 0x3u << (2u * (unsigned) i);
  }
 }
 return mask12 & GYRO_KERN_LAYER_MASK;
}

static uint32_t gyro_kern_step_state(uint32_t state24, uint32_t byte) {
 uint32_t intron = (byte & 0xFFu) ^ GYRO_KERN_GENE_MIC_S;
 uint32_t m12 = gyro_kern_micro_ref_to_mask12((intron >> 1) & 0x3Fu);
 uint32_t a12 = (state24 >> 12) & GYRO_KERN_LAYER_MASK;
 uint32_t b12 = state24 & GYRO_KERN_LAYER_MASK;
 uint32_t a_mut = (a12 ^ m12) & GYRO_KERN_LAYER_MASK;
 uint32_t invert_a = (intron & 0x01u) ? GYRO_KERN_COMPLEMENT : 0u;
 uint32_t invert_b = (intron & 0x80u) ? GYRO_KERN_COMPLEMENT : 0u;
 uint32_t a_next = (b12 ^ invert_a) & GYRO_KERN_LAYER_MASK;
 uint32_t b_next = (a_mut ^ invert_b) & GYRO_KERN_LAYER_MASK;
 return (a_next << 12) | b_next;
}

static uint32_t gyro_kern_chirality6(uint32_t state24) {
 uint32_t a12 = (state24 >> 12) & GYRO_KERN_LAYER_MASK;
 uint32_t b12 = state24 & GYRO_KERN_LAYER_MASK;
 uint32_t diff = (a12 ^ b12) & GYRO_KERN_LAYER_MASK;
 uint32_t out = 0u;
 int i;
 for (i = 0; i < 6; ++i) {
  uint32_t pair = (diff >> (2u * (unsigned) i)) & 0x3u;
  if (pair == 0x3u) {
   out |= 1u << (unsigned) i;
  }
 }
 return out;
}

static uint32_t gyro_kern_residue_chi(uint32_t r) {
 uint32_t byte = (r & 0xFFu) ^ GYRO_KERN_GENE_MIC_S;
 return gyro_kern_chirality6(gyro_kern_step_state(GYRO_KERN_GENE_MAC_REST, byte));
}

static uint32_t gyro_kern_byte_family_payload(int family, int payload6) {
 uint32_t intron = (((uint32_t) family >> 1) & 1u) << 7;
 intron |= ((uint32_t) payload6 & 0x3Fu) << 1;
 intron |= (uint32_t) family & 1u;
 return intron ^ GYRO_KERN_GENE_MIC_S;
}

static int gyro_kern_is_on_equality_horizon(uint32_t state24) {
 uint32_t a12 = (state24 >> 12) & GYRO_KERN_LAYER_MASK;
 uint32_t b12 = state24 & GYRO_KERN_LAYER_MASK;
 return a12 == b12;
}

static int gyro_kern_is_on_complement_horizon(uint32_t state24) {
 uint32_t a12 = (state24 >> 12) & GYRO_KERN_LAYER_MASK;
 uint32_t b12 = state24 & GYRO_KERN_LAYER_MASK;
 return a12 == (b12 ^ GYRO_KERN_COMPLEMENT);
}

static int gyro_kern_horizon128(uint32_t state24) {
 uint32_t chi = gyro_kern_chirality6(state24) & 0x3Fu;
 uint32_t d_rest = gyro_kern_popcount32(state24 ^ GYRO_KERN_GENE_MAC_REST);
 uint32_t d_swap = gyro_kern_popcount32(state24 ^ GYRO_KERN_GENE_MAC_SWAP);
 uint32_t z2 = (d_rest <= d_swap) ? 0u : 1u;

 if (gyro_kern_is_on_equality_horizon(state24)) {
  return (int) chi;
 }
 if (gyro_kern_is_on_complement_horizon(state24)) {
  return (int) (64 + chi);
 }
 return (int) (chi | (z2 << 6));
}

static int gyro_kern_native_cell_count(uint64_t n) {
 int bits = 0;
 uint64_t x = n;
 while (x) {
  ++bits;
  x >>= 1;
 }
 if (bits <= 0) {
  return 1;
 }
 return (bits + 5) / 6;
}

static uint8_t gyro_kern_phase_link(int cell, uint64_t n) {
 uint64_t slice = (n >> (6u * (unsigned) cell)) & 63u;
 return (uint8_t) (gyro_kern_residue_chi((uint32_t) slice) & 0x3Fu);
}

static uint8_t gyro_kern_cell_byte(uint64_t y, int cell, const uint8_t *phases) {
 uint64_t limb = (y >> (6u * (unsigned) cell)) & 63u;
 uint8_t payload = (uint8_t) ((limb ^ (uint64_t) phases[cell]) & 63u);
 return (uint8_t) gyro_kern_byte_family_payload(cell & 3, (int) payload);
}

static int gyro_kern_horizon_at_cell(uint64_t y, int cell, const uint8_t *phases) {
 uint8_t byte = gyro_kern_cell_byte(y, cell, phases);
 return gyro_kern_horizon128(gyro_kern_step_state(GYRO_KERN_GENE_MAC_REST, byte));
}

static uint64_t gyro_horizon_pack_one_key(
 uint64_t y,
 int n_cells,
 const uint8_t *phases)
{
 uint64_t word = 0u;
 int c;
 for (c = 0; c < n_cells; ++c) {
  int h = gyro_kern_horizon_at_cell(y, c, phases);
  word |= ((uint64_t) (h & 0x7F)) << (GYRO_HORIZON_KEY_BITS * (unsigned) c);
 }
 return word;
}

GYROSCOPIC_EXPORT int gyroscopic_horizon_pack_keys_u64(
 uint64_t n,
 uint64_t *keys_out,
 int cap)
{
 int n_cells;
 uint8_t phases[GYRO_HORIZON_MAX_CELLS];
 uint64_t y;

 if (keys_out == NULL || n <= 1u) {
  return -1;
 }
 if (cap < (int) n || n > (uint64_t) SIZE_MAX / sizeof(uint64_t)) {
  return -1;
 }
 n_cells = gyro_kern_native_cell_count(n);
 if (n_cells <= 0 || n_cells > GYRO_HORIZON_MAX_CELLS) {
  return -1;
 }
 int c;
 for (c = 0; c < n_cells; ++c) {
  phases[c] = gyro_kern_phase_link(c, n);
 }
 for (y = 0u; y < n; ++y) {
  keys_out[y] = gyro_horizon_pack_one_key(y, n_cells, phases);
 }
 return n_cells;
}

GYROSCOPIC_EXPORT int gyroscopic_horizon_n_cells_u64(uint64_t n)
{
 if (n <= 1u) {
  return -1;
 }
 return gyro_kern_native_cell_count(n);
}

GYROSCOPIC_EXPORT uint64_t gyroscopic_horizon_key_u64(uint64_t n, uint64_t y)
{
 int n_cells;
 uint8_t phases[GYRO_HORIZON_MAX_CELLS];

 if (n <= 1u) {
  return 0u;
 }
 n_cells = gyro_kern_native_cell_count(n);
 if (n_cells <= 0 || n_cells > GYRO_HORIZON_MAX_CELLS) {
  return 0u;
 }
 {
  int c;
  for (c = 0; c < n_cells; ++c) {
   phases[c] = gyro_kern_phase_link(c, n);
  }
 }
 return gyro_horizon_pack_one_key(y % n, n_cells, phases);
}

#define GYRO_CHIR_HASH_CAP 262144u

typedef struct {
 uint8_t used;
 uint64_t chi;
 uint64_t x;
} GyroChiSlot;

static uint64_t gyro_gcd_u64(uint64_t a, uint64_t b)
{
 while (b != 0u) {
  uint64_t t = a % b;
  a = b;
  b = t;
 }
 return a;
}

static uint32_t gyro_shor_peel_order(uint64_t a, uint64_t n, uint64_t g)
{
 uint64_t r = g;
 uint64_t p;

 if (r < 2u || gyroscopic_exp_mod_ladder(a, r, n) != 1u) {
  return 0u;
 }
 p = 2u;
 while (p * p <= r) {
  while (r % p == 0u && gyroscopic_exp_mod_ladder(a, r / p, n) == 1u) {
   r /= p;
  }
  if (p == 2u) {
   p = 3u;
  } else {
   p += 2u;
  }
 }
 if (r >= (uint64_t) UINT32_MAX) {
  return 0u;
 }
 return (uint32_t) r;
}

/* Period via injective horizon chart χ(a^x): collision + gcd (O(√r) queries, O(1) RAM). */
GYROSCOPIC_EXPORT uint32_t gyroscopic_shor_period_chirality_u64(
 uint64_t base,
 uint64_t n,
 uint32_t max_samples)
{
 uint64_t a;
 uint64_t g_acc;
 uint64_t x;
 uint64_t y;
 uint64_t chi;
 uint64_t prev_x;
 GyroChiSlot *tab;
 uint32_t samples;
 uint32_t i;
 uint32_t cap;
 uint32_t target;

 if (n <= 1u) {
  gyro_shor_path_set("FAIL_CLOSED");
  return 0u;
 }
 a = base % n;
 if (a == 0u || gyro_gcd_u64(a, n) != 1u) {
  gyro_shor_path_set("FAIL_CLOSED");
  return 0u;
 }
 cap = GYRO_CHIR_HASH_CAP;
 tab = (GyroChiSlot *) calloc((size_t) cap, sizeof(GyroChiSlot));
 if (tab == NULL) {
  gyro_shor_path_set("FAIL_CLOSED");
  return 0u;
 }
 target = max_samples;
 if (target == 0u) {
  int bits = 0;
  uint64_t t = n;
  while (t) {
   ++bits;
   t >>= 1;
  }
  target = (uint32_t) cap;
  if ((uint32_t) bits * 4096u > target) {
   target = (uint32_t) bits * 4096u;
  }
  if (target > 1048576u) {
   target = 1048576u;
  }
 }
 g_acc = 0u;
 x = 1u;
 for (i = 0; i < target; ++i) {
  size_t idx;
  size_t probe;

  y = gyroscopic_exp_mod_ladder(a, x, n);
  chi = gyroscopic_horizon_key_u64(n, y);
  idx = (size_t) ((chi ^ (chi >> 17) ^ (chi >> 33)) % (uint64_t) cap);
  for (probe = 0; probe < (size_t) cap; ++probe) {
   size_t slot = (idx + probe) % (size_t) cap;
   if (!tab[slot].used) {
    tab[slot].used = 1u;
    tab[slot].chi = chi;
    tab[slot].x = x;
    break;
   }
   if (tab[slot].chi == chi) {
    uint64_t d;

    prev_x = tab[slot].x;
    d = (x >= prev_x) ? (x - prev_x) : (prev_x - x);
    if (d != 0u) {
     g_acc = (g_acc == 0u) ? d : gyro_gcd_u64(g_acc, d);
    }
    break;
   }
  }
  x = x * 6364136223846793005u + 1442695040888963407u;
  if (x == 0u) {
   x = 1u;
  }
 }
 free(tab);
 if (g_acc < 2u) {
  gyro_shor_path_set("FAIL_CLOSED");
  return 0u;
 }
 {
  uint32_t r = gyro_shor_peel_order(a, n, g_acc);
  if (r <= 1u) {
   gyro_shor_path_set("FAIL_CLOSED");
   return 0u;
  }
  gyro_shor_path_set("CHIRALITY_COLLISION");
  return r;
 }
}

/* Horizon tensor map + drift (uses unified gyro_dp_step_digit). */

static size_t gyro_horizon_tensor_map_size(int n_cells) {
 size_t s = 1u;
 int c;
 for (c = 0; c < n_cells; ++c) {
  s *= (size_t) GYRO_HORIZON_DIM;
 }
 return s;
}

static int gyro_horizon_tensor_build_y_map(
 uint64_t n,
 const uint64_t *keys,
 int n_cells,
 int32_t *y_map,
 size_t map_size)
{
 uint64_t y;
 int g[GYRO_HORIZON_MAX_CELLS];

 memset(y_map, -1, map_size * sizeof(int32_t));
 for (y = 0u; y < n; ++y) {
  size_t idx;
  int c;
  gyro_horizon_tensor_cells_of_key(keys[y], n_cells, g);
  for (c = 0; c < n_cells; ++c) {
   if (g[c] < 0 || g[c] >= GYRO_HORIZON_DIM) {
    return -3;
   }
  }
  idx = gyro_horizon_tensor_idx(g, n_cells);
  if (idx >= map_size || y_map[idx] >= 0) {
   return -3;
  }
  y_map[idx] = (int32_t) y;
 }
 return 0;
}

static double gyro_horizon_tensor_mag2_y1_core(
 uint64_t base,
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 const uint64_t *keys,
 int n_cells)
{
 uint64_t a;
 int B;
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64];
 size_t map_size;
 int32_t *y_map;
 GyroDpState st;
 int g_one[GYRO_HORIZON_MAX_CELLS];
 size_t one_idx;
 double *re;
 double *im;
 double mag2;

 if (keys == NULL || n <= 1u || Q <= 1u || k >= Q || n_cells < 2 || n_cells > 4) {
  return 0.0;
 }
 a = base % n;
 if (a == 0u) {
  return 0.0;
 }

 map_size = gyro_horizon_tensor_map_size(n_cells);
 y_map = (int32_t *) malloc(map_size * sizeof(int32_t));
 re = (double *) calloc(map_size, sizeof(double));
 im = (double *) calloc(map_size, sizeof(double));
 if (y_map == NULL || re == NULL || im == NULL) {
  free(y_map);
  free(re);
  free(im);
  return 0.0;
 }
 if (gyro_horizon_tensor_build_y_map(n, keys, n_cells, y_map, map_size) != 0) {
  free(y_map);
  free(re);
  free(im);
  return 0.0;
 }

 B = gyro_shor_digit_count(Q);
 gyro_shor_build_powers(a, n, B, powers);
 gyro_horizon_tensor_cells_of_key(keys[1], n_cells, g_one);
 one_idx = gyro_horizon_tensor_idx(g_one, n_cells);

 memset(&st, 0, sizeof(st));
 st.mode = GYRO_DP_TENSOR;
 st.state_len = map_size;
 st.y_map = y_map;
 st.keys = keys;
 st.n_cells = n_cells;
 if (!gyro_dp_suffix_run(n, Q, k, B, powers, &st, one_idx, re, im)) {
  free(y_map);
  free(re);
  free(im);
  return 0.0;
 }

 mag2 = gyro_dp_mag2_at(re, im, one_idx);
 free(y_map);
 free(re);
 free(im);
 return mag2;
}

static int gyro_horizon_tensor_step_drift_core(
 uint64_t base,
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 const uint64_t *keys,
 int n_cells,
 double *out_exact_mag2,
 double *out_tensor_mag2,
 int out_cap)
{
 uint64_t a;
 int B;
 uint64_t powers[GYRO_SHOR_DP_MAX_DIGITS][64];
 size_t map_size;
 int32_t *y_map;
 GyroDpState exact_st;
 GyroDpState tensor_st;
 double *cur_re;
 double *cur_im;
 double *nxt_re;
 double *nxt_im;
 double *psi_re;
 double *psi_im;
 double *t_nxt_re;
 double *t_nxt_im;
 int g_one[GYRO_HORIZON_MAX_CELLS];
 size_t one_idx;
 int j;

 if (keys == NULL || out_exact_mag2 == NULL || out_tensor_mag2 == NULL) {
  return -1;
 }
 if (n <= 1u || Q <= 1u || n_cells < 2 || n_cells > 4) {
  return -2;
 }
 a = base % n;
 if (a == 0u) {
  return -1;
 }
 B = gyro_shor_digit_count(Q);
 if (out_cap < B) {
  return -1;
 }
 gyro_shor_build_powers(a, n, B, powers);

 map_size = gyro_horizon_tensor_map_size(n_cells);
 y_map = (int32_t *) malloc(map_size * sizeof(int32_t));
 cur_re = (double *) calloc((size_t) n, sizeof(double));
 cur_im = (double *) calloc((size_t) n, sizeof(double));
 nxt_re = (double *) calloc((size_t) n, sizeof(double));
 nxt_im = (double *) calloc((size_t) n, sizeof(double));
 psi_re = (double *) calloc(map_size, sizeof(double));
 psi_im = (double *) calloc(map_size, sizeof(double));
 t_nxt_re = (double *) calloc(map_size, sizeof(double));
 t_nxt_im = (double *) calloc(map_size, sizeof(double));
 if (y_map == NULL || cur_re == NULL || cur_im == NULL || nxt_re == NULL || nxt_im == NULL
  || psi_re == NULL || psi_im == NULL || t_nxt_re == NULL || t_nxt_im == NULL) {
  free(y_map);
  free(cur_re);
  free(cur_im);
  free(nxt_re);
  free(nxt_im);
  free(psi_re);
  free(psi_im);
  free(t_nxt_re);
  free(t_nxt_im);
  return -1;
 }
 if (gyro_horizon_tensor_build_y_map(n, keys, n_cells, y_map, map_size) != 0) {
  free(y_map);
  free(cur_re);
  free(cur_im);
  free(nxt_re);
  free(nxt_im);
  free(psi_re);
  free(psi_im);
  free(t_nxt_re);
  free(t_nxt_im);
  return -3;
 }

 memset(&exact_st, 0, sizeof(exact_st));
 exact_st.mode = GYRO_DP_EXACT;
 exact_st.state_len = (size_t) n;
 memset(&tensor_st, 0, sizeof(tensor_st));
 tensor_st.mode = GYRO_DP_TENSOR;
 tensor_st.state_len = map_size;
 tensor_st.y_map = y_map;
 tensor_st.keys = keys;
 tensor_st.n_cells = n_cells;

 cur_re[1] = 1.0;
 gyro_horizon_tensor_cells_of_key(keys[1], n_cells, g_one);
 one_idx = gyro_horizon_tensor_idx(g_one, n_cells);
 psi_re[one_idx] = 1.0;

 for (j = 0; j < B; ++j) {
  double wj_re;
  double wj_im;

  gyro_dp_phase_wj(k, Q, j, &wj_re, &wj_im);
  gyro_dp_step_digit(
   n, powers[j], wj_re, wj_im, &exact_st, cur_re, cur_im, nxt_re, nxt_im);
  gyro_dp_step_digit(
   n, powers[j], wj_re, wj_im, &tensor_st, psi_re, psi_im, t_nxt_re, t_nxt_im);
  gyro_dp_buf_swap(&cur_re, &cur_im, &nxt_re, &nxt_im);
  gyro_dp_buf_swap(&psi_re, &psi_im, &t_nxt_re, &t_nxt_im);

  out_exact_mag2[j] = gyro_dp_mag2_at(cur_re, cur_im, 1u);
  out_tensor_mag2[j] = gyro_dp_mag2_at(psi_re, psi_im, one_idx);
 }

 free(y_map);
 free(cur_re);
 free(cur_im);
 free(nxt_re);
 free(nxt_im);
 free(psi_re);
 free(psi_im);
 free(t_nxt_re);
 free(t_nxt_im);
 return B;
}


GYROSCOPIC_EXPORT double gyroscopic_horizon_tensor_mag2_y1_u64(
 uint64_t base,
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 const uint64_t *keys,
 int n_cells)
{
 return gyro_horizon_tensor_mag2_y1_core(base, n, Q, k, keys, n_cells);
}

GYROSCOPIC_EXPORT int gyroscopic_horizon_tensor_step_drift_u64(
 uint64_t base,
 uint64_t n,
 uint64_t Q,
 uint64_t k,
 const uint64_t *keys,
 int n_cells,
 double *out_exact_mag2,
 double *out_tensor_mag2,
 int out_cap)
{
 if (n_cells < 2 || n_cells > 4) {
  return -2;
 }
 return gyro_horizon_tensor_step_drift_core(
  base, n, Q, k, keys, n_cells, out_exact_mag2, out_tensor_mag2, out_cap);
}

/* ── DLP: dual-oracle 2D suffix (exact Z_N or horizon tensor) ── */

static double gyro_dlp_2d_mag2_core(
 uint64_t base_g,
 uint64_t base_h,
 uint64_t n,
 uint64_t Q,
 uint64_t k1,
 uint64_t k2,
 const uint64_t *keys,
 int n_cells)
{
 int B;
 uint64_t powers_g[GYRO_SHOR_DP_MAX_DIGITS][64];
 uint64_t powers_h[GYRO_SHOR_DP_MAX_DIGITS][64];
 size_t state_len;
 size_t one_idx;
 int32_t *y_map;
 GyroDpState st;
 int g_one[GYRO_HORIZON_MAX_CELLS];
 double *cur_re;
 double *cur_im;
 double *nxt_re;
 double *nxt_im;
 int j;
 double mag2;

 if (n <= 1u || Q <= 1u || k1 >= Q || k2 >= Q) {
  return 0.0;
 }
 if (base_g == 0u || base_h == 0u) {
  return 0.0;
 }
 if (n_cells < 1 || n_cells > GYRO_HORIZON_MAX_CELLS) {
  return 0.0;
 }

 B = gyro_shor_digit_count(Q);
 gyro_shor_build_powers(base_g % n, n, B, powers_g);
 {
  uint64_t h_inv;
  if (!gyro_mod_inv_u64(base_h % n, n, &h_inv)) {
   return 0.0;
  }
  gyro_shor_build_powers(h_inv, n, B, powers_h);
 }

 y_map = NULL;
 memset(&st, 0, sizeof(st));
 if (n_cells < 2) {
  state_len = (size_t) n;
  one_idx = (size_t) (1u % n);
  st.mode = GYRO_DP_EXACT;
  st.state_len = state_len;
 } else {
  if (keys == NULL) {
   return 0.0;
  }
  state_len = gyro_horizon_tensor_map_size(n_cells);
  y_map = (int32_t *) malloc(state_len * sizeof(int32_t));
  if (y_map == NULL) {
   return 0.0;
  }
  if (gyro_horizon_tensor_build_y_map(n, keys, n_cells, y_map, state_len) != 0) {
   free(y_map);
   return 0.0;
  }
  st.mode = GYRO_DP_TENSOR;
  st.state_len = state_len;
  st.y_map = y_map;
  st.keys = keys;
  st.n_cells = n_cells;
  gyro_horizon_tensor_cells_of_key(keys[1], n_cells, g_one);
  one_idx = gyro_horizon_tensor_idx(g_one, n_cells);
 }

 cur_re = (double *) calloc(state_len, sizeof(double));
 cur_im = (double *) calloc(state_len, sizeof(double));
 nxt_re = (double *) calloc(state_len, sizeof(double));
 nxt_im = (double *) calloc(state_len, sizeof(double));
 if (cur_re == NULL || cur_im == NULL || nxt_re == NULL || nxt_im == NULL) {
  free(y_map);
  free(cur_re);
  free(cur_im);
  free(nxt_re);
  free(nxt_im);
  return 0.0;
 }

 cur_re[one_idx] = 1.0;

 for (j = 0; j < B; ++j) {
  double tw1_re[64];
  double tw1_im[64];
  double tw2_re[64];
  double tw2_im[64];

  gyro_dp_radix_branch_twiddles(k1, Q, j, tw1_re, tw1_im);
  gyro_dp_radix_branch_twiddles(k2, Q, j, tw2_re, tw2_im);
  gyro_dp_step_digit_2d(
   n, powers_g[j], powers_h[j],
   tw1_re, tw1_im, tw2_re, tw2_im,
   &st, cur_re, cur_im, nxt_re, nxt_im);
  gyro_dp_buf_swap(&cur_re, &cur_im, &nxt_re, &nxt_im);
 }

 mag2 = gyro_dp_mag2_at(cur_re, cur_im, one_idx);
 free(y_map);
 free(cur_re);
 free(cur_im);
 free(nxt_re);
 free(nxt_im);
 return mag2;
}

GYROSCOPIC_EXPORT double gyroscopic_dlp_2d_tensor_mag2_u64(
 uint64_t base_g,
 uint64_t base_h,
 uint64_t n,
 uint64_t Q,
 uint64_t k1,
 uint64_t k2,
 const uint64_t *keys,
 int n_cells)
{
 return gyro_dlp_2d_mag2_core(base_g, base_h, n, Q, k1, k2, keys, n_cells);
}

