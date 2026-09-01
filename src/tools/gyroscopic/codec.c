#include "codec.h"

#include "constants.h"
#include "kernel.h"
#include "ledger.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Layer owned-call receipts (defined in layer.c). */
void hqvm_norm_ruler_commit_inc(void);
void hqvm_rope_codec_row_inc(void);
void hqvm_ffn_shell_gate_inc(void);

/*
 * Codecs + dyad arithmetic. Finite charts (norm ruler, RoPE ticks, shell FFN) are
 * medium faces when selected. to_f32/from_f32 are interoperability adapters only.
 */

/* ===== Integer-owned finite binary32 chart ===== */

typedef struct hqvm_dyad_parts {
    uint64_t sig;
    int exp2;
    uint32_t sign;
} hqvm_dyad_parts;

static int dyad_unpack(hqvm_dyad32_t x, hqvm_dyad_parts *p) {
    const uint32_t ef = (x.bits >> 23) & 0xffu;
    const uint32_t frac = x.bits & 0x7fffffu;
    if (ef == 0xffu) return -1;
    p->sign = x.bits >> 31;
    if (ef == 0) {
        p->sig = frac;
        p->exp2 = -149;
    } else {
        p->sig = (uint64_t)(0x800000u | frac);
        p->exp2 = (int)ef - 150;
    }
    return 0;
}

static int dyad_msb64(uint64_t x) {
    int n = -1;
    while (x) { x >>= 1; ++n; }
    return n;
}

static uint64_t dyad_shr_jam(uint64_t x, unsigned d) {
    if (d == 0) return x;
    if (d >= 64) return x ? 1u : 0u;
    return (x >> d) | ((x & ((((uint64_t)1) << d) - 1u)) != 0u);
}

static uint64_t dyad_round_shr_even(uint64_t x, unsigned d) {
    uint64_t q, rem, half;
    if (d == 0) return x;
    if (d > 64) return 0;
    if (d == 64) {
        const uint64_t h = ((uint64_t)1) << 63;
        return x > h ? 1u : 0u;
    }
    q = x >> d;
    rem = x & ((((uint64_t)1) << d) - 1u);
    half = ((uint64_t)1) << (d - 1u);
    if (rem > half || (rem == half && (q & 1u))) ++q;
    return q;
}

static int dyad_pack(uint32_t sign, uint64_t sig, int exp2, hqvm_dyad32_t *out) {
    int top, e;
    uint64_t q;
    if (!out) return -1;
    if (!sig) {
        out->bits = sign << 31;
        return 0;
    }
    top = dyad_msb64(sig);
    e = top + exp2;
    if (e > 127) return -2;
    if (e >= -126) {
        const int shift = top - 23;
        q = shift > 0 ? dyad_round_shr_even(sig, (unsigned)shift)
                      : sig << (unsigned)(-shift);
        if (q == 0x1000000u) {
            q >>= 1;
            ++e;
            if (e > 127) return -2;
        }
        out->bits = (sign << 31) | ((uint32_t)(e + 127) << 23)
                  | ((uint32_t)q & 0x7fffffu);
        return 0;
    }
    {
        const int shift = -(exp2 + 149);
        q = shift > 0 ? dyad_round_shr_even(sig, (unsigned)shift)
                      : sig << (unsigned)(-shift);
        if (q >= 0x800000u) {
            out->bits = (sign << 31) | 0x00800000u;
        } else {
            out->bits = (sign << 31) | (uint32_t)q;
        }
    }
    return 0;
}

int hqvm_dyad32_is_finite(hqvm_dyad32_t x) { return (x.bits & 0x7f800000u) != 0x7f800000u; }
int hqvm_dyad32_sign(hqvm_dyad32_t x) { return (int)(x.bits >> 31); }
hqvm_dyad32_t hqvm_dyad32_abs(hqvm_dyad32_t x) { x.bits &= 0x7fffffffu; return x; }
int hqvm_dyad32_is_zero(hqvm_dyad32_t x) { return (x.bits & 0x7fffffffu) == 0; }

hqvm_dyad32_t hqvm_dyad32_from_f32(float x) {
    hqvm_dyad32_t d;
    memcpy(&d.bits, &x, sizeof(d.bits));
    return d;
}

float hqvm_dyad32_to_f32(hqvm_dyad32_t x) {
    float f;
    memcpy(&f, &x.bits, sizeof(f));
    return f;
}

int hqvm_dyad32_from_i32(int32_t x, hqvm_dyad32_t *out) {
    const uint32_t sign = x < 0;
    const uint64_t mag = sign ? (uint64_t)(-(int64_t)x) : (uint64_t)x;
    return dyad_pack(sign, mag, 0, out);
}

int hqvm_dyad32_pack_i128(uint32_t sign, uint64_t sig, int exp2, hqvm_dyad32_t *out) {
    return dyad_pack(sign ? 1u : 0u, sig, exp2, out);
}

int hqvm_dyad32_add(hqvm_dyad32_t a, hqvm_dyad32_t b, hqvm_dyad32_t *out) {
    hqvm_dyad_parts pa, pb, t;
    uint64_t sa, sb, sig;
    uint32_t sign;
    int d;
    if (dyad_unpack(a, &pa) != 0 || dyad_unpack(b, &pb) != 0 || !out) return -1;
    if (!pa.sig) { *out = b; return 0; }
    if (!pb.sig) { *out = a; return 0; }
    if (pa.exp2 < pb.exp2) { t = pa; pa = pb; pb = t; }
    d = pa.exp2 - pb.exp2;
    sa = pa.sig << 3;
    sb = dyad_shr_jam(pb.sig << 3, (unsigned)d);
    if (pa.sign == pb.sign) {
        sig = sa + sb;
        sign = pa.sign;
    } else if (sa >= sb) {
        sig = sa - sb;
        sign = pa.sign;
    } else {
        sig = sb - sa;
        sign = pb.sign;
    }
    return dyad_pack(sign, sig, pa.exp2 - 3, out);
}

int hqvm_dyad32_mul(hqvm_dyad32_t a, hqvm_dyad32_t b, hqvm_dyad32_t *out) {
    hqvm_dyad_parts pa, pb;
    if (dyad_unpack(a, &pa) != 0 || dyad_unpack(b, &pb) != 0 || !out) return -1;
    return dyad_pack(pa.sign ^ pb.sign, pa.sig * pb.sig, pa.exp2 + pb.exp2, out);
}

int hqvm_dyad32_div(hqvm_dyad32_t a, hqvm_dyad32_t b, hqvm_dyad32_t *out) {
    hqvm_dyad_parts pa, pb;
    uint64_t numerator, q, rem;
    if (dyad_unpack(a, &pa) != 0 || dyad_unpack(b, &pb) != 0 || !out || !pb.sig) return -1;
    if (!pa.sig) return dyad_pack(pa.sign ^ pb.sign, 0, 0, out);
    numerator = pa.sig << 31;
    q = numerator / pb.sig;
    rem = numerator % pb.sig;
    if (rem) q |= 1u;
    return dyad_pack(pa.sign ^ pb.sign, q, pa.exp2 - pb.exp2 - 31, out);
}

int hqvm_dyad32_mul_rational(
    hqvm_dyad32_t x, int32_t num, int32_t den, hqvm_dyad32_t *out)
{
    hqvm_dyad32_t n, d, gain, product;
    int rc;
    if (den == 0 || !out) return -1;
    if ((rc = hqvm_dyad32_from_i32(num, &n)) != 0 ||
        (rc = hqvm_dyad32_from_i32(den, &d)) != 0 ||
        (rc = hqvm_dyad32_div(n, d, &gain)) != 0 ||
        (rc = hqvm_dyad32_mul(x, gain, &product)) != 0) return rc;
    *out = product;
    return 0;
}

/* ===== Norm ===== */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int hqvm_norm_codec_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_NORM_CODEC");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

int hqvm_norm_commit_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_NORM_COMMIT");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

float hqvm_rms_gain(const float *x, int64_t n, float eps) {
    int64_t i;
    double ss = 0.0;
    if (!x || n <= 0) return 1.0f;
    for (i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    return (float)(1.0 / sqrt(ss / (double)n + (double)eps));
}

/* Fixed-point RMS gain: Q16 accumulation + rsqrt LUT + 1 Newton step. */
#define HQVM_RSQRT_LUT 256
static float s_rsqrt_lut[HQVM_RSQRT_LUT];
static int s_rsqrt_init = 0;

static void hqvm_rsqrt_lut_init(void) {
    int i;
    if (s_rsqrt_init) return;
    for (i = 0; i < HQVM_RSQRT_LUT; ++i) {
        const double m = (0.5 + (double)i) / (double)HQVM_RSQRT_LUT; /* (0,1] */
        const double x = 1.0 + m; /* mantissa in [1,2) */
        s_rsqrt_lut[i] = (float)(1.0 / sqrt(x));
    }
    s_rsqrt_init = 1;
}

float hqvm_rms_gain_fixed(const float *x, int64_t n, float eps) {
    int64_t i;
    int64_t ss_q = 0;
    double amax = 0.0;
    double mean;
    int exp2;
    double mant;
    int idx;
    float y, xmean;
    if (!x || n <= 0) return 1.0f;
    hqvm_rsqrt_lut_init();

    /* Normalize before integer accumulation. Direct Q16 squaring overflows
     * int64 for valid late-layer residuals (RMS > ~23 at n=4096). Q15 on
     * x/amax bounds the sum by n*32767^2 without losing the fixed moment. */
    for (i = 0; i < n; ++i) {
        const double ax = fabs((double)x[i]);
        if (ax > amax) amax = ax;
    }
    if (amax == 0.0) return (float)(1.0 / sqrt((double)eps));
    for (i = 0; i < n; ++i) {
        const double z = (double)x[i] / amax;
        const int64_t q = (int64_t)round(z * 32767.0);
        ss_q += q * q;
    }
    mean = amax * amax * (double)ss_q /
        (32767.0 * 32767.0 * (double)n) + (double)eps;
    if (mean <= 0.0) return 1.0f;
    /* Normalize mean = 2^exp2 * mant, mant in [1,2) */
    mant = mean;
    exp2 = 0;
    while (mant >= 2.0) { mant *= 0.5; exp2++; }
    while (mant < 1.0 && mant > 0.0) { mant *= 2.0; exp2--; }
    idx = (int)floor((mant - 1.0) * (double)HQVM_RSQRT_LUT);
    if (idx < 0) idx = 0;
    if (idx >= HQVM_RSQRT_LUT) idx = HQVM_RSQRT_LUT - 1;
    y = s_rsqrt_lut[idx];
    /* scale: 1/sqrt(2^exp2 * mant) = 2^(-exp2/2) / sqrt(mant) */
    if (exp2 & 1) {
        y *= (float)(1.0 / sqrt(2.0));
        exp2--;
    }
    y *= (float)ldexp(1.0, -exp2 / 2);
    xmean = (float)mean;
    /* One Newton: y <- y*(1.5 - 0.5*x*y^2) */
    y = y * (1.5f - 0.5f * xmean * y * y);
    return y;
}

/* g0: geomean of Norm gains, or GYRO_NORM_G0, else 1.0 once noted. */
static float s_norm_g0 = -1.0f;
static int s_norm_g0_from_gains = 0;

/* pow2Delta LUT for ticks in [-512, 512] */
#define HQVM_POW2_DELTA_SPAN 512
static float s_pow2_delta[2 * HQVM_POW2_DELTA_SPAN + 1];
static int s_pow2_init = 0;

static void hqvm_pow2_delta_init(void) {
    int k;
    const double Delta = (double)APERTURE_GAP;
    if (s_pow2_init) return;
    for (k = -HQVM_POW2_DELTA_SPAN; k <= HQVM_POW2_DELTA_SPAN; ++k) {
        s_pow2_delta[k + HQVM_POW2_DELTA_SPAN] =
            (float)pow(2.0, (double)k * Delta);
    }
    s_pow2_init = 1;
}

float hqvm_norm_pow2_delta(int16_t n) {
    int k = (int)n;
    hqvm_pow2_delta_init();
    if (k < -HQVM_POW2_DELTA_SPAN) k = -HQVM_POW2_DELTA_SPAN;
    if (k > HQVM_POW2_DELTA_SPAN) k = HQVM_POW2_DELTA_SPAN;
    return s_pow2_delta[k + HQVM_POW2_DELTA_SPAN];
}

void hqvm_norm_set_g0(float g0) {
    if (g0 > 0.0f) s_norm_g0 = g0;
}

float hqvm_norm_geomean_gains(const float *g, int64_t n) {
    int64_t i, c = 0;
    double slog = 0.0;
    if (!g || n <= 0) return 1.0f;
    for (i = 0; i < n; ++i) {
        const float a = fabsf(g[i]);
        if (a > 0.0f) {
            slog += log((double)a);
            c++;
        }
    }
    if (c <= 0) return 1.0f;
    return (float)exp(slog / (double)c);
}

void hqvm_norm_set_g0_from_gains(const float *g, int64_t n) {
    float g0;
    if (s_norm_g0_from_gains) return;
    g0 = hqvm_norm_geomean_gains(g, n);
    if (g0 > 0.0f) {
        s_norm_g0 = g0;
        s_norm_g0_from_gains = 1;
        fprintf(stderr, "[hqvm-norm] g0=%.6g from geomean(|g|) n=%lld\n",
                (double)g0, (long long)n);
        fflush(stderr);
    }
}

float hqvm_norm_g0(void) {
    if (s_norm_g0 > 0.0f) return s_norm_g0;
    {
        const char *e = getenv("GYRO_NORM_G0");
        if (e && e[0]) {
            float v = (float)atof(e);
            if (v > 0.0f) {
                s_norm_g0 = v;
                return s_norm_g0;
            }
        }
    }
    s_norm_g0 = 1.0f;
    {
        static int s_g0_note = 0;
        if (!s_g0_note) {
            fprintf(stderr,
                "[hqvm-norm] g0=1.0 (awaiting geomean from Norm weights or GYRO_NORM_G0)\n");
            fflush(stderr);
            s_g0_note = 1;
        }
    }
    return s_norm_g0;
}

/* Encode gain on the Delta-ruler (Formalism Â§7): n = round(log2(g/g0)/Delta).
 * Signed int16; never clamp negatives to 0. Delta is APERTURE_GAP. */
int16_t hqvm_norm_encode_gain16(float g, float g0, float Delta) {
    double n_g;
    long q;
    if (g <= 0.0f || g0 <= 0.0f || Delta <= 0.0f) return 0;
    n_g = log((double)g / (double)g0) / (0.6931471805599453 * (double)Delta);
    q = (long)floor(n_g + 0.5);
    if (q < -32768L) q = -32768L;
    if (q > 32767L) q = 32767L;
    return (int16_t)q;
}

float hqvm_norm_decode_gain16(int16_t n, float g0, float Delta) {
    (void)Delta;
    return g0 * hqvm_norm_pow2_delta(n);
}

/* Committed-chart access: LUT entry for a mantissa in [1,2). */
float hqvm_norm_rsqrt_mantissa(double mant) {
    int idx;
    if (mant < 1.0) mant = 1.0;
    if (mant >= 2.0) mant = 2.0 - 1e-12;
    hqvm_rsqrt_lut_init();
    idx = (int)floor((mant - 1.0) * (double)HQVM_RSQRT_LUT);
    if (idx < 0) idx = 0;
    if (idx >= HQVM_RSQRT_LUT) idx = HQVM_RSQRT_LUT - 1;
    return s_rsqrt_lut[idx];
}

/* Commit an inverse RMS gain onto the Delta-ruler; decodes the ruler tick. */
float hqvm_norm_commit_gain(float inv_gain) {
    const float gg0 = 1.0f;
    const int16_t ns = hqvm_norm_encode_gain16(inv_gain, gg0, (float)APERTURE_GAP);
    return hqvm_norm_decode_gain16(ns, gg0, (float)APERTURE_GAP);
}

/* Learned weight as its own ruler value (two-reference discipline: tensor
 * geomean reference supplied by the caller). */
float hqvm_norm_weight_commuted(float w, float ref, float Delta) {
    const float aw = fabsf(w);
    if (aw <= 0.0f) return 0.0f;
    {
        const int16_t n16 = hqvm_norm_encode_gain16(aw > 0.0f ? aw : ref, ref, Delta);
        const float wl = hqvm_norm_decode_gain16(n16, ref, Delta);
        return (w < 0.0f) ? -wl : wl;
    }
}

void hqvm_norm_apply_gain_ruler(float *w, int64_t n, float g0, float Delta) {
    int64_t i;
    if (!w || n <= 0 || g0 <= 0.0f) return;
    for (i = 0; i < n; ++i) {
        const float gi = fabsf(w[i]);
        const int16_t n16 = hqvm_norm_encode_gain16(gi > 0.0f ? gi : g0, g0, Delta);
        const float gh = hqvm_norm_decode_gain16(n16, g0, Delta);
        w[i] = (w[i] < 0.0f) ? -gh : gh;
    }
}

uint16_t hqvm_norm_encode_gain12(float g, float g0, float Delta) {
    int16_t n = hqvm_norm_encode_gain16(g, g0, Delta);
    if (n < 0) return 0;
    if (n > (int16_t)LAYER_MASK_12) return (uint16_t)LAYER_MASK_12;
    return (uint16_t)n;
}

float hqvm_norm_decode_gain12(uint16_t q, float g0, float Delta) {
    return hqvm_norm_decode_gain16((int16_t)q, g0, Delta);
}

uint8_t hqvm_norm_encode_gain(float g, float g0, float Delta) {
    int16_t n = hqvm_norm_encode_gain16(g, g0, Delta);
    if (n < 0) return 0;
    if (n > 255) return 255;
    return (uint8_t)n;
}

float hqvm_norm_decode_gain(uint8_t q, float g0, float Delta) {
    return hqvm_norm_decode_gain16((int16_t)q, g0, Delta);
}

void hqvm_norm_codec_shadow(const float *x, int64_t n, float eps, float Delta) {
    static int s_print = 0;
    int64_t i;
    double dot = 0.0, na = 0.0, nb = 0.0;
    float g, gh, g0;
    int16_t q;
    float *a, *b;
    if (!x || n <= 0) return;
    if (Delta <= 0.0f) Delta = (float)APERTURE_GAP;
    g0 = hqvm_norm_g0();
    a = (float *) malloc((size_t)n * sizeof(float));
    b = (float *) malloc((size_t)n * sizeof(float));
    if (!a || !b) { free(a); free(b); return; }
    g = hqvm_rms_gain(x, n, eps);
    q = hqvm_norm_encode_gain16(g, g0, Delta);
    gh = hqvm_norm_decode_gain16(q, g0, Delta);
    for (i = 0; i < n; ++i) {
        a[i] = x[i] * g;
        b[i] = x[i] * gh;
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (s_print < 40 && na > 0.0 && nb > 0.0) {
        const float rel = (g > 0.0f) ? (float)fabs((double)(gh - g) / (double)g) : 0.0f;
        fprintf(stderr, "[hqvm-norm-codec] cos=%.6f g=%.5f g_hat=%.5f n16=%d |g_hat-g|/g=%.4f\n",
            dot / (sqrt(na) * sqrt(nb)), g, gh, (int)q, rel);
        s_print++;
    }
    free(a);
    free(b);
}


/* ===== RoPE ===== */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float s_rope_cos[HQVM_ROPE_TICKS];
static float s_rope_sin[HQVM_ROPE_TICKS];
static int16_t s_rope_cos_q14[HQVM_ROPE_TICKS];
static int16_t s_rope_sin_q14[HQVM_ROPE_TICKS];
static uint16_t s_rope_dtheta[HQVM_ROPE_MAX_FREQ];
static int64_t s_rope_n_dims = 0;
static int s_rope_init = 0;
static int s_rope_dtheta_init = 0;

void hqvm_rope_codec_init(void) {
    int t;
    if (s_rope_init) return;
    for (t = 0; t < HQVM_ROPE_TICKS; ++t) {
        const float th = 2.0f * (float)GYRO_M_PI * (float)t / (float)HQVM_ROPE_TICKS;
        s_rope_cos[t] = cosf(th);
        s_rope_sin[t] = sinf(th);
        s_rope_cos_q14[t] = (int16_t)floorf(s_rope_cos[t] * 16384.0f + 0.5f);
        s_rope_sin_q14[t] = (int16_t)floorf(s_rope_sin[t] * 16384.0f + 0.5f);
    }
    s_rope_init = 1;
}

void hqvm_rope_init_dtheta(
    int64_t n_dims, float theta_scale, float freq_scale, const float *freq_factors)
{
    int64_t k, nfreq;
    double theta = 1.0;
    hqvm_rope_codec_init();
    if (n_dims <= 0) return;
    nfreq = n_dims / 2;
    if (nfreq > HQVM_ROPE_MAX_FREQ) nfreq = HQVM_ROPE_MAX_FREQ;
    for (k = 0; k < nfreq; ++k) {
        double th_k = theta;
        if (freq_factors) th_k /= (double)freq_factors[k];
        th_k *= (double)freq_scale;
        /* tick increment per position on T_256^(turn) */
        {
            long d = (long)floor(th_k / (2.0 * GYRO_M_PI) * (double)HQVM_ROPE_TICKS + 0.5);
            if (d < 0) d = 0;
            if (d > 65535L) d = 65535L;
            s_rope_dtheta[k] = (uint16_t)d;
        }
        theta *= (double)theta_scale;
    }
    s_rope_n_dims = n_dims;
    s_rope_dtheta_init = 1;
}

void hqvm_rope_ticks_from_pos(int64_t pos, int64_t n_dims, uint8_t *ticks_out) {
    int64_t k, nfreq;
    if (!ticks_out || n_dims <= 0) return;
    if (!s_rope_dtheta_init || s_rope_n_dims != n_dims) {
        /* Fallback: unit turn ticks from pos only (still no atan2). */
        nfreq = n_dims / 2;
        for (k = 0; k < nfreq; ++k) {
            ticks_out[k] = (uint8_t)((pos * (k + 1)) & 255);
        }
        return;
    }
    nfreq = n_dims / 2;
    if (nfreq > HQVM_ROPE_MAX_FREQ) nfreq = HQVM_ROPE_MAX_FREQ;
    for (k = 0; k < nfreq; ++k) {
        ticks_out[k] = (uint8_t)(((uint64_t)pos * (uint64_t)s_rope_dtheta[k]) & 255ull);
    }
}

int hqvm_rope_codec_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_ROPE_CODEC");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (s) hqvm_rope_codec_init();
    }
    return s;
}

uint8_t hqvm_rope_encode_tick(float theta) {
    float w = theta / (2.0f * (float)GYRO_M_PI);
    long t;
    w -= (float)floor(w);
    t = (long)floor(w * (float)HQVM_ROPE_TICKS + 0.5f);
    return (uint8_t)(t & 255);
}

static uint64_t s_rope_codec_calls = 0;
static uint64_t s_rope_stock_calls = 0;
static float s_max_sin_diff = 0.0f;
static float s_max_cos_diff = 0.0f;

void hqvm_rope_codec_audit_reset(void) {
    s_rope_codec_calls = 0;
    s_rope_stock_calls = 0;
    s_max_sin_diff = 0.0f;
    s_max_cos_diff = 0.0f;
}

void hqvm_rope_stock_inc(void) {
    s_rope_stock_calls++;
}

void hqvm_rope_codec_counters_get(uint64_t *codec_calls, uint64_t *stock_calls) {
    if (codec_calls) *codec_calls = s_rope_codec_calls;
    if (stock_calls) *stock_calls = s_rope_stock_calls;
}

void hqvm_rope_codec_audit_report(void) {
    fprintf(stderr,
        "[hqvm-rope-codec] rope_codec_calls=%llu rope_stock_calls=%llu "
        "max_abs_sin_diff=%.6g max_abs_cos_diff=%.6g\n",
        (unsigned long long)s_rope_codec_calls,
        (unsigned long long)s_rope_stock_calls,
        s_max_sin_diff, s_max_cos_diff);
}

void hqvm_rope_apply_pair(
    float x0, float x1, uint8_t tick, float sin_sign, float *y0, float *y1)
{
    /* Tick index is finite; Q14 multiply is available via GYRO_ROPE_Q14=1. */
    static int s_q14 = -1;
    if (s_q14 < 0) {
        const char *e = getenv("GYRO_ROPE_Q14");
        s_q14 = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (s_q14) {
        const int32_t c = (int32_t)s_rope_cos_q14[tick];
        const int32_t s = (int32_t)((float)s_rope_sin_q14[tick] * sin_sign);
        const float inv = 1.0f / 16384.0f;
        *y0 = (x0 * (float)c - x1 * (float)s) * inv;
        *y1 = (x0 * (float)s + x1 * (float)c) * inv;
    } else {
        const float c = s_rope_cos[tick];
        const float s = s_rope_sin[tick] * sin_sign;
        *y0 = x0 * c - x1 * s;
        *y1 = x0 * s + x1 * c;
    }
    s_rope_codec_calls++;
}

void hqvm_rope_apply_row(
    const float *src, float *dst, const uint8_t *ticks, int64_t n_dims,
    int64_t n_offset, int is_neox, float sin_sign)
{
    int64_t i;
    if (!src || !dst || !ticks) return;
    if (is_neox) {
        for (i = 0; i < n_dims / 2; ++i) {
            float y0, y1;
            hqvm_rope_apply_pair(src[i], src[i + n_offset], ticks[i], sin_sign, &y0, &y1);
            dst[i] = y0;
            dst[i + n_offset] = y1;
        }
    } else {
        for (i = 0; i < n_dims; i += 2) {
            float y0, y1;
            hqvm_rope_apply_pair(src[i], src[i + 1], ticks[i / 2], sin_sign, &y0, &y1);
            dst[i] = y0;
            dst[i + 1] = y1;
        }
    }
}

void hqvm_rope_codec_shadow(
    const float *src, const float *dst_stock, const float *cache,
    int64_t n_dims, int64_t n_offset, int is_neox, float sin_sign)
{
    static int s_print = 0;
    int64_t i, npairs;
    double dot = 0.0, na = 0.0, nb = 0.0;
    float max_sd = 0.0f, max_cd = 0.0f;
    if (!src || !dst_stock || !cache || n_dims <= 0) return;
    if (!s_rope_init) hqvm_rope_codec_init();
    npairs = n_dims / 2;
    for (i = 0; i < npairs; ++i) {
        const float c = cache[2 * i];
        const float s = cache[2 * i + 1];
        const float theta = atan2f(s, c);
        const uint8_t tick = hqvm_rope_encode_tick(theta);
        const float cl = s_rope_cos[tick];
        const float sl = s_rope_sin[tick] * sin_sign;
        float x0, x1, y0, y1, s0, s1;
        float sd = fabsf(s - sl), cd = fabsf(c - cl);
        if (sd > max_sd) max_sd = sd;
        if (cd > max_cd) max_cd = cd;
        if (sd > s_max_sin_diff) s_max_sin_diff = sd;
        if (cd > s_max_cos_diff) s_max_cos_diff = cd;
        if (is_neox) {
            x0 = src[i]; x1 = src[i + n_offset];
            s0 = dst_stock[i]; s1 = dst_stock[i + n_offset];
        } else {
            x0 = src[2 * i]; x1 = src[2 * i + 1];
            s0 = dst_stock[2 * i]; s1 = dst_stock[2 * i + 1];
        }
        hqvm_rope_apply_pair(x0, x1, tick, sin_sign, &y0, &y1);
        dot += (double)y0 * s0 + (double)y1 * s1;
        na += (double)y0 * y0 + (double)y1 * y1;
        nb += (double)s0 * s0 + (double)s1 * s1;
    }
    if (s_print < 20 && na > 0.0 && nb > 0.0) {
        fprintf(stderr, "[hqvm-rope-codec] cos=%.6f max_sin_diff=%.6g max_cos_diff=%.6g pairs=%lld\n",
            dot / (sqrt(na) * sqrt(nb)), max_sd, max_cd, (long long)npairs);
        s_print++;
    }
}


/* ===== FFN joint law (Theory_Drop Runtime §4.1.2) =====
 * Shell-only `up * simplex(λ^N)` discards controller magnitude — do not revive.
 * Documented production L2: silu(gate)·up·(1+Δ·m)·(1+0.25Δ·m_req).
 * SiLU here is the pretrained controller chart; shell/family/request enter only
 * as aperture gains (Analysis §7.4 joint-law parallel). Opt-in GYRO_FFN_NATIVE=1.
 * Eventual carrier-native nonlinearity (Analysis §7.7) is separate work — not a
 * softstep / mag-LUT / scale-sweep stand-in for SiLU. */
static int s_ffn_gate_init = 0;
static uint64_t s_stock_silu_calls = 0;

void hqvm_ffn_shell_gate_init(void) {
    if (s_ffn_gate_init) return;
    s_ffn_gate_init = 1;
}

int hqvm_ffn_shell_gate_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_FFN_NATIVE");
        if (!(e && e[0] == '1')) e = getenv("GYRO_FFN_SHELL_GATE");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (s) hqvm_ffn_shell_gate_init();
    }
    return s;
}

void hqvm_stock_silu_inc(void) {
    s_stock_silu_calls++;
}

uint64_t hqvm_stock_silu_calls(void) {
    return s_stock_silu_calls;
}

/* Documented FFN L2 joint law. Not shell-only. Not a SiLU LUT costume for Ω. */
void hqvm_ffn_shell_gate_apply_native(
    float *dst, const float *gate, const float *up, int64_t n,
    uint8_t fam, uint8_t Nc)
{
    int64_t b, i;
    const float Delta = (float)APERTURE_GAP;
    float m_req;
    float g_req;
    if (!dst || !gate || !up || n <= 0) return;
    hqvm_ffn_shell_gate_init();
    fam = (uint8_t)(fam & 3);
    if (Nc > 6) Nc = 6;
    m_req = ((float)Nc - 3.0f) / 3.0f;
    g_req = 1.0f + 0.25f * Delta * m_req;

    for (b = 0; b + 64 <= n; b += 64) {
        uint64_t signs = 0;
        uint8_t chi;
        int Ng, Nf;
        float m, g_shell, gain;
        for (i = 0; i < 64; ++i) {
            if (gate[b + i] >= 0.0f) signs |= (1ull << i);
        }
        chi = gyroscopic_chirality_from_signs64(signs);
#if defined(_MSC_VER)
        Ng = (int)__popcnt((unsigned)(chi & 63));
#else
        Ng = __builtin_popcount((unsigned)(chi & 63));
#endif
        if (Ng > 6) Ng = 6;
        /* Family rotates shell index (Genealogy phase); m = (N-3)/3. */
        Nf = (Ng + (int)fam) % 7;
        m = ((float)Nf - 3.0f) / 3.0f;
        g_shell = 1.0f + Delta * m;
        gain = g_shell * g_req;
        for (i = 0; i < 64; ++i) {
            const float z = gate[b + i];
            const float silu = z / (1.0f + expf(-z));
            dst[b + i] = silu * up[b + i] * gain;
        }
    }
    if (b < n) {
        int Ng = (int)Nc;
        int Nf = (Ng + (int)fam) % 7;
        float m = ((float)Nf - 3.0f) / 3.0f;
        float gain = (1.0f + Delta * m) * g_req;
        for (; b < n; ++b) {
            const float z = gate[b];
            const float silu = z / (1.0f + expf(-z));
            dst[b] = silu * up[b] * gain;
        }
    }
}

void hqvm_residual_gain_q16(uint8_t Nc, int32_t *num, int32_t *den) {
    const int32_t gap = HQVM_APERTURE_GAP_Q16;
    const int32_t nc = (int32_t)(Nc <= 6u ? Nc : 6u);
    /* gain = 1 + gap_q16/65536 * (Nc-3)/3 = (3*65536 + gap*(Nc-3)) / (3*65536) */
    if (num) *num = 3 * 65536 + gap * (nc - 3);
    if (den) *den = 3 * 65536;
}

float hqvm_residual_gain_from_Nc(uint8_t Nc) {
    int32_t num = 0, den = 1;
    hqvm_residual_gain_q16(Nc, &num, &den);
    return (float)num / (float)den;
}

void hqvm_manifold_gain_q16(
    uint8_t chi_bit0, uint8_t p0, uint8_t chi_bit1, uint8_t p1,
    int32_t *num, int32_t *den)
{
    const int32_t gap = HQVM_APERTURE_GAP_Q16;
    const int s0 = ((p0 ^ chi_bit0) & 1u) ? 1 : -1;
    const int s1 = ((p1 ^ chi_bit1) & 1u) ? 1 : -1;
    /* 1 + gap * 0.5 * (s0+s1) = 1 + gap * half_sum / 65536, half_sum in {-1,0,1} */
    const int32_t half_sum = (int32_t)((s0 + s1) / 2);
    if (num) *num = 65536 + gap * half_sum;
    if (den) *den = 65536;
}

float hqvm_manifold_gain_from_bits(
    uint8_t chi_bit0, uint8_t p0, uint8_t chi_bit1, uint8_t p1)
{
    int32_t num = 0, den = 1;
    hqvm_manifold_gain_q16(chi_bit0, p0, chi_bit1, p1, &num, &den);
    return (float)num / (float)den;
}

int hqvm_ffn_native_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_FFN_NATIVE");
        if (!(e && e[0] == '1')) e = getenv("GYRO_FFN_SHELL_GATE");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

int hqvm_norm_ruler_dyad(
    const hqvm_dyad32_t *x_in,
    hqvm_dyad32_t *x_out,
    int64_t n,
    const float *g,
    float g0)
{
    static int s_plain = -1;
    float *tmp = NULL;
    float scale;
    int64_t i;

    hqvm_gate_counters_inc_norm();
    if (!x_in || !x_out || n <= 0) return -1;

    if (s_plain < 0) {
        const char *e = getenv("GYRO_NATIVE_NORM");
        s_plain = (e && strcmp(e, "plain") == 0) ? 1 : 0;
        if (s_plain) {
            fprintf(stderr, "[hqvm-norm] plain RMS (no Delta-ruler)\n");
            fflush(stderr);
        } else {
            fprintf(stderr, "[hqvm-norm] mode=delta-ruler\n");
            fflush(stderr);
        }
    }

    tmp = (float *)malloc((size_t)n * sizeof(float));
    if (!tmp) return -1;
    for (i = 0; i < n; ++i) tmp[i] = hqvm_dyad32_to_f32(x_in[i]);

    if (s_plain) {
        scale = hqvm_rms_gain(tmp, n, 1e-6f);
        for (i = 0; i < n; ++i) {
            const float wi = g ? g[i] : 1.0f;
            x_out[i] = hqvm_dyad32_from_f32(tmp[i] * scale * wi);
        }
        free(tmp);
        return 0;
    }

    scale = hqvm_rms_gain_fixed(tmp, n, 1e-6f);
    hqvm_norm_ruler_commit_inc();
    {
        const int16_t ns = hqvm_norm_encode_gain16(scale, 1.0f, (float)APERTURE_GAP);
        scale = hqvm_norm_decode_gain16(ns, 1.0f, (float)APERTURE_GAP);
    }
    for (i = 0; i < n; ++i) {
        float wi = 1.0f;
        if (g) {
            const float gi = fabsf(g[i]);
            const float g_ref = (g0 > 0.0f) ? g0 : 1.0f;
            const int16_t n16 = hqvm_norm_encode_gain16(
                gi > 0.0f ? gi : g_ref, g_ref, (float)APERTURE_GAP);
            wi = hqvm_norm_decode_gain16(n16, g_ref, (float)APERTURE_GAP);
            if (g[i] < 0.0f) wi = -wi;
        }
        x_out[i] = hqvm_dyad32_from_f32(tmp[i] * scale * wi);
    }
    free(tmp);
    return 0;
}

static float rope_freq_base(void) {
    static float s = -1.0f;
    if (s < 0.0f) {
        const char *e = getenv("GYRO_ROPE_FREQ_BASE");
        s = (e && e[0]) ? (float)atof(e) : 1000000.0f;
        if (s <= 0.0f) s = 1000000.0f;
        fprintf(stderr, "[hqvm-rope] freq_base=%.0f\n", (double)s);
        fflush(stderr);
    }
    return s;
}

static float rope_freq_scale(void) {
    static float s = -1.0f;
    if (s < 0.0f) {
        const char *e = getenv("GYRO_ROPE_FREQ_SCALE");
        s = (e && e[0]) ? (float)atof(e) : 0.25f;
        if (s <= 0.0f) s = 0.25f;
        fprintf(stderr, "[hqvm-rope] freq_scale=%g\n", (double)s);
        fflush(stderr);
    }
    return s;
}

static void rope_yarn_ticks(int32_t token_pos, uint8_t *ticks) {
    const float freq_base = rope_freq_base();
    const float freq_scale = rope_freq_scale();
    const int np = HQVM_HEAD_DIM / 2;
    const int n_dims = HQVM_HEAD_DIM;
    const float n_ctx_orig = 16384.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;
    const float ext_factor = 1.0f;
    float corr0, corr1;
    float theta = (float)token_pos;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    int i;
    hqvm_rope_codec_init();
    {
        const float start = (float)n_dims * logf(n_ctx_orig / (beta_fast * 2.0f * (float)GYRO_M_PI))
            / (2.0f * logf(freq_base));
        const float end = (float)n_dims * logf(n_ctx_orig / (beta_slow * 2.0f * (float)GYRO_M_PI))
            / (2.0f * logf(freq_base));
        corr0 = start < 0.0f ? 0.0f : floorf(start);
        corr1 = end > (float)(n_dims - 1) ? (float)(n_dims - 1) : ceilf(end);
    }
    for (i = 0; i < np; ++i) {
        const float theta_extrap = theta;
        const float theta_interp = freq_scale * theta_extrap;
        float ramp = 1.0f - ((float)i - corr0) / (corr1 - corr0 + 1e-3f);
        float th;
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        ramp *= ext_factor;
        th = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
        ticks[i] = hqvm_rope_encode_tick(th);
        theta *= theta_scale;
    }
}

static void rope_apply_head_dyad(hqvm_dyad32_t *row, const uint8_t *ticks) {
    int i;
    const int np = HQVM_HEAD_DIM / 2;
    for (i = 0; i < np; ++i) {
        float x0 = hqvm_dyad32_to_f32(row[2 * i]);
        float x1 = hqvm_dyad32_to_f32(row[2 * i + 1]);
        float y0, y1;
        hqvm_rope_apply_pair(x0, x1, ticks[i], 1.0f, &y0, &y1);
        row[2 * i] = hqvm_dyad32_from_f32(y0);
        row[2 * i + 1] = hqvm_dyad32_from_f32(y1);
    }
    hqvm_rope_codec_row_inc();
}

static void rope_apply_heads_float(
    hqvm_dyad32_t *Q, hqvm_dyad32_t *K,
    int32_t n_heads, int32_t n_kv, int32_t token_pos)
{
    const float freq_base = rope_freq_base();
    const float freq_scale = rope_freq_scale();
    const int np = HQVM_HEAD_DIM / 2;
    const int n_dims = HQVM_HEAD_DIM;
    const float n_ctx_orig = 16384.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;
    const float ext_factor = 1.0f;
    const float mscale = 1.0f;
    float corr0, corr1;
    float cos_t[64], sin_t[64];
    float theta = (float)token_pos;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    int i, h, kv_h;
    {
        const float start = (float)n_dims * logf(n_ctx_orig / (beta_fast * 2.0f * (float)GYRO_M_PI))
            / (2.0f * logf(freq_base));
        const float end = (float)n_dims * logf(n_ctx_orig / (beta_slow * 2.0f * (float)GYRO_M_PI))
            / (2.0f * logf(freq_base));
        corr0 = start < 0.0f ? 0.0f : floorf(start);
        corr1 = end > (float)(n_dims - 1) ? (float)(n_dims - 1) : ceilf(end);
    }
    for (i = 0; i < np; ++i) {
        const float theta_extrap = theta;
        const float theta_interp = freq_scale * theta_extrap;
        float ramp = 1.0f - ((float)i - corr0) / (corr1 - corr0 + 1e-3f);
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        ramp *= ext_factor;
        {
            const float th = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
            cos_t[i] = cosf(th) * mscale;
            sin_t[i] = sinf(th) * mscale;
        }
        theta *= theta_scale;
    }
    for (h = 0; h < n_heads; ++h) {
        hqvm_dyad32_t *qh = Q + h * HQVM_HEAD_DIM;
        for (i = 0; i < np; ++i) {
            const float x0 = hqvm_dyad32_to_f32(qh[2 * i]);
            const float x1 = hqvm_dyad32_to_f32(qh[2 * i + 1]);
            qh[2 * i] = hqvm_dyad32_from_f32(x0 * cos_t[i] - x1 * sin_t[i]);
            qh[2 * i + 1] = hqvm_dyad32_from_f32(x0 * sin_t[i] + x1 * cos_t[i]);
        }
    }
    for (kv_h = 0; kv_h < n_kv; ++kv_h) {
        hqvm_dyad32_t *kh = K + kv_h * HQVM_HEAD_DIM;
        for (i = 0; i < np; ++i) {
            const float x0 = hqvm_dyad32_to_f32(kh[2 * i]);
            const float x1 = hqvm_dyad32_to_f32(kh[2 * i + 1]);
            kh[2 * i] = hqvm_dyad32_from_f32(x0 * cos_t[i] - x1 * sin_t[i]);
            kh[2 * i + 1] = hqvm_dyad32_from_f32(x0 * sin_t[i] + x1 * cos_t[i]);
        }
    }
}

int hqvm_rope_qk_dyad(
    hqvm_dyad32_t *Q,
    hqvm_dyad32_t *K,
    int32_t n_heads,
    int32_t gqa_ratio,
    int32_t token_pos)
{
    static int s_mode = -1;
    int32_t n_kv;
    int h, kv_h;
    uint8_t ticks[HQVM_ROPE_MAX_FREQ];

    hqvm_gate_counters_inc_rope();
    if (!Q || !K || n_heads <= 0 || gqa_ratio <= 0) return -1;

    if (s_mode < 0) {
        const char *e = getenv("GYRO_NATIVE_ROPE");
        if (e && e[0] == '0') s_mode = 1;
        else if (e && strcmp(e, "float") == 0) s_mode = 2;
        else s_mode = 0;
        fprintf(stderr, "[hqvm-rope] mode=%s\n",
            s_mode == 1 ? "skip" : (s_mode == 2 ? "float" : "tick"));
        fflush(stderr);
    }

    n_kv = n_heads / gqa_ratio;
    if (n_kv <= 0) n_kv = 1;

    if (s_mode == 1) return 0;

    if (s_mode == 2) {
        rope_apply_heads_float(Q, K, n_heads, n_kv, token_pos);
        return 0;
    }

    rope_yarn_ticks(token_pos, ticks);
    for (h = 0; h < n_heads; ++h) {
        rope_apply_head_dyad(Q + h * HQVM_HEAD_DIM, ticks);
    }
    for (kv_h = 0; kv_h < n_kv; ++kv_h) {
        rope_apply_head_dyad(K + kv_h * HQVM_HEAD_DIM, ticks);
    }
    return 0;
}

static void stock_swiglu(float *dst, const float *gate, const float *up, int64_t n) {
    int64_t i;
    for (i = 0; i < n; ++i) {
        const float z = gate[i];
        const float sig = 1.0f / (1.0f + expf(-z));
        dst[i] = z * sig * up[i];
    }
}

int hqvm_ffn_gate_dyad(
    const hqvm_dyad32_t *gate,
    const hqvm_dyad32_t *up,
    hqvm_dyad32_t *dst,
    int64_t n,
    uint8_t fam,
    uint8_t Nc)
{
    static int s_logged = 0;
    float g64[64], u64[64], d64[64];
    int64_t b, i;
    const int native = hqvm_ffn_native_enabled();

    hqvm_gate_counters_inc_swiglu();
    if (!gate || !up || !dst || n <= 0) return -1;

    if (!s_logged) {
        fprintf(stderr, "[hqvm-ffn] mode=%s\n",
            native ? "NATIVE-shell (opt-in)" : "stock-SwiGLU");
        fflush(stderr);
        s_logged = 1;
    }

    for (b = 0; b < n; b += 64) {
        const int64_t chunk = (b + 64 <= n) ? 64 : (n - b);
        for (i = 0; i < chunk; ++i) {
            g64[i] = hqvm_dyad32_to_f32(gate[b + i]);
            u64[i] = hqvm_dyad32_to_f32(up[b + i]);
        }
        if (native) {
            hqvm_ffn_shell_gate_apply_native(d64, g64, u64, chunk, fam, Nc);
            hqvm_ffn_shell_gate_inc();
        } else {
            stock_swiglu(d64, g64, u64, chunk);
        }
        for (i = 0; i < chunk; ++i) {
            dst[b + i] = hqvm_dyad32_from_f32(d64[i]);
        }
    }
    return 0;
}

