#include "codec.h"

#include "constants.h"
#include "kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Encode gain on the Delta-ruler (Formalism §7): n = round(log2(g/g0)/Delta).
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
    const float c = s_rope_cos[tick];
    const float s = s_rope_sin[tick] * sin_sign;
    *y0 = x0 * c - x1 * s;
    *y1 = x0 * s + x1 * c;
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


/* ===== SiLU ===== */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float s_silu_lut[HQVM_SILU_BINS];
static float s_silu_clip = 0.0f;
static int s_silu_init = 0;

static float silu_exact(float x) {
    return x / (1.0f + expf(-x));
}

void hqvm_silu_codec_init(void) {
    hqvm_silu_codec_init_range(10.0f);
}

void hqvm_silu_codec_init_range(float clip) {
    int i;
    if (clip <= 0.0f) clip = 10.0f;
    s_silu_clip = clip;
    for (i = 0; i < HQVM_SILU_BINS; ++i) {
        const float xc = -clip + (2.0f * clip) * ((float)i + 0.5f) / (float)HQVM_SILU_BINS;
        s_silu_lut[i] = silu_exact(xc);
    }
    s_silu_init = 1;
}

int hqvm_silu_codec_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_SILU_CODEC");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (s) hqvm_silu_codec_init();
    }
    return s;
}

void hqvm_silu_apply(float *x, int64_t n, float clip) {
    int64_t i;
    if (!x || n <= 0) return;
    if (!s_silu_init || clip != s_silu_clip) hqvm_silu_codec_init_range(clip);
    for (i = 0; i < n; ++i) {
        float v = x[i];
        int b;
        if (v < -clip) v = -clip;
        if (v > clip) v = clip;
        b = (int)(((v + clip) / (2.0f * clip)) * (float)HQVM_SILU_BINS);
        if (b < 0) b = 0;
        if (b >= HQVM_SILU_BINS) b = HQVM_SILU_BINS - 1;
        x[i] = s_silu_lut[b];
    }
}

void hqvm_swiglu_apply(
    float *dst, const float *gate, const float *up, int64_t n, float clip)
{
    int64_t i;
    if (!dst || !gate || !up || n <= 0) return;
    if (!s_silu_init || clip != s_silu_clip) hqvm_silu_codec_init_range(clip);
    for (i = 0; i < n; ++i) {
        float v = gate[i];
        int b;
        if (v < -clip) v = -clip;
        if (v > clip) v = clip;
        b = (int)(((v + clip) / (2.0f * clip)) * (float)HQVM_SILU_BINS);
        if (b < 0) b = 0;
        if (b >= HQVM_SILU_BINS) b = HQVM_SILU_BINS - 1;
        dst[i] = s_silu_lut[b] * up[i];
    }
}

void hqvm_silu_codec_shadow(const float *x, int64_t n, float clip) {
    static int s_print = 0;
    int64_t i;
    double dot = 0.0, na = 0.0, nb = 0.0, maxerr = 0.0, maxabs = 0.0;
    if (!x || n <= 0) return;
    if (!s_silu_init || clip != s_silu_clip) hqvm_silu_codec_init_range(clip);
    for (i = 0; i < n; ++i) {
        float v = x[i], ve = silu_exact(v), vl;
        int b;
        float vc = v;
        double ax = v < 0 ? -(double)v : (double)v;
        if (ax > maxabs) maxabs = ax;
        if (vc < -clip) vc = -clip;
        if (vc > clip) vc = clip;
        b = (int)(((vc + clip) / (2.0f * clip)) * (float)HQVM_SILU_BINS);
        if (b < 0) b = 0;
        if (b >= HQVM_SILU_BINS) b = HQVM_SILU_BINS - 1;
        vl = s_silu_lut[b];
        dot += (double)ve * vl;
        na += (double)ve * ve;
        nb += (double)vl * vl;
        {
            double e = (double)ve - vl;
            if (e < 0) e = -e;
            if (e > maxerr) maxerr = e;
        }
    }
    if (s_print < 40 && na > 0.0 && nb > 0.0) {
        fprintf(stderr, "[hqvm-silu-codec] cos=%.6f maxerr=%.5f maxabs=%.3f n=%lld\n",
            dot / (sqrt(na) * sqrt(nb)), maxerr, maxabs, (long long)n);
        s_print++;
    }
}

void hqvm_swiglu_codec_shadow(
    const float *gate, const float *up, int64_t n, float clip)
{
    static int s_print = 0;
    int64_t i;
    double dot = 0.0, na = 0.0, nb = 0.0, maxabs = 0.0;
    if (!gate || !up || n <= 0) return;
    if (!s_silu_init || clip != s_silu_clip) hqvm_silu_codec_init_range(clip);
    for (i = 0; i < n; ++i) {
        float g = gate[i], ve = silu_exact(g) * up[i], vl;
        int b;
        float vc = g;
        double ax = g < 0 ? -(double)g : (double)g;
        if (ax > maxabs) maxabs = ax;
        if (vc < -clip) vc = -clip;
        if (vc > clip) vc = clip;
        b = (int)(((vc + clip) / (2.0f * clip)) * (float)HQVM_SILU_BINS);
        if (b < 0) b = 0;
        if (b >= HQVM_SILU_BINS) b = HQVM_SILU_BINS - 1;
        vl = s_silu_lut[b] * up[i];
        dot += (double)ve * vl;
        na += (double)ve * ve;
        nb += (double)vl * vl;
    }
    if (s_print < 40 && na > 0.0 && nb > 0.0) {
        fprintf(stderr, "[hqvm-swiglu-codec] cos=%.6f maxabs=%.3f n=%lld\n",
            dot / (sqrt(na) * sqrt(nb)), maxabs, (long long)n);
        s_print++;
    }
}


/* ===== Aperture Norm / SiLU ===== */
int hqvm_aperture_silu_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_APERTURE_SILU");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

/* Aperture RMSNorm: scale = 1 / (rms * (1 - eps_shell)), eps_shell from shell
 * popcount of the row's chirality. Keeps the magnitude channel; enforces that
 * the norm cannot collapse energy to zero (Delta as irreducible opening). */
void hqvm_aperture_rms_scale(float *row, int64_t n, float Delta) {
    int64_t i;
    double ss = 0.0;
    float rms, scale;
    uint64_t signs = 0;
    uint8_t chi;
    int shell;
    float eps;
    if (!row || n <= 0) return;
    for (i = 0; i < n; ++i) ss += (double)row[i] * (double)row[i];
    rms = (float)sqrt(ss / (double)n);
    if (rms <= 0.0f) return;
    for (i = 0; i < 64 && i < n; ++i) if (row[i] >= 0.0f) signs |= (1ull << i);
    chi = gyroscopic_chirality_from_signs64(signs);
    shell = gyroscopic_chirality_distance(chi, 0);
    eps = (float)(6 - shell) * Delta;
    if (eps > 0.25f) eps = 0.25f;
    if (eps < 0.0f) eps = 0.0f;
    scale = 1.0f / (rms * (1.0f - eps));
    for (i = 0; i < n; ++i) row[i] *= scale;
}

/* Aperture SiLU: y = silu(x)*(1-eps) + eps*x. The identity aperture keeps the
 * gate from fully closing (Gate-F cannot annihilate the signal). */
void hqvm_aperture_silu(float *row, int64_t n, float Delta) {
    int64_t i;
    const float eps = Delta;
    if (!row || n <= 0) return;
    for (i = 0; i < n; ++i) {
        const float x = row[i];
        const float silu = x / (1.0f + expf(-x));
        row[i] = silu * (1.0f - eps) + eps * x;
    }
}

/* ===== FFN shell gate (QuBEC occupation; not SiLU LUT) ===== */
#define HQVM_LAMBDA_MAX 8.0f
static float s_gate_factor[4][7];
static int s_ffn_gate_init = 0;
static uint64_t s_stock_silu_calls = 0;

static float hqvm_lambda_from_Nc(uint8_t Nc) {
    /* m = (Nc-3)/3; lambda = (1+m)/(1-m); Nc=6 -> lambda_max */
    static const float lam[7] = {
        0.0f, 0.2f, 0.5f, 1.0f, 2.0f, 5.0f, HQVM_LAMBDA_MAX
    };
    if (Nc > 6) Nc = 6;
    return lam[Nc];
}

void hqvm_ffn_shell_gate_init(void) {
    int fam, N;
    if (s_ffn_gate_init) return;
    /* Theory table: factors from normalized lambda^N mass (lambda=1 baseline),
     * scaled into (0,1]; fam rotates the monotone profile by K4 phase. */
    for (fam = 0; fam < 4; ++fam) {
        float sum = 0.0f;
        float raw[7];
        for (N = 0; N < 7; ++N) {
            /* Base monotone in N; fam shifts emphasis */
            const int Np = (N + fam) % 7;
            raw[N] = (float)(1 + Np);
            sum += raw[N];
        }
        for (N = 0; N < 7; ++N) {
            s_gate_factor[fam][N] = raw[N] / sum;
        }
    }
    s_ffn_gate_init = 1;
}

int hqvm_ffn_shell_gate_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_FFN_SHELL_GATE");
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

void hqvm_ffn_shell_gate_apply(
    float *dst, const float *gate, const float *up, int64_t n,
    uint8_t fam, uint8_t Nc)
{
    int64_t b, i;
    float lam, lam_pow[7];
    float wN[7];
    float Z = 0.0f;
    if (!dst || !gate || !up || n <= 0) return;
    hqvm_ffn_shell_gate_init();
    fam = (uint8_t)(fam & 3);
    lam = hqvm_lambda_from_Nc(Nc);
    lam_pow[0] = 1.0f;
    for (i = 1; i <= 6; ++i) lam_pow[i] = lam_pow[i - 1] * lam;
    for (i = 0; i < 7; ++i) {
        wN[i] = s_gate_factor[fam][i] * lam_pow[i];
        Z += wN[i];
    }
    if (Z <= 0.0f) Z = 1.0f;
    for (i = 0; i < 7; ++i) wN[i] /= Z;

    for (b = 0; b + 64 <= n; b += 64) {
        uint64_t signs = 0;
        uint8_t chi;
        int Ng;
        float f;
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
        f = wN[Ng];
        for (i = 0; i < 64; ++i) {
            dst[b + i] = up[b + i] * f;
        }
    }
    /* Tail < 64: use last shell factor / mean */
    for (; b < n; ++b) {
        dst[b] = up[b] * wN[3];
    }
}
