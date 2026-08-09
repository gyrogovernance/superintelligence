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

/* Encode gain on the Delta-ruler (Formalism §7): n_g = log2(g/g0)/Delta.
 * Quantize into the 12-bit GENE_Mac frame (LAYER_MASK_12). Delta is the ruler
 * UNIT (APERTURE_GAP); not an invented fine step. g0=1 (gain is dimensionless). */
uint16_t hqvm_norm_encode_gain12(float g, float g0, float Delta) {
    float n_g;
    long q;
    if (g <= 0.0f || g0 <= 0.0f || Delta <= 0.0f) return 0;
    n_g = (float)(log(g / g0) / (0.6931471805599453 * (double)Delta));
    q = (long)floor((double)n_g + 0.5);
    if (q < 0) q = 0;
    if (q > (long)LAYER_MASK_12) q = (long)LAYER_MASK_12;
    return (uint16_t)q;
}

float hqvm_norm_decode_gain12(uint16_t q, float g0, float Delta) {
    return (float)((double)g0 * pow(2.0, (double)q * (double)Delta));
}

uint8_t hqvm_norm_encode_gain(float g, float g0, float Delta) {
    uint16_t q12 = hqvm_norm_encode_gain12(g, g0, Delta);
    return (uint8_t)(q12 > 255 ? 255 : q12);
}

float hqvm_norm_decode_gain(uint8_t q, float g0, float Delta) {
    return hqvm_norm_decode_gain12((uint16_t)q, g0, Delta);
}

void hqvm_norm_codec_shadow(const float *x, int64_t n, float eps, float Delta) {
    static int s_print = 0;
    int64_t i;
    double dot = 0.0, na = 0.0, nb = 0.0;
    float g, gh, g0 = 1.0f;
    uint16_t q;
    float *a, *b;
    if (!x || n <= 0) return;
    if (Delta <= 0.0f) Delta = (float)APERTURE_GAP;
    a = (float *) malloc((size_t)n * sizeof(float));
    b = (float *) malloc((size_t)n * sizeof(float));
    if (!a || !b) { free(a); free(b); return; }
    g = hqvm_rms_gain(x, n, eps);
    q = hqvm_norm_encode_gain12(g, g0, Delta);
    gh = hqvm_norm_decode_gain12(q, g0, Delta);
    for (i = 0; i < n; ++i) {
        a[i] = x[i] * g;
        b[i] = x[i] * gh;
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (s_print < 40 && na > 0.0 && nb > 0.0) {
        fprintf(stderr, "[hqvm-norm-codec] cos=%.6f g=%.5f g_hat=%.5f q12=%u rel=%.4f\n",
            dot / (sqrt(na) * sqrt(nb)), g, gh, (unsigned)q,
            g > 0.0f ? (gh - g) / g : 0.0f);
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
static int s_rope_init = 0;

void hqvm_rope_codec_init(void) {
    int t;
    if (s_rope_init) return;
    for (t = 0; t < HQVM_ROPE_TICKS; ++t) {
        const float th = 2.0f * (float)GYRO_M_PI * (float)t / (float)HQVM_ROPE_TICKS;
        s_rope_cos[t] = cosf(th);
        s_rope_sin[t] = sinf(th);
    }
    s_rope_init = 1;
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
