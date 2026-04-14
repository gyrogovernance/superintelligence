#include "gyrograph_policy.h"

#include "gyrolabe.h"

#include <stdlib.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <pthread.h>
#endif

static int env_truthy(const char * v) {
    if (v == NULL || v[0] == '\0') {
        return 0;
    }
    return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
}

static void gyro_policy_load(gyro_policy * out) {
    const char * mode_env = getenv("GGML_GYROSCOPIC");
#if defined(GGML_USE_GYROSCOPIC) || defined(GGML_USE_GYROSCOPIC_GRAPH)
    if (mode_env == NULL || mode_env[0] == '\0') {
        out->mode = GYRO_MODE_GYROSCOPIC;
    } else {
        out->mode = env_truthy(mode_env) ? GYRO_MODE_GYROSCOPIC : GYRO_MODE_STOCK;
    }
#else
    out->mode = env_truthy(mode_env) ? GYRO_MODE_GYROSCOPIC : GYRO_MODE_STOCK;
#endif
    out->strict = env_truthy(getenv("GGML_GYROSCOPIC_STRICT")) ? 1 : 0;
    out->trace = env_truthy(getenv("GGML_GYROSCOPIC_TRACE")) ? 1 : 0;
}

static gyro_policy g_policy_storage;

#if defined(_WIN32)
static INIT_ONCE g_gyro_policy_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK gyro_policy_once_cb(PINIT_ONCE po, PVOID pv, PVOID * ctx) {
    (void) po;
    (void) pv;
    (void) ctx;
    gyro_policy_load(&g_policy_storage);
    return TRUE;
}
#else
static pthread_once_t g_gyro_policy_once = PTHREAD_ONCE_INIT;

static void gyro_policy_load_once(void) {
    gyro_policy_load(&g_policy_storage);
}
#endif

const gyro_policy * gyro_policy_get(void) {
#if defined(_WIN32)
    PVOID ctx = NULL;
    InitOnceExecuteOnce(&g_gyro_policy_once, gyro_policy_once_cb, NULL, &ctx);
#else
    pthread_once(&g_gyro_policy_once, gyro_policy_load_once);
#endif
    return &g_policy_storage;
}

void gyromatmul_runtime_query(gyromatmul_runtime_caps * out_caps) {
    if (out_caps == NULL) {
        return;
    }

    out_caps->avx2_enabled = 0u;
#if defined(__AVX2__)
    out_caps->avx2_enabled = 1u;
#endif

#if defined(__F16C__) || (defined(_MSC_VER) && defined(__AVX2__))
    out_caps->f16c_enabled = 1u;
#else
    out_caps->f16c_enabled = 0u;
#endif

#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
    out_caps->fma_enabled = 1u;
#else
    out_caps->fma_enabled = 0u;
#endif

    out_caps->reserved = 0u;
}
