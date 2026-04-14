#pragma once

#if !defined(GYROLABE_EXPORT)
#if defined(_WIN32) || defined(_WIN64)
#  define GYROLABE_EXPORT __declspec(dllexport)
#else
#  define GYROLABE_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

GYROLABE_EXPORT void gyrolabe_wht16_f32_inplace(float * x);

GYROLABE_EXPORT void gyrolabe_chi_gauge_input_wht(const float * x64, float xwht[4][16]);

GYROLABE_EXPORT void gyrolabe_apply_spectral_tile64_gauge4_lifted(
    const float xwht[4][16],
    const float * gauge_eigen256,
    float * y64
);

GYROLABE_EXPORT void gyrolabe_apply_spectral_tile64_gauge4(
    const float * x64,
    const float * gauge_eigen256,
    float * y64
);

#ifdef __cplusplus
}
#endif
