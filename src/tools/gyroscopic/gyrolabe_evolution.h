#ifndef GYROLABE_EVOLUTION_H
#define GYROLABE_EVOLUTION_H

#include <stdint.h>

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

GYROLABE_EXPORT void gyrolabe_chirality_evolve_n(
    const int32_t chi_hist[64],
    const int32_t byte_ensemble[64],
    uint64_t n_steps,
    int32_t out[64]
);

GYROLABE_EXPORT void gyrolabe_shell_evolve_n(
    const int32_t shell_hist[7],
    const float eigenvals[7],
    uint64_t n_steps,
    int32_t out[7]
);

GYROLABE_EXPORT int gyrolabe_horizon_proximity(
    const int32_t chi_hist[64],
    float threshold
);

GYROLABE_EXPORT void gyrolabe_anisotropy_extract(
    const int32_t byte_ensemble[256],
    float eta_vec[6]
);

#ifdef __cplusplus
}
#endif

#endif
