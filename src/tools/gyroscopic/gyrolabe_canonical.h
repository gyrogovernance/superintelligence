#ifndef GYROLABE_CANONICAL_H
#define GYROLABE_CANONICAL_H

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

GYROLABE_EXPORT void gyrolabe_canonical_decompose(
    uint32_t omega12,
    uint8_t *c,
    uint8_t *chi,
    uint8_t *N
);
GYROLABE_EXPORT uint32_t gyrolabe_canonical_reconstruct(uint8_t c, uint8_t chi);
GYROLABE_EXPORT uint32_t gyrolabe_shell_population(uint8_t N);
GYROLABE_EXPORT uint8_t gyrolabe_chi_from_omega12(uint32_t omega12);
GYROLABE_EXPORT uint8_t gyrolabe_shell_from_chi(uint8_t chi);

#ifdef __cplusplus
}
#endif

#endif
