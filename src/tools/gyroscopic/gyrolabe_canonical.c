#include "gyrolabe_canonical.h"

#if defined(_MSC_VER)
#include <intrin.h>
static inline uint8_t popcnt6(uint8_t x) {
    return (uint8_t)__popcnt((unsigned int)(x & 0x3F));
}
#else
static inline uint8_t popcnt6(uint8_t x) {
    return (uint8_t)__builtin_popcount((unsigned int)(x & 0x3F));
}
#endif

static const uint8_t C6[7] = {1, 6, 15, 20, 15, 6, 1};

void gyrolabe_canonical_decompose(uint32_t omega12, uint8_t *c, uint8_t *chi, uint8_t *N) {
    uint32_t w = omega12 & 0xFFFu;
    uint8_t u = (uint8_t)((w >> 6) & 0x3Fu);
    uint8_t v = (uint8_t)(w & 0x3Fu);
    if (c) {
        *c = u;
    }
    if (chi) {
        *chi = (uint8_t)(u ^ v);
    }
    if (N) {
        *N = popcnt6((uint8_t)(u ^ v));
    }
}

uint32_t gyrolabe_canonical_reconstruct(uint8_t c, uint8_t chi) {
    uint8_t u = (uint8_t)(c & 0x3Fu);
    uint8_t v = (uint8_t)((u ^ chi) & 0x3Fu);
    return ((uint32_t)u << 6) | v;
}

uint32_t gyrolabe_shell_population(uint8_t N) {
    if (N > 6u) {
        return 0u;
    }
    return 64u * (uint32_t)C6[N];
}

uint8_t gyrolabe_chi_from_omega12(uint32_t omega12) {
    uint32_t w = omega12 & 0xFFFu;
    return (uint8_t)(((w >> 6) ^ w) & 0x3Fu);
}

uint8_t gyrolabe_shell_from_chi(uint8_t chi) {
    return popcnt6(chi);
}
