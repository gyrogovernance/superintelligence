#ifndef GYROLABE_APERTURE_H
#define GYROLABE_APERTURE_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GYROLABE_M_A 0.19947114020071635
#define GYROLABE_DELTA_BU 0.195342176580
#define GYROLABE_RHO (GYROLABE_DELTA_BU / GYROLABE_M_A)
#define GYROLABE_DELTA (1.0 - GYROLABE_RHO)
#define GYROLABE_Q_G (4.0 * M_PI)
#define GYROLABE_B 65536

#define GYROLABE_DEPTH4_BITS 48
#define GYROLABE_APERTURE_Q256 5

static inline double gyrolabe_aperture_gap(void) { return GYROLABE_DELTA; }
static inline double gyrolabe_closure_ratio(void) { return GYROLABE_RHO; }
static inline int gyrolabe_depth4_quantization(void) {
    return (int)(GYROLABE_DEPTH4_BITS * GYROLABE_DELTA + 0.5);
}

#ifdef __cplusplus
}
#endif

#endif
