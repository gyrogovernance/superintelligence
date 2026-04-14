#ifndef GYROGRAPH_POLICY_H
#define GYROGRAPH_POLICY_H

#include "gyrograph.h"

/*
 * Gyroscopic env: GGML_GYROSCOPIC (mode; unset defaults to gyroscopic when built
 * with GGML_USE_GYROSCOPIC / GGML_USE_GYROSCOPIC_GRAPH), GGML_GYROSCOPIC_TRACE.
 */

typedef enum {
    GYRO_MODE_STOCK = 0,
    GYRO_MODE_GYROSCOPIC = 1,
} gyro_mode_t;

typedef struct gyro_policy {
    gyro_mode_t mode;
    int strict;
    int trace;
} gyro_policy;

#ifdef __cplusplus
extern "C" {
#endif

const gyro_policy * gyro_policy_get(void);

#ifdef __cplusplus
}
#endif

#endif
