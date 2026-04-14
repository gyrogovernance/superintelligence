#ifndef GYROGRAPH_TYPES_H
#define GYROGRAPH_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// === Moment (SDK §5.4, Spec 1.1) ===
typedef struct gyrograph_moment {
    uint64_t step;
    uint32_t state24;
    uint8_t last_byte;
    uint8_t ledger_len;
    uint8_t ledger[256];
    uint16_t parity_O12;
    uint16_t parity_E12;
    uint8_t parity_bit;
    uint32_t omega_sig;
    uint8_t q_transport6;
} gyrograph_moment;

// === WordSignature (SDK §5.1.3) ===
typedef struct gyrograph_word_signature {
    uint8_t parity;
    uint16_t tau_a12;
    uint16_t tau_b12;
} gyrograph_word_signature;

// === SLCP Record (GyroGraph §13) ===
typedef struct gyrograph_slcp_record {
    int64_t cell_id;
    uint64_t step;
    int32_t omega12;
    uint32_t state24;
    uint8_t last_byte;
    uint8_t family;
    uint8_t micro_ref;
    uint8_t q6;
    uint8_t chi6;
    uint8_t shell;
    uint16_t horizon_distance;
    uint16_t ab_distance;
    int32_t omega_sig;
    uint16_t parity_O12;
    uint16_t parity_E12;
    uint8_t parity_bit;
    uint32_t resonance_key;
    uint32_t current_resonance;
    float spectral64[64];
    float gauge_spectral[4];
    float shell_spectral[7];
} gyrograph_slcp_record;

// === Constitutional Chart (SDK §5.3) ===
typedef struct gyrograph_constitutional {
    uint32_t rest_distance;
    uint32_t horizon_distance;
    uint32_t ab_distance;
    uint8_t on_complement_horizon;
    uint8_t on_equality_horizon;
    float a_density;
    float b_density;
    uint32_t complementarity_sum;
} gyrograph_constitutional;

// === Operator Analysis Report (GyroLabe §3.3) ===
// Numeric values follow this enum (Python bridge index order). They are not the same
// encoding as gyro_class_id_t in gyrolabe_registry.h: never cast between them by number.
typedef enum {
    GYROLABE_CLASS_GENERIC = 0,
    GYROLABE_CLASS_SHELL_RADIAL = 1,
    GYROLABE_CLASS_SHELL_GAUGE = 2,
    GYROLABE_CLASS_CHI_INVARIANT = 3,
    GYROLABE_CLASS_CHI_GAUGE = 4,
} gyrolabe_operator_class_t;

typedef struct gyrolabe_operator_report {
    gyrolabe_operator_class_t op_class;
    /* Profiling only: ||P_Q||_F / ||B||_F for the structured part (0 if generic). */
    float scr;
    float defect_norm;
    float eigenvalues_256[256];
    uint8_t eigenvalues_valid;
} gyrolabe_operator_report;

// === Resonance Profiles (GyroGraph §10) ===
typedef enum {
    GYROGRAPH_PROFILE_CHIRALITY = 0,
    GYROGRAPH_PROFILE_SHELL = 1,
    GYROGRAPH_PROFILE_HORIZON_CLASS = 2,
    GYROGRAPH_PROFILE_OMEGA_COINCIDENCE = 3,
    GYROGRAPH_PROFILE_SIGNATURE = 4,
    GYROGRAPH_PROFILE_Q_TRANSPORT = 5,
} gyrograph_resonance_profile_t;

#ifdef __cplusplus
}
#endif

#endif