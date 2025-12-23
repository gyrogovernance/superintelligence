"""
Minimal CGM constants for the Router kernel.

These constants parameterize the canonical aperture A* used in the
Superintelligence Index and are copied from the Common Governance Model
without simulation-specific coefficients.
"""

import numpy as np

M_A = 1 / (2 * np.sqrt(2 * np.pi))
DELTA_BU = 0.195342176580
A_STAR = 1 - (DELTA_BU / M_A)
Q_G = 4 * np.pi


