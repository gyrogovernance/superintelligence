(.venv) PS F:\Development\superintelligence> python -m pytest -v -s tests/
========================= test session starts ==========================
platform win32 -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- F:\Development\superintelligence\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: F:\Development\superintelligence
plugins: anyio-4.12.0
collected 146 items                                                     

tests/physics/test_physics_1.py::TestStateRepresentation::test_pack_unpack_archetype PASSED
tests/physics/test_physics_1.py::TestStateRepresentation::test_pack_unpack_invertible PASSED
tests/physics/test_physics_1.py::TestStateRepresentation::test_component_isolation PASSED
tests/physics/test_physics_1.py::TestTranscription::test_gene_mic_s_constant PASSED
tests/physics/test_physics_1.py::TestTranscription::test_transcription_involution PASSED
tests/physics/test_physics_1.py::TestTranscription::test_transcription_bijective PASSED
tests/physics/test_physics_1.py::TestTranscription::test_specific_transcriptions PASSED
tests/physics/test_physics_1.py::TestExpansion::test_expansion_deterministic PASSED
tests/physics/test_physics_1.py::TestExpansion::test_expansion_injective PASSED
tests/physics/test_physics_1.py::TestExpansion::test_type_b_mask_always_zero PASSED
tests/physics/test_physics_1.py::TestExpansion::test_precomputed_table_matches PASSED
tests/physics/test_physics_1.py::TestFIFOGyration::test_gyration_is_asymmetric PASSED
tests/physics/test_physics_1.py::TestCGMChirality::test_gyration_not_pure_swap PASSED
tests/physics/test_physics_1.py::TestCGMChirality::test_gyration_asymmetry PASSED
tests/physics/test_physics_1.py::TestInvariants::test_state_space_boundedness PASSED
tests/physics/test_physics_1.py::TestInvariants::test_determinism PASSED 
tests/physics/test_physics_1.py::TestInvariants::test_all_bytes_are_operations PASSED
tests/physics/test_physics_1.py::TestClosedFormDepthLaws::test_step_is_bijective_with_explicit_inverse PASSED
tests/physics/test_physics_1.py::TestClosedFormDepthLaws::test_depth2_decoupling_closed_form PASSED
tests/physics/test_physics_1.py::TestClosedFormDepthLaws::test_depth2_commutes_iff_same_byte PASSED
tests/physics/test_physics_1.py::TestClosedFormDepthLaws::test_trajectory_closed_form_arbitrary_length PASSED
tests/physics/test_physics_1.py::TestCSOperatorAndSeparatorLemmas::test_separator_lemma_x_then_AA_updates_A_only PASSED
tests/physics/test_physics_1.py::TestCSOperatorAndSeparatorLemmas::test_separator_lemma_AA_then_x_updates_B_only PASSED
tests/physics/test_physics_1.py::TestCSOperatorAndSeparatorLemmas::test_depth4_alternation_identity_all_pairs_on_archetype PASSED
tests/physics/test_physics_1.py::TestInverseConjugationForm::test_inverse_is_conjugation_by_R PASSED
tests/physics/test_physics_1.py::TestQuantumGravityManifold::test_holographic_area_scaling
==========
PILLAR 2: Holographic Area/Entropy Scaling
==========
  Horizon Area (States): 256
  Boundary Layer Volume: 65280
  Total Atmosphere (Horizon + Boundary): 65536
  Expansion Ratio (Boundary/Area): 255.00
  ✓ Verified: The Horizon is holographically complete (Boundary = Bulk). 
PASSED
tests/physics/test_physics_2.py::TestComplementSymmetryKernelWide::test_complement_symmetry_commutes_with_byte_actions
==========
SYMMETRY: Complement map commutes with byte actions (kernel-wide)        
==========
  Tested bytes: ['0x00', '0x01', '0x42', '0xaa', '0xff', '0x55', '0x12', '0x34']
  ✓ Verified: Complement symmetry commutes with sampled byte actions on all Ω states.
PASSED
tests/physics/test_physics_2.py::TestKernelIntrinsicMaskCoordinates::test_exhaustive_step_law_in_mask_coordinates_all_states_all_bytes
==========
KERNEL CLOSED FORM: Exhaustive (u,v) step law check on atlas
==========
  |Ω| = 65,536 states
  Checking all 256 bytes across all Ω states (16,777,216 transitions) ...
  Total u-mismatches: 0
  Total v-mismatches: 0
  ✓ Verified: Kernel dynamics is exactly (u_next, v_next) = (v, u XOR m_b) on real atlas.
PASSED
tests/physics/test_physics_2.py::TestKernelCommutatorAsTranslation::test_exhaustive_commutator_translation_all_byte_pairs
==========
COMMUTATOR TRANSLATION: Exhaustive K(x,y) over all 256×256 pairs
==========
  Start indices tested: [0, 32768, 65535]
  Word: [x_inv, y_inv, x, y] where inv(z) = [0xAA, z, 0xAA]
  Expectation: delta24(x,y) = ((m_x XOR m_y)<<12) | (m_x XOR m_y)        
  ✓ start_idx=0: all 65,536 commutators match expected translation mask
  ✓ start_idx=32768: all 65,536 commutators match expected translation mask
  ✓ start_idx=65535: all 65,536 commutators match expected translation mask

----------
INTRINSIC DISTRIBUTIONS (ordered byte pairs):
  d = m_x XOR m_y (12-bit)
  commutator delta24 flips d in both halves -> dist24 = 2*popcount(d)    
----------
  popcount(d) distribution (w in 0..12):
    w= 0: count=   256  prob=0.003906
    w= 1: count=  1024  prob=0.015625
    w= 2: count=  2560  prob=0.039062
    w= 3: count=  5120  prob=0.078125
    w= 4: count=  7936  prob=0.121094
    w= 5: count= 10240  prob=0.156250
    w= 6: count= 11264  prob=0.171875
    w= 7: count= 10240  prob=0.156250
    w= 8: count=  7936  prob=0.121094
    w= 9: count=  5120  prob=0.078125
    w=10: count=  2560  prob=0.039062
    w=11: count=  1024  prob=0.015625
    w=12: count=   256  prob=0.003906

  dist24 distribution (0..24, even only expected):
    dist= 0: count=   256  prob=0.003906
    dist= 2: count=  1024  prob=0.015625
    dist= 4: count=  2560  prob=0.039062
    dist= 6: count=  5120  prob=0.078125
    dist= 8: count=  7936  prob=0.121094
    dist=10: count= 10240  prob=0.156250
    dist=12: count= 11264  prob=0.171875
    dist=14: count= 10240  prob=0.156250
    dist=16: count=  7936  prob=0.121094
    dist=18: count=  5120  prob=0.078125
    dist=20: count=  2560  prob=0.039062
    dist=22: count=  1024  prob=0.015625
    dist=24: count=   256  prob=0.003906

  Defect energy landscape (12D ±1 embedding):
    w_min = 1 (bits out of 12)
    θ_min = arccos(1 - w_min/6) = 0.585685543 rad
    E[w] = 6.000000
    E[θ(w)] = 1.570796 rad

  CGM reference anchors:
    CGM δ_BU = 0.195342177 rad
    CGM A*   = 0.020699554

  A*-search (kernel-only, intrinsic probability masses):
    A* ≈ 0.020700
    Closest prob to A* in popcount(d):
      w=1  prob=0.015625  |diff|=0.005075

  Kernel-native aperture: A_kernel = P(w<=1) = 0.019531250 (compare A*=0.020699554)
PASSED
tests/physics/test_physics_2.py::TestKernelMonodromyBaseFiber::test_bu_dual_pole_monodromy_base_closure_fiber_defect
==========
CGM-ANCHORED KERNEL MONODROMY: base closure, fiber defect
==========
  CS threshold s_p = π/2      = 1.570796
  UNA threshold u_p = 1/√2    = 0.707107
  ONA threshold o_p = π/4     = 0.785398
  BU threshold m_a            = 0.199471140
  CGM δ_BU                    = 0.195342177
  CGM A* = 1 - δ_BU/m_a       = 0.020699554
  ✓ Verified on sampled states: W=[x,y,x,z] closes u and shifts v by (m_y XOR m_z).

  Fiber defect statistics over ALL pole pairs (y,z):
    mean popcount(m_y XOR m_z): 6.000000 (out of 12)
    var  popcount(m_y XOR m_z): 5.000000
    mean fiber-angle θ_v:       1.570796 rad (hypercube-angle mapping)   

  Kernel-native 'openness' diagnostics (no claims):
    P(w<=1) = 0.019531250   (compare A*=0.020699554)
    Var(w)/256 = 0.019531250 (compare A*=0.020699554)
PASSED
tests/physics/test_physics_2.py::TestCGMThresholdAnatomyInKernel::test_weight1_primitives_and_anatomical_locations
==========
CGM ANATOMY: Primitive minimal moves (weight-1 masks)
==========
  Count of weight-1 masks: 4 (expected 4)
  byte=0x2a  intron=0x80  mask=0x080  pop=1  at (frame=1, row=0, col=1, bit=7)
  byte=0x8a  intron=0x20  mask=0x020  pop=1  at (frame=0, row=2, col=1, bit=5)
  byte=0xba  intron=0x10  mask=0x010  pop=1  at (frame=0, row=2, col=0, bit=4)
  byte=0xea  intron=0x40  mask=0x040  pop=1  at (frame=1, row=0, col=0, bit=6)
  ✓ Verified: exactly 4 primitive directions exist, each is a single anatomical bit.
PASSED
tests/physics/test_physics_2.py::TestKernelCGMThresholdAnchors::test_cs_anchor_mean_fiber_angle_is_pi_over_2_exact
==========
CGM ANCHOR: CS threshold as exact mean fiber-angle
==========
  Mean θ over code C: 1.570796326795 rad
  π/2:                1.570796326795 rad
  Difference:         -2.220446049250e-16 rad
  (This equality follows from θ(12-w)=π-θ(w) and symmetric P(w).)        
  ✓ Verified: CS anchor s_p=π/2 is intrinsic (exact) in the kernel's defect-angle geometry.
PASSED
tests/physics/test_physics_2.py::TestKernelCGMThresholdAnchors::test_monodromy_hierarchy_bridge_theta_min_delta_omega
==========
CGM HIERARCHY BRIDGE: θ_min -> SU2 holonomy, δ, ω
==========
  θ_min (kernel) = arccos(5/6) = 0.585685543 rad
  SU(2) holonomy (CGM)         = 0.587901000 rad
  θ_min - SU2_holonomy         = -0.002215457 rad

  3-row anatomical split:
  δ_kernel := θ_min/3          = 0.195228514 rad
  δ_BU (CGM)                   = 0.195342177 rad
  δ_kernel - δ_BU              = -0.000113662 rad

  Dual-pole split:
  ω_kernel := δ_kernel/2       = 0.097614257 rad
  ω(ONA↔BU) (CGM)              = 0.097671000 rad
  ω_kernel - ω_CGM             = -0.000056743 rad
  ✓ Diagnostic complete: kernel produces the right hierarchy scales without tuning.
PASSED
tests/physics/test_physics_2.py::TestKernelCGMThresholdAnchors::test_discrete_aperture_shadow_A_kernel
==========
CGM APERTURE SHADOW: A_kernel vs A*
==========
  A_kernel = 5/256              = 0.019531250000
  closure_kernel = 1 - A_kernel = 0.980468750000

  A*_CGM                         = 0.020699553813
  closure_CGM                    = 0.979300446187

  A_kernel - A*_CGM              = -0.001168303813
  closure_kernel - closure_CGM   = +0.001168303813
  ✓ Verified: kernel has an intrinsic discrete small-openness constant A_kernel=5/256.
PASSED
tests/physics/test_physics_3.py::TestKernelByteCyclesAndEigenphases::test_reference_byte_cycle_decomposition_and_eigenphases
Reference byte 0xAA - cycle and eigenphase structure
----------------------------------------------------
  n                          : 65536
  fixed points (1-cycles)    : 256
  2-cycles                   : 32640
  eigenvalue multiplicity 1  : 32896
  eigenvalue multiplicity -1 : 32640
PASSED
tests/physics/test_physics_3.py::TestKernelByteCyclesAndEigenphases::test_all_nonreference_bytes_are_pure_4_cycles
Non-reference bytes: fixed points of T_b^2
------------------------------------------
  bytes tested : 255
  anomalies    : 0
  anomaly list : []

Representative non-reference byte 0x42 - cycle and eigenphase structure  
-----------------------------------------------------------------------  
  4-cycles                   : 16384
  eigenvalue multiplicity 1  : 16384
  eigenvalue multiplicity i  : 16384
  eigenvalue multiplicity -1 : 16384
  eigenvalue multiplicity -i : 16384
PASSED
tests/physics/test_physics_3.py::TestMaskCodeDualityAndFourierSupport::test_generator_bytes_span_all_masks_and_print
Mask code spanning set from intron-basis bytes
----------------------------------------------
  generator bytes       : ['0xab', '0xa8', '0xae', '0xa2', '0xba', '0x8a', '0xea', '0x2a']
  generator masks (hex) : ['0x101', '0x202', '0x404', '0x808', '0x010', '0x020', '0x040', '0x080']
  reachable masks       : 256
  actual masks          : 256
  match                 : True

  Context note: mask code has dimension 8 over GF(2), hence |C| = 2^8 = 256.
PASSED
tests/physics/test_physics_3.py::TestMaskCodeDualityAndFourierSupport::test_code_duality_sizes_and_macwilliams_identity
Code duality and MacWilliams identity
-------------------------------------
  n                 : 12
  |C|               : 256
  |C_perp|          : 16
  |C|*|C_perp|      : 4096
  2^12              : 4096
  MacWilliams match : True
  B_actual weights  : {0: 1, 2: 4, 4: 6, 6: 4, 8: 1}
PASSED
tests/physics/test_physics_3.py::TestMaskCodeDualityAndFourierSupport::test_walsh_spectrum_support_equals_dual_code
Walsh spectrum of mask-code indicator
-------------------------------------
  unique W(s) values    : [0, 256]
  counts by value       : {0: 4080, 256: 16}
  support size W(s)=256 : 16
  expected |C_perp|     : 16
  support equals C_perp : True
PASSED
tests/physics/test_physics_3.py::TestAtlasShellDistributionsFromCodeEnumerator::test_horizon_distance_shells_equal_256_times_code_enumerator      
Horizon distance shells over Ω (exact)
-----------------------------------
  w= 0 : count=   256  prob=0.003906
  w= 1 : count=  1024  prob=0.015625
  w= 2 : count=  2560  prob=0.039062
  w= 3 : count=  5120  prob=0.078125
  w= 4 : count=  7936  prob=0.121094
  w= 5 : count= 10240  prob=0.156250
  w= 6 : count= 11264  prob=0.171875
  w= 7 : count= 10240  prob=0.156250
  w= 8 : count=  7936  prob=0.121094
  w= 9 : count=  5120  prob=0.078125
  w=10 : count=  2560  prob=0.039062
  w=11 : count=  1024  prob=0.015625
  w=12 : count=   256  prob=0.003906
PASSED
tests/physics/test_physics_3.py::TestAtlasShellDistributionsFromCodeEnumerator::test_archetype_distance_shells_equal_convolution
Archetype distance shells over Ω (exact)
---------------------------------------
  dist= 0 : count=     1  prob=0.000015
  dist= 1 : count=     8  prob=0.000122
  dist= 2 : count=    36  prob=0.000549
  dist= 3 : count=   120  prob=0.001831
  dist= 4 : count=   322  prob=0.004913
  dist= 5 : count=   728  prob=0.011108
  dist= 6 : count=  1428  prob=0.021790
  dist= 7 : count=  2472  prob=0.037720
  dist= 8 : count=  3823  prob=0.058334
  dist= 9 : count=  5328  prob=0.081299
  dist=10 : count=  6728  prob=0.102661
  dist=11 : count=  7728  prob=0.117920
  dist=12 : count=  8092  prob=0.123474
  dist=13 : count=  7728  prob=0.117920
  dist=14 : count=  6728  prob=0.102661
  dist=15 : count=  5328  prob=0.081299
  dist=16 : count=  3823  prob=0.058334
  dist=17 : count=  2472  prob=0.037720
  dist=18 : count=  1428  prob=0.021790
  dist=19 : count=   728  prob=0.011108
  dist=20 : count=   322  prob=0.004913
  dist=21 : count=   120  prob=0.001831
  dist=22 : count=    36  prob=0.000549
  dist=23 : count=     8  prob=0.000122
  dist=24 : count=     1  prob=0.000015
PASSED
tests/physics/test_physics_3.py::TestCGMUnitsBridgeDiagnostics::test_kernel_to_cgm_units_bridge_prints
CGM Units ↔ Kernel bridge (diagnostic)
--------------------------------------
  A_kernel                        : 0.019531250000 (exact 5/256=0.019531250000)
  A*_CGM                          : 0.020699553813
  A_kernel - A*_CGM               : -0.001168303813
                                  :
  theta_min                       : 0.585685543457  (arccos(5/6))        
  delta_kernel                    : 0.195228514486  (theta_min/3)        
  delta_BU_CGM                    : 0.195342176600
  delta_kernel - delta_BU_CGM     : -0.000113662114
  omega_kernel                    : 0.097614257243  (delta_kernel/2)     
                                  :
  m_a_kernel                      : 0.199117528719
  m_a_CGM                         : 0.199471140201
  m_a_kernel - m_a_CGM            : -0.000353611482
                                  :
  Q_G_kernel                      : 12.611043312527
  Q_G_CGM = 4π                    : 12.566370614359
  Q_G_kernel - 4π                 : +0.044672698168
                                  :
  K_QG_kernel                     : 3.944394893067  ((π/4)/m_a_kernel)   
  K_QG_CGM((π/4)/m_a)             : 3.937402486431
  K_QG_CGM(pi^2/sqrt(2pi))        : 3.937402486431
                                  :
  alpha_kernel                    : 0.007295641839  (delta_kernel^4 / m_a_kernel)
  alpha_CGM (paper)               : 0.007297352563
  alpha_CGM (units)               : 0.007299734000
  alpha_kernel - alpha_CGM(paper) : -0.000001710724
PASSED
tests/physics/test_physics_3.py::TestActionGroupPresentationInUV::test_word_action_depends_only_on_parity_OE_and_prints
Action group presentation check (u,v)
-------------------------------------
  probe states        : 8192
  random words tested : 200
  mismatching words   : 0
PASSED
tests/physics/test_physics_3.py::TestActionGroupPresentationInUV::test_only_two_linear_parts_exist_identity_or_swap PASSED
tests/physics/test_physics_3.py::TestClosedFormShellPolynomials::test_mask_weight_enumerator_is_closed_form_and_prints
Mask weight enumerator: observed vs closed form
-----------------------------------------------
  observed    : [1, 4, 10, 20, 31, 40, 44, 40, 31, 20, 10, 4, 1]
  closed form : [1, 4, 10, 20, 31, 40, 44, 40, 31, 20, 10, 4, 1]
  match       : True
PASSED
tests/physics/test_physics_3.py::TestClosedFormShellPolynomials::test_archetype_distance_enumerator_is_closed_form_and_prints
Archetype distance enumerator: observed vs closed form
------------------------------------------------------
  observed first 13    : [1, 8, 36, 120, 322, 728, 1428, 2472, 3823, 5328, 6728, 7728, 8092]
  observed last 12     : [7728, 6728, 5328, 3823, 2472, 1428, 728, 322, 120, 36, 8, 1]
  closed form first 13 : [1, 8, 36, 120, 322, 728, 1428, 2472, 3823, 5328, 6728, 7728, 8092]
  closed form last 12  : [7728, 6728, 5328, 3823, 2472, 1428, 728, 322, 120, 36, 8, 1]
  match                : True
PASSED
tests/physics/test_physics_3.py::TestClosedFormShellPolynomials::test_uv_ir_symmetry_in_shells_and_prints
UV/IR shell symmetry (exact in Ω)
---------------------------------
  max |count[d] - count[24-d]| : 0
  symmetry holds               : True
PASSED
tests/physics/test_physics_4.py::TestApertureConstraintHalfInteger::test_aperture_constraint_product_equals_half
==========
APERTURE CONSTRAINT: Q_G × m_a² = 1/2
==========
  A_kernel = 0.019531250000 (5/256)
  closure  = 0.980468750000 (251/256)
  θ_min    = 0.585685543457 rad
  δ_kernel = 0.195228514486 rad
  m_a_kernel = 0.199117528719

  4π × m_a_kernel² = 0.498228826233
  Expected (CGM)   = 0.500000000000
  Deviation        = 0.001771173767
  Relative error   = 0.354235%

  Q_G_kernel (from 1/(2m_a²)) = 12.611043312527
  Q_G_CGM (4π)                = 12.566370614359
  ✓ Aperture constraint 4π × m_a² ≈ 0.5 holds within 1%
PASSED
tests/physics/test_physics_4.py::TestHolonomyGroupIsCode::test_achievable_fiber_defects_equal_mask_code
==========
HOLONOMY GROUP = MASK CODE
==========
  |Achievable defects| = 256
  |Mask code C|        = 256
  Sets equal           = True
  XOR closure verified = True
  ✓ Holonomy group is isomorphic to (Z/2)^8
PASSED
tests/physics/test_physics_4.py::TestOpticalConjugacyShellProduct::test_shell_conjugacy_product_structure
==========
OPTICAL CONJUGACY: SHELL PRODUCTS
==========
  Mean distance d        = 12.000000 (expected 12.0)
  Mean product d(24-d)   = 134.000000
  Max product (at d=12)  = 144 = 144 = 12²

  Identity: E[d(24-d)] = 144 - Var(d)
  Var(d)               = 10.000000
  144 - Var(d)         = 134.000000
  (Expected: Var(d) = 10, from Var(w1) + Var(w2) = 5 + 5)

  4π²                    = 39.478418
  mean_product / 4π²     = 3.394260
  144 / 4π²              = 3.647563
  Shell symmetry (d ↔ 24-d) = True
  ✓ Optical conjugacy structure verified
  ✓ Second-moment identity: E[d(24-d)] = 144 - Var(d) = 134
PASSED
tests/physics/test_physics_4.py::TestKQGCommutatorScale::test_commutator_scale_kqg
==========
COMMUTATOR SCALE K_QG
==========
  m_a_kernel       = 0.199117528719
  K_QG_kernel      = (π/4)/m_a = 3.944394893067
  K_QG_CGM         = π²/√(2π) = 3.937402486431
  Difference       = +0.006992406637
  Relative error   = 0.177589%

  S_ONA_kernel     = 3.944394893067
  K_QG = S_ONA?    : 3.944395 vs 3.944395
  (In CGM: K_QG = S_CS/2 = S_ONA)
PASSED
tests/physics/test_physics_4.py::TestNeutrinoScaleInvariant::test_neutrino_scale_invariant_from_kernel
==========
NEUTRINO-SCALE-LIKE KERNEL INVAIRANT
==========
  A_kernel        = 0.019531250000  (5/256)
  closure         = 0.980468750000  (251/256)
  theta_min       = 0.585685543457 rad  (arccos(5/6))
  delta_kernel    = 0.195228514486 rad  (theta_min/3)
  m_a_kernel      = 0.199117528719
  alpha_kernel    = 0.007295641839

  R_nu (kernel)   = δ^6 / (m_a^2 * α) = 0.191415457562
  (Purely kernel-defined, no external scales used)
PASSED
tests/physics/test_physics_4.py::TestRestrictedAlphabetPhaseTransition::test_rank_orbit_theorem
==========
RANK/ORBIT THEOREM: RESTRICTED ALPHABET
==========
  t= 0: |allowed|=  1, rank=0, pred=     1, orbit=     1, match=True     
  t= 1: |allowed|=  5, rank=4, pred=   256, orbit=   256, match=True     
  t= 2: |allowed|= 15, rank=8, pred= 65536, orbit= 65536, match=True
  t= 3: |allowed|= 35, rank=8, pred= 65536, orbit= 65536, match=True
  t= 4: |allowed|= 66, rank=8, pred= 65536, orbit= 65536, match=True
  t= 5: |allowed|=106, rank=8, pred= 65536, orbit= 65536, match=True
  t= 6: |allowed|=150, rank=8, pred= 65536, orbit= 65536, match=True
  t= 7: |allowed|=190, rank=8, pred= 65536, orbit= 65536, match=True
  t= 8: |allowed|=221, rank=8, pred= 65536, orbit= 65536, match=True
  t= 9: |allowed|=241, rank=8, pred= 65536, orbit= 65536, match=True
  t=10: |allowed|=251, rank=8, pred= 65536, orbit= 65536, match=True
  t=11: |allowed|=255, rank=8, pred= 65536, orbit= 65536, match=True
  t=12: |allowed|=256, rank=8, pred= 65536, orbit= 65536, match=True

  Mismatches: 0
PASSED
tests/physics/test_physics_4.py::TestRestrictedAlphabetPhaseTransition::test_nucleation_barrier_critical_threshold
==========
NUCLEATION BARRIER: CRITICAL THRESHOLD
==========
  Rank progression by weight threshold t:
    t= 0: rank=0, 2^(2*rank)=1
    t= 1: rank=4, 2^(2*rank)=256
    t= 2: rank=8, 2^(2*rank)=65536
    t= 3: rank=8, 2^(2*rank)=65536
    t= 4: rank=8, 2^(2*rank)=65536
    t= 5: rank=8, 2^(2*rank)=65536
    t= 6: rank=8, 2^(2*rank)=65536
    t= 7: rank=8, 2^(2*rank)=65536
    t= 8: rank=8, 2^(2*rank)=65536
    t= 9: rank=8, 2^(2*rank)=65536
    t=10: rank=8, 2^(2*rank)=65536
    t=11: rank=8, 2^(2*rank)=65536
    t=12: rank=8, 2^(2*rank)=65536

  Critical threshold: t=2 (rank jumps to 8)
  Below t=2: bubble sub-ontology (strict subset of Ω)
  At t>=2: full Ω accessible (65536 states)
PASSED
tests/physics/test_physics_4.py::TestRestrictedAlphabetPhaseTransition::test_minimal_generator_count
==========
MINIMAL GENERATOR COUNT
==========
  Sampling random byte subsets to find minimal generators:
    k= 4: max_rank=4, full_rank_count=0/100
    k= 5: max_rank=5, full_rank_count=0/100
    k= 6: max_rank=6, full_rank_count=0/100
    k= 7: max_rank=7, full_rank_count=0/100
    k= 8: max_rank=8, full_rank_count=43/100
    k= 9: max_rank=8, full_rank_count=67/100
    k=10: max_rank=8, full_rank_count=76/100
    k=12: max_rank=8, full_rank_count=96/100
    k=16: max_rank=8, full_rank_count=99/100
    k=32: max_rank=8, full_rank_count=100/100

  Minimum k with full rank observed: 8
  (Code dimension = 8, so theoretical minimum is k >= 8)
PASSED
tests/physics/test_physics_4.py::TestRestrictedAlphabetPhaseTransition::test_weight2_bridge_masks_extend_rank
==========
WEIGHT-2 BRIDGE MASKS
==========
  |U1|           = 16 (rank 4)
  |W2 inside U1| = 6
  |W2 bridge|    = 4
  Combined rank  = 8
PASSED
tests/test_aci_cli.py::test_a_cold_start_builds_atlas_and_templates PASSED
tests/test_aci_cli.py::test_b_compile_program_into_artifacts PASSED
tests/test_aci_cli.py::test_c_tamper_detection PASSED
tests/test_aci_cli.py::test_d_determinism PASSED
tests/test_aci_cli.py::test_e_skipped_attestations_in_report PASSED
tests/test_app.py::TestDomainLedgers::test_aperture_zero_when_ledger_is_zero PASSED
tests/test_app.py::TestDomainLedgers::test_decompose_reconstructs_y PASSED
tests/test_app.py::TestDomainLedgers::test_aperture_scale_invariant PASSED
tests/test_app.py::TestCoordinator::test_coordinator_replay_determinism PASSED
tests/test_app.py::TestCoordinator::test_event_binding_records_kernel_moment PASSED
tests/test_app.py::TestCoordinator::test_coordinator_reset PASSED        
tests/test_app.py::TestCoordinator::test_coordinator_status_structure PASSED
tests/test_app.py::TestHodgeProjections::test_programor_identities PASSED
tests/test_app.py::TestHodgeProjections::test_cycle_component_in_kernel_of_B PASSED
tests/test_app.py::TestHodgeProjections::test_cycle_basis_is_in_kernel_and_unit_norm PASSED
tests/test_measurement.py::TestMeasurementCollapse::test_scalar_collapse_loses_aperture_distinguishability
==========
PROOF: SCALAR BLINDNESS
==========
  State 1 (Collapsed):   Scalar = 4.000000, A = 0.500000
  State 2 (Distributed): Scalar = 4.000000, A = 0.625000
  Scalar distinguishes states? False
  ✓ Proven: Scalar aggregation discards structural information
PASSED
tests/test_measurement.py::TestMeasurementCollapse::test_scalar_sum_cannot_detect_A_star_proximity
==========
PROOF: ALIGNMENT INVISIBILITY
==========
  Aligned State:    Scalar = 0.363198, |A - A*| = 0.000000
  Misaligned State: Scalar = 0.363198, |A - A*| = 0.479300
  ✓ Proven: Scalar evaluation cannot detect alignment
PASSED
tests/test_measurement.py::TestMeasurementCollapse::test_single_axis_structural_lock_vs_multi_axis_freedom
==========
PROOF: STRUCTURAL LOCK VS FREEDOM
==========
  Target A* = 0.020700
  [Single Axis Strategy]
    Mag   1.0 -> A = 0.500000 (Locked)
    Mag  10.0 -> A = 0.500000 (Locked)
    Mag 100.0 -> A = 0.500000 (Locked)
  [Multi-Axis Strategy]
    Constructed y -> A = 0.020700 (Aligned)
  ✓ Proven: Single-axis optimization cannot achieve alignment.
  ✓ Proven: Epistemic enumeration enables A*.
PASSED
tests/test_measurement.py::TestMeasurementCollapse::test_kernel_A_kernel_vs_app_A_star
==========
KERNEL ↔ APP APERTURE BRIDGE
==========
  A_kernel (discrete) = 0.019531250000 (5/256)
  A* (CGM continuous) = 0.020700000000
  Relative difference = 5.6%
  ✓ Kernel discrete structure approximates CGM continuous target
PASSED
tests/test_measurement.py::TestMeasurementCollapse::test_epistemic_aperture_is_scale_invariant
==========
PROOF: EPISTEMIC SELF-NORMALIZATION
==========
  Scalar: 6.2000 -> 620.0000 (Scales with input)
  Aperture: 0.359484777518 -> 0.359484777518 (Invariant)
  ✓ Epistemic measurement is self-normalizing
PASSED
tests/test_moments.py::test_router_static_structure_anchors
----------
Router Anchors
----------
Ontology size |Ω|: 65,536
Byte alphabet: 256
PASSED
tests/test_moments.py::test_aperture_shadow_a_kernel_close_to_a_star     
----------
Aperture Shadow
----------
A_kernel (exact): 0.019531250000 (5/256)
A* (CGM):         0.020699553813
Relative diff:    5.644101%
PASSED
tests/test_moments.py::test_atomic_second_anchor_constant
----------
Atomic Second Anchor
----------
Cs-133 hyperfine frequency: 9,192,631,770 Hz
PASSED
tests/test_moments.py::test_mu_definition_and_base_rate_base60
----------
Base-60 MU Anchor
----------
Seconds/minute: 60
Minutes/hour:   60
MU/minute:      1
MU/hour:        60
PASSED
tests/test_moments.py::test_uhi_amounts_daily_and_annual
----------
UHI
----------
UHI hours/day:   4
UHI MU/day:      240
UHI MU/year:     87,600
PASSED
tests/test_moments.py::test_tier_multipliers_from_uhi
----------
Tier Multipliers
----------
UHI (MU/year): 87,600
tier_1: 1× -> 87,600 MU/year
tier_2: 2× -> 175,200 MU/year
tier_3: 3× -> 262,800 MU/year
tier_4: 60× -> 5,256,000 MU/year
PASSED
tests/test_moments.py::test_tier4_accessible_mnemonic_one_per_second_for_four_hours_day
----------
Tier 4 Mnemonic
----------
Tier 4 (MU/year):        5,256,000
4 hours/day in seconds:  14,400
Mnemonic annual seconds: 5,256,000
PASSED
tests/test_moments.py::test_illustrative_work_week_is_not_the_definition_of_tiers
----------
Tier Definition vs Illustrative Work Week
----------
Tier 2 increment (MU/year):         87,600
Illustrative 4×4 work MU/year:      49,920
Note: Tiers are defined by UHI multipliers, not by this schedule.        
PASSED
tests/test_moments.py::test_fiat_capacity_upper_bound_from_atomic_and_kernel_rates
----------
Fiat Capacity Upper Bound
----------
Atomic Hz:          9,192,631,770
Kernel steps/sec:   2,400,000
F_total (/sec):     22.06 quadrillion (22,062,316,248,000,000)
PASSED
tests/test_moments.py::test_millennium_uhi_feasibility_under_conservative_mapping
----------
Millennium UHI Feasibility (Conservative Mapping)
----------
Conservative demonstration mapping: 1 micro-state == 1 MU
Population:                      8,100,000,000
UHI per person per year (MU):    87,600
Needed per year (MU):            709.56 trillion (709,560,000,000,000)   
Available per year (units):      695,757.21 quintillion (695,757,205,196,927,974,506,496)
Used % over 1000 years:        0.0000001020%
Surplus over 1000 years (units): 695,757,204.49 quintillion (695,757,204,487,367,991,733,780,480)
PASSED
tests/test_moments.py::test_notional_surplus_allocation_12_divisions     
----------
Notional Surplus Allocation (12 Divisions)
----------
Horizon years:     1,000
Divisions:         12 (3 domains × 4 capacities)
Surplus (units):   695,757,204.49 quintillion (695,757,204,487,367,991,733,780,480)
Per division:      57,979,767.04 quintillion (57,979,767,040,614,002,174,459,904)

Sample divisions:
  Economy      × GM    : 57,979,767.04 quintillion (57,979,767,040,614,002,174,459,904)
  Economy      × ICu   : 57,979,767.04 quintillion (57,979,767,040,614,002,174,459,904)
  Economy      × IInter: 57,979,767.04 quintillion (57,979,767,040,614,002,174,459,904)
  Economy      × ICo   : 57,979,767.04 quintillion (57,979,767,040,614,002,174,459,904)
  Employment   × GM    : 57,979,767.04 quintillion (57,979,767,040,614,002,174,459,904)
  Employment   × ICu   : 57,979,767.04 quintillion (57,979,767,040,614,002,174,459,904)
PASSED
tests/test_others.py::TestAtlasBuildingValidation::test_atlas_exists PASSED
tests/test_others.py::TestAtlasBuildingValidation::test_ontology_file_exists PASSED
tests/test_others.py::TestAtlasBuildingValidation::test_epistemology_file_exists PASSED
tests/test_others.py::TestAtlasBuildingValidation::test_phenomenology_file_exists PASSED
tests/test_others.py::TestAtlasBuildingValidation::test_ontology_size PASSED
tests/test_others.py::TestAtlasBuildingValidation::test_ontology_sortedPPASSED
tests/test_others.py::TestAtlasBuildingValidation::test_archetype_in_ontology PASSED
tests/test_others.py::TestOntologyStructure::test_ontology_is_cartesian_product_of_two_256_sets PASSED
tests/test_others.py::TestPhenomenologyValidation::test_archetype_stored_correctly PASSED
tests/test_others.py::TestPhenomenologyValidation::test_gene_mic_s_stored
 PASSED
tests/test_others.py::TestPhenomenologyValidation::test_xform_masks_count
 PASSED
tests/test_others.py::TestPhenomenologyValidation::test_xform_masks_match_constants PASSED
tests/test_others.py::TestEdgeCases::test_all_zeros_state PASSED
tests/test_others.py::TestEdgeCases::test_all_ones_state PASSED
tests/test_others.py::TestEdgeCases::test_mask_boundary_values PASSED    
tests/test_others.py::TestEdgeCases::test_repeated_byte_application PASSED
tests/test_others.py::TestPerformance::test_step_performance
  Steps/sec: 1,651,010
  Time per step: 0.61 μs
PASSED
tests/test_others.py::TestPerformance::test_kernel_step_performance      
  Kernel steps/sec: 1,095,194
  Time per step: 0.91 μs
PASSED
tests/test_others.py::TestPerformance::test_aperture_measurement_performance
  Aperture measurements/sec: 510,882
  Time per measurement: 0.00 ms
PASSED
tests/test_others.py::TestInvariantValidation::test_unique_mask_count PASSED
tests/test_others.py::TestInvariantValidation::test_all_b_masks_zero PASSED
tests/test_others.py::TestInvariantValidation::test_a_mask_coverage PASSED
tests/test_plugins.py::TestAnalytics::test_plugins_analytics_matches_domainledger_aperture PASSED
tests/test_plugins.py::TestAnalytics::test_hodge_decompose_reconstruction
 PASSED
tests/test_plugins.py::TestAnalytics::test_hodge_decompose_zero_vector PASSED
tests/test_plugins.py::TestAPIAdapters::test_parse_domain PASSED
tests/test_plugins.py::TestAPIAdapters::test_parse_edge_id PASSED        
tests/test_plugins.py::TestAPIAdapters::test_event_from_dict PASSED      
tests/test_plugins.py::TestAPIAdapters::test_event_to_dict PASSED        
tests/test_plugins.py::TestFrameworkPlugins::test_thm_displacement_plugin
 PASSED
tests/test_plugins.py::TestFrameworkPlugins::test_thm_displacement_plugin_ignores_domain_parameter PASSED
tests/test_plugins.py::TestFrameworkPlugins::test_gyroscope_workmix_plugin PASSED
tests/test_plugins.py::TestFrameworkPlugins::test_gyroscope_workmix_plugin_infer_intel PASSED
tests/test_plugins.py::TestFrameworkPlugins::test_plugin_context_meta PASSED
tests/test_routing.py::TestAtlasLoading::test_ontology_loaded PASSED     
tests/test_routing.py::TestAtlasLoading::test_epistemology_loaded PASSED 
tests/test_routing.py::TestAtlasLoading::test_phenomenology_loaded PASSED
tests/test_routing.py::TestAtlasLoading::test_archetype_found PASSED     
tests/test_routing.py::TestAtlasLoading::test_archetype_index_convention 
  Archetype at index 43,605 / 65,536
PASSED
tests/test_routing.py::TestStateTransitions::test_initial_state_is_archetype PASSED
tests/test_routing.py::TestStateTransitions::test_single_byte_transition PASSED
tests/test_routing.py::TestStateTransitions::test_all_bytes_produce_transitions PASSED
tests/test_routing.py::TestStateTransitions::test_transitions_deterministic PASSED
tests/test_routing.py::TestStateTransitions::test_reset_returns_to_archetype PASSED
tests/test_routing.py::TestStateTransitions::test_step_counter_increments_and_resets PASSED
tests/test_routing.py::TestMultiStepRouting::test_two_byte_sequence PASSED
tests/test_routing.py::TestMultiStepRouting::test_payload_routing PASSED 
tests/test_routing.py::TestMultiStepRouting::test_order_matters PASSED   
tests/test_routing.py::TestSignatureProperties::test_signature_fields PASSED
tests/test_routing.py::TestSignatureProperties::test_signature_hex_format
 PASSED
tests/test_routing.py::TestSignatureProperties::test_signature_consistency PASSED
tests/test_routing.py::TestReachability::test_archetype_reachable_from_itself PASSED
tests/test_routing.py::TestReachability::test_random_walk_stays_in_ontology PASSED
tests/test_routing.py::TestAtlasGlobalGroupFacts::test_each_byte_column_is_permutation PASSED
tests/test_routing.py::TestAtlasGlobalGroupFacts::test_bfs_radius_two_from_archetype PASSED
tests/test_routing.py::TestAtlasGlobalGroupFacts::test_depth4_alternation_identity_on_all_states_for_selected_pairs PASSED
tests/test_routing.py::TestHorizonFixedPoints::test_R0xAA_fixed_points_match_horizon_set_and_count PASSED
tests/test_routing.py::TestRowFanoutDistinctness::test_row_fanout_is_256_for_all_states PASSED
tests/test_routing.py::TestEpistemologyMatchesVectorizedStep::test_epistemology_matches_vectorized_step_for_all_states_all_bytes PASSED
==========
ROUTING TEST SUMMARY
==========
Ontology size: 65,536 states
Epistemology shape: (65536, 256)
Archetype state: 0xaaa555
Archetype index: 43605
Archetype A12: 0xaaa
==========

==========
OVERALL TEST SUMMARY
==========
Atlas loaded: YES
  Ontology: 65,536 states
  Epistemology: (65536, 256)
  Phenomenology: 6 arrays

Constants:
  GENE_MIC_S: 0xaa
  Archetype: 0xaaa555
  Unique masks: 256
==========

==========
PHYSICS 4 DASHBOARD - CGM EMERGENCE
==========
  EXISTING:
  ✓ Reference byte cycles and eigenphases
  ✓ Code duality and MacWilliams
  ✓ Walsh spectrum support
  ✓ Shell distributions from enumerator
  ✓ CGM constants bridge (δ, m_a, Q_G, α)
  ✓ Group presentation in (u,v)

  NEW EXPLORATIONS:
  • Aperture constraint: Q_G × m_a² = 1/2
  • Holonomy group = mask code
  • Optical conjugacy shell products
  • K_QG commutator scale
  • Neutrino-scale-like invariant R_nu
  • Restricted alphabet phase transition (rank/orbit theorem, bridge masks)
==========

==========
PHYSICS 3 DASHBOARD - KERNEL STRUCTURE TOWARDS CGM UNITS
==========
  ✓ Reference byte 0xAA: involution with 256 fixed points and 32640 2-cycles
  ✓ Non-reference bytes: proven pure 4-cycle permutations on Ω
  ✓ Permutation-unitary eigenphases implied and printed
  ✓ Mask code spanning set via intron basis bytes (2^8 = 256 masks)      
  ✓ Exact linear-code duality: |C|*|C_perp| = 2^12
  ✓ MacWilliams identity verified for C and C_perp
  ✓ Walsh spectrum support theorem verified: support equals C_perp       
  ✓ Atlas shell distributions forced by code enumerator (exact)
  ✓ CGM Units bridge diagnostics printed (A, m_a, Q_G, K_QG, alpha)      
==========

==========
KERNEL CGM EMERGENCE DASHBOARD (Physics 3)
==========
  ✓ Closed-form (u,v) phase-space dynamics: (u_next, v_next) = (v, u XOR m_b)
  ✓ Commutator K(x,y) is global translation: s_out = s XOR ((d<<12)|d), d=m_x XOR m_y
  ✓ Kernel monodromy: base closure + fiber defect (CGM-anchored)
  ✓ Complement symmetry commutes with byte actions (global commuting automorphism)
  ✓ CGM threshold anatomy: mask code cartography (2×3×2 decomposition)   
  ✓ Kernel -> CGM invariant reconstruction (δ, m_a, Q_G, α)
  ✓ CS anchor: mean fiber-angle = π/2 (exact theorem)
  ✓ Monodromy hierarchy bridge: θ_min → δ → ω (kernel-native scales)     
  ✓ Discrete aperture shadow: A_kernel = 5/256 vs A*
==========

==========
PHYSICS TEST SUMMARY
==========
Unique masks: 256 / 256
Non-zero A masks: 255 / 256
Non-zero B masks: 0 / 256 (expected 0)
==========


========================= 146 passed in 11.45s ========================= 
(.venv) PS F:\Development\superintelligence> 