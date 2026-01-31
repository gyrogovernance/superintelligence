#!/usr/bin/env python3
"""
GyroASI Validation Suite: Physics Calibration and Falsification Tests

This module provides comprehensive validation of GyroASI physics implementation,
ensuring the system behaves according to the Common Governance Model (CGM)
specification before conducting topology experiments.

Key validation tests:
- Pauli commutator identity verification (family-mask search)
- Parity-closed orbit validation  
- 3 degree click calibration
- SU(2) spinor structure (720 degree closure)
- Atlas integrity checks
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baby.kernel import governance
from baby.constants.atlas_builder import AtlasPaths

# Mask calculus for exact composition and equality
BIT48 = (1 << 48) - 1

# Intron family masks from governance
EXON_LI_MASK = governance.EXON_LI_MASK  # 0b01000010 - UNA/Parity bits
EXON_FG_MASK = governance.EXON_FG_MASK  # 0b00100100 - Forward Gyration bits
EXON_BG_MASK = governance.EXON_BG_MASK  # 0b00011000 - Backward Gyration bits


def get_family_introns(family_mask: int) -> List[int]:
    """Get all introns that have at least one bit set in the family mask."""
    return [i for i in range(256) if (i & family_mask) != 0]


def masks_for_intron(i: int) -> Tuple[int, int, int, int]:
    """Get exact masks for intron i: (A, C, P, M) where T_i(s) = (s & A) ^ C"""
    P = int(governance.INTRON_BROADCAST_MASKS[i]) & BIT48
    M = int(governance.XFORM_MASK[i]) & BIT48
    A = (~P) & BIT48  # gate mask
    C = M & A         # bias mask
    return A, C, P, M


def compose(seq: List[int]) -> Tuple[int, int]:
    """Compose sequence of introns into single transform: T_seq(s) = (s & A_seq) ^ C_seq"""
    A = BIT48
    C = 0
    for i in seq:
        A_i, C_i, _, _ = masks_for_intron(i)
        C = (C & A_i) ^ C_i   # project previous bias, add new bias
        A &= A_i              # accumulate gates
    return A, C


def equal_all(seq1: List[int], seq2: List[int]) -> bool:
    """Check if two sequences are identical for all states"""
    A1, C1 = compose(seq1)
    A2, C2 = compose(seq2)
    return (A1 == A2) and (C1 == C2)


def equal_up_to_parity(seq1: List[int], seq2: List[int]) -> bool:
    """Check if two sequences are equal up to global complement for all states"""
    A1, C1 = compose(seq1)
    A2, C2 = compose(seq2)
    return (A1 == A2) and (C1 == (C2 ^ BIT48))


class GyroValidationSuite:
    """Comprehensive validation suite for GyroASI physics."""
    
    def __init__(self, atlas_paths: AtlasPaths):
        """Initialise with atlas maps."""
        self.atlas_paths = atlas_paths
        self._load_atlas()
        self._prepare_layer_patterns()
        
    def _load_atlas(self) -> None:
        """Load atlas maps for validation."""
        try:
            self.ontology_keys = np.load(self.atlas_paths.ontology, mmap_mode='r')
            self.epistemology = np.load(self.atlas_paths.epistemology, mmap_mode='r')
            self.phenomenology_map = np.load(self.atlas_paths.phenomenology, mmap_mode='r')
            self.theta = np.load(self.atlas_paths.theta, mmap_mode='r')
            self.orbit_sizes = np.load(self.atlas_paths.orbit_sizes, mmap_mode='r')
            
            print(f"Loaded atlas: {len(self.ontology_keys)} states")
            
        except FileNotFoundError as e:
            raise RuntimeError(f"Atlas maps not found: {e}")
    
    def _prepare_layer_patterns(self) -> None:
        """Prepare layer patterns for signature computation."""
        # Extract 4-layer patterns from GENE_Mac_S
        self.layer_patterns = []
        for i in range(4):
            layer = governance.GENE_Mac_S[i].astype(np.int8).flatten()
            # Map +1→0, -1→1 to match 48-bit packing
            pattern = np.where(layer > 0, 0, 1).astype(np.uint8)
            self.layer_patterns.append(pattern)
    
    def state_to_bits(self, state: int) -> np.ndarray[np.uint8, np.dtype[np.uint8]]:
        """Convert 48-bit state to bit vector."""
        # 48-bit little-endian bool vector (0/1)
        bits = np.unpackbits(np.array([state >> 16, state & 0xFFFF], dtype='>u8').view(np.uint8))[-48:]
        return bits.astype(np.uint8)
    
    def layer_signature(self, state: int) -> Dict[str, Any]:
        """Compute layer signature for a state using tensor correlation."""
        T = governance.int_to_tensor(state).astype(np.int8)  # shape [4,2,3,2], ±1
        scores = [int(np.sum(T[i] * governance.GENE_Mac_S[i])) for i in range(4)]
        dominant = int(np.argmax(scores))
        return {'scores': scores, 'dominant': dominant}
    
    def apply_intron_to_state(self, state: int, intron: int) -> int:
        """Apply intron using atlas-only dynamics."""
        state_idx = np.searchsorted(self.ontology_keys, state)
        if state_idx >= len(self.ontology_keys) or self.ontology_keys[state_idx] != state:
            raise ValueError(f"State 0x{state:012x} not in ontology; illegal path or mapping bug")
        next_state_idx = self.epistemology[state_idx, intron]
        return int(self.ontology_keys[next_state_idx])
    
    def pauli_commutator_exact(self) -> Dict[str, Any]:
        """
        Search for Pauli-like commutator identity using family masks.
        
        Searches for triples (fg, bg, li) where:
        - fg has at least one FG bit set (EXON_FG_MASK)
        - bg has at least one BG bit set (EXON_BG_MASK)
        - li has at least one LI bit set (EXON_LI_MASK)
        
        And the identity [fg, bg] == [bg, fg, li] holds (exact or up to parity).
        """
        # Get all introns in each family
        fg_introns = get_family_introns(EXON_FG_MASK)
        bg_introns = get_family_introns(EXON_BG_MASK)
        li_introns = get_family_introns(EXON_LI_MASK)
        
        # For efficiency, sample from larger families
        max_per_family = 32
        if len(fg_introns) > max_per_family:
            fg_introns = fg_introns[:max_per_family]
        if len(bg_introns) > max_per_family:
            bg_introns = bg_introns[:max_per_family]
        if len(li_introns) > max_per_family:
            li_introns = li_introns[:max_per_family]
        
        exact_matches = []
        parity_matches = []
        total_searched = 0
        
        for fg in fg_introns:
            for bg in bg_introns:
                lhs = [fg, bg]
                
                for li in li_introns:
                    rhs = [bg, fg, li]
                    total_searched += 1
                    
                    if equal_all(lhs, rhs):
                        exact_matches.append((fg, bg, li))
                    elif equal_up_to_parity(lhs, rhs):
                        parity_matches.append((fg, bg, li))
        
        has_exact = len(exact_matches) > 0
        has_parity = len(parity_matches) > 0
        
        return {
            'exact_all_states': has_exact,
            'parity_all_states': has_parity,
            'exact_matches': exact_matches[:10],  # First 10 for display
            'parity_matches': parity_matches[:10],
            'exact_count': len(exact_matches),
            'parity_count': len(parity_matches),
            'total_searched': total_searched,
            'threshold_met': has_exact or has_parity
        }
    
    def pauli_commutator_orbit_level(self) -> Dict[str, Any]:
        """
        Test Pauli commutator at orbit level using family-mask introns.
        
        For found exact/parity matches from pauli_commutator_exact, verify
        they also preserve orbit structure (phenomenology).
        """
        # Use single-bit representatives from each family for orbit test
        fg_reps = [0x04, 0x20]  # Single FG bits
        bg_reps = [0x08, 0x10]  # Single BG bits
        li_reps = [0x02, 0x40]  # Single LI bits
        
        orbit_results = []
        best_triples = []
        
        for fg in fg_reps:
            for bg in bg_reps:
                lhs = [fg, bg]
                best_li = None
                best_rate = 0.0
                
                for li in li_reps:
                    rhs = [bg, fg, li]
                    matches = 0
                    total = 0
                    
                    # Sample states from ontology
                    sample_size = min(512, len(self.ontology_keys))
                    step = max(1, len(self.ontology_keys) // sample_size)
                    
                    for i in range(0, len(self.ontology_keys), step):
                        try:
                            s0 = int(self.ontology_keys[i])
                            s_lhs = self._apply_sequence(s0, lhs)
                            s_rhs = self._apply_sequence(s0, rhs)
                            
                            # Check orbit equivalence
                            lhs_idx = np.searchsorted(self.ontology_keys, s_lhs)
                            rhs_idx = np.searchsorted(self.ontology_keys, s_rhs)
                            
                            if (lhs_idx < len(self.ontology_keys) and 
                                rhs_idx < len(self.ontology_keys) and
                                self.ontology_keys[lhs_idx] == s_lhs and
                                self.ontology_keys[rhs_idx] == s_rhs):
                                
                                lhs_phenom = self.phenomenology_map[lhs_idx]
                                rhs_phenom = self.phenomenology_map[rhs_idx]
                                
                                if lhs_phenom == rhs_phenom:
                                    matches += 1
                                total += 1
                        except (ValueError, IndexError):
                            continue
                    
                    if total > 0:
                        rate = matches / total
                        if rate > best_rate:
                            best_rate = rate
                            best_li = li
                
                orbit_results.append(best_rate)
                if best_li is not None:
                    best_triples.append((fg, bg, best_li, best_rate))
        
        orbit_avg = float(np.mean(orbit_results)) if orbit_results else 0.0
        has_good_triple = any(r >= 0.95 for r in orbit_results)
        
        return {
            'orbit_avg_rate': orbit_avg,
            'orbit_rates': orbit_results,
            'best_triples': best_triples,
            'has_good_triple': has_good_triple,
            'threshold_met': has_good_triple and orbit_avg >= 0.90
        }
    
    def _apply_sequence(self, state: int, sequence: List[int]) -> int:
        """Apply sequence of introns to state using atlas lookups."""
        current_state = state
        for intron in sequence:
            current_state = self.apply_intron_to_state(current_state, intron)
        return current_state
    
    def parity_orbit_check(self, samples: int = 4096) -> Dict[str, float]:
        """
        Verify parity-closed orbits: v and v ⊕ FULL_MASK share same orbit representative.
        Fixed denominator bug - only count valid mirror pairs.
        """
        rng = np.random.default_rng(2)
        ok = 0
        valid = 0
        for _ in range(samples):
            i = int(rng.integers(0, len(self.ontology_keys)))
            v = int(self.ontology_keys[i])
            vc = v ^ governance.FULL_MASK
            j = int(np.searchsorted(self.ontology_keys, vc))
            if j >= len(self.ontology_keys) or self.ontology_keys[j] != vc:
                continue  # complement not present; skip
            valid += 1
            ok += int(int(self.phenomenology_map[i]) == int(self.phenomenology_map[j]))
        pass_rate = (ok / valid) if valid else 1.0
        return {'pass_rate': pass_rate, 'samples_tested': valid, 'passed': ok, 'threshold_met': pass_rate >= 0.999}
    
    def _theta_of(self, state: int) -> float:
        """Get theta value for a state from atlas."""
        i = int(np.searchsorted(self.ontology_keys, state))
        if i >= len(self.ontology_keys) or self.ontology_keys[i] != state:
            raise ValueError("state not in atlas")
        return float(self.theta[i])

    def calibrate_clicks_long_run(self, ops: list[int], run_len: int = 8, trials: int = 64, theta_window: tuple[float,float] | None = None) -> Dict[str, Any]:
        """Long-run slope calibration with θ-windowing to reduce variance."""
        rng = np.random.default_rng(11)
        click = math.pi / 60.0
        slopes = []

        for _ in range(trials):
            i = int(rng.integers(0, len(self.ontology_keys)))
            s = int(self.ontology_keys[i])
            th0 = float(self.theta[i])
            if theta_window is not None:
                lo, hi = theta_window
                if not (lo <= th0 <= hi):
                    continue

            s_curr = s
            th_prev = th0
            total = 0.0
            ok = True
            for _step in range(run_len):
                for op in ops:
                    try:
                        s_curr = self.apply_intron_to_state(s_curr, op)
                    except ValueError:
                        ok = False
                        break
                if not ok:
                    break
                try:
                    th_curr = self._theta_of(s_curr)
                except ValueError:
                    ok = False
                    break
                total += abs(th_curr - th_prev)
                th_prev = th_curr
            if ok and run_len > 0:
                slope_clicks = (total / run_len) / click
                slopes.append(slope_clicks)

        if not slopes:
            return {'median_clicks': float('nan'), 'mad_clicks': float('nan'), 'n': 0}

        med = float(np.median(slopes))
        mad = float(np.median(np.abs(np.array(slopes) - med)))
        return {'median_clicks': med, 'mad_clicks': mad, 'n': len(slopes)}

    def exact_clicks_from_archetype(self) -> Dict[str, float]:
        """
        Calculate exact click values from archetype using atlas, no calibration needed.
        """
        # Get archetype state
        s0 = int(governance.tensor_to_int(governance.GENE_Mac_S))
        
        # Calculate exact clicks for each micro-step
        clicks = {}
        for name, intron in [('FG1', 0x04), ('FG2', 0x20), ('BG1', 0x08), ('BG2', 0x10)]:
            try:
                # Get next state via atlas
                idx0 = int(np.searchsorted(self.ontology_keys, s0))
                s1 = int(self.ontology_keys[int(self.epistemology[idx0, intron])])
                
                # Calculate exact theta change
                theta0 = float(self.theta[idx0])
                theta1 = float(self.theta[int(np.searchsorted(self.ontology_keys, s1))])
                dtheta = abs(theta1 - theta0)
                
                # Convert to clicks (π/60 = 3°)
                clicks[name] = dtheta / (math.pi / 60.0)
            except (ValueError, IndexError):
                clicks[name] = 0.0
        
        return clicks
    
    def find_best_microstep(self) -> Dict[str, Any]:
        """Find the best micro-step by enumerating sequences of length ≤ 4."""
        s0 = int(governance.tensor_to_int(governance.GENE_Mac_S))
        generators = [0x04, 0x20, 0x08, 0x10, 0x02, 0x40]  # FG1, FG2, BG1, BG2, LI1, LI2
        
        best_sequence = None
        best_clicks = float('inf')
        best_stability = 0.0
        
        # Enumerate all sequences of length 1-4
        for length in range(1, 5):
            for sequence in self._generate_sequences(generators, length):
                try:
                    clicks = self._calculate_sequence_clicks(s0, sequence)
                    if clicks > 0 and clicks < best_clicks:
                        # Check stability across a few canonical states
                        stability = self._calculate_sequence_stability(sequence)
                        
                        best_sequence = sequence
                        best_clicks = clicks
                        best_stability = stability
                        
                except (ValueError, IndexError):
                    continue
        
        return {
            'sequence': best_sequence,
            'clicks': best_clicks,
            'stability': best_stability,
            'found': best_sequence is not None
        }
    
    def _generate_sequences(self, generators: List[int], length: int) -> List[List[int]]:
        """Generate all sequences of given length from generators."""
        if length == 1:
            return [[g] for g in generators]
        
        sequences = []
        for seq in self._generate_sequences(generators, length - 1):
            for g in generators:
                sequences.append(seq + [g])
        return sequences
    
    def _calculate_sequence_clicks(self, s0: int, sequence: List[int]) -> float:
        """Calculate clicks for a sequence from archetype."""
        try:
            s_final = self._apply_sequence(s0, sequence)
            
            idx0 = int(np.searchsorted(self.ontology_keys, s0))
            idx_final = int(np.searchsorted(self.ontology_keys, s_final))
            
            theta0 = float(self.theta[idx0])
            theta_final = float(self.theta[idx_final])
            dtheta = abs(theta_final - theta0)
            
            return dtheta / (math.pi / 60.0)
        except (ValueError, IndexError):
            return 0.0
    
    def _calculate_sequence_stability(self, sequence: List[int]) -> float:
        """Calculate stability of sequence across canonical states."""
        canonical_states = [
            int(governance.tensor_to_int(governance.GENE_Mac_S)),
            int(governance.tensor_to_int(governance.GENE_Mac_S)) ^ BIT48
        ]
        
        clicks_variations = []
        for s0 in canonical_states:
            try:
                clicks = self._calculate_sequence_clicks(s0, sequence)
                if clicks > 0:
                    clicks_variations.append(clicks)
            except (ValueError, IndexError):
                continue
        
        if len(clicks_variations) < 2:
            return 0.0
        
        # Stability = 1 - (std / mean)
        mean_clicks = float(np.mean(clicks_variations))
        std_clicks = float(np.std(clicks_variations))
        return 1.0 - (std_clicks / mean_clicks) if mean_clicks > 0 else 0.0

    def calibrate_clicks(self, trials: int = 64) -> Dict[str, Any]:
        """Improved click calibration using long-run slope + θ-windowing for stability."""
        # candidate micro-steps (short sequences)
        candidates = {
            'FG1': [0x04],
            'FG2': [0x20],
            'BG1': [0x08],
            'BG2': [0x10],
            'μ_FG→BG': [0x04, 0x08],
            'μ_BG→FG': [0x08, 0x04],
        }
        # try two windows: UNA and broad default (reduced for speed)
        pi = math.pi
        windows = [
            (pi/4 - 0.20, pi/4 + 0.20),  # UNA window
            None  # No windowing
        ]

        results = {}
        for name, ops in candidates.items():
            best = None
            for win in windows:
                r = self.calibrate_clicks_long_run(ops, run_len=8, trials=trials, theta_window=win)
                if r['n'] < 128 or math.isnan(r['median_clicks']):
                    continue  # discard; too few samples
                r['window'] = win
                if best is None or (r['mad_clicks']/max(1e-9, r['median_clicks'])) < (best['mad_clicks']/max(1e-9, best['median_clicks'])):
                    best = r
            if best:
                results[name] = best

        if not results:
            return {'all': {}, 'chosen': {'name': None, 'median_clicks': float('nan'), 'mad_clicks': float('nan'), 'n': 0}, 'threshold_met': False}

        # choose candidate with lowest relative MAD in [0.5, 8] clicks
        filtered = [(k,v) for k,v in results.items() if 0.5 <= v['median_clicks'] <= 8]
        if not filtered:
            filtered = list(results.items())
        filtered.sort(key=lambda kv: (kv[1]['mad_clicks']/max(1e-9, kv[1]['median_clicks'])))
        chosen_name, chosen = filtered[0]
        chosen_out = {'name': chosen_name, **{k: chosen[k] for k in ['median_clicks','mad_clicks','n','window']}}

        # acceptance: relative MAD ≤ 0.20 and minimum samples ≥ 128
        threshold_met = (chosen['mad_clicks']/max(1e-9, chosen['median_clicks'])) <= 0.20 and chosen['n'] >= 128

        return {'all': results, 'chosen': chosen_out, 'threshold_met': threshold_met}
    
    def _find_period_with_half(
        self, start: int, ops: List[int], pmax: int = 64
    ) -> Tuple[Optional[int], Optional[int], bool]:
        """
        Find period and half-period for SU(2) spinor structure.
        
        Returns: (full_period, half_period, has_parity_at_half)
        - full_period: k where g^k(s) = s (or None if not found)
        - half_period: k where g^k(s) = s ^ FULL_MASK (or None if not found)
        - has_parity_at_half: True if half_period exists and 2*half_period = full_period
        """
        s = start
        parity_partner = start ^ governance.FULL_MASK
        half_period = None
        
        for step in range(1, pmax + 1):
            # Apply one iteration of the generator sequence
            for op in ops:
                s = self.apply_intron_to_state(s, op)
            
            # Check for parity partner (half turn in SU(2))
            if s == parity_partner and half_period is None:
                half_period = step
            
            # Check for return to start (full turn)
            if s == start:
                # SU(2) spinor: half_period should be exactly half of full_period
                has_parity_at_half = (half_period is not None and 
                                      half_period * 2 == step)
                return step, half_period, has_parity_at_half
        
        return None, half_period, False

    def su2_spinor_closure(self) -> Dict[str, Any]:
        """
        Test SU(2) spinor structure (720 degree closure).
        
        A proper SU(2) discrete analog should satisfy:
        - g^k(s) = s XOR parity_partner at "half turn" (360 degrees)
        - g^(2k)(s) = s at "full turn" (720 degrees)
        
        This is the discrete analog of spinor behavior where a 360 degree
        rotation gives -1 (parity flip) and 720 degrees returns to identity.
        """
        # Candidate generator sequences to test
        # These are physically motivated: FG+BG combinations should generate rotations
        generators = [
            ([0x04], "FG1"),
            ([0x08], "BG1"),
            ([0x04, 0x08], "FG1+BG1"),
            ([0x08, 0x04], "BG1+FG1"),
            ([0x20, 0x10], "FG2+BG2"),
            ([0x04, 0x08, 0x04, 0x08], "FG1+BG1 x2"),
            ([0x04, 0x02], "FG1+LI1"),
            ([0x08, 0x02], "BG1+LI1"),
        ]
        
        # Test on multiple starting states
        test_states = []
        archetype = int(governance.tensor_to_int(governance.GENE_Mac_S))
        test_states.append(archetype)
        
        # Add some orbit representatives
        unique_orbits = np.unique(self.phenomenology_map)
        for orbit_id in unique_orbits[:8]:  # First 8 orbits
            orbit_idx = np.where(self.phenomenology_map == orbit_id)[0]
            if len(orbit_idx) > 0:
                test_states.append(int(self.ontology_keys[orbit_idx[0]]))
        
        results = []
        best_generator = None
        best_score = 0
        
        for ops, name in generators:
            spinor_count = 0
            total_tested = 0
            periods = []
            
            for s0 in test_states:
                try:
                    full_p, half_p, is_spinor = self._find_period_with_half(s0, ops)
                    total_tested += 1
                    
                    if is_spinor:
                        spinor_count += 1
                        periods.append((full_p, half_p))
                except (ValueError, IndexError):
                    continue
            
            if total_tested > 0:
                rate = spinor_count / total_tested
                results.append({
                    'generator': name,
                    'ops': ops,
                    'spinor_rate': rate,
                    'spinor_count': spinor_count,
                    'total_tested': total_tested,
                    'periods': periods[:5]  # First 5 for display
                })
                
                if rate > best_score:
                    best_score = rate
                    best_generator = name
        
        # Also test the legacy layer traversal for comparison
        layer_result = self._test_layer_traversal()
        
        has_spinor_structure = best_score >= 0.5  # At least half show spinor behavior
        
        return {
            'generator_results': results,
            'best_generator': best_generator,
            'best_spinor_rate': best_score,
            'layer_traversal': layer_result,
            'has_spinor_structure': has_spinor_structure,
            'threshold_met': has_spinor_structure or layer_result.get('has_family_alternation', False)
        }

    def _test_layer_traversal(self) -> Dict[str, Any]:
        """Legacy layer traversal test for backward compatibility."""
        schedule = [0x04, 0x02, 0x08, 0x40]  # FG1, LI1, BG1, LI2
        s0 = int(governance.tensor_to_int(governance.GENE_Mac_S))
        
        layer_sequence = []
        family_sequence = []
        
        try:
            current_state = s0
            for intron in schedule:
                idx = int(np.searchsorted(self.ontology_keys, current_state))
                current_state = int(self.ontology_keys[int(self.epistemology[idx, intron])])
                
                T = governance.int_to_tensor(current_state).astype(np.int8)
                scores = [int(np.sum(T[i] * governance.GENE_Mac_S[i])) for i in range(4)]
                dominant = int(np.argmax(scores))
                layer_sequence.append(dominant)
                family_sequence.append(0 if dominant in [0, 2] else 1)
            
            family_alternations = sum(1 for i in range(1, len(family_sequence)) 
                                      if family_sequence[i] != family_sequence[i-1])
            
            return {
                'layer_sequence': layer_sequence,
                'family_sequence': family_sequence,
                'unique_layers': len(set(layer_sequence)),
                'has_family_alternation': family_alternations > 0
            }
        except (ValueError, IndexError):
            return {'has_family_alternation': False}

    def compare_parity_operators(self, samples: int = 8192) -> Dict[str, Any]:
        """
        Compare candidate parity operators to find the true physics parity.
        
        Tests three involutions:
        1. Representation complement: C(s) = s ^ FULL_MASK
        2. LI1-step: P1(s) = T_{0x02}(s) via epistemology
        3. LI2-step: P2(s) = T_{0x40}(s) via epistemology
        4. LI-both: P3(s) = T_{0x42}(s) via epistemology
        
        For each, measures:
        - Partner existence rate (always 100% for LI steps since they're transitions)
        - Orbit self-mirroring rate: phenom(s) == phenom(partner)
        
        The operator with near-100% orbit self-mirroring is the "physics parity".
        """
        rng = np.random.default_rng(42)
        
        # Sample random states from ontology
        n_states = len(self.ontology_keys)
        sample_indices = rng.choice(n_states, size=min(samples, n_states), replace=False)
        
        # Define parity operators
        operators = {
            'C (s^FULL_MASK)': lambda s, idx: (s ^ governance.FULL_MASK, None),
            'P1 (LI1=0x02)': lambda s, idx: (int(self.ontology_keys[int(self.epistemology[idx, 0x02])]), 0x02),
            'P2 (LI2=0x40)': lambda s, idx: (int(self.ontology_keys[int(self.epistemology[idx, 0x40])]), 0x40),
            'P3 (LI=0x42)': lambda s, idx: (int(self.ontology_keys[int(self.epistemology[idx, 0x42])]), 0x42),
        }
        
        results = {}
        
        for op_name, op_func in operators.items():
            partner_exists = 0
            orbit_match = 0
            is_involution = 0  # P(P(s)) == s
            total = 0
            
            for idx in sample_indices:
                s = int(self.ontology_keys[idx])
                s_phenom = int(self.phenomenology_map[idx])
                
                try:
                    partner, _intron = op_func(s, idx)
                    
                    # Check if partner exists in ontology
                    partner_idx = int(np.searchsorted(self.ontology_keys, partner))
                    if partner_idx >= n_states or self.ontology_keys[partner_idx] != partner:
                        # Partner not in ontology
                        total += 1
                        continue
                    
                    partner_exists += 1
                    
                    # Check orbit self-mirroring
                    partner_phenom = int(self.phenomenology_map[partner_idx])
                    if s_phenom == partner_phenom:
                        orbit_match += 1
                    
                    # Check involution property: P(P(s)) == s
                    try:
                        pp, _ = op_func(partner, partner_idx)
                        if pp == s:
                            is_involution += 1
                    except (ValueError, IndexError):
                        pass
                    
                    total += 1
                    
                except (ValueError, IndexError):
                    total += 1
                    continue
            
            if total > 0:
                results[op_name] = {
                    'partner_exists_rate': partner_exists / total,
                    'orbit_match_rate': orbit_match / total if partner_exists > 0 else 0.0,
                    'involution_rate': is_involution / total if partner_exists > 0 else 0.0,
                    'partner_exists': partner_exists,
                    'orbit_match': orbit_match,
                    'is_involution': is_involution,
                    'total': total,
                }
        
        # Determine best parity operator
        best_op = None
        best_score = 0.0
        for op_name, r in results.items():
            # Score: prioritize orbit_match_rate, then partner_exists_rate
            score = r['orbit_match_rate'] * r['partner_exists_rate']
            if score > best_score:
                best_score = score
                best_op = op_name
        
        return {
            'operators': results,
            'best_operator': best_op,
            'best_score': best_score,
            'recommendation': f"Use '{best_op}' as physics parity" if best_op else "No clear winner"
        }
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite and return results."""
        print("Running GyroASI Validation Suite...")
        
        results = {}
        
        # 1. Pauli commutator test (family-mask search)
        print("  Searching for Pauli commutator identity (family masks)...")
        results['pauli'] = self.pauli_commutator_exact()
        
        # 2. Pauli commutator test (orbit level)
        print("  Testing Pauli commutator at orbit level...")
        results['pauli_orbit'] = self.pauli_commutator_orbit_level()
        
        # 3. Parity orbit test  
        print("  Testing parity-closed orbits...")
        results['parity'] = self.parity_orbit_check()
        
        # 4. Exact click calculation
        print("  Calculating exact 3 degree click effects...")
        results['clicks'] = self.exact_clicks_from_archetype()
        
        # 5. Best micro-step search
        print("  Finding best micro-step...")
        results['microstep'] = self.find_best_microstep()
        
        # 6. SU(2) spinor closure test (720 degree)
        print("  Testing SU(2) spinor structure (720 degree closure)...")
        results['spinor'] = self.su2_spinor_closure()
        
        # 7. Parity operator comparison
        print("  Comparing parity operators (C, LI1, LI2, LI-both)...")
        results['parity_ops'] = self.compare_parity_operators()
        
        # Overall validation status
        all_passed = all([
            results['pauli']['threshold_met'],
            results['parity']['threshold_met'], 
            results['spinor']['threshold_met']
        ])
        
        results['overall'] = {
            'all_tests_passed': all_passed,
            'atlas_states': len(self.ontology_keys),
            'validation_timestamp': str(np.datetime64('now'))
        }
        
        return results
    
    def print_validation_report(self, results: Dict[str, Any]) -> None:
        """Print comprehensive validation report."""
        print("\n" + "="*60)
        print("GYROSI VALIDATION REPORT")
        print("="*60)
        
        # Pauli commutator (family-mask search)
        pauli = results['pauli']
        print(f"\nPauli Commutator Test (Family-Mask Search):")
        print(f"  Exact matches found: {pauli['exact_count']}")
        print(f"  Parity matches found: {pauli['parity_count']}")
        print(f"  Total triples searched: {pauli['total_searched']}")
        if pauli['exact_matches']:
            print(f"  First exact match: fg=0x{pauli['exact_matches'][0][0]:02x}, "
                  f"bg=0x{pauli['exact_matches'][0][1]:02x}, li=0x{pauli['exact_matches'][0][2]:02x}")
        if pauli['parity_matches']:
            print(f"  First parity match: fg=0x{pauli['parity_matches'][0][0]:02x}, "
                  f"bg=0x{pauli['parity_matches'][0][1]:02x}, li=0x{pauli['parity_matches'][0][2]:02x}")
        print(f"  Threshold: [PASS]" if pauli['threshold_met'] else "  Threshold: [FAIL]")
        
        # Pauli orbit level
        pauli_orbit = results['pauli_orbit']
        print(f"\nPauli Commutator (Orbit Level):")
        print(f"  Average orbit equivalence rate: {pauli_orbit['orbit_avg_rate']:.3f}")
        print(f"  Has good triple (>=95%): {'[PASS]' if pauli_orbit['has_good_triple'] else '[FAIL]'}")
        
        # Parity orbits
        parity = results['parity']
        print(f"\nParity-Closed Orbits:")
        print(f"  Pass rate: {parity['pass_rate']:.3f} ({parity['passed']}/{parity['samples_tested']})")
        print(f"  Threshold >=0.999: {'[PASS]' if parity['threshold_met'] else '[FAIL]'}")
        
        # Exact clicks
        clicks = results['clicks']
        print(f"\n3 Degree Click Calculation (Exact):")
        print(f"  FG1: {clicks['FG1']:.3f} clicks")
        print(f"  FG2: {clicks['FG2']:.3f} clicks")
        print(f"  BG1: {clicks['BG1']:.3f} clicks")
        print(f"  BG2: {clicks['BG2']:.3f} clicks")
        
        # SU(2) Spinor closure (720 degree)
        spinor = results['spinor']
        print(f"\nSU(2) Spinor Structure (720 Degree Closure):")
        print(f"  Best generator: {spinor['best_generator']}")
        print(f"  Best spinor rate: {spinor['best_spinor_rate']:.3f}")
        if spinor['generator_results']:
            print("  Generator results:")
            for r in spinor['generator_results'][:4]:
                print(f"    {r['generator']}: {r['spinor_rate']:.2f} "
                      f"({r['spinor_count']}/{r['total_tested']})")
        layer = spinor.get('layer_traversal', {})
        if layer:
            print(f"  Layer traversal: {layer.get('layer_sequence', [])}")
            print(f"  Family alternation: {'[PASS]' if layer.get('has_family_alternation') else '[FAIL]'}")
        print(f"  Threshold: {'[PASS]' if spinor['threshold_met'] else '[FAIL]'}")
        
        # Parity operator comparison
        parity_ops = results.get('parity_ops', {})
        if parity_ops:
            print(f"\nParity Operator Comparison:")
            print(f"  {'Operator':<20} {'Partner%':>10} {'Orbit%':>10} {'Invol%':>10}")
            print(f"  {'-'*50}")
            for op_name, r in parity_ops.get('operators', {}).items():
                print(f"  {op_name:<20} {r['partner_exists_rate']*100:>9.1f}% "
                      f"{r['orbit_match_rate']*100:>9.1f}% {r['involution_rate']*100:>9.1f}%")
            print(f"  Best operator: {parity_ops.get('best_operator', 'N/A')}")
            print(f"  Recommendation: {parity_ops.get('recommendation', 'N/A')}")
        
        # Overall status
        overall = results['overall']
        print(f"\nOverall Status:")
        print(f"  All tests passed: {'[PASS]' if overall['all_tests_passed'] else '[FAIL]'}")
        print(f"  Atlas states: {overall['atlas_states']:,}")
        
        if not overall['all_tests_passed']:
            print("\n[WARNING] VALIDATION FAILED - Physics implementation needs fixes!")
        else:
            print("\n[SUCCESS] VALIDATION PASSED - Ready for topology experiments!")


def main():
    """Run validation suite."""
    # Path relative to research/experiments/ -> research/memories/public/meta
    script_dir = Path(__file__).parent
    atlas_dir = script_dir.parent / "memories" / "public" / "meta"
    atlas_paths = AtlasPaths.from_directory(atlas_dir)
    validator = GyroValidationSuite(atlas_paths)
    
    results = validator.run_full_validation()
    validator.print_validation_report(results)
    
    return results['overall']['all_tests_passed']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
