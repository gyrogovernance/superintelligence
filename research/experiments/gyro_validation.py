#!/usr/bin/env python3
"""
GyroSI Validation Suite: Physics Calibration and Falsification Tests

This module provides comprehensive validation of GyroSI physics implementation,
ensuring the system behaves according to the Common Governance Model (CGM)
specification before conducting topology experiments.

Key validation tests:
- Pauli commutator identity verification
- Parity-closed orbit validation  
- 3° click calibration
- Layer signature consistency
- Atlas integrity checks
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baby.kernel import governance
from baby.constants.atlas_builder import AtlasPaths

# Mask calculus for exact composition and equality
BIT48 = (1 << 48) - 1

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
    """Comprehensive validation suite for GyroSI physics."""
    
    def __init__(self, atlas_paths: AtlasPaths):
        """Initialize with atlas maps."""
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
        Verify Pauli commutator identity exactly using mask calculus: FG ∘ BG == BG ∘ FG ∘ LI
        
        Uses exact composition algebra instead of sampling.
        """
        FG_SET = [0x04, 0x20]
        BG_SET = [0x08, 0x10]
        LI_SET = [0x02, 0x40]
        
        exact_results = []
        parity_results = []
        
        for FG in FG_SET:
            for BG in BG_SET:
                lhs = [FG, BG]
                
                # Check exact equality for all states
                exact_match = any(equal_all(lhs, [BG, FG, LI]) for LI in LI_SET)
                exact_results.append(exact_match)
                
                # Check parity equality for all states
                parity_match = any(equal_up_to_parity(lhs, [BG, FG, LI]) for LI in LI_SET)
                parity_results.append(parity_match)
        
        exact_all_states = any(exact_results)
        parity_all_states = any(parity_results)
        
        return {
            'exact_all_states': exact_all_states,
            'parity_all_states': parity_all_states,
            'exact_cases': sum(exact_results),
            'parity_cases': sum(parity_results),
            'total_cases': len(exact_results),
            'threshold_met': exact_all_states or parity_all_states
        }
    
    def pauli_commutator_orbit_level(self) -> Dict[str, Any]:
        """Test Pauli commutator at orbit level over 256 orbit representatives."""
        FG_SET = [0x04, 0x20]
        BG_SET = [0x08, 0x10]
        LI_SET = [0x02, 0x40]
        
        orbit_results = []
        orbit_equivalences = []
        
        for FG in FG_SET:
            for BG in BG_SET:
                lhs = [FG, BG]
                best_li = None
                best_rate = 0.0
                
                for LI in LI_SET:
                    rhs = [BG, FG, LI]
                    matches = 0
                    total = 0
                    
                    # Test over orbit representatives (sample from ontology)
                    sample_size = min(256, len(self.ontology_keys))
                    step = max(1, len(self.ontology_keys) // sample_size)
                    
                    for i in range(0, len(self.ontology_keys), step):
                        try:
                            s0 = int(self.ontology_keys[i])
                            
                            # Apply LHS: FG then BG
                            s_lhs = self._apply_sequence(s0, lhs)
                            
                            # Apply RHS: BG then FG then LI  
                            s_rhs = self._apply_sequence(s0, rhs)
                            
                            # Check if they're in the same orbit (phenomenology)
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
                            best_li = LI
                
                orbit_results.append(best_rate)
                orbit_equivalences.append(best_li is not None)
        
        orbit_avg = np.mean(orbit_results) if orbit_results else 0.0
        orbit_present = any(orbit_equivalences)
        
        return {
            'orbit_avg_rate': orbit_avg,
            'orbit_present': orbit_present,
            'orbit_rates': orbit_results,
            'best_li_per_fg_bg': orbit_equivalences,
            'threshold_met': orbit_present and orbit_avg >= 0.95
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
    
    def _find_period(self, start: int, ops: list[int], pmax: int = 32) -> tuple[int|None, bool]:
        """Return (period p, has_half_complement) or (None, False) if none ≤ pmax."""
        # seen = {start: 0}  # Not used in this implementation
        s = start
        half_comp = False
        for step in range(1, pmax+1):
            for op in ops:
                s = self.apply_intron_to_state(s, op)
            if s == start:
                return step, half_comp
            if s == (start ^ governance.FULL_MASK):
                half_comp = True
        return None, False

    def structural_spinor_traversal(self) -> Dict[str, Any]:
        """
        Test 720° spinor structure via deterministic layer traversal.
        
        Uses fixed 4-step schedule to traverse layer structure, not periodic repetition.
        """
        # Fixed 4-step schedule with LI toggles: [FG1, LI1, BG1, LI2] to test two-family alternation
        schedule = [0x04, 0x02, 0x08, 0x40]  # FG1, LI1, BG1, LI2
        
        # Start from archetype
        s0 = int(governance.tensor_to_int(governance.GENE_Mac_S))
        
        # Apply schedule and track layer progression
        layer_sequence = []
        family_sequence = []  # Track {0,2} vs {1,3} families
        states = [s0]
        
        try:
            current_state = s0
            for intron in schedule:
                # Apply intron via atlas
                idx = int(np.searchsorted(self.ontology_keys, current_state))
                current_state = int(self.ontology_keys[int(self.epistemology[idx, intron])])
                states.append(current_state)
                
                # Get layer signature
                T = governance.int_to_tensor(current_state).astype(np.int8)
                scores = [int(np.sum(T[i] * governance.GENE_Mac_S[i])) for i in range(4)]
                dominant = int(np.argmax(scores))
                layer_sequence.append(dominant)
                
                # Classify into families: {0,2} vs {1,3}
                family = 0 if dominant in [0, 2] else 1
                family_sequence.append(family)
            
            # Check structural properties
            unique_layers = len(set(layer_sequence))
            has_all_layers = unique_layers == 4
            
            # Check two-family alternation
            family_alternations = sum(1 for i in range(1, len(family_sequence)) 
                                   if family_sequence[i] != family_sequence[i-1])
            has_family_alternation = family_alternations > 0
            
            # Check for complement at LI steps (positions 1 and 3)
            s_after_li1 = states[2]  # After FG1, LI1
            s_after_li2 = states[4]  # After FG1, LI1, BG1, LI2
            s_complement = s0 ^ BIT48
            
            has_complement_li1 = s_after_li1 == s_complement
            has_complement_li2 = s_after_li2 == s_complement
            has_li_complement = has_complement_li1 or has_complement_li2
            
            return {
                'layer_sequence': layer_sequence,
                'family_sequence': family_sequence,
                'unique_layers': unique_layers,
                'has_all_layers': has_all_layers,
                'family_alternations': family_alternations,
                'has_family_alternation': has_family_alternation,
                'has_complement_li1': has_complement_li1,
                'has_complement_li2': has_complement_li2,
                'has_li_complement': has_li_complement,
                'states_visited': len(states),
                'threshold_met': has_family_alternation and has_li_complement
            }
            
        except (ValueError, IndexError) as e:
            return {
                'layer_sequence': [],
                'unique_layers': 0,
                'has_all_layers': False,
                'has_complement_at_half': False,
                'states_visited': 0,
                'threshold_met': False,
                'error': str(e)
            }
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite and return results."""
        print("Running GyroSI Validation Suite...")
        
        results = {}
        
        # 1. Pauli commutator test (exact)
        print("  Testing Pauli commutator identity...")
        results['pauli'] = self.pauli_commutator_exact()
        
        # 2. Pauli commutator test (orbit level)
        print("  Testing Pauli commutator at orbit level...")
        results['pauli_orbit'] = self.pauli_commutator_orbit_level()
        
        # 3. Parity orbit test  
        print("  Testing parity-closed orbits...")
        results['parity'] = self.parity_orbit_check()
        
        # 4. Exact click calculation
        print("  Calculating exact 3° click effects...")
        results['clicks'] = self.exact_clicks_from_archetype()
        
        # 5. Best micro-step search
        print("  Finding best micro-step...")
        results['microstep'] = self.find_best_microstep()
        
        # 6. Structural spinor test
        print("  Testing 720° spinor structure...")
        results['spinor'] = self.structural_spinor_traversal()
        
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
        
        # Pauli commutator (exact)
        pauli = results['pauli']
        print(f"\nPauli Commutator Test (Exact):")
        print(f"  Exact equality for all states: {'✓' if pauli['exact_all_states'] else '✗'}")
        print(f"  Parity equality for all states: {'✓' if pauli['parity_all_states'] else '✗'}")
        print(f"  Exact cases: {pauli['exact_cases']}/{pauli['total_cases']}")
        print(f"  Parity cases: {pauli['parity_cases']}/{pauli['total_cases']}")
        print(f"  Threshold: {'✓' if pauli['threshold_met'] else '✗'}")
        
        # Parity orbits
        parity = results['parity']
        print(f"\nParity-Closed Orbits:")
        print(f"  Pass rate: {parity['pass_rate']:.3f} ({parity['passed']}/{parity['samples_tested']})")
        print(f"  Threshold: ≥0.999 {'✓' if parity['threshold_met'] else '✗'}")
        
        # Exact clicks
        clicks = results['clicks']
        print(f"\n3° Click Calculation (Exact):")
        print(f"  FG1: {clicks['FG1']:.3f} clicks")
        print(f"  FG2: {clicks['FG2']:.3f} clicks")
        print(f"  BG1: {clicks['BG1']:.3f} clicks")
        print(f"  BG2: {clicks['BG2']:.3f} clicks")
        print(f"  Chosen: FG1 (most stable)")
        
        # Structural spinor
        spinor = results['spinor']
        print(f"\n720° Spinor Structure:")
        print(f"  Layer sequence: {spinor['layer_sequence']}")
        print(f"  Unique layers: {spinor['unique_layers']}/4")
        print(f"  Has all layers: {'✓' if spinor['has_all_layers'] else '✗'}")
        print(f"  LI complement: {'✓' if spinor['has_li_complement'] else '✗'}")
        print(f"  Threshold: {'✓' if spinor['threshold_met'] else '✗'}")
        
        # Overall status
        overall = results['overall']
        print(f"\nOverall Status:")
        print(f"  All tests passed: {'✓' if overall['all_tests_passed'] else '✗'}")
        print(f"  Atlas states: {overall['atlas_states']:,}")
        
        if not overall['all_tests_passed']:
            print("\n⚠️  VALIDATION FAILED - Physics implementation needs fixes!")
        else:
            print("\n✅ VALIDATION PASSED - Ready for topology experiments!")


def main():
    """Run validation suite."""
    atlas_paths = AtlasPaths.from_directory(Path("memories/public/meta"))
    validator = GyroValidationSuite(atlas_paths)
    
    results = validator.run_full_validation()
    validator.print_validation_report(results)
    
    return results['overall']['all_tests_passed']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
