"""
Suffix Adaptor (Step 2): Aho-Corasick automaton over byte_trie.

Replaces the expensive backward scan in expand_byte_ids() with O(1) state
transitions. Exact, portable (.npz), same semantics as the trie.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as script from any cwd: blomo_port must be on path for common
_adaptors_dir = Path(__file__).resolve().parent
_blomo_port = _adaptors_dir.parent
if str(_blomo_port) not in sys.path:
    sys.path.insert(0, str(_blomo_port))

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from common import PROJECT_ROOT, bolmo_reset_local_caches, load_bolmo


@dataclass
class SuffixAutomaton:
    """
    Aho-Corasick automaton for suffix resolution.

    Arrays:
      - next_state: [num_states, 256] -> next state for each byte
      - best_eid: [num_states] -> longest-token-id ending at this state
    """
    next_state: np.ndarray  # [num_states, 256], dtype=int32
    best_eid: np.ndarray    # [num_states], dtype=int32
    num_states: int

    def save(self, path: Path) -> None:
        np.savez(path, next_state=self.next_state, best_eid=self.best_eid, num_states=self.num_states)

    @classmethod
    def load(cls, path: Path) -> "SuffixAutomaton":
        data = np.load(path)
        return cls(
            next_state=data["next_state"],
            best_eid=data["best_eid"],
            num_states=int(data["num_states"]),
        )


def expand_byte_ids_with_automaton(
    byte_ids: list[int],
    automaton: SuffixAutomaton,
    tokenizer: Any,
) -> list[int]:
    """Same semantics as tokenizer.expand_byte_ids(byte_ids), O(1) per byte. For use by module_7 etc."""
    offset = int(getattr(tokenizer, "offset", 4))
    unk_id = int(getattr(tokenizer.hf_tokenizer, "unk_token_id", 0))
    result: list[int] = []
    state = 0
    for byte_id in byte_ids:
        if byte_id < offset:
            result.append(byte_id)
            state = 0
            continue
        if byte_id >= offset + 256:
            byte_val = byte_id - (offset + 256)
        else:
            byte_val = byte_id - offset
        state = int(automaton.next_state[state, byte_val])
        eid = int(automaton.best_eid[state])
        result.append(unk_id if eid == -1 else eid)
    return result


def _norm_trie_key_to_byteval(k: int, tok: Any) -> Optional[int]:
    """Normalize byte_trie edge key to byte value 0..255."""
    k = int(k)
    bpe_end = int(getattr(tok, "bpe_token_end_id", 3))
    if k == bpe_end:
        return None
    off = int(getattr(tok, "offset", 4))
    boundary_offset = off + 256
    if k >= boundary_offset:
        k -= boundary_offset
    if k < off or k >= off + 256:
        return None
    return k - off


def build_suffix_automaton(tokenizer: Any) -> SuffixAutomaton:
    """Build Aho-Corasick automaton from tokenizer.byte_trie."""
    print("Building suffix automaton from byte_trie...")

    root_keys = [k for k in tokenizer.byte_trie.keys() if k != tokenizer.TOKEN_ID_KEY]
    if root_keys:
        rmin, rmax = min(root_keys), max(root_keys)
        print(f"  Trie root key range: {rmin}..{rmax} (sample: {sorted(root_keys)[:10]})")

    print("  Extracting byte sequences (normalized, forward)...")
    sequences: list[tuple[list[int], int, int]] = []  # (forward_seq, token_id, length)

    def extract_sequences(trie_dict: dict, current_rev: list[int]) -> None:
        for key, val in trie_dict.items():
            if key == tokenizer.TOKEN_ID_KEY:
                fwd = list(reversed(current_rev))
                sequences.append((fwd, int(val), len(fwd)))
                continue
            b = _norm_trie_key_to_byteval(key, tokenizer)
            if b is None:
                continue
            current_rev.append(b)
            extract_sequences(val, current_rev)
            current_rev.pop()

    extract_sequences(tokenizer.byte_trie, [])
    print(f"  Found {len(sequences)} byte-to-token mappings")

    next_state_dict = {0: {}}
    best_eid_dict = {0: -1}
    best_len_dict = {0: -1}

    for seq, token_id, L in sequences:
        node = 0
        for b in seq:
            if b not in next_state_dict[node]:
                new_node = len(next_state_dict)
                next_state_dict[node][b] = new_node
                next_state_dict[new_node] = {}
                best_eid_dict[new_node] = -1
                best_len_dict[new_node] = -1
            node = next_state_dict[node][b]
        if L > best_len_dict.get(node, -1):
            best_len_dict[node] = L
            best_eid_dict[node] = token_id

    num_states = len(next_state_dict)
    print(f"  Built trie with {num_states} states")

    fail_link: dict[int, int] = {0: 0}
    from collections import deque
    queue = deque()
    for b, node in next_state_dict[0].items():
        fail_link[node] = 0
        queue.append(node)

    while queue:
        current = queue.popleft()
        for byte_val, child in next_state_dict[current].items():
            f = fail_link[current]
            while f != 0 and byte_val not in next_state_dict[f]:
                f = fail_link[f]
            if byte_val in next_state_dict[f]:
                fail_link[child] = next_state_dict[f][byte_val]
            else:
                fail_link[child] = 0
            f = fail_link[child]
            if best_len_dict.get(f, -1) > best_len_dict.get(child, -1):
                best_len_dict[child] = best_len_dict[f]
                best_eid_dict[child] = best_eid_dict[f]
            queue.append(child)

    print(f"  Computed {len(fail_link)} failure links")

    next_state = np.full((num_states, 256), -1, dtype=np.int32)
    best_eid = np.full(num_states, -1, dtype=np.int32)

    for node in range(num_states):
        best_eid[node] = best_eid_dict.get(node, -1)
        for byte_val in range(256):
            if byte_val in next_state_dict[node]:
                next_state[node, byte_val] = next_state_dict[node][byte_val]
            elif node != 0:
                f = fail_link[node]
                while f != 0 and byte_val not in next_state_dict[f]:
                    f = fail_link[f]
                if byte_val in next_state_dict[f]:
                    next_state[node, byte_val] = next_state_dict[f][byte_val]
                else:
                    next_state[node, byte_val] = 0
            else:
                next_state[node, byte_val] = 0

    print(f"  Final automaton: {num_states} states, 256 byte transitions each")
    return SuffixAutomaton(next_state=next_state, best_eid=best_eid, num_states=num_states)


def test_automaton(tokenizer: Any, automaton: SuffixAutomaton, num_test_cases: int = 100) -> bool:
    """Test that automaton produces same results as expand_byte_ids."""
    from common import encode_no_specials
    print("\n[Testing automaton against reference expand_byte_ids]")
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "123456789",
        "aaaaaa",
        "test",
        "xyz" * 50,
        "The theory of relativity",
        "import numpy as np",
        "def foo():",
        "class Bar:",
    ]
    all_passed = True
    total_positions = 0
    for text in test_texts:
        byte_ids = encode_no_specials(tokenizer, text)
        ref_expanded = tokenizer.expand_byte_ids(byte_ids)
        auto_expanded = []
        state = 0
        for byte_id in byte_ids:
            if byte_id < tokenizer.offset:
                auto_expanded.append(byte_id)
                state = 0
                continue
            if byte_id >= tokenizer.offset + 256:
                byte_val = byte_id - (tokenizer.offset + 256)
            else:
                byte_val = byte_id - tokenizer.offset
            state = int(automaton.next_state[state, byte_val])
            best_eid = int(automaton.best_eid[state])
            if best_eid == -1:
                best_eid = tokenizer.hf_tokenizer.unk_token_id
            auto_expanded.append(best_eid)
        total_positions += len(byte_ids)
        if ref_expanded != auto_expanded:
            print(f"  FAIL: '{text[:30]}...'")
            print(f"    byte_ids: {byte_ids[:10]}...")
            print(f"    ref: {ref_expanded[:10]}...")
            print(f"    auto: {auto_expanded[:10]}...")
            for i, (r, a) in enumerate(zip(ref_expanded, auto_expanded)):
                if r != a:
                    print(f"    First mismatch at pos {i}: ref={r}, auto={a}")
                    break
            all_passed = False
        else:
            print(f"  PASS: '{text[:30]}...' ({len(byte_ids)} positions)")

    print(f"\n[Summary] {total_positions} positions tested")
    if all_passed:
        print("  All tests PASSED - automaton matches reference exactly")
    else:
        print("  Some tests FAILED - automaton needs debugging")
    return all_passed


def benchmark_automaton(tokenizer: Any, automaton: SuffixAutomaton, text: str, iterations: int = 10):
    """Benchmark automaton vs reference expand_byte_ids."""
    import time
    print(f"\n[Benchmark: automaton vs reference]")
    enc = tokenizer(text, return_tensors=None, add_special_tokens=False)
    if hasattr(enc, 'input_ids'):
        byte_ids = enc.input_ids
    elif isinstance(enc, dict):
        byte_ids = enc["input_ids"]
    else:
        byte_ids = enc
    if hasattr(byte_ids, 'tolist'):
        byte_ids = byte_ids.tolist()
    if hasattr(byte_ids, '__iter__') and not isinstance(byte_ids, (str, bytes)):
        byte_ids = list(byte_ids)
    flat = byte_ids[0] if (byte_ids and len(byte_ids) > 0 and isinstance(byte_ids[0], list)) else byte_ids
    byte_ids = [int(x) for x in flat] if isinstance(flat, list) else [int(flat)]

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = tokenizer.expand_byte_ids(byte_ids)
    ref_time = time.perf_counter() - t0

    def automaton_expand(ids: list[int]) -> list[int]:
        result = []
        state = 0
        for byte_id in ids:
            if byte_id < tokenizer.offset:
                result.append(byte_id)
                state = 0
                continue
            if byte_id >= tokenizer.offset + 256:
                byte_val = byte_id - (tokenizer.offset + 256)
            else:
                byte_val = byte_id - tokenizer.offset
            state = int(automaton.next_state[state, byte_val])
            best_eid = int(automaton.best_eid[state])
            if best_eid == -1:
                best_eid = tokenizer.hf_tokenizer.unk_token_id
            result.append(best_eid)
        return result

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = automaton_expand(byte_ids)
    auto_time = time.perf_counter() - t0
    speedup = ref_time / auto_time if auto_time > 0 else float('inf')
    print(f"  Text length: {len(byte_ids)} bytes")
    print(f"  Iterations: {iterations}")
    print(f"  Reference expand_byte_ids: {ref_time*1000:.2f}ms total")
    print(f"  Automaton O(1) per byte: {auto_time*1000:.2f}ms total")
    print(f"  Speedup: {speedup:.2f}x")
    return speedup


def main():
    print("=" * 10)
    print("SUFFIX ADAPTOR (Step 2)")
    print("=" * 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = PROJECT_ROOT / "data" / "models" / "Bolmo-1B"
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_bolmo(model_dir, device)
    automaton = build_suffix_automaton(tokenizer)
    save_path = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "suffix_automaton.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    automaton.save(save_path)
    print(f"\nSaved automaton to: {save_path}")
    print(f"  File size: {save_path.stat().st_size / 1024:.1f} KB")
    passed = test_automaton(tokenizer, automaton)
    benchmark_text = "The quick brown fox jumps over the lazy dog. " * 100
    benchmark_automaton(tokenizer, automaton, benchmark_text, iterations=100)
    print("\n" + "=" * 10)
    print("SUMMARY")
    print("=" * 10)
    print(f"Automaton states: {automaton.num_states}")
    print(f"Memory footprint: {automaton.next_state.nbytes / 1024 / 1024:.2f} MB")
    print("=" * 10)
    return automaton if passed else None


if __name__ == "__main__":
    import sys
    result = main()
    sys.exit(0 if result is not None else 1)
