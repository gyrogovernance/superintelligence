"""
CSM (Common Source Moment) derivation and storage model analysis.

Computes physical information capacity, CSM for k-byte Router states,
and storage requirements for various encoding models.
"""

import argparse
import math


N_PHYS = 3.25e30


def csm(k_bytes: int) -> float:
    """Common Source Moment: microcells per Router state for k-byte keys."""
    return N_PHYS / (2 ** (8 * k_bytes))


def bits_of(x: float) -> float:
    """Information capacity in bits."""
    return math.log2(x)


def human_bytes(x: float) -> str:
    """Format bytes with decimal prefixes (KB, MB, GB, ...)."""
    units = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    i = 0
    while x >= 1000.0 and i < len(units) - 1:
        x /= 1000.0
        i += 1
    return f"{x:.3f} {units[i]}"


def total_bits_for_s(s_bits: float) -> float:
    """Total bits for N_phys microcells at s bits per cell."""
    return N_PHYS * s_bits


def total_bytes_for_s(s_bits: float) -> float:
    """Total bytes for N_phys microcells at s bits per cell."""
    return total_bits_for_s(s_bits) / 8.0


def bytes_backing_per_router_state(k_bytes: int, s_bits: float) -> float:
    """Bytes needed per Router state when backing with s bits per microcell."""
    return csm(k_bytes) * (s_bits / 8.0)


def run_full_report(k_range: range = range(1, 17), s_values: dict | None = None) -> None:
    """Print the full CSM derivation and model analysis."""
    if s_values is None:
        s_values = {
            "2 bits": 2,
            "6 bits": 6,
            "2 bytes (16 bits)": 16,
            "4 bytes (32 bits)": 32,
        }

    bits_per_label = bits_of(N_PHYS)
    bytes_per_label = bits_per_label / 8.0
    words32_per_label = bits_per_label / 32.0

    print(f"N_phys = {N_PHYS:.3e} microcells")
    print(f"log2(N_phys) = {bits_per_label:.6f} bits "
          f"(~{bytes_per_label:.6f} bytes, ~{words32_per_label:.6f} 32-bit words)")
    print()

    # CSM table
    print("CSM (microcells per Router state):")
    for k in k_range:
        omega = 2 ** (8 * k)
        print(f"  k={k} bytes: |Omega|={omega:,}  CSM={csm(k):.3e}")
    print()

    # Physical capacity
    max_full_bytes = int(bits_per_label // 8)
    ceil_bytes = math.ceil(bits_per_label / 8.0)
    print("Physical capacity:")
    print(f"  max full bytes before CSM < 1 = {max_full_bytes}")
    print(f"  ceil bytes to uniquely label = {ceil_bytes}")
    print()

    # Baseline models
    bits_bitmap = total_bits_for_s(1)
    bytes_bitmap = total_bytes_for_s(1)
    bits_naive = N_PHYS * bits_per_label
    bytes_naive = bits_naive / 8.0
    words32_naive = bits_naive / 32.0

    print("Baseline models:")
    print(f"  Bitmap (1 bit/cell): {bytes_bitmap:.6e} bytes ({human_bytes(bytes_bitmap)})")
    print(f"  Naive labels: {bytes_naive:.6e} bytes ({human_bytes(bytes_naive)})")
    print(f"    32-bit words = {words32_naive:.6e}")
    print()

    # s-bit models
    for name, s in s_values.items():
        b = total_bits_for_s(s)
        B = total_bytes_for_s(s)
        w32 = b / 32.0
        print(f"  {name}: {B:.6e} bytes ({human_bytes(B)}), {w32:.6e} 32-bit words")
    print()

    # Router backing
    for k in (2, 4):
        print(f"Router k={k} bytes (CSM={csm(k):.6e} cells/state):")
        for name, s in s_values.items():
            bp = bytes_backing_per_router_state(k, s)
            print(f"  {name}: {bp:.6e} bytes ({human_bytes(bp)})")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSM derivation and storage model analysis"
    )
    parser.add_argument(
        "--k-max", type=int, default=16,
        help="Max k for CSM table (default: 16)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Short report (k=1..4 only)"
    )
    args = parser.parse_args()

    k_range = range(1, 5) if args.quick else range(1, args.k_max + 1)
    run_full_report(k_range=k_range)


if __name__ == "__main__":
    main()
