# gyrocrypt no-drift contract

## Mission

**Gyroscopic Holonomy Relation Finding** on GENE_Mac — not classical Shor/QFT simulation dressed as quantum.

- **Production:** `kernel/shor.py` → native F_{G_X} spectral readout (`native.c`)
- **Audit / falsification:** `kernel/audit.py` — same native scorers, explicit reference path
- **OPEN research:** `kernel/holonomy.py` — byte oracle compiler; fail-closed until bridge closes
- **Verify only (hot path):** `exp_mod_ladder` / `gcd` at end of factorization
- **Simon:** `kernel/simon.py` (GF(2)^{6B} HSP on correct group)

## Production rules

1. `kernel/shor.py` must **never** import `kernel/holonomy.py`.
2. Do not delete or gut `native.c` Shor/horizon/DLP spectral core while holonomy compiler is open.
3. Holonomy `compile_factor_operator` / `gyro_period` stay **fail-closed** until real byte compiler lands.
4. `dlp_solve` / `gyro_dlp` fail closed until dual-flow holonomy readout exists.

## Open milestone

`compile_factor_operator(N,a)` must compile multi-cell QuBEC holonomy:
`inject_residue_multicell(y) → inject_residue_multicell(a·y mod N)` via `MultiCellRouter`
byte-ledger (carry-coupled), with period from Ω spectral closure — **without** classical
`pow()` / coset enumeration in the holonomy hot path.

**Classical coset + CQFT in `native.c` is audit/falsification only** (`kernel/audit.py`).
It must not be deleted while the holonomy compiler is open, and must not be routed
through `kernel/shor.period()` until the QuBEC compiler closes at scale.
