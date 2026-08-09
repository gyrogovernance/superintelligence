"""Gyroscopic helpers: build, bench, run, gates.

- ``build_*.ps1`` / package ``build.py`` — native kernel / llama.cpp builds
- ``bench_gyroscopic_llama`` — stock vs gyroscopic generation bench
- ``run_bonsai`` — chat entry (ledger+KV; ``--incomplete-forward`` for unfinished-site stress only)
- ``gates`` — canonical acceptance gates (NavPad §7): ledger / kv / codecs / causal / forward-probe

Production ledger: package-root ``ledger.py`` (auto-ensured by ``production_gyroscopic_env``).
C sources: ``kernel``, ``ledger``, ``attn``, ``codec``.
"""
