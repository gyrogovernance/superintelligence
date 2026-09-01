"""Helper modules for the hQVM autoencoder.

Flat package: every helper is ``<domain>_<role>.py`` where the domain tells
you what the file is actually for:

- ``training_``: the training loop and loss definitions;
- ``evals_``: everything that reads a model or dataset and judges it - running
  evaluations, computing metrics, verifying invariants, building eval corpora,
  and computing readouts.

There is no separate ``analysis_`` or ``experiments_`` domain - anything that reads a model or dataset and judges it lives under ``evals_`` so a new user never has to guess which of three near-synonym buckets a function lives in. The subdomains are:

- ``evals_run``: checkpoint loading, eval runners, reports, and verification
  (equivariance/audit);
- ``evals_metrics``: metric functions (probes, psi_hat, shadow invariance) plus
  kernel-exact readouts;
- ``evals_datasets``: eval dataset and corpus builders.

All evaluation work lives in the ``evals_`` domain. Helpers import root
models and other helpers acyclically.
"""