# Noetiko Convergence — Reproducibility Toolkit (Papers I–III)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18394099.svg)](https://doi.org/10.5281/zenodo.18394099)
![CI](https://github.com/Andre-Kappe-NOETIKO/Noetiko-convergence/actions/workflows/python-ci.yml/badge.svg)

Noetiko Convergence is a conservative, operational, and falsifiable implementation scaffold for reproducible analysis and simulation of **phase synchronization under noise**. The toolkit is designed for **auditable metrics**, **surrogate testing**, and **clear reporting artifacts** in open-science workflows.

It is explicitly **not** a mechanism-identification claim.

---

## Paper series (Zenodo)

- **Paper I — Energetics / Metastable information persistence**
  - DOI: https://doi.org/10.5281/zenodo.18379906
- **Paper II — Coordination dynamics / coupled oscillators**
  - DOI: https://doi.org/10.5281/zenodo.18379950
- **Paper III — Measurement-to-model protocols**
  - DOI: https://doi.org/10.5281/zenodo.18379991
- **Synthesis — Framework consolidation**
  - DOI: https://doi.org/10.5281/zenodo.18384275

Software release (this repository):
- **Software DOI (v0.2.0):** https://doi.org/10.5281/zenodo.18394099

---

## Scope & non-claims (SNO)

- No semantic conclusions (e.g., “health”, “repair”, “consciousness”) are inferred from synchronization statistics.
- Reduced parameters (e.g., K, D, κ) are **effective descriptors**, not microscopic coupling channels.
- Claims are restricted to **measurement-to-model contracts** with documented phase maps, admissibility gates, surrogate baselines, null hypotheses, and reporting minima.

---

## What is included

- Phase extraction (Hilbert analytic phase) with optional band-pass preprocessing
- Quality gates (amplitude / SNR proxy gates; unwrap consistency checks)
- Surrogate controls:
  - Fourier phase randomization (preserves marginal spectra)
  - Cross-channel time-shift surrogates (destroys coordination while preserving marginals)
- Coordination metrics (Kuramoto order parameter r(t), PLV, metastability index)
- Kuramoto sanity simulations (Euler–Maruyama; global coupling)
- Switching statistics helpers (dwell times and transition rates between low/high-r regimes)
- A proof-of-protocol demo script (synthetic by default)

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
