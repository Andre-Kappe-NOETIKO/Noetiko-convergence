# Noetiko Convergence — Reproducibility Toolkit (Papers I–IV)

![CI](https://github.com/Andre-Kappe-NOETIKO/Noetiko-convergence/actions/workflows/python-ci.yml/badge.svg)

A conservative, operational, and falsifiable implementation scaffold for the **Noetiko Convergence** paper series:
a reproducibility-first protocol stack for metastable information and coordination in open systems.

## Canonical papers (Zenodo)

- **Paper I — Noetiko Convergence I: Dimensional Delimitation and the Energetics of Biological Information Stability**  
  DOI: https://doi.org/10.5281/zenodo.18379906

- **Paper II — Noetiko Convergence II: Operational Phase-Coordination Signatures in Coupled Oscillator Networks**  
  DOI: https://doi.org/10.5281/zenodo.18379950

- **Paper III — Noetiko Convergence III: Measurement-to-Model Protocols for Metastable Coordination in Open Systems**  
  DOI: https://doi.org/10.5281/zenodo.18379991

- **Paper IV (Synthesis) — Noetiko Convergence: A Conservative, Falsifiable Framework for Metastable Information and Coordination in Open Systems (Synthesis)**  
  DOI: https://doi.org/10.5281/zenodo.18384275

![CI](...dein-ci-badge...)

---

## What this repository is (and is not)

This repository provides a conservative, operational, and falsifiable implementation scaffold for the **Noetiko Convergence** series:

- **Paper I**: metastable information persistence under open-system constraints (effective stability scale \(\kappa\), switching statistics, barrier-like inference)
- **Paper II**: coordination dynamics (Kuramoto-type reduced models) with explicit guardrails, surrogate controls, and counter-class discrimination
- **Paper III**: measurement-to-model protocols (phase extraction admissibility gates, reporting checklist, proof-of-protocol demo template)
- **Paper IV (Synthesis)**: unified vocabulary + axiomatic spine + protocol stack + falsification benchmarks linking Papers I–III

The code is intended to support **reproducible analysis** and **methodological stress-tests**.  
It is explicitly **not** a mechanism-identification claim.

---

## Scope & non-claims (SNO)

- No semantic conclusions (e.g., “health”, “repair”, “consciousness”) are inferred from synchronization statistics.
- Reduced parameters (e.g., \(K\), \(D\), \(\kappa\)) are **effective descriptors**, not microscopic coupling channels.
- Claims are restricted to **measurement-to-model contracts** with documented phase maps, admissibility gates, surrogate baselines, null hypotheses, and reporting minima.

---

## What is included

- Phase extraction (Hilbert analytic phase) with optional band-pass preprocessing
- Quality gates (amplitude/SNR proxy gates; unwrap consistency checks)
- Surrogate controls
  - Fourier phase randomization (preserves marginal spectra)
  - Cross-channel time-shift surrogates (destroys coordination while preserving marginals)
- Coordination metrics (Kuramoto order parameter \(r(t)\), PLV, metastability index)
- Kuramoto sanity simulations (Euler–Maruyama; global coupling)
- Switching statistics helpers (dwell times and transition rates between low/high-\(r\) regimes)
- A **proof-of-protocol** demo script (synthetic by default)

---

## Quickstart

### 1) Install (recommended)

If your repository uses a `pyproject.toml` with extras (as in `.[dev]`), use:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
