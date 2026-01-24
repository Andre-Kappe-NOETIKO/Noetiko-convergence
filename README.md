# Noetiko Convergence — Reproducibility Toolkit (Papers I–III)

![CI](https://github.com/Andre-Kappe-NOETIKO/Noetiko-convergence/actions/workflows/python-ci.yml/badge.svg)

This repository provides a conservative, operational, and falsifiable implementation scaffold for the **Noetiko Convergence** trilogy:

- **Paper I**: metastable information persistence under open‑system constraints (effective stability scale \(\kappa\), switching statistics, barrier‑like inference)
- **Paper II**: coordination dynamics (Kuramoto‑type reduced models) with explicit guardrails, surrogate controls, and counter‑class discrimination
- **Paper III**: measurement‑to‑model protocols (phase extraction admissibility gates, reporting checklist, proof‑of‑protocol demo template)

The code is intended to support **reproducible analysis** and **methodological stress‑tests**. It is explicitly not a mechanism‑identification claim.

## Scope & non‑claims (SNO)

- No semantic conclusions (e.g., “health”, “repair”, “consciousness”) are inferred from synchronization statistics.
- Reduced parameters (e.g., \(K\), \(D\), \(\kappa\)) are **effective descriptors**, not microscopic coupling channels.
- Claims are restricted to **measurement‑to‑model contracts** with documented phase maps, admissibility gates, surrogate baselines, null hypotheses, and reporting minima.

## What is included

- Phase extraction (Hilbert analytic phase) with optional band‑pass preprocessing
- Quality gates (amplitude/SNR proxy gates; unwrap consistency checks)
- Surrogate controls
  - Fourier phase randomization (preserves marginal spectra)
  - Cross‑channel time‑shift surrogates (destroys coordination while preserving marginals)
- Coordination metrics (Kuramoto order parameter \(r(t)\), PLV, metastability index)
- Kuramoto sanity simulations (Euler–Maruyama; global coupling)
- Switching statistics helpers (dwell times and transition rates between low/high‑\(r\) regimes)
- A **proof‑of‑protocol** demo script (synthetic by default)

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional (Paper I quantum tooling):

```bash
pip install -r requirements-quantum.txt
```

### 2) Run the proof‑of‑protocol demo

```bash
python -m noetiko_convergence.demo --out results/demo
```

This produces:

- `results/demo/demo_r_surrogates.png`
- `results/demo/gate_summary.txt`

## Repository layout

- `src/noetiko_convergence/` — core library (phase, gates, surrogates, metrics, simulations)
- `tests/` — unit tests (operational correctness)
- `docs/` — project notes (ethics, limitations, reporting contract)
- `results/` — outputs (git‑ignored by default)

## Citation

If you use this repository, please cite the trilogy preprints and the software DOI (if minted):

- Andre Kappe, *Noetiko Convergence* (Papers I–III), preprints, 2026.
- Software DOI: `10.5281/zenodo.<placeholder>`

ORCID: https://orcid.org/0009-0001-2799-379X

## License

- Code: Apache-2.0 (see `LICENSE`)
- Text/paper artifacts: please add an explicit license header in `/papers` if you include them here.
