# Mathematical notes (Papers I–III)

This repository intentionally avoids speculative mechanism claims. Mathematical material is organized to mirror the trilogy:

- **Paper I (stability / persistence):**
  - Open‑system language (effective reduced dynamics)
  - Switching statistics; Arrhenius/Kramers‑type fits as *model checks*
  - Effective stability scale \(\kappa\) as an inferred barrier‑like parameter (coarse‑graining dependent)

- **Paper II (coordination / dynamics):**
  - Kuramoto‑type reduced descriptions as a conservative scaffold
  - Counter‑classes (hypersynchrony, clustered synchronization) require auxiliary metrics beyond \(r\)

- **Paper III (protocols):**
  - Measurement‑to‑model contract: phases are *operational* only after admissibility gates
  - Mandatory surrogates and null hypotheses

For the implementational mapping, see `src/noetiko_convergence/`:

- `phase.py` — analytic phase extraction (Hilbert) + optional band‑pass preprocessing
- `quality_gates.py` — amplitude/SNR proxy gates; unwrap consistency checks
- `surrogates.py` — phase randomization and time‑shift surrogates
- `metrics.py` — order parameter \(r\), PLV, metastability index
- `kuramoto.py` — Euler–Maruyama simulation for sanity checks
- `switching.py` — dwell times and transition rates for bistable \(r(t)\) regimes

If you add deeper theory notes, keep them consistent with the SNO contract in the root README.
