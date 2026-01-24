# Ethics, risks, and responsible use

This repository supports operational analysis of **coordination statistics** and **switching behavior** in reduced phase descriptions.
It is designed to prevent common misuse patterns by making falsifiers, surrogates, and explicit non‑claims first‑class.

## Non‑claims (must be stated in any downstream use)

- High coordination (e.g., high order parameter \(r\)) does **not** imply “health”, “repair”, “consciousness”, or clinical benefit.
- Reduced parameters (\(K\), \(D\), \(\kappa\)) are **effective** and context‑dependent; they are not microscopic coupling channels.
- Any functional interpretation requires **independent external metrics** and domain‑specific validation.

## Typical failure modes this repo is designed to catch

- Filter leakage / shared reference artifacts misread as “synchronization”
- Phase estimation in low‑SNR windows (invalid phases)
- Nonstationarity confounded with regime shifts
- Over‑interpretation of a single metric (\(r\)) without counter‑class discriminators

## Minimum safeguards

When reporting coordination results, include:

1. A fixed, documented phase map \(\Phi\) and admissibility gates.
2. At least two surrogate baselines (phase randomization and time shift).
3. Explicit null outcomes (failed gates or retained null hypotheses).
4. A clear statement of what is **not** claimed.

## Clinical / bio‑intervention disclaimer

This software is not a medical device and is not intended for diagnosis, treatment, or clinical decision making.
