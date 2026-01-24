# Security policy

This repository is not a network service by default. Security concerns typically relate to:

- accidental commits of secrets (API keys, tokens)
- supply‑chain issues in dependencies

## Reporting

If you discover a vulnerability or a secret leak, please report it privately to:

- andre.kappe@noetiko.tech

## Recommendation for maintainers

- Use GitHub secret scanning on public release.
- Keep `results/` and raw datasets out of git (see `.gitignore`).
