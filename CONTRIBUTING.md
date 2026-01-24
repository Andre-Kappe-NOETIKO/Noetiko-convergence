# Contributing

This repository is a research‑oriented reproducibility toolkit for the **Noetiko Convergence** trilogy (Papers I–III).

## Development workflow

1. Fork and clone.
2. Create a feature branch.
3. Install dev dependencies:

```bash
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt
```

4. Run tests:

```bash
pytest -q
```

## Research standards (non‑negotiable)

- Changes must preserve the **SNO contract**: operational definitions, admissibility gates, surrogate baselines, explicit nulls.
- Do not add semantic/clinical claims to documentation or code comments.
- Prefer minimal dependencies and transparent implementations over opaque “black boxes”.

## What is welcome

- Additional surrogate generators (IAAFT, block surrogates) with clear documentation
- Additional quality gates (stationarity tests, colored‑noise checks) with conservative defaults
- Reproducible example notebooks/scripts that mirror Paper II/III reporting
