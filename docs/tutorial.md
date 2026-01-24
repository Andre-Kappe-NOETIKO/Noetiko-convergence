# Tutorial (minimal)

## 1) Compute phases and r(t)

```python
import numpy as np
from noetiko_convergence.phase import hilbert_phase
from noetiko_convergence.metrics import order_parameter

res = hilbert_phase(x, fs_hz=fs, band_hz=(6,10))
op = order_parameter(res.phase, axis=0)
r_t = op.r
```

## 2) Run surrogates

```python
from noetiko_convergence.surrogates import phase_randomization_surrogate, time_shift_surrogate

x_pr = phase_randomization_surrogate(x)
x_ts = time_shift_surrogate(x, min_shift=int(fs*2))
```

## 3) Switching statistics (if bimodal)

```python
from noetiko_convergence.switching import switching_statistics
stats = switching_statistics(r_t, fs=fs, r_L=0.3, r_H=0.6)
```

## 4) Proof‑of‑protocol demo

```bash
python -m noetiko_convergence.demo --out results/demo
```
