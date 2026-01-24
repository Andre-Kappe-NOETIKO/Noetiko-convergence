import numpy as np
from noetiko_convergence.kuramoto import simulate_kuramoto_em
from noetiko_convergence.metrics import order_parameter


def test_kuramoto_sync_increases_with_K():
    rng = np.random.default_rng(1)
    n = 50
    omega = rng.normal(0.0, 0.5, size=n)
    dt = 0.01
    steps = 5000

    low = simulate_kuramoto_em(omega, K=0.2, D=0.02, dt=dt, n_steps=steps, rng=rng)
    high = simulate_kuramoto_em(omega, K=3.0, D=0.02, dt=dt, n_steps=steps, rng=rng)

    r_low = order_parameter(low.theta, axis=0).r
    r_high = order_parameter(high.theta, axis=0).r

    assert float(np.mean(r_high[-1000:])) > float(np.mean(r_low[-1000:]))
