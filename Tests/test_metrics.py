import numpy as np
from noetiko_convergence.metrics import order_parameter


def test_order_parameter_basic():
    # identical phases -> r=1
    theta = np.zeros((10, 100))
    res = order_parameter(theta, axis=0)
    assert np.allclose(res.r, 1.0)

    # uniform random phases -> r small on average
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=(200, 500))
    res = order_parameter(theta, axis=0)
    assert float(np.mean(res.r)) < 0.2
