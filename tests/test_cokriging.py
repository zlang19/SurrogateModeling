import numpy as np
import torch

from surrogatemodeling.methods.acquisition import greedy_batch
from surrogatemodeling.methods.cokriging import ConfigBiasKernel
from surrogatemodeling.methods.gp_common import IndependentGPs


def test_bias_kernel_is_zero_at_hf_and_across_configs():
    k = ConfigBiasKernel(n_configs=3, hf=2, x_dims=2)
    with torch.no_grad():
        k.raw_scales[:] = 0.0
    x = torch.tensor([[0.1, 0.2, 0.0], [0.1, 0.2, 1.0], [0.1, 0.2, 2.0], [0.15, 0.2, 0.0]], dtype=torch.float64)
    K = k.forward(x, x).detach().numpy()
    assert K[2, 2] == 0  # high fidelity has no bias
    assert K[0, 1] == 0 and K[0, 2] == 0  # different configs: independent biases
    assert K[0, 3] > 0  # same config, nearby points: correlated bias
    np.testing.assert_allclose(np.diag(K), k.forward(x, x, diag=True).detach().numpy())


def test_greedy_batch_accepts_per_row_costs():
    rng = np.random.default_rng(0)
    A = rng.random((30, 30))
    K = A @ A.T / 30
    tau2 = np.full((1, 1, 20), 0.01)
    costs = np.where(np.arange(20) < 10, 0.1, 1.0)[None, :]  # first 10 rows cheap
    chosen = greedy_batch([K], 10, tau2, costs, k=5, budget=0.35, cost_aware=True)
    assert sum(costs[0, c] for c, _ in chosen) <= 0.35 + 1e-9
    assert all(c < 10 for c, _ in chosen)


def test_loo_calibration_is_near_one_for_a_well_specified_gp():
    rng = np.random.default_rng(1)
    X = rng.random((80, 2))
    y = np.sin(4 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.standard_normal(80)
    gp = IndependentGPs(calibrate=True)
    gp.fit(X, y[:, None], np.full((80, 1), 0.05**2))
    assert 0.7 < gp.cal[0] < 1.4
