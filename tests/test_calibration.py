import numpy as np
import torch

from surrogatemodeling.core.metrics import ncrps
from surrogatemodeling.core.protocols import Prediction
from surrogatemodeling.methods.cokriging import ConfigScaledNoise
from surrogatemodeling.methods.gp_common import IndependentGPs


def test_ncrps_is_proper_and_reduces_to_mae():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((20000, 1)) * 2.0
    mean = Y + rng.standard_normal(Y.shape) * 0.5  # errors ~ N(0, 0.5^2)
    honest = ncrps(Prediction(mean, np.full(Y.shape, 0.25)), Y)
    too_narrow = ncrps(Prediction(mean, np.full(Y.shape, 0.25 / 16)), Y)
    too_wide = ncrps(Prediction(mean, np.full(Y.shape, 0.25 * 16)), Y)
    assert honest < too_narrow and honest < too_wide  # proper: honest error bars score best
    mae = np.mean(np.abs(Y - mean)) / np.std(Y)
    assert np.isclose(ncrps(Prediction(mean, None), Y)[0], mae)


def test_pooled_noise_scale_is_shared_across_configs():
    n = ConfigScaledNoise(torch.ones(5, dtype=torch.float64), n_configs=4, pooled=True)
    assert n.raw_scale.numel() == 1 and n.scale.shape == (4,)
    assert torch.allclose(n.scale, n.scale[0].expand(4))


def test_log_outputs_round_trip_positive_data():
    rng = np.random.default_rng(1)
    X = rng.random((60, 2))
    y = np.exp(1 + X[:, :1] + 0.5 * X[:, 1:]) * (1 + 0.01 * rng.standard_normal((60, 1)))
    gp = IndependentGPs(log_outputs=True)
    gp.fit(X, y, (0.01 * y) ** 2)
    assert gp._log[0]
    mean, var = gp.predict(X)
    assert np.all(var > 0)
    assert np.sqrt(np.mean((mean - y) ** 2)) / y.std() < 0.05
