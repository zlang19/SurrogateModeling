import numpy as np

from surrogatemodeling.problems.base import NoiseModel


def test_empirical_sigma_matches_declared_and_scales_with_fidelity():
    noise = NoiseModel(rel=0.02, abs=0.1, sigma_dof=20)
    rng = np.random.default_rng(0)
    y = np.full((50000, 1), 10.0)
    for fid in (1.0, 0.25):
        f = np.full(len(y), fid)
        obs = noise.observe(y, f, rng)
        expected = np.sqrt((0.02 * 10) ** 2 + 0.1**2) / np.sqrt(fid)
        assert abs(np.std(obs.y - y) / expected - 1) < 0.02
        # Reported sigma is an unbiased variance estimate, not the exact value.
        assert abs(np.mean(obs.sigma**2) / expected**2 - 1) < 0.02
        assert np.std(obs.sigma) > 0
