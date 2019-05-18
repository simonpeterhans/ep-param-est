import numpy as np
from scipy import stats as ss

from tbd import Distribution


class NormalDistribution(Distribution):
    def __init__(self, name, x_min, x_max, mu_min, mu_max, sigma_min, sigma_max):
        """
        TBA
        :param name:
        :param x_min:
        :param x_max:
        :param mu_min:
        :param mu_max:
        :param sigma_min:
        :param sigma_max:
        """
        super().__init__(name)
        self.x_min = x_min
        self.x_max = x_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # TODO Add check here once we know what ranges it works best for.

    def gen_data(self, n_param_samples, grid_size_per_sample):
        """
        TBA
        :param n_param_samples:
        :param grid_size_per_sample:
        :return:
        """
        mu = ss.uniform(self.mu_min, self.mu_max - self.mu_min).rvs(n_param_samples)
        sigma = ss.uniform(self.sigma_min, self.sigma_max - self.sigma_min).rvs(n_param_samples)
        x = np.linspace(self.x_min, self.x_max, grid_size_per_sample)
        # TODO Make sure that distributions don't get off the grid (truncated)?

        sampled_params = np.concatenate((mu[..., np.newaxis], sigma[..., np.newaxis]), 1)
        sampled_grid = ss.norm(mu[:, np.newaxis], sigma[:, np.newaxis]).logpdf(x[np.newaxis, :])

        return sampled_params, sampled_grid
