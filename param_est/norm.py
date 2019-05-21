import numpy as np
from scipy import stats as ss

from core import Distribution


class NormalDistribution(Distribution):
    """
    Class to sample from a normal distribution.
    """

    def __init__(self, name, x_min, x_max, mu_min, mu_max, sigma_min, sigma_max):
        """
        Initializes the object by setting the relevant parameters.

        :param name: The identifier of the object.
        :param x_min: Minimum value to be sampled.
        :param x_max: Maximum value to be sampled.
        :param mu_min: Minimum mu value to sample when generating parameters.
        :param mu_max: Maximum mu value to sample when generating parameters.
        :param sigma_min: Minimum sigma value to sample when generating parameters.
        :param sigma_max: Maximum sigma value to sample when generating parameters.
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
        Abstract method to generate data.

        :param n_param_samples: Number of thetas (parameter vectors) to generate.
        :param grid_size_per_sample: Number of samples to generate from the generated parameter
        vectors, i.e., the grid size.
        :return: The sampled parameters and the grid sampled from the sampled parameters.
        """
        mu = ss.uniform(self.mu_min, self.mu_max - self.mu_min).rvs(n_param_samples)
        sigma = ss.uniform(self.sigma_min, self.sigma_max - self.sigma_min).rvs(n_param_samples)
        x = np.linspace(self.x_min, self.x_max, grid_size_per_sample)
        # TODO Make sure that distributions don't get off the grid (truncated)?

        sampled_params = np.concatenate((mu[..., np.newaxis], sigma[..., np.newaxis]), 1)
        sampled_grid = ss.norm(mu[:, np.newaxis], sigma[:, np.newaxis]).logpdf(x[np.newaxis, :])

        return sampled_params, sampled_grid

    def gen_fun(self, n_param_samples, grid_size_per_sample):
        """
        Generator function for this distribution to dynamically generate (yield) data.

        :param n_param_samples: Number of thetas (parameter vectors) to generate.
        :param grid_size_per_sample: Number of samples to generate from the generated parameter
        vectors, i.e., the grid size.
        :return: The sampled parameters and the grid sampled from the sampled parameters.
        """
        # TODO This could potentially be in the parent class.
        while 1:
            yield self.gen_data(n_param_samples, grid_size_per_sample)
