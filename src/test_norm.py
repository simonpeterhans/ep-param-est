import os
from itertools import product

import numpy as np
import pandas as pd
from keras.utils import plot_model

from norm import NormalDistribution
from tbd import create_model, LossHistory

path = 'output/'

# Number of parameter vectors sampled and number of samples from the distributions from the sampled parameters.
n_param_vectors = [100000, 500000, 1000000]
grid_sizes = np.linspace(50, 250, 5, dtype=int)
params = product(n_param_vectors, grid_sizes)

# NN Settings.
n_epochs = 1
n_dense_layers = 20
dense_scaling = 1
depth = 25
n_kernels = 15
kernel_size = 40

# ND Settings.
x_min = -10
x_max = 10
mu_min = -3
mu_max = 3
sigma_min = 0.5
sigma_max = 3

os.chdir("..")
if not os.path.exists(path):
    os.makedirs(path)

for theta, desired_grid_size in params:
    model, grid_size = create_model(desired_grid_size, n_dense_layers, dense_scaling, depth, n_kernels, kernel_size)

    name = str(theta) + '-' + str(grid_size)

    model.compile(optimizer='adam', loss='mean_squared_error')
    plot_model(model, show_shapes=True, to_file=path + name + '-p.png')

    nd = NormalDistribution(x_min, x_max, mu_min, mu_max, sigma_min, sigma_max)
    sampled_params, sampled_grid = nd.gen_data(theta, grid_size)

    history = LossHistory()
    model.fit(sampled_params, sampled_grid, shuffle=True, batch_size=64, epochs=n_epochs, callbacks=[history])
    model.save(path  + name + '-m.h5')

    pd.DataFrame(history.losses).to_csv(path + name + '-l.csv', header=False)
