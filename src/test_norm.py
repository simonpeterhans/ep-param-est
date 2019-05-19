import datetime
import logging
import os
from itertools import product

import numpy as np
import pandas as pd
from keras.utils import plot_model

from norm import NormalDistribution
from tbd import ModelParameters, create_model, LossHistory

test_name = 'norm'
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
path = os.path.join('output', test_name + '-' + now)  # TODO Where/How to store files?

os.chdir("..")
if not os.path.exists(path):
    os.makedirs(path)

# TODO Implement proper logging.
logging.basicConfig(filename=test_name + '.log', level=logging.INFO)

# Parameter settings and number of samples per sampled parameter vector.
n_param_vectors = [150000]
grid_sizes = np.linspace(50, 150, 3, dtype=int)
combinations = product(n_param_vectors, grid_sizes)

# NN Settings.
n_epochs = 5
batch_size = 64
n_dense_layers = 20
dense_scaling = 1
depth = 25
n_kernels = 15
kernel_size = 40

# ND Settings.
# As these are fixed in this example, we could instead simply create one NormalDistribution object
# instead; however, once we run tests on different sets of parameters for this distribution
# we'll have to create a new one anyway.
x_min = -10
x_max = 10
mu_min = -3
mu_max = 3
sigma_min = 0.5
sigma_max = 3

param_list = []
dist_list = []
loss_list = []

for thetas, desired_grid_size in combinations:
    # TODO Consider wrapping the whole keras model and its method in a class.
    # Create model parameters and add to list.
    p = ModelParameters(n_epochs, batch_size, desired_grid_size, n_dense_layers, dense_scaling,
                        depth, n_kernels, kernel_size)
    p.name = str(thetas) + '-' + str(p.grid_size)
    # TODO Add the number of training samples into this.
    param_list.append(p.get_params())

    model = create_model(p)

    model.compile(optimizer='adam', loss='mean_squared_error')
    plot_model(model, show_shapes=True, to_file=os.path.join(path, p.name + '-plot.png'))

    # Create distribution and add to list.
    nd = NormalDistribution(p.name, x_min, x_max, mu_min, mu_max, sigma_min, sigma_max)
    dist_list.append(nd.get_params())
    sampled_params, sampled_grid = nd.gen_data(thetas, p.grid_size)

    history = LossHistory()

    model.fit(sampled_params, sampled_grid, shuffle=True, batch_size=p.batch_size, epochs=n_epochs,
              callbacks=[history])
    """model.fit_generator(nd.gen_fun(p.batch_size, p.grid_size),
                        steps_per_epoch=np.ceil(thetas / p.batch_size), shuffle=True,
                        epochs=n_epochs, callbacks=[history]"""
    model.save(os.path.join(path, p.name + '-model.h5'))

    loss_list.append(pd.DataFrame(history.losses, columns=[p.name]))

    # TODO Append number of sampled thetas to the parameter table.
    pd.concat(param_list, ignore_index=True).to_csv(path + '-params.csv')
    pd.concat(dist_list, ignore_index=True).to_csv(path + '-dists.csv')
    pd.concat(loss_list, axis=1).to_csv(path + '-losses.csv')
