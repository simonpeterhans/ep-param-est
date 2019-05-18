import datetime
import logging
import os
from itertools import product

import numpy as np
import pandas as pd
from keras.utils import plot_model

from norm import NormalDistribution
from tbd import ModelParameters, create_model, LossHistory

# TODO Implement proper logging.


test_name = 'norm'
path = os.path.join('output', test_name)  # TODO Where/How to store files?
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S-")

os.chdir("..")
if not os.path.exists(path):
    os.makedirs(path)

logging.basicConfig(filename=test_name + '.log', level=logging.INFO)

# Parameter settings and number of samples per sampled parameter vector.
n_param_vectors = [100000]
grid_sizes = np.linspace(50, 250, 3, dtype=int)
combinations = product(n_param_vectors, grid_sizes)

# NN Settings.
n_epochs = 2
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

for theta, desired_grid_size in combinations:
    name = str(theta) + '-' + str(desired_grid_size)

    p = ModelParameters(name, n_epochs, desired_grid_size, n_dense_layers, dense_scaling, depth,
                        n_kernels, kernel_size)

    # Add params of model to list.
    param_list.append(p.get_params())

    model = create_model(p)

    # TODO Do this in tbd.py and use kwargs.
    model.compile(optimizer='adam', loss='mean_squared_error')
    plot_model(model, show_shapes=True, to_file=os.path.join(path, name + '-plot.png'))

    nd = NormalDistribution(name, x_min, x_max, mu_min, mu_max, sigma_min, sigma_max)
    dist_list.append(nd.get_params())

    sampled_params, sampled_grid = nd.gen_data(theta, p.grid_size)

    history = LossHistory()  # TODO Do this in tbd.py and use kwargs.
    model.fit(sampled_params, sampled_grid, shuffle=True, batch_size=64, epochs=n_epochs,
              callbacks=[history])
    model.save(os.path.join(path, name + '-model.h5'))

    # TODO Do this in tbd.py and use kwargs.
    loss_list.append(pd.DataFrame(history.losses, columns=[name]))

pd.concat(param_list, ignore_index=True).to_csv(path + now + 'params.csv')
pd.concat(dist_list, ignore_index=True).to_csv(path + now + 'dists.csv')
pd.concat(loss_list, axis=1).to_csv(path + now + 'losses.csv')
