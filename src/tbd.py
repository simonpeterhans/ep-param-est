# TODO Find a name for our module!
import random
import string
from abc import abstractmethod

import pandas as pd
from keras import Model, Input
from keras import backend
from keras.callbacks import Callback
from keras.layers import Dense, Concatenate, Reshape, Conv2D, Lambda


class ModelParameters(object):
    """
    TBA
    """

    def __init__(self, n_epochs, batch_size, desired_grid_size, n_dense_layers, dense_scaling,
                 depth, n_kernels, kernel_size):
        """
        TBA
        :param n_epochs:
        :param batch_size:
        :param desired_grid_size:
        :param n_dense_layers:
        :param dense_scaling:
        :param depth:
        :param n_kernels:
        :param kernel_size:
        """
        # TODO Consider using default values in function signature.
        self.name = self.__id_generator(6)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_dense_layers = n_dense_layers
        self.dense_scaling = dense_scaling
        self.depth = depth
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size

        self.out_dense_2 = (desired_grid_size + kernel_size - 1) // n_dense_layers
        self.grid_size = self.out_dense_2 * n_dense_layers - kernel_size + 1

        # TODO Give feedback about parameter settings (especially grid_size).

    @staticmethod
    def __id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        """
        TBA
        :param size:
        :param chars:
        :return:
        """
        return ''.join(random.choice(chars) for _ in range(size))

    def get_params(self):
        """
        TBA
        :return:
        """
        return pd.DataFrame.from_records([vars(self)], columns=vars(self).keys())


class Distribution(object):
    """
    TBA
    """

    def __init__(self, name):
        """
        TBA
        :param name:
        """
        self.name = name

    @abstractmethod
    def gen_data(self, n_vectors, n_samples):
        """
        TBA
        :param n_vectors:
        :param n_samples:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def gen_fun(self, n_vectors, n_samples):
        """
        TBA
        :param n_vectors:
        :param n_samples:
        :return:
        """
        raise NotImplementedError

    def get_params(self):
        """
        TBA
        :return:
        """
        return pd.DataFrame.from_records([vars(self)], columns=vars(self).keys())


class LossHistory(Callback):
    """
    TBA
    """

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


def create_model(p):
    """
    TBA
    :type p: ModelParameters
    :return:
    """
    model = Input(shape=(2,), name='input')

    dense_1, dense_2, dense_reshape = [], [], []

    for i in range(p.n_dense_layers):
        dense_1.append(Dense(p.dense_scaling * p.depth, activation='relu',
                             name='level_1_dense_{:02d}'.format(i))(model))

        dense_2.append(Dense(p.out_dense_2 * p.depth, activation='relu',
                             name='level_2_dense_{:02d}'.format(i))(dense_1[-1]))

        dense_reshape.append(Reshape(target_shape=(p.out_dense_2, p.depth, 1))(dense_2[-1]))

    concat = Concatenate(axis=-1)(dense_reshape)
    dense_reshape = Reshape(target_shape=(p.grid_size + p.kernel_size - 1, p.depth, 1))(concat)

    conv = Conv2D(filters=p.n_kernels, kernel_size=(p.kernel_size, p.depth), strides=1)(
        dense_reshape)

    avg = Lambda(lambda x: backend.sum(x, axis=-1), output_shape=lambda d: (d[0], d[1]))(conv)
    avg_reshape = Reshape(target_shape=(p.grid_size,))(avg)

    model = Model(model, avg_reshape)

    return model
