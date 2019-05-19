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
    Class to store the model parameters in.
    """

    def __init__(self, n_samples, n_epochs, batch_size, desired_grid_size, n_dense_layers,
                 dense_scaling, depth, n_kernels, kernel_size, name=None):
        """
        Sets the initial values for the parameters. Calculates out_dense_2 and adjusts
        desired_grid_size to grid_size so the model dimensions are correct.

        :param n_samples: Total number of samples to train this network on.
        :param n_epochs: Number of epochs to use when using this parameters on training a model.
        :param batch_size: Batch size.
        :param desired_grid_size: Desired size of the grid; may get adjusted and is available as
        grid_size.
        :param n_dense_layers: Number of dense layers per first and second level.
        :param dense_scaling: Scaling for the number of neurons in the first dense layer
        (dense_scaling * depth)
        :param depth: Model depth.
        :param n_kernels: Number of kernels to use.
        :param kernel_size: Size of the kernels.
        :param name: Can be set on initialization or afterwards. If not set, a random string of
        length 6 is generated and used as name instead.
        """
        # TODO Consider using default values in function signature.
        if name is None:
            self.name = self.__id_generator(6)

        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
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
        Used to generate a random ID for the model so we can identify it (usually it's smarter to
        manually set a name).

        :param size: Length of the id.
        :param chars: Selection of the characters to use, defaults to upper-case letters and digits.
        :return: The generated string.
        """
        return ''.join(random.choice(chars) for _ in range(size))

    def to_df(self):
        """
        Creates a pandas data frame of the set of parameters and their values.

        :return: The created data frame.
        """
        return pd.DataFrame.from_records([vars(self)], columns=vars(self).keys())


class Distribution(object):
    """
    Abstract class for distribution objects.
    """

    def __init__(self, name):
        """
        Creates a distribution object.

        :param name: The identifier of the object.
        """
        self.name = name

    @abstractmethod
    def gen_data(self, n_vectors, n_samples):
        """
        Abstract method to generate data.

        :param n_vectors: Number of thetas (parameter vectors) to generate.
        :param n_samples: Number of samples to generate from the generated parameter vectors.
        :return: The generated data (parameter vectors and their sampled data).
        """
        raise NotImplementedError

    @abstractmethod
    def gen_fun(self, n_vectors, n_samples):
        """
        Generator function for this distribution to dynamically generate data.

        :param n_vectors: Number of thetas (parameter vectors) to generate.
        :param n_samples: Number of samples to generate from the generated parameter vectors.
        :return: Yields the generated data (parameter vectors and their sampled data).
        """
        raise NotImplementedError

    def to_df(self):
        """
        Creates a pandas data frame of the set of parameters and their values.

        :return: The created data frame.
        """
        return pd.DataFrame.from_records([vars(self)], columns=vars(self).keys())


class LossHistory(Callback):
    """
    Loss history object to record the development of the loss value during training;
    as of now, the loss is recorded after every epoch.
    """

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


def create_model(p):
    """
    Creates the model to predict the likelihood function when given parameters of a distribution.

    :type p: ModelParameters
    :param p: A parameter object to specify the parameters of this model.
    :return: The created (uncompiled) model.
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
