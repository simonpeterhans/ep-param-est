from abc import abstractmethod

import keras
import pandas as pd
from keras import Input
from keras import backend
from keras.callbacks import Callback
from keras.layers import Dense, Concatenate, Reshape, Conv2D, Lambda
from keras.utils import plot_model


class Model(keras.Model):
    """
    Keras model that also holds the parameter values as attributes and generates the following
    structure upon initialization.

    TODO MODEL STRUCTURE/DESCRIPTION
    """

    class LossHistory(Callback):
        """
        Loss history object to record the development of the loss value during training after every
        epoch.
        """

        def __init__(self):
            super().__init__()
            self.history = None

        def on_train_begin(self, logs={}):
            self.losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))

    def __init__(self, n_samples, desired_grid_size, n_epochs, batch_size, n_dense_layers,
                 dense_scaling, depth, n_kernels, kernel_size, name=None):
        """
        Sets the initial values for the parameters. Calculates out_dense_2 and adjusts
        desired_grid_size to grid_size so the model dimensions are correct.

        :param n_samples: Total number of samples to train this network on.
        :param desired_grid_size: Desired size of the grid; may get adjusted and is available as
        grid_size.
        :param n_epochs: Number of epochs to use when using this parameters on training a model.
        :param batch_size: Batch size.
        :param n_dense_layers: Number of dense layers per first and second level.
        :param dense_scaling: Scaling for the number of neurons in the first dense layer
        (dense_scaling * depth)
        :param depth: Model depth.
        :param n_kernels: Number of kernels to use.
        :param kernel_size: Size of the kernels.
        :param name: Can be set on initialization or afterwards.
        """
        super().__init__()

        self.model_name = name
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

        self.history = self.LossHistory()

        self.__create_model()

        # TODO Give feedback about parameter settings (especially grid_size).

    def loss_to_csv(self, path):
        """
        Creates a pandas data frame of the loss values obtained during the training and stores the
        data frame in a .csv file.

        :param path: The path to the .csv file to create.
        """
        pd.DataFrame(self.history.history['loss'], columns=[self.model_name]).to_csv(path)

    def params_to_csv(self, path):
        """
        Creates a pandas data frame of the set of parameters which are not inherited from
        keras.Model and stores the data frame in a .csv file.

        :param path: The path to the .csv file to create.
        """
        select = {
            'name': self.model_name,
            'n_samples': self.n_samples,
            'grid_size': self.grid_size,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'n_dense_layers': self.n_dense_layers,
            'dense_scaling': self.dense_scaling,
            'depth': self.depth,
            'n_kernels': self.n_kernels,
            'kernel_size': self.kernel_size,
            'out_dense_2': self.out_dense_2
        }

        df = pd.DataFrame.from_records([select], columns=select.keys())
        df.to_csv(path)

    def plot(self, **kwargs):
        """
        Wrapper method for the plot_model method from keras.utils for convenience.

        :param kwargs: The arguments to pass on to plot_model.
        """
        plot_model(self, **kwargs)

    def __create_model(self):
        """
        Creates the model to predict the likelihood function when given parameters of a
        distribution.
        """
        input_layer = Input(shape=(2,), name='input')

        dense_1, dense_2, dense_reshape = [], [], []

        for i in range(self.n_dense_layers):
            dense_1.append(Dense(int(self.dense_scaling * self.depth), activation='relu',
                                 name='level_1_dense_{:02d}'.format(i))(input_layer))

            dense_2.append(Dense(self.out_dense_2 * self.depth, activation='relu',
                                 name='level_2_dense_{:02d}'.format(i))(dense_1[-1]))

            dense_reshape.append(
                Reshape(target_shape=(self.out_dense_2, self.depth, 1))(dense_2[-1]))

        concat = Concatenate(axis=-1)(dense_reshape)
        dense_reshape = Reshape(
            target_shape=(self.grid_size + self.kernel_size - 1, self.depth, 1))(concat)

        conv = Conv2D(filters=self.n_kernels, kernel_size=(self.kernel_size, self.depth),
                      strides=1)(dense_reshape)

        avg = Lambda(lambda x: backend.sum(x, axis=-1), output_shape=lambda d: (d[0], d[1]))(conv)
        avg_reshape = Reshape(target_shape=(self.grid_size,))(avg)

        super().__init__(input_layer, avg_reshape)


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

    def to_csv(self, file):
        """
        Creates a pandas data frame of the set of parameters and their values and stores the data
        frame in a .csv file.

        :param file: The path to the .csv file to create.
        """
        df = pd.DataFrame.from_records([vars(self)], columns=vars(self).keys())
        df.to_csv(file)
