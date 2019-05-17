# TODO Find a name for our module!

from keras import Model, Input
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Dense, Concatenate, Reshape, Conv2D, Lambda


class Distribution(object):
    def gen_data(self, n_vectors, n_samples):
        """
        TBA
        :param n_vectors:
        :param n_samples:
        :return:
        """
        raise NotImplementedError('Distribution subclasses must override gen_data().')


class LossHistory(Callback):
    """
    TBA
    """

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


def create_model(desired_grid_size, n_dense_layers, dense_scaling, depth, n_kernels, kernel_size):
    """
    TBA
    :param desired_grid_size:
    :param n_dense_layers:
    :param dense_scaling:
    :param depth:
    :param n_kernels:
    :param kernel_size:
    :return:
    """
    out_dense_2 = (desired_grid_size + kernel_size - 1) // n_dense_layers
    grid_size = out_dense_2 * n_dense_layers - kernel_size + 1

    # TODO Give feedback about parameter settings (especially grid_size).

    model = Input(shape=(2,), name='input')
    print("Shape after input:", model.shape)  # TODO Replace these by a logger.

    dense_1 = []
    dense_2 = []
    dense_reshape = []

    for i in range(n_dense_layers):
        dense_1.append(Dense(dense_scaling * depth, activation='relu', name='level_1_dense_{:02d}'.format(i))(model))
        dense_2.append(
            Dense(out_dense_2 * depth, activation='relu', name='level_2_dense_{:02d}'.format(i))(dense_1[-1]))
        dense_reshape.append(Reshape(target_shape=(out_dense_2, depth, 1))(dense_2[-1]))

    concat = Concatenate(axis=-1)(dense_reshape)
    print("Shape after concat:", concat.shape)

    dense_reshape = Reshape(target_shape=(grid_size + kernel_size - 1, depth, 1))(concat)
    print("Shape after reshape:", dense_reshape.shape)

    conv = Conv2D(filters=n_kernels, kernel_size=(kernel_size, depth), strides=1)(dense_reshape)

    avg = Lambda(lambda x: K.sum(x, axis=-1), output_shape=lambda d: (d[0], d[1]))(conv)
    print("Shape after avg:", avg.shape)

    avg_reshape = Reshape(target_shape=(grid_size,))(avg)
    print("Shape after reshape:", dense_reshape.shape)

    model = Model(model, avg_reshape)

    return model, grid_size
