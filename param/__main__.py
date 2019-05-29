import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime

import numpy as np

from param.core import Model
from param.dists.norm import NormalDistribution


def main(args):
    """
    Example of how the module can be used.

    :param args: The arguments passed upon calling the module.
    """
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    path = os.path.join(args.path, args.name + '-' + now)

    if not os.path.exists(path):
        os.makedirs(path)

    model = Model(args.n_samples, args.grid_size, args.epochs, args.batch_size, args.dense_layers,
                  args.dense_factor, args.depth, args.kernels, args.kernel_size, args.name)
    model.params_to_csv(os.path.join(path, 'model-params.csv'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    if args.plot_model or args.preview:
        model.plot(show_shapes=True, to_file=os.path.join(path, 'model-plot.png'))

    if args.preview:
        exit(0)

    if args.dist == 'norm':
        dist = NormalDistribution(args.name, *args.dist_args)
    else:
        dist = None  # Silences warning.
        print("Distribution " + args.dist + " doesn't exist - terminating.")
        exit(1)
    dist.to_csv(os.path.join(path, 'dist-params.csv'))

    sampled_params, sampled_grid = dist.gen_data(model.n_samples, model.grid_size)

    if not args.gen_mode:
        model.fit(sampled_params, sampled_grid, shuffle=True, batch_size=model.batch_size,
                  epochs=model.n_epochs, callbacks=[model.history])
    else:
        model.fit_generator(dist.gen_fun(model.batch_size, model.grid_size),
                            steps_per_epoch=np.ceil(model.n_samples / model.batch_size),
                            shuffle=True, epochs=model.n_epochs, callbacks=[model.history])

    model.save(os.path.join(path, 'model.h5'))
    model.loss_to_csv(os.path.join(path, 'loss.csv'))


def parse_args():  # TODO Add epilog with detailed description.
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, epilog='')

    parser.add_argument('--preview', action='store_true',
                        help='compiles and plots model, stores parameters, but no actual training')
    parser.add_argument('--dist', metavar='d', default='norm', type=str, required=True,
                        help='the distribution to sample the grid from')
    parser.add_argument('--dist-args', metavar='par', nargs='*', type=float, required=True,
                        help='the parameters for the selected distribution')
    parser.add_argument('--gen-mode', action='store_true',
                        help='generates data on the fly; --n-samples samples generated per epoch')
    parser.add_argument('--n-samples', metavar='n', default=500000, type=int,
                        help='number of samples to train the model on')
    parser.add_argument('--grid-size', metavar='x', default=250, type=int,
                        help='the size of the grid')
    parser.add_argument('--epochs', metavar='n', default=25, type=int, help='number of epochs')
    parser.add_argument('--batch-size', metavar='x', type=int, default=64, help='number of batches')
    parser.add_argument('--dense-layers', metavar='n', type=int, default=25,
                        help='number of dense layers per level')
    parser.add_argument('--dense-factor', metavar='f', type=float, default=1.0,
                        help='scaling of the first dense layer level')
    parser.add_argument('--depth', metavar='x', default=25, type=int, help='depth of the model')
    parser.add_argument('--kernels', metavar='n', default=25, type=int,
                        help='number of kernels to use')
    parser.add_argument('--kernel-size', metavar='x', default=25, type=int,
                        help='size of the kernels')
    parser.add_argument('--name', metavar='n', type=str,
                        help='if not set, the name will be deduced from the distribution used')
    parser.add_argument('--path', metavar='p', default='~/output', type=str,
                        help='path to store the output in; ~/output/ used if not set')
    parser.add_argument('--plot-model', action='store_true', help='saves the model as a .png file')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.dist
    args.path = os.path.expanduser(args.path)

    return args


if __name__ == '__main__':
    main(parse_args())
