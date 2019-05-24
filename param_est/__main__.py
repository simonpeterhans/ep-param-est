import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime

from core import Model
from norm import NormalDistribution


def main(args):
    """
    TBA

    :param args:
    :return:
    """
    now = datetime.now().strftime("%Y%m%d%H%M%S")

    path = os.path.join(os.getcwd(), 'output', args.name + '-' + now)

    if not os.path.exists(path):
        os.makedirs(path)

    model = Model(args.n_samples, args.grid_size, args.epochs, args.batch_size, args.dense_layers,
                  args.dense_factor, args.depth, args.kernels, args.kernel_size, args.name)

    model.compile(optimizer='adam', loss='mean_squared_error')

    if args.plot_model:
        model.plot(show_shapes=True, to_file=os.path.join(path, 'model-plot.png'))

    if args.dist == 'norm':
        dist = NormalDistribution(args.name, *args.dist_args)
    else:  # TODO Handle this.
        dist = NormalDistribution(args.name, *args.dist_args)

    sampled_params, sampled_grid = dist.gen_data(model.n_samples, model.grid_size)

    model.fit(sampled_params, sampled_grid, shuffle=True, batch_size=model.batch_size,
              epochs=model.n_epochs, callbacks=[model.history])
    """model.fit_generator(dist.gen_fun(model.batch_size, model.grid_size),
                        steps_per_epoch=np.ceil(model.n_samples / model.batch_size), shuffle=True,
                        epochs=model.n_epochs, callbacks=[history])"""

    # Export files.
    model.save(os.path.join(path, 'model.h5'))
    dist.to_csv(os.path.join(path, 'dist-params.csv'))
    model.params_to_csv(os.path.join(path, 'model-params.csv'))
    model.loss_to_csv(os.path.join(path, 'loss.csv'))



def parse_args():
    """
    TBA

    :return:
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, epilog='tba')

    parser.add_argument('--dist', metavar='name', default='norm', type=str, required=True,
                        help='the distribution to sample the grid from')
    parser.add_argument('--dist-args', metavar='par', nargs='*', type=float, required=True,
                        help='the parameters for the selected distribution')
    parser.add_argument('--n-samples', metavar='n', default=100000, type=int,
                        help='number of samples to train the model on')
    parser.add_argument('--grid-size', metavar='x', default=100, type=int,
                        help='the size of the grid')
    parser.add_argument('--epochs', metavar='n', default=2, type=int, help='number of epochs')
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
    parser.add_argument('--name', metavar='modelname', type=str,
                        help='if not set, the name will be deduced from the distribution used')
    parser.add_argument('--plot-model', action='store_true', help='saves the model as a .png file')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.dist

    return args


if __name__ == '__main__':
    main(parse_args())
