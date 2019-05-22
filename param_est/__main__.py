from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, epilog='tba')

parser.add_argument('--n-samples', metavar='n', default=1000000,
                    help='number of samples to train the model on')
parser.add_argument('--grid-size', metavar='x', default=500, help='the size of the grid')
parser.add_argument('--epochs', metavar='n', default=50, help='number of epochs')
parser.add_argument('--batch-size', metavar='x', default=64, help='number of batches')
parser.add_argument('--dense-layers', metavar='n', default=25,
                    help='number of dense layers per level')
parser.add_argument('--dense-factor', metavar='f', default=1.0,
                    help='scaling of the first dense layer level')
parser.add_argument('--depth', metavar='x', default=25, help='depth of the model')
parser.add_argument('--kernels', metavar='n', default=25, help='number of kernels to use')
parser.add_argument('--kernel-size', metavar='x', default=25, help='size of the kernels')
parser.add_argument('--dist', metavar='name', default='norm', type=str,
                    help='the distribution to sample the grid from')
parser.add_argument('--dist-args', metavar='par', nargs='*',
                    help='the parameters for the selected distribution')
parser.add_argument('--store-params', action='store_true', help='saves the parameters used')
parser.add_argument('--store-model', action='store_true', help='saves the model')
parser.add_argument('--plot-model', action='store_true', help='saves the model as a .png file')

args = parser.parse_args()

print(args)
