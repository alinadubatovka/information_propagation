import argparse
import csv
import sys

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from cifar10_dataset import load_cifar10
from training import train_configuration

parser = argparse.ArgumentParser(description='Perform train specified configuration (layer size N, depth L, '
                                             'and initialization) on MNIST or CIFAR-10, sigma_w is critical (sqrt(2 / N)).')
parser.add_argument('--n', default=200, metavar='N', type=int, nargs=1, help='size of inner layers', required=True)
parser.add_argument('--l', metavar='L', type=int, nargs=1, help='number of inner layers', required=True)
parser.add_argument('--init', metavar='init mode', type=str, nargs=1, help='initialization strategy: He, GSM, Ortho',
                    required=True)
parser.add_argument('--sample_size', default=100, metavar='number of samples', type=int, nargs=1,
                    help='total number of sample trials')
parser.add_argument('--filename', metavar='file name', type=str, nargs=1, help='name of the result CSV file')
parser.add_argument('--sigma_b', default=0, metavar='sigma_b', type=float, nargs=1, help='variance of bias init')
parser.add_argument('--num_epoch', default=200, metavar='num_epoch', type=int, nargs=1, help='number of epochs')
parser.add_argument('--dataset', default="mnist", metavar='dataset', type=str, nargs=1,
                    help='dataset for training: mnist, cifar10')

args = parser.parse_args()

N = args.n[0]
L = args.l[0]
init_mode = args.init[0]
sample_size = args.sample_size[0]
sigma_b = args.sigma_b
num_epoch = args.num_epoch[0]
dataset_name = args.dataset[0]

if dataset_name == "mnist":
    dataset = mnist_data.read_data_sets("mnist_data", one_hot=True, reshape=True)
elif dataset_name == "cifar10":
    dataset = load_cifar10(reshape=True, crop_size=28)
else:
    raise ValueError("Unknown dataset name: %s" % dataset_name)

if init_mode not in ("He", "GSM", "Ortho"):
    raise ValueError("Unknown initialization: %s" % init_mode)

if args.filename is not None:
    filename = args.filename[0]
else:
    filename = '%s/%s_init/N_%d_L_%d_%s_%d_trials.csv' % (dataset, init_mode, N, L, init_mode, sample_size)

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    # Critical point sigma_w = sqrt(2 / N)
    sigma_w = (2.0 / N) ** 0.5
    for sample_num in range(sample_size):
        test_acc, train_time = train_configuration(N, L, sigma_w, sigma_b, init_mode, dataset, num_epoch, dataset_name)
        writer.writerow([sample_num, N, L, sigma_w, sigma_b, test_acc, train_time])
        print("Layers size: %d, Depth: %d, Sample number: %d, Sigma_W: %f, Max test accuracy: %f, Time: %f" % (
        N, L, sample_num, sigma_w, test_acc, train_time))
        sys.stdout.flush()
