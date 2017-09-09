import argparse
import torch
from torchvision import transforms
import sys
sys.path.append('../')
import utility
from influence_function import s_test
from trainer_cifar10 import Cifar10Trainer
import resnet
from optimizers import MomentumSGD

# to evade error, disabled cudnn
# error is like this:
# RuntimeError: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.

# torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                    help='-1 means cpu, otherwise gpu id')
parser.add_argument('--save_path', type=str, default='./s_test', metavar='N',
                    help='s_test will be saved here')
parser.add_argument('--load_model', default=None, metavar='N',
                    help='pretrained model')
parser.add_argument('--train_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for s_test (default: 1)')
parser.add_argument('--t', type=int, default=5000, metavar='N',
                    help='t')
parser.add_argument('--r', type=int, default=1, metavar='N',
                    help='r')
parser.add_argument('--damp', type=float, default=0.01, metavar='N',
                    help='damp')
parser.add_argument('--scale', type=float, default=25.0, metavar='N',
                    help='scaling')
parser.add_argument('--debug', default=True, metavar='N',
                    help='run test_one_epoch before processing s_test')
parser.add_argument('--start', type=float, default=0, metavar='N',
                    help='index starts from this')
parser.add_argument('--k', '-k', type=int, default=1,
                    help='width hyperparameter')
parser.add_argument('--N', '-n', type=int, default=18,
                    help='total layers: 17 * 3 * 3 + 1')
parser.add_argument('--multiplier', '-multiplier', type=int, default=4,
                    help='channel of last block of bottleneck (1x1 conv)')

args = parser.parse_args().__dict__
print(args)
t = args.pop('t')
r = args.pop('r')
damp = args.pop('damp')
scale = args.pop('scale')
debug = args.pop('debug')
start = args.pop('start')
k = args.pop('k')
N = args.pop('N')
multiplier = args.pop('multiplier')

model = resnet.ResidualNetwork(10, out_channels=(16 * k, 32 * k, 64 * k), N=(N, N, N))
optimizer = MomentumSGD(model, 0, 0)
args['model'], args['optimizer'] = model, optimizer

args['train_transform'] = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                                (0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010)),
                                             ])
main = Cifar10Trainer(**args)

if args['gpu'] >= 0:
    main.to_gpu()

if debug is True:
    print(main.test_one_epoch())

for i in utility.create_progressbar(main.test_loader.dataset.test_data.shape[0], desc='z_test', start=start):
    z_test, t_test = main.test_loader.dataset[i]
    z_test = main.test_loader.collate_fn([z_test])
    t_test = main.test_loader.collate_fn([t_test])
    for ii in utility.create_progressbar(r, desc='r'):
        s_test_vec = s_test(z_test, t_test, model, main.train_loader, gpu=args['gpu'], damp=damp, scale=scale, repeat=t)
        s_test_vec = [s.cpu() for s in s_test_vec]
        torch.save(s_test_vec, '{}/{}_{}.s_test'.format(main.save_path, i, ii))
