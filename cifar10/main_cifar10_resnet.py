import argparse
import sys
sys.path.append('../')
import resnet
from trainer_cifar10 import Cifar10Trainer
from optimizers import MomentumSGD

parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                    help='-1 means cpu, otherwise gpu id')
parser.add_argument('--save_path', type=str, default='./log', metavar='N',
                    help='log and model will be saved here')
parser.add_argument('--load_model', default=None, metavar='N',
                    help='pretrained model')
parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='epochs start from this number')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--k', '-k', type=int, default=1,
                    help='width hyperparameter')
parser.add_argument('--N', '-n', type=int, default=18,
                    help='total layers: 17 * 3 * 3 + 1')
parser.add_argument('--multiplier', '-multiplier', type=int, default=4,
                    help='channel of last block of bottleneck (1x1 conv)')

args = parser.parse_args().__dict__
print('Args')
print('    {}'.format(args))
lr = args.pop('lr')
momentum = args.pop('momentum')
k = args.pop('k')
N = args.pop('N')
multiplier = args.pop('multiplier')

# define model
model = resnet.ResidualNetwork(10, out_channels=(16 * k, 32 * k, 64 * k), N=(N, N, N))
print('Model')
print('    name: {}'.format(model.name))
print('    parameters: {}'.format(model.count_parameters()))

# define parameters
optimizer = MomentumSGD(model, lr=lr, momentum=momentum, schedule=[100, 150], lr_decay=0.1)
optimizer.info()

args['model'], args['optimizer'] = model, optimizer

main = Cifar10Trainer(**args)
main.run()
