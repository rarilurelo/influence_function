import argparse
import torch.optim as optim
from net import Net
from trainer_mnist import MnistTrainer

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                    help='-1 means cpu, otherwise gpu id')
parser.add_argument('--save_path', type=str, default='./', metavar='N',
                    help='log and model will be saved here')
parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='epochs start from this number')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

args = parser.parse_args().__dict__
print(args)
lr = args.pop('lr')
momentum = args.pop('momentum')

model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
args['model'], args['optimizer'] = model, optimizer
main = MnistTrainer(**args)
main.run()
