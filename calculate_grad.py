import argparse
import torch
import utility
from influence_function import grad_z
from trainer_mnist import MnistTrainer
from net import Net
from optimizers import MomentumSGD

# to evade error, disabled cudnn
# error is like this:
# RuntimeError: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.

torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                        help='-1 means cpu, otherwise gpu id')
    parser.add_argument('--save_path', type=str, default='./log', metavar='N',
                        help='s_test will be saved here')
    parser.add_argument('--load_model', default=None, metavar='N',
                        help='pretrained model')
    parser.add_argument('--train_batch_size', type=int, default=1, metavar='N',
                        help='input batch size for s_test (default: 1)')
    parser.add_argument('--debug', default=True, metavar='N',
                        help='run test_one_epoch before processing s_test')
    parser.add_argument('--start', type=float, default=0, metavar='N',
                        help='index starts from this')

args = parser.parse_args().__dict__
print(args)
debug = args.pop('debug')
start = args.pop('start')

model = Net()
optimizer = MomentumSGD(model, 0, 0)
args['model'], args['optimizer'] = model, optimizer
main = MnistTrainer(**args)

if args['gpu'] >= 0:
    main.to_gpu()

if debug is True:
    print(main.test_one_epoch())

for i in utility.create_progressbar(main.train_loader.dataset.train_data.shape[0], desc='z', start=start):
    z, t = main.train_loader.dataset[i]
    z = main.train_loader.collate_fn([z])
    t = main.train_loader.collate_fn([t])
    grad_z_vec = grad_z(z, t, model, gpu=-1)
    grad_z_vec = [g.cpu() for g in grad_z_vec]
    torch.save(grad_z_vec, '{}/{}.grad_z'.format(main.save_path, i))
