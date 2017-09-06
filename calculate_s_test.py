import argparse
import torch
import utility
from influence_function import s_test
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
                        help='log and model will be saved here')
    parser.add_argument('--load_model', default=None, metavar='N',
                        help='pretrained model')
    parser.add_argument('--train_batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--t', type=int, default=5000, metavar='N',
                        help='t')
    parser.add_argument('--r', type=int, default=10, metavar='N',
                        help='r')
    parser.add_argument('--damp', type=float, default=0.01, metavar='N',
                        help='damp')
    parser.add_argument('--scale', type=float, default=25.0, metavar='N',
                        help='scaling')
    parser.add_argument('--debug', default=True, metavar='N',
                        help='run test_one_epoch')

args = parser.parse_args().__dict__
print(args)
t = args.pop('t')
r = args.pop('r')
damp = args.pop('damp')
scale = args.pop('scale')
debug = args.pop('debug')

model = Net()
optimizer = MomentumSGD(model, 0, 0)
args['model'], args['optimizer'] = model, optimizer
main = MnistTrainer(**args)

if args['gpu'] >= 0:
    main.to_gpu()

if debug is True:
    print(main.test_one_epoch())

for i in utility.create_progressbar(main.test_loader.dataset.test_data.shape[0], desc='z_test'):
    z_test, t_test = main.test_loader.dataset[i]
    z_test = main.test_loader.collate_fn([z_test])
    t_test = main.test_loader.collate_fn([t_test])
    for ii in utility.create_progressbar(r, desc='r'):
        s_test_vec = s_test(z_test, t_test, model, main.train_loader, gpu=args['gpu'], damp=damp, scale=scale, repeat=t)
        s_test_vec = [s.cpu() for s in s_test_vec]
        torch.save(s_test_vec, '{}/{}_{}.s_test'.format(main.save_path, i, ii))
