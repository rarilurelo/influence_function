import six
import argparse
import torch
import numpy as np
import utility
from trainer_mnist import MnistTrainer
import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt
plt.switch_backend('agg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize influence function')
    parser.add_argument('--z', type=str, default='./grad_z', metavar='N',
                        help='directory that contains gradient of z')
    parser.add_argument('--s_test', type=str, default='./s_test', metavar='N',
                        help='root directory')
    parser.add_argument('--s_tests_id', type=str, default='0', metavar='N',
                        help='index of test data')
    parser.add_argument('--r', type=int, default=10, metavar='N',
                        help='take average on r')
    parser.add_argument('--n', type=int, default=50000, metavar='N',
                        help='n')
    parser.add_argument('--save_figure', type=str, default='./mnist.jpg', metavar='N',
                        help='save path')

args = parser.parse_args().__dict__
print(args)

z = utility.remove_slash(args['z'])
s_test = utility.remove_slash(args['s_test'])
s_tests_id = args['s_tests_id']
r = args['r']
n = args['n']
save_figure = args['save_figure']
s_tests = []
for i in six.moves.range(r):
    s_tests.append('{}/{}_{}.s_test'.format(s_test, s_tests_id, i))

grad_z = []
grad_z_iter_len = 60000

for i in utility.create_progressbar(grad_z_iter_len, desc='loading grad_z'):
    grad_z.append(torch.load('{}/{}.grad_z'.format(z, i)))

# take sum
e_s_test = torch.load('{}'.format(s_tests[0]))
for i in utility.create_progressbar(len(s_tests), desc='loading s_tests', start=1):
    e_s_test = [i + j for i, j in six.moves.zip(e_s_test, torch.load('{}'.format(s_tests[0])))]
# average
e_s_test = [i / len(s_tests) for i in e_s_test]

influence = []
for i in utility.create_progressbar(grad_z_iter_len, desc='caluculating influence'):

harmful = np.argsort(influence)
helpful = harmful[::-1]

main = MnistTrainer(None, None)


def to_subplot(axes, picture, title):
    axes.axis('off')
    axes.imshow(picture, cmap='gray', interpolation='nearest')
    axes.set_title(title)


fig, axes = plt.subplots(nrows=4, ncols=4)
to_subplot(axes[0, 0], main.test_loader.dataset.test_data[int(s_tests_id)].numpy(), 'test:{}'.format(main.test_loader.dataset.test_labels[int(s_tests_id)]))
for i, ii, iii in [[0, 1, 0], [0, 2, 1], [0, 3, 2], [1, 0, 3], [1, 1, 4], [1, 2, 5], [1, 3, 6]]:
    to_subplot(axes[i, ii], main.train_loader.dataset.train_data[helpful[iii]].numpy(), '{}'.format(main.train_loader.dataset.train_labels[helpful[iii]]) + ':{0:+9.2e}'.format(influence[helpful[iii]]))
for i, ii, iii in [[2, 0, 0], [2, 1, 1], [2, 2, 2], [2, 3, 3], [3, 0, 4], [3, 1, 5], [3, 2, 6], [3, 3, 7]]:
    to_subplot(axes[i, ii], main.train_loader.dataset.train_data[harmful[iii]].numpy(), '{}'.format(main.train_loader.dataset.train_labels[harmful[iii]]) + ':{0:+9.2e}'.format(influence[harmful[iii]]))

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.2, hspace=0.4)
fig.savefig(save_figure)
