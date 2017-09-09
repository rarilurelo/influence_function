import six
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import functools
from nutszebra_initialization import Initialization as initializer


def select_way(way, channel_in, channel_out):
    if way == 'ave':
        n_i = channel_in
        n_i_next = channel_out
    if way == 'forward':
        n_i = channel_in
        n_i_next = None
    if way == 'backward':
        n_i = None
        n_i_next = channel_out
    return n_i, n_i_next


def weight_relu_initialization(link, mean=0.0, relu_a=0.0, way='forward'):
    dim = len(link.weight.data.shape)
    if dim == 2:
        # fc layer
        channel_out, channel_in = link.weight.data.shape
        y_k, x_k = 1, 1
    elif dim == 4:
        # conv layer
        channel_out, channel_in, y_k, x_k = link.weight.data.shape
    n_i, n_i_next = select_way(way, channel_in * y_k * x_k, channel_out * y_k * x_k)
    # calculate variance
    variance = initializer.variance_relu(n_i, n_i_next, a=relu_a)
    # orthogonal matrix
    w = []
    for i in six.moves.range(channel_out):
        w.append(initializer.orthonorm(mean, variance, (channel_in, y_k * x_k), initializer.gauss, np.float32))
    return np.reshape(w, link.weight.data.shape)


def bias_initialization(conv, constant=0.0):
    return initializer.const(conv.bias.data.shape, constant=constant, dtype=np.float32)


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

    def __setitem__(self, key, value):
        super(NN, self).__setattr__(key, value)
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def _count_parameters(self, shape):
        return functools.reduce(lambda a, b: a * b, shape)

    def count_parameters(self):
        return sum([self._count_parameters(p.data.shape) for p in self.parameters()])

    def global_average_pooling(self, x):
        batch, ch, height, width = x.data.shape
        x = F.avg_pool2d(x, (height, width), 1, 0)
        return x.view(batch, ch)


class DoNothing(object):

    def __call__(self, *args, **kwargs):
        return args[0]

    def weight_initialization(self):
        pass

    def count_parameters(self):
        return 0


class Bridge(NN):

    def __init__(self, in_channel, pad, pool_flag=True):
        super(Bridge, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.pad = pad
        self.pool_flag = pool_flag

    def weight_initialization(self):
        pass

    @staticmethod
    def concatenate_zero_pad(x, pad):
        N, _, H, W = x.data.shape
        x_pad = Variable(torch.zeros(N, pad, H, W), volatile=x.volatile)
        if x.data.type() == 'torch.cuda.FloatTensor':
            x_pad = x_pad.cuda(x.data.get_device())
        x = torch.cat((x, x_pad), 1)
        return x

    def forward(self, x):
        x = self.bn(x)
        if self.pool_flag:
            x = F.avg_pool2d(x, 1, 2, 0)
        return self.concatenate_zero_pad(x, self.pad)


class BN_ReLU_Conv(NN):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, stride, pad)
        self.bn = nn.BatchNorm2d(in_channel)

    def weight_initialization(self):
        self.conv.weight.data = torch.FloatTensor(weight_relu_initialization(self.conv))
        self.conv.bias.data = torch.FloatTensor(bias_initialization(self.conv, constant=0))

    def forward(self, x):
        return self.conv(F.relu(self.bn(x)))


class ResBlock(NN):

    def __init__(self, in_channel, out_channel, bridge=DoNothing(), n=18, stride_at_first_layer=2, multiplier=4):
        super(ResBlock, self).__init__()
        for i in six.moves.range(n):
            self['bn_relu_conv1_{}'.format(i)] = BN_ReLU_Conv(in_channel, out_channel, 1, stride_at_first_layer, 0)
            self['bn_relu_conv2_{}'.format(i)] = BN_ReLU_Conv(out_channel, out_channel)
            self['bn_relu_conv3_{}'.format(i)] = BN_ReLU_Conv(out_channel, int(multiplier * out_channel), 1, 1, 0)
            stride_at_first_layer = 1
            in_channel = int(multiplier * out_channel)
        self.bridge = bridge
        self.n = n

    def weight_initialization(self):
        self.bridge.weight_initialization()
        for i in six.moves.range(self.n):
            self['bn_relu_conv1_{}'.format(i)].weight_initialization()
            self['bn_relu_conv2_{}'.format(i)].weight_initialization()
            self['bn_relu_conv3_{}'.format(i)].weight_initialization()

    def __call__(self, x):
        for i in six.moves.range(self.n):
            h = self['bn_relu_conv1_{}'.format(i)](x)
            h = self['bn_relu_conv2_{}'.format(i)](h)
            h = self['bn_relu_conv3_{}'.format(i)](h)
            if i == 0:
                x = self.bridge(x)
            x = h + x
        return x


class ResidualNetwork(NN):

    def __init__(self, category_num, out_channels=(16, 32, 64), N=(18, 18, 18), multiplier=4):
        super(ResidualNetwork, self).__init__()
        # first conv
        self.conv1 = nn.Conv2d(3, out_channels[0], 3, 1, 1)
        in_channel = out_channels[0]
        # first block's stride is 1
        strides = [1] + [2] * (len(out_channels) - 1)
        # create resblock
        for i, out_channel, n, stride in six.moves.zip(six.moves.range(len(out_channels)), out_channels, N, strides):
            print('{}:, {}, {}'.format(i, in_channel, out_channel))
            bridge = Bridge(in_channel, out_channel * multiplier - in_channel, pool_flag=stride == 2)
            self['res_block{}'.format(i)] = ResBlock(in_channel, out_channel, n=n, stride_at_first_layer=stride, multiplier=multiplier, bridge=bridge)
            in_channel = int(out_channel * multiplier)
        self.bn_relu_conv = BN_ReLU_Conv(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        # arguments
        self.out_channels = out_channels
        # name of model
        self.name = 'residual_network_{}_{}_{}_{}'.format(category_num, out_channels, N, multiplier)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(bias_initialization(self.conv1, constant=0))
        for i in six.moves.range(len(self.out_channels)):
            self['res_block{}'.format(i)].weight_initialization()
        self.bn_relu_conv.weight_initialization()

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        for i in six.moves.range(len(self.out_channels)):
            h = self['res_block{}'.format(i)](h)
        h = self.bn_relu_conv(F.relu(h))
        h = self.global_average_pooling(h)
        return h

    def calc_loss(self, y, t):
        y = F.log_softmax(y)
        loss = F.nll_loss(y, t, weight=None, size_average=False)
        return loss
