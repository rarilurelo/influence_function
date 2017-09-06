import six
import torch
from torch.autograd import grad


def hvp(y, w, v):
    first_grads = grad(y, w, create_graph=True)
    grad_v = 0
    for g, v in six.moves.zip(first_grads, v):
        grad_v += torch.sum(g * v)
    return grad(grad_v, w, create_graph=True)


if __name__ == '__main__':
    import argparse
    import torch.optim as optim
    from net import Net
    from trainer_mnist import MnistTrainer
    from torch.autograd import Variable
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                        help='-1 means cpu, otherwise gpu id')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')
    momentum = args.pop('momentum')
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args['model'], args['optimizer'] = model, optimizer
    main = MnistTrainer(**args)
    main.model.train()
    for x, t in main.train_loader:
        x, t = Variable(x, volatile=False), Variable(t, volatile=False)
        y = main.model(x)
        loss = F.nll_loss(y, t, weight=None, size_average=True)
        v = grad(loss, list(model.parameters()), retain_graph=True, create_graph=True)
        hv = hvp(loss, list(model.parameters()), v)
        print(hv)
        break
