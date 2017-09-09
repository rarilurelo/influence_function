import six
import utility
from hvp import hvp
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad


def s_test(z_test, t_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0, repeat=5000):
    v = grad_z(z_test, t_test, model, gpu)
    h_estimates = v.copy()

    for i in utility.create_progressbar(repeat, desc='s_test'):
        for x, t in z_loader:
            x, t = Variable(x, volatile=False), Variable(t, volatile=False)
            if gpu >= 0:
                x, t = x.cuda(gpu), t.cuda(gpu)
            y = model(x)
            loss = model.calc_loss(y, t)
            hv = hvp(loss, list(model.parameters()), h_estimates)
            h_estimate = [_v + (1 - damp) * h_estimate - _hv / scale for _v, h_estimate, _hv in six.moves.zip(v, h_estimates, hv)]
            break
    return h_estimate


def grad_z(z, t, model, gpu=-1):
    model.eval()
    # initialize
    z, t = Variable(z, volatile=False), Variable(t, volatile=False)
    if gpu >= 0:
        z, t = z.cuda(gpu), t.cuda(gpu)
        model.cuda(gpu)
    y = model(z)
    loss = model.calc_loss(y, t)
    return list(grad(loss, list(model.parameters()), create_graph=True))


if __name__ == '__main__':
    from trainer_mnist import MnistTrainer
    from net import Net
    from optimizers import MomentumSGD
    model = Net()
    optimizer = MomentumSGD(model, 0, 0)
    main = MnistTrainer(model, optimizer, train_batch_size=1)
    z_test, t_test = main.test_loader.dataset[0]
    z_test, t_test = main.test_loader.collate_fn([z_test]), main.test_loader.collate_fn([t_test])
    test = s_test(z_test, t_test, model, main.train_loader, gpu=-1)
