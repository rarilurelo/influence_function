import six
import utility
from hvp import hvp
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad


def s_test(z_test, t_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0, repeat=5000):
    # prepate v
    z_test, t_test = Variable(z_test, volatile=False), Variable(t_test, volatile=False)
    if gpu >= 0:
        z_test, t_test = z_test.cuda(gpu), t_test.cuda(gpu)
    y_test = model(z_test)
    loss = F.nll_loss(y_test, t_test, weight=None, size_average=True)
    v = list(grad(loss, list(model.parameters()), create_graph=True))
    h_estimates = v.copy()

    for i in utility.create_progressbar(repeat, desc='s_test'):
        for x, t in z_loader:
            x, t = Variable(x, volatile=False), Variable(t, volatile=False)
            if gpu >= 0:
                x, t = x.cuda(gpu), t.cuda(gpu)
            y = model(x)
            loss = F.nll_loss(y, t, weight=None, size_average=True)
            hv = hvp(loss, list(model.parameters()), h_estimates)
            h_estimate = [_v + (1 - damp) * h_estimate - _hv / scale for _v, h_estimate, _hv in six.moves.zip(v, h_estimates, hv)]
            break
    return h_estimate


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
