import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import torch.nn.functional as F
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def hvp(loss, params, vec):
    """
    loss: loss
    params: pramas
    vec: vector of multiplied, vec is same shape of params
    """
    params = list(params)
    vec = [Variable(v.data) for v in vec]
    grads = grad(loss, params, create_graph=True)
    sum_vec = [torch.sum(g*v) for g, v in zip(grads, vec)]
    sums = sum(sum_vec)
    _hvp = grad(sums, list(params), create_graph=True)
    return _hvp


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    model = Net()
    model.cpu()
    model.load_state_dict(torch.load(args.model))

    #from torch.utils import data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    print(model)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = F.nll_loss(output, target)
        #grads = grad(loss, model.parameters(), retain_graph=True, create_graph=True)
        #grads = [g.detach() for g in grads]
        _hvp = hvp(loss, model.parameters(), model.parameters())
        print(_hvp)
