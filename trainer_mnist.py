from __future__ import print_function
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import utility


class MnistTrainer(object):

    def __init__(self, model, optimizer, gpu=-1, save_path='./', train_transform=None, test_transform=None, train_batch_size=64, test_batch_size=256, start_epoch=1, epochs=200, seed=1):
        self.model, self.optimizer = model, optimizer
        self.gpu, self.save_path = gpu, utility.remove_slash(save_path)
        self.train_transform, self.test_transform = train_transform, test_transform
        self.train_batch_size, self.test_batch_size = train_batch_size, test_batch_size
        self.start_epoch, self.epochs, self.seed = start_epoch, epochs, seed
        # load mnist
        self.init_dataset()
        # initialize seed
        self.init_seed()

    def init_transform(self):
        if self.train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        if self.test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    def init_dataset(self):
        # initialize transform
        self.init_transform()
        # arguments for gpu mode
        kwargs = {}
        if self.check_gpu():
            kwargs = {'num_workers': 1,
                      'pin_memory': True}

        # load dataset
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=self.train_transform),
            batch_size=self.train_batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                           transform=self.test_transform),
            batch_size=self.test_batch_size, shuffle=False, **kwargs)

    def check_gpu(self):
        return self.gpu >= 0 and torch.cuda.is_available()

    def init_seed(self):
        torch.manual_seed(self.seed)
        if self.check_gpu():
            torch.cuda.manual_seed(self.seed)

    def to_gpu(self):
        if self.check_gpu():
            self.model.cuda(self.gpu)

    def to_cpu(self):
        self.model.cpu()

    def train_one_epoch(self):
        self.to_gpu()
        self.model.train()
        sum_loss = 0
        for x, t in self.train_loader:
            if self.check_gpu():
                x, t = x.cuda(), t.cuda()
            x, t = Variable(x, volatile=False), Variable(t, volatile=False)
            self.optimizer.zero_grad()
            y = self.model(x)
            loss = F.nll_loss(y, t, weight=None, size_average=True)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss.cpu().data[0] * self.train_batch_size
        self.to_cpu()
        # save model
        return sum_loss / len(self.train_loader.dataset)

    def test_one_epoch(self):
        self.to_gpu()
        self.model.eval()
        sum_loss = 0
        accuracy = 0
        for x, t in self.test_loader:
            if self.check_gpu():
                x, t = x.cuda(), t.cuda()
            x, t = Variable(x, volatile=True), Variable(t, volatile=True)
            y = self.model(x)
            # loss
            loss = F.nll_loss(y, t, weight=None, size_average=False)
            sum_loss += loss.cpu().data[0]
            # accuracy
            y = y.data.max(1, keepdim=True)[1]
            accuracy += y.eq(t.data.view_as(y)).cpu().sum()
        sum_loss /= len(self.test_loader.dataset)
        accuracy /= len(self.test_loader.dataset)
        return sum_loss, accuracy

    def save_model(self):
        pass

    def run(self):
        for i in utility.create_progressbar(self.epochs + 1, desc='epoch', stride=1, start=self.start_epoch):
            print('train {}: loss->{}'.format(i, self.train_one_epoch()))
            self.save_model()
            print('test {}: (loss, accu)->{}'.format(i, self.test_one_epoch()))
