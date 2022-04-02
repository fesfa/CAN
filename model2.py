import torch.nn as nn
import torch

class D2(nn.Module):
    def __init__(self, opt):
        super(D2, self).__init__()

        self.discrim = nn.Linear(2048, 1)
        self.apply(weights_init)

    def forward(self, x):

        disc = self.discrim(x)
        return disc


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(1024, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.fc3 = nn.Linear(opt.attSize + opt.nz, 1024)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc3(h))
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h


class classifier(nn.Module):
    def __init__(self, opt):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax()

    def forward(self, input):
        m = self.lrelu(self.fc1(input))
        m = self.lrelu(self.fc2(m))
        m = self.logic(self.fc3(m))
        return m

class RNet1(nn.Module):
    def __init__(self, opt):
        super(RNet1, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.fc3 = nn.Linear(opt.resSize, 4096)
        self.apply(weights_init)

    def forward(self,  x_g):
        h = self.lrelu(self.fc3(x_g))
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h