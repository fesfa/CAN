import torch.nn as nn
import torch

# G
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


# G
class G2(nn.Module):
    def __init__(self, opt):
        super(G2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.fc3 = nn.Linear(opt.attSize, 1024)
        self.fc1 = nn.Linear(1024, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.apply(weights_init)

    def forward(self, att):
        h = self.lrelu(self.fc3(att))
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h


# D
class D2(nn.Module):
    def __init__(self, opt):
        super(D2, self).__init__()
        self.discriminator = nn.Linear(opt.mapSize, 1)
        self.hidden = nn.Linear(opt.mapSize, 1024)
        self.classifier = nn.Linear(1024, opt.nclass_seen)
        self.logic = nn.LogSoftmax(dim=1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU()
        self.mapping = nn.Linear(opt.mapSize, 4096)
        self.mapping2 = nn.Linear(4096, opt.mapSize)
        self.discrim = nn.Linear(2048, 1)
        self.apply(weights_init)

    def forward(self, x):
        m = self.lrelu(self.mapping(x))
        m = self.lrelu(self.mapping2(m))
        disc = self.discrim(m)
        return m, disc

class M1(nn.Module):
    def __init__(self, opt):
        super(M1, self).__init__()
        self.hidden = nn.Linear(opt.mapSize, 1024)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.mapping = nn.Linear(opt.resSize, 4096)
        self.mapping2 = nn.Linear(4096, opt.mapSize)
        self.apply(weights_init)

    def forward(self, x):
        m = self.lrelu(self.mapping(x))
        m = self.lrelu(self.mapping2(m))
        m = self.hidden(m)
        return m

class M2(nn.Module):
    def __init__(self, opt):
        super(M2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.mapping = nn.Linear(opt.attSize, 2048)
        self.mapping2 = nn.Linear(2048, 1024)
        self.apply(weights_init)

    def forward(self, x):
        m = self.lrelu(self.mapping(x))
        m = self.mapping2(m)
        return m

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2048, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.fc(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# F类
class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize+opt.attSize, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h

# 回归
class regress(nn.Module):
    def __init__(self, opt):
        super(regress, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, 1024)
        self.fc2 = nn.Linear(1024, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        return self.fc2(h)

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