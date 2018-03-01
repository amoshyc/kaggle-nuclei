import math

import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def conv(in_c, out_c, k, s=1):
    convolution = nn.Conv2d(in_c, out_c, k, stride=s, padding=(k - 1) // 2)
    init.xavier_normal(convolution.weight, math.sqrt(2))
    return nn.Sequential(convolution, nn.ReLU())
    # return nn.Sequential(convolution, nn.ReLU(), nn.BatchNorm2d(out_c))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pre = conv(c, c, 3)
        self.c1 = conv(c, c // 2, 1)
        self.c2 = conv(c // 2, c // 2, 3)
        self.c3 = conv(c // 2, c, 1)

    def forward(self, x):
        x1 = self.pre(x)

        x2 = F.max_pool2d(x1, 2)
        x2 = self.c1(x2)
        x2 = self.c2(x2)
        x2 = self.c3(x2)
        x2 = F.upsample(x2, scale_factor=2)

        return x1 + x2


class Hourglass(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pre1 = nn.Sequential(ResBlock(c), ResBlock(c))
        self.pre2 = nn.Sequential(ResBlock(c), ResBlock(c), ResBlock(c))
        self.pre3 = nn.Sequential(ResBlock(c), ResBlock(c), ResBlock(c))

        self.mid1 = nn.Sequential(ResBlock(c), ResBlock(c), ResBlock(c))
        self.mid2 = nn.Sequential(ResBlock(c), ResBlock(c), ResBlock(c))
        self.mid3 = nn.Sequential(ResBlock(c), ResBlock(c), ResBlock(c))
        self.mid4 = nn.Sequential(ResBlock(c), ResBlock(c), ResBlock(c))

        self.post1 = nn.Sequential(ResBlock(c))
        self.post2 = nn.Sequential(ResBlock(c))
        self.post3 = nn.Sequential(ResBlock(c), ResBlock(c))

    def forward(self, x):
        x1 = self.pre1(x)
        x2 = self.pre2(F.max_pool2d(x1, 2))
        x3 = self.pre3(F.max_pool2d(x2, 2))
        z = F.max_pool2d(x3, 2)

        x1 = self.mid1(x1)
        x2 = self.mid2(x2)
        x3 = self.mid3(x3)
        z = self.mid4(z)

        x3 = self.post3(F.upsample(z, scale_factor=2) + x3)
        x2 = self.post2(F.upsample(x3, scale_factor=2) + x2)
        x1 = self.post1(F.upsample(x2, scale_factor=2) + x1)

        return x1


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64
        self.pre = nn.Sequential(conv(3, 32, 3), conv(32, c, 3))
        self.hg = Hourglass(c)
        self.post = nn.Sequential(conv(c, 32, 3), conv(32, 2, 3))

    def forward(self, x):
        x = self.pre(x)
        x = self.hg(x)
        x = self.post(x)
        seg = F.sigmoid(x[:, 0, ...])
        tag = x[:, 1, ...]
        return seg, tag


def test_model():
    model = AE().cuda()
    inp = Variable(T.zeros(1, 3, 448, 448).cuda(), requires_grad=False)
    seg, tag = model(inp)
    print('Pass')
    input('Press to continue')