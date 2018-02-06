import math

import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block_1 = self.__conv_block(3, 32)
        self.down_block_2 = self.__conv_block(32, 64)
        self.down_block_3 = self.__conv_block(64, 128)

        self.center_1 = self.__conv(32, 32, (1, 1), p=0)
        self.center_2 = self.__conv(64, 64, (1, 1), p=0)
        self.center_3 = self.__conv(128, 128, (1, 1), p=0)
        self.center_4 = self.__conv(128, 128, (3, 3), p=1)

        self.up_block_1 = self.__conv_block(64, 3)
        self.up_block_2 = self.__conv_block(128, 32)
        self.up_block_3 = self.__conv_block(256, 64)

        self.res = self.__conv(3, 1, (3, 3), p=1)

    def __conv(self, in_c, out_c, k, p=1):
        conv = nn.Conv2d(in_c, out_c, k, padding=p)
        init.xavier_normal(conv.weight, math.sqrt(2))
        return nn.Sequential(conv, nn.ReLU(), nn.BatchNorm2d(out_c))

    def __conv_block(self, in_c, out_c):
        return nn.Sequential(
            self.__conv(in_c, out_c, (3, 3), p=1),
            self.__conv(out_c, out_c, (3, 3), p=1),
        )

    def forward(self, x):
        out1 = self.down_block_1(x)
        down1 = F.max_pool2d(out1, kernel_size=2)
        out2 = self.down_block_2(down1)
        down2 = F.max_pool2d(out2, kernel_size=2)
        out3 = self.down_block_3(down2)
        z = F.max_pool2d(out3, kernel_size=2)

        z = self.center_4(z)
        out3 = self.center_3(out3)
        out2 = self.center_2(out2)
        out1 = self.center_1(out1)

        up3 = F.upsample(z, scale_factor=2, mode='bilinear')
        out3 = self.up_block_3(T.cat((out3, up3), 1))
        up2 = F.upsample(out3, scale_factor=2, mode='bilinear')
        out2 = self.up_block_2(T.cat((out2, up2), 1))
        up1 = F.upsample(out2, scale_factor=2, mode='bilinear')
        out1 = self.up_block_1(T.cat((out1, up1), 1))

        res = F.sigmoid(self.res(out1))
        return res


def test_model():
    model = UNet().cuda()
    inp = Variable(T.zeros(10, 3, 448, 448).cuda(), requires_grad=False)
    out = model(inp)
    assert out.size() == (10, 1, 448, 448)
    print('Pass')