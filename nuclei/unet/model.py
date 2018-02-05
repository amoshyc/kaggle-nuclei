import math

import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block_1 = self.__conv_block(1, 64)
        self.down_block_2 = self.__conv_block(64, 128)
        self.down_block_3 = self.__conv_block(128, 256)
        self.down_block_4 = self.__conv_block(256, 512)

        self.center_1 = self.__center_block(64)
        self.center_2 = self.__center_block(128)
        self.center_3 = self.__center_block(256)
        self.center_4 = self.__center_block(512)
        self.center_5 = self.__center_block(1024)

        self.up_block_1 = self.__conv_block(128, 64)
        self.up_block_2 = self.__conv_block(256, 128)
        self.up_block_3 = self.__conv_block(512, 256)
        self.up_block_4 = self.__conv_block(1024, 512)

        self.last = self.__conv(64, 2)

    def __conv(self, in_c, out_c, k):
        conv = nn.Conv2d(in_c, out_c, k, padding=1)
        init.xavier_normal(conv.weight, math.sqrt(2))
        return nn.Sequential(conv, nn.ReLU())

    def __conv_block(self, in_c, out_c):
        return nn.Sequential(
            self.__conv(in_c, out_c, (3, 3)),
            self.__conv(out_c, out_c, (3, 3)),
            self.__conv(out_c, out_c, (3, 3))
        )

    def __center_block(self, c):
        return nn.Sequential(
            self.__conv(c, c, (1, 1))
        )

    def forward(self, x):
        out1 = self.down_block_1(x)
        down1 = F.max_pool2d(out1)
        out2 = self.down_block_2(x)
        down2 = F.max_pool2d(out2)
        out3 = self.down_block_3(x)
        down3 = F.max_pool2d(out3)
        out4 = self.down_block_4(x)
        down4 = F.max_pool2d(out4)

        z = self.center_5(down4)
        out1 = self.center_1(out1)
        out2 = self.center_2(out2)
        out3 = self.center_3(out3)
        out4 = self.center_4(out4)

        up4 = F.upsample(z, scale_factor=2, mode='bilinear')
        out4 = self.up_block_4(T.cat((out4, up4), 1))
        up3 = F.upsample(out4, scale_factor=2, mode='bilinear')
        out3 = self.up_block_3(T.cat((out3, up3), 1))
        up2 = F.upsample(out3, scale_factor=2, mode='bilinear')
        out2 = self.up_block_2(T.cat((out2, up2), 1))
        up1 = F.upsample(out2, scale_factor=2, mode='bilinear')
        out1 = self.up_block_1(T.cat((out1, up1), 1))

        res = self.last(out1)
        return res


def test_model():
    model = UNet()
    inp = Variable(torch.zeros(10, 3, 512, 512), requires_grad=False)
    out = model(inp)
    assert out.size() == (10, 2, 512, 512)