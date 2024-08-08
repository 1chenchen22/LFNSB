import os

from torch import nn
import torch
from networks import LFN
from torch.nn import Module



class Spatial_bias(nn.Module):
    def __init__(self, channel):
        super(Spatial_bias, self).__init__()
        sb_in_plane = 66
        self.reduce_r = 2
        self.num_sb = sb_in_plane - 2
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(channel, sb_in_plane, kernel_size=1, bias = False, stride = 1),
            nn.BatchNorm2d(sb_in_plane),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.reduce_r)
        )
        self.sb_conv = nn.Conv1d((self.reduce_r**2), (self.reduce_r**2), 3, padding = 0)

    def forward(self, x_):
        x_ = self.feature_reduction(x_)
        x = torch.reshape(x_, (x_.shape[0], x_.shape[1], 1, -1)).squeeze().permute(0, 2, 1)
        x = self.sb_conv(x).unsqueeze(2)
        x = x.permute(0, 3, 2, 1).reshape(x_.shape[0], self.num_sb, self.reduce_r, self.reduce_r)
        # print("x",x.shape)
        return x
class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Flatten(Module):
    def forward(self, input):
       return input.reshape(input.size(0), -1)



class LFNSB2(nn.Module):
    def __init__(self, num_class=7, num_head=1, pretrained=True):
        super(LFNSB2, self).__init__()
        net = LFN.mfn()
        # if pretrained:
        #     net = torch.load(os.path.join('./pretrained/', "lfn.pth"))
        self.features = nn.Sequential(*list(net.children())[:-4])
        self.sb=Spatial_bias(256)
        self.flatten = Flatten()

        self.fc = nn.Linear(256, num_class)
        self.bn = nn.BatchNorm1d(num_class)
    def forward(self, x):
        x = self.features(x)
        y=self.sb(x)

        y = self.flatten(y)

        out = self.fc(y)
        return out, x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordAtt(512, 512)

    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()

        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding=(0, 0))
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding=(0, 0))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.Linear_h(x)
        x_w = self.Linear_w(x)
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        # print("x_h",x_h.shape)
        # print("x_w", x_w.shape)
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = x_w * x_h

        return y