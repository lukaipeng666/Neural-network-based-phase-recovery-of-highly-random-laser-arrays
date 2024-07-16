from collections import OrderedDict
import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class upsample2(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=False):
        super(upsample2, self).__init__()
        self.up = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5) if drop_out else nn.Identity()
        )

    def forward(self, x):
        return self.up(x)


class conv_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_net, self).__init__()
        self.Con_3 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.Con_3(x)


class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Conv_net(nn.Module):
    def __init__(self):
        super(Conv_net, self).__init__()
        self.L1 = conv_net(1, 1024)
        self.L2 = conv_net(1024, 1024)
        self.L3 = conv_net(1024, 1024)
        self.L4 = upsample2(1024, 1024)
        self.L5 = upsample2(1024, 1024)
        self.L6 = upsample2(1024, 1)
        self.tanh = nn.Tanh()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Linear):
                if w.bias is not None:
                    nn.init.zeros_(w.bias)

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.tanh(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()
        self.layer1 = downsample(1, 64)
        self.layer2 = downsample(64, 128)
        self.layer3 = downsample(128, 256)
        self.layer4 = downsample(256, 512)
        self.layer5 = downsample(512, 512)
        self.layer6 = downsample(512, 512)
        self.layer7 = downsample(512, 256)

        self.up_1 = upsample2(256, 256)
        self.up_2 = upsample2(768, 1024)
        self.up_3 = upsample2(1536, 1024)
        self.up_4 = upsample2(1536, 1024)
        self.up_5 = upsample2(1280, 512)
        self.up_6 = upsample2(640, 256)
        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=320, out_channels=1, kernel_size=2, stride=2),
            nn.Tanh()
        )

        self.L1 = Conv_net()
        self.L2 = Conv_net()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d) or isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out' if isinstance(w, nn.Conv2d) else 'fan_in')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        down_1 = self.layer1(x)
        down_2 = self.layer2(down_1)
        down_3 = self.layer3(down_2)
        down_4 = self.layer4(down_3)
        down_5 = self.layer5(down_4)
        down_6 = self.layer6(down_5)
        down_7 = self.layer7(down_6)
        up_1 = self.up_1(down_7)
        up_2 = self.up_2(torch.cat((up_1, down_6), dim=1))
        up_3 = self.up_3(torch.cat((up_2, down_5), dim=1))
        up_4 = self.up_4(torch.cat((up_3, 0.8 * down_4), dim=1))
        up_5 = self.up_5(torch.cat((up_4, 0.4 * down_3), dim=1))
        up_6 = self.up_6(torch.cat((up_5, 0.2 * down_2), dim=1))
        out = self.last_Conv(torch.cat((up_6, 0.1 * down_1), dim=1))
        out_10 = self.L1(out)

        return out, out_10



class unet_D(nn.Module):
    def __init__(self):
        super(unet_D, self).__init__()

        def base_Conv_bn_lkrl(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        D_dic = OrderedDict()
        in_channels = 2
        out_channels = 64
        for i in range(4):
            if i < 3:
                D_dic.update({'layer_{}'.format(i + 1): base_Conv_bn_lkrl(in_channels, out_channels, 2)})
            else:
                D_dic.update({'layer_{}'.format(i + 1): base_Conv_bn_lkrl(in_channels, out_channels, 1)})
            in_channels = out_channels
            out_channels *= 2
        D_dic.update({'last_layer': nn.Conv2d(512, 1, 4, 2, 1)})  # [batch,1,30,30]
        self.D_model = nn.Sequential(D_dic)

    def forward(self, x1, x2):
        in_x = torch.cat([x1, x2], dim=1)
        return self.D_model(in_x)
