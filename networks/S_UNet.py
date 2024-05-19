import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, padding=1, dropout=0.0, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=padding, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, padding=1, dropout=0.0, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.maxpool_dconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, mid_channels, padding, dropout, bias)
        )

    def forward(self, x):
        return self.maxpool_dconv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, dropout=0.0,bias=False):
        super(Encoder, self).__init__()
        self.inc = DoubleConv(in_channels, base_filters, padding=193, dropout=dropout, bias=bias)
        self.down1 = DownSampleLayer(base_filters, base_filters * 2, padding=97, dropout=dropout, bias=bias)
        self.down2 = DownSampleLayer(base_filters * 2, base_filters * 4, padding=49, dropout=dropout, bias=bias)
        self.down3 = DownSampleLayer(base_filters * 4, base_filters * 8, padding=25, dropout=dropout, bias=bias)
        self.down4 = DownSampleLayer(base_filters * 8, base_filters * 16, padding=13, dropout=dropout, bias=bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, dropout=0, mode='nearest', Transpose=False, bias=False):
        super(UpSampleLayer, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2, bias=bias)
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=mode),
                                    nn.Conv2d(in_channels, in_channels//2, kernel_size=1, bias=bias))
        self.conv = DoubleConv(in_channels, out_channels, padding=padding, dropout=dropout, bias=bias)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入可能具有不同的空间维度，因此我们需要填充
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, base_filters=16,bias=False):
        super(Decoder, self).__init__()
        self.up1 = UpSampleLayer(base_filters * 16, base_filters * 8, padding=25, dropout=0, bias=bias)
        self.up2 = UpSampleLayer(base_filters * 8, base_filters * 4, padding=49, dropout=0, bias=bias)
        self.up3 = UpSampleLayer(base_filters * 4, base_filters * 2, padding=97, dropout=0, bias=bias)
        self.up4 = UpSampleLayer(base_filters * 2, base_filters, padding=193, dropout=0, bias=bias)

        self.outc = nn.Conv2d(base_filters, 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.tanh(x)


class S_UNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=16, bias=False):
        super(S_UNet, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, base_filters=base_filters, bias=bias)
        self.decoder = Decoder(base_filters=base_filters, bias=bias)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        y_hat = self.decoder(x1, x2, x3, x4, x5)
        return y_hat


if __name__ == '__main__':
    X = torch.randn((64, 1, 384, 384))  # 输入图像
    base_filters = 32
    net = S_UNet(in_channels=1, base_filters=base_filters, bias=True)
    summary(net, input_size=(1, 1, 384, 384))



