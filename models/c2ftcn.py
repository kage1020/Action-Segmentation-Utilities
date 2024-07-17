import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def upsize(x: Tensor, size: int):
    return F.interpolate(x, size=size, mode="linear", align_corners=True)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.max_pool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class SPPBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1, padding=0)

    def _layer(self, x: Tensor, pool: nn.Module):
        x = F.upsample(pool(x), size=x.size(2), mode="linear", align_corners=True)
        return x

    def forward(self, x: Tensor) -> Tensor:
        layer1 = self._layer(x, self.pool1)
        layer2 = self._layer(x, self.pool2)
        layer3 = self._layer(x, self.pool3)
        layer4 = self._layer(x, self.pool4)

        out = torch.cat([layer1, layer2, layer3, layer4, x], 1)
        return out


class C2F_TCN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        num_features: int,
    ):
        """
        Parameters:
            num_channels: number of channels in the input data
            num_classes: number of classes in the output data
            num_features: number of features in the output data
        """
        super().__init__()
        self.inconv = InConv(num_channels, 256)
        self.down1 = Down(256, 256)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 128)
        self.down4 = Down(128, 128)
        self.down5 = Down(128, 128)
        self.down6 = Down(128, 128)
        self.up = Up(256 + 1 * 4, 128)
        self.outconv0 = OutConv(128, num_classes)
        self.proj0 = OutConv(128, num_features)
        self.up0 = Up(256, 128)
        self.outconv1 = OutConv(128, num_classes)
        self.proj1 = OutConv(128, num_features)
        self.up1 = Up(256, 128)
        self.outconv2 = OutConv(128, num_classes)
        self.proj2 = OutConv(128, num_features)
        self.up2 = Up(256 + 128, 128)
        self.outconv3 = OutConv(128, num_classes)
        self.proj3 = OutConv(128, num_features)
        self.up3 = Up(256 + 128, 128)
        self.outconv4 = OutConv(128, num_classes)
        self.proj4 = OutConv(128, num_features)
        self.up4 = Up(256 + 128, 128)
        self.outconv5 = OutConv(128, num_classes)
        self.proj5 = OutConv(128, num_features)
        self.spp = SPPBlock(128)

    def ensemble(self, y: list[Tensor]) -> Tensor:
        ensemble_weights = [1, 1, 1, 1, 1, 1]
        num_videos = y[0].shape[-1]
        ensemble_prob = (
            F.softmax(y[0], dim=1) * ensemble_weights[0] / sum(ensemble_weights)
        )

        for i, y_i in enumerate(y[1:]):
            logit = upsize(y_i, num_videos)
            ensemble_prob += (
                F.softmax(logit, dim=1)
                * ensemble_weights[i + 1]
                / sum(ensemble_weights)
            )

        return torch.log(ensemble_prob + 1e-8)

    def forward(self, x: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x7 = self.spp(x7)
        x = self.up(x7, x6)
        y0 = self.outconv0(x)
        p0 = self.proj0(x)
        x = self.up0(x, x5)
        y1 = self.outconv1(x)
        p1 = self.proj1(x)
        x = self.up1(x, x4)
        y2 = self.outconv2(x)
        p2 = self.proj2(x)
        x = self.up2(x, x3)
        y3 = self.outconv3(x)
        p3 = self.proj3(x)
        x = self.up3(x, x2)
        y4 = self.outconv4(x)
        p4 = self.proj4(x)
        x = self.up4(x, x1)
        y5 = self.outconv5(x)
        p5 = self.proj5(x)

        p_list = [p5, p4, p3, p2, p1, p0]
        y_list = [y5, y4, y3, y2, y1, y0]
        num_videos = p5.shape[-1]
        p_list = [upsize(p, num_videos) for p in p_list]
        p_list = [p / torch.norm(p, dim=1, keepdim=True) for p in p_list]

        features = torch.cat(
            [feat * math.sqrt(wt) for (wt, feat) in zip(weights, p_list)], dim=1
        )
        total_norm = math.sqrt(sum(weights))
        features = features / total_norm
        return features, self.ensemble(y_list)
