import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class MaxPool3dSamePadding(nn.MaxPool3d):
    stride: tuple[int, int, int]
    kernel_size: tuple[int, int, int]

    def __init__(
        self,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: int = 0,
        name: str = "max_pool3d_same_padding",
    ):
        """
        Parameters:
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding of the convolution.
            name: A string (optional). The name of this module.
        """
        super().__init__(kernel_size=kernel_size, stride=stride, padding=0)
        self.padding = padding
        self.name = name

    def _pad(self, dim: int, size: int) -> int:
        """
        Parameters:
            dim: Dimension to pad.
            size: Size of dimmension.
        Returns:
            Padding size.
        """
        if size % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (size % self.stride[dim]), 0)

    def forward(self, x: Tensor) -> Tensor:
        _, _, t, h, w = x.shape

        pad_t = self._pad(0, t)
        pad_h = self._pad(1, h)
        pad_w = self._pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)

        x = F.pad(x, pad=pad)
        x = super().forward(x)

        return x


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: tuple[int, int, int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: int = 0,
        activation_fn=F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
        name: str = "unit_3d",
    ):
        """
        Parameters:
            in_channels: Number of channels of input videos.
            output_channels: Number of channels of output videos.
            kernel_shape: Shape of the convolution kernel.
            stride: Stride of the convolution.
            padding: Padding of the convolution.
            activation_fn: Activation function.
            use_batch_norm: Whether to use batch normalization.
            use_bias: Whether to use bias.
            name: A string (optional). The name of this module.
        """
        super().__init__()

        self._out_channels = out_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._padding = padding
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self.name = name

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._out_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._out_channels, eps=0.001, momentum=0.01)

    def _pad(self, dim: int, size: int) -> int:
        """
        Parameters:
            dim: Dimension to pad.
            size: Size of dimmension.
        Returns:
            Padding size.
        """
        if size % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (size % self._stride[dim]), 0)

    def forward(self, x: Tensor):
        _, _, t, h, w = x.shape

        pad_t = self._pad(0, t)
        pad_h = self._pad(1, h)
        pad_w = self._pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)

        x = F.pad(x, pad=pad)
        x = self.conv3d(x)

        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)

        return x


class InceptionModule(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: list[int], name: str = "inception_module"
    ):
        super().__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_shape=(1, 1, 1),
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[1],
            kernel_shape=(1, 1, 1),
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            out_channels=out_channels[2],
            kernel_shape=(3, 3, 3),
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[3],
            kernel_shape=(1, 1, 1),
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            out_channels=out_channels[4],
            kernel_shape=(3, 3, 3),
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[5],
            kernel_shape=(1, 1, 1),
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class I3D(nn.Module):
    """
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up to a designated `final_endpoint` are returned in a dictionary as the second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes: int = 400,
        spatial_squeeze: bool = True,
        final_endpoint: str = "Logits",
        name: str = "inception_i3d",
        in_channels: int = 3,
        dropout_keep_prob: float = 0.5,
    ):
        """
        Parameters:
            num_classes: The number of outputs in the logit layer (default 400, which matches the Kinetics dataset).
            spatial_squeeze: Whether to squeeze the spatial dimensions for the logits before returning (default True).
            final_endpoint: The model contains many possible endpoints. `final_endpoint` specifies the last endpoint for the model to be built up to. In addition to the output at `final_endpoint`, all the outputs at endpoints up to `final_endpoint` will also be returned, in a dictionary. `final_endpoint` must be one of InceptionI3d.VALID_ENDPOINTS (default 'Logits').
            name: A string (optional). The name of this module.
            in_channels: Number of channels of input videos, default 3.
            dropout_keep_prob: Probability for the nn.Dropout layer (float in [0, 1)).
        """
        super().__init__()

        assert (
            final_endpoint in self.VALID_ENDPOINTS
        ), f"Unknown final endpoint {final_endpoint}"
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            out_channels=64,
            kernel_shape=(7, 7, 7),
            stride=(2, 2, 2),
            padding=3,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0, name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            out_channels=64,
            kernel_shape=(1, 1, 1),
            padding=0,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            out_channels=192,
            kernel_shape=(3, 3, 3),
            padding=1,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0, name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(
            in_channels=192,
            out_channels=[64, 96, 128, 16, 32, 32],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(
            in_channels=64 + 128 + 32 + 32,
            out_channels=[128, 128, 192, 32, 96, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0, name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(
            in_channels=128 + 192 + 96 + 64,
            out_channels=[192, 96, 208, 16, 48, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(
            in_channels=192 + 208 + 48 + 64,
            out_channels=[160, 112, 224, 24, 64, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(
            in_channels=160 + 224 + 64 + 64,
            out_channels=[128, 128, 256, 24, 64, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(
            in_channels=128 + 256 + 64 + 64,
            out_channels=[112, 144, 288, 32, 64, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(
            in_channels=112 + 288 + 64 + 64,
            out_channels=[256, 160, 320, 32, 128, 128],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(
            in_channels=256 + 320 + 128 + 128,
            out_channels=[256, 160, 320, 32, 128, 128],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            in_channels=256 + 320 + 128 + 128,
            out_channels=[384, 192, 384, 48, 128, 128],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Logits"
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            out_channels=self._num_classes,
            kernel_shape=(1, 1, 1),
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x: Tensor) -> Tensor:
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        logits: Tensor = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = logits.squeeze(3).squeeze(3)

        return logits  # (B, T, C)

    def extract_features(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x: Input video, shape (B, C, T, H, W).
        Returns:
            Features, shape (B, T, C).
        """
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        return self.avg_pool(x)
