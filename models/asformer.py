import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from copy import deepcopy
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        """
        Parameters:
            d_model: dimension of model
            max_len: maximum length of sequence
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = rearrange(pe, "l d -> 1 d l")
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x: (batch size, feature map dimension, number of frames)
        Returns:
            x: (batch size, feature map dimension, number of frames)
        """
        return x + self.pe[:, :, 0 : x.shape[2]]


class ConvFeedForward(nn.Module):
    def __init__(self, dilation: int, in_channels: int, out_channels: int):
        """
        Parameters:
            dilation: dilation rate
            in_channels: number of input channels
            out_channels: number of output channels
        Returns:
            out: (batch size, feature map dimension, number of frames)
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, 3, dilation=dilation, padding=dilation
            ),
            nn.ReLU(),
        )

    def forward(self, x) -> Tensor:
        """
        Parameters:
            x: (batch size, feature map dimension, number of frames)
        Returns:
            x: (batch size, feature map dimension, number of frames)
        """
        x = self.layer(x)
        return x


class FCFeedForward(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters:
            in_channels: number of input channels
            out_channels: number of output channels
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x: (batch size, feature map dimension, number of frames)
        Returns:
            x: (batch size, feature map dimension, number of frames)
        """
        x = self.layer(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        r1: int,
        r2: int,
        r3: int,
        dilation: int,
        att_type: str,
        stage: str,
    ):
        """
        Parameters:
            q_dim: dimension of query
            k_dim: dimension of key
            v_dim: dimension of value
            r1: expansion rate for query
            r2: expansion rate for key
            r3: expansion rate for value
            dilation: dilation rate
            att_type: type of attention, it can be 'encoder' or 'decoder'
            stage: stage of the model, it can be 'normal_att', 'block_att', 'sliding_att'
        """
        super().__init__()
        assert stage in ["encoder", "decoder"]
        assert att_type in ["normal_att", "block_att", "sliding_att"]
        self.query_conv = nn.Conv1d(
            in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1
        )
        self.key_conv = nn.Conv1d(
            in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1
        )
        self.value_conv = nn.Conv1d(
            in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1
        )
        self.conv_out = nn.Conv1d(
            in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dilation = dilation
        self.att_type = att_type
        self.stage = stage

        self.window_mask = self._construct_window_mask()

    def _construct_window_mask(self) -> Tensor:
        """
        Returns:
            window_mask: (1, dilation, dilation + 2 * (dilation // 2))
        """
        window_mask = torch.zeros(
            (1, self.dilation, self.dilation + 2 * (self.dilation // 2))
        )
        for i in range(self.dilation):
            window_mask[:, i, i : i + self.dilation] = 1
        return window_mask

    def _normal_att(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters:
            q: (batch size, feature dimension, number of frames)
            k: (batch size, feature dimension, number of frames)
            v: (batch size, feature dimension, number of frames)
            mask: (batch size, number of action classes, number of frames)
        Returns:
            output: (batch size, feature dimension, number of frames)
        """
        B, _, T = q.size()
        padding_mask = torch.ones((B, 1, T)) * mask[:, 0:1, :]
        output = self.dot(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:T]
        return output * mask[:, 0:1, :]

    def _block_wise_att(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters:
            q: (batch size, feature dimension, number of frames)
            k: (batch size, feature dimension, number of frames)
            v: (batch size, feature dimension, number of frames)
            mask: (batch size, number of action classes, number of frames)
        """
        B, c1, T = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        nb = T // self.dilation
        if T % self.dilation != 0:
            q = torch.cat(
                [q, torch.zeros((B, c1, self.dilation - T % self.dilation))], dim=-1
            )
            k = torch.cat(
                [k, torch.zeros((B, c2, self.dilation - T % self.dilation))], dim=-1
            )
            v = torch.cat(
                [v, torch.zeros((B, c3, self.dilation - T % self.dilation))], dim=-1
            )
            nb += 1

        padding_mask = torch.cat(
            [
                torch.ones((B, 1, T)) * mask[:, 0:1, :],
                torch.zeros((B, 1, self.dilation * nb - T)),
            ],
            dim=-1,
        )

        q = rearrange(q, "b c (n d) -> (b n) c d", n=nb, d=self.dilation)
        padding_mask = rearrange(
            padding_mask, "b 1 (n d) -> (b n) 1 d", n=nb, d=self.dilation
        )
        k = rearrange(k, "b c (n d) -> (b n) c d", n=nb, d=self.dilation)
        v = rearrange(v, "b c (n d) -> (b n) c d", n=nb, d=self.dilation)

        output = self.dot(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))

        output = rearrange(output, "(b n) c d -> b c (n d)", n=nb, d=self.dilation)
        output = output[:, :, 0:T]
        return output * mask[:, 0:1, :]

    def _sliding_window_att(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor
    ) -> Tensor:
        """
        Parameters:
            q: (batch size, feature dimension, number of frames)
            k: (batch size, feature dimension, number of frames)
            v: (batch size, feature dimension, number of frames)
            mask: (batch size, number of action classes, number of frames)
        """
        B, c1, T = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        assert B == 1
        num_block = T // self.dilation
        # padding zeros for the last block
        if T % self.dilation != 0:
            q = torch.cat(
                [
                    q,
                    torch.zeros((B, c1, self.dilation - T % self.dilation)).to(
                        q.device
                    ),
                ],
                dim=-1,
            )
            k = torch.cat(
                [
                    k,
                    torch.zeros((B, c2, self.dilation - T % self.dilation)).to(
                        k.device
                    ),
                ],
                dim=-1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros((B, c3, self.dilation - T % self.dilation)).to(
                        v.device
                    ),
                ],
                dim=-1,
            )
            num_block += 1
        padding_mask = torch.cat(
            [
                torch.ones((B, 1, T)).to(mask.device) * mask[:, 0:1, :],
                torch.zeros((B, 1, self.dilation * num_block - T)).to(mask.device),
            ],
            dim=-1,
        )

        # sliding window approach, by splitting query and key into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = rearrange(q, "b c (n d) -> (b n) c d", n=num_block, d=self.dilation)
        # sliding window approach for key
        # 1. add paddings at the start and end
        k = torch.cat(
            [
                torch.zeros(B, c2, self.dilation // 2).to(k.device),
                k,
                torch.zeros(B, c2, self.dilation // 2).to(k.device),
            ],
            dim=-1,
        )
        v = torch.cat(
            [
                torch.zeros(B, c3, self.dilation // 2).to(v.device),
                v,
                torch.zeros(B, c3, self.dilation // 2).to(v.device),
            ],
            dim=-1,
        )
        padding_mask = torch.cat(
            [
                torch.zeros(B, 1, self.dilation // 2).to(padding_mask.device),
                padding_mask,
                torch.zeros(B, 1, self.dilation // 2).to(padding_mask.device),
            ],
            dim=-1,
        )

        # 2. reshape key_proj of shape (B*num_block, c1, 2*self.dilation)
        k = torch.cat(
            [
                k[
                    :,
                    :,
                    i * self.dilation : (i + 1) * self.dilation
                    + (self.dilation // 2) * 2,
                ]
                for i in range(num_block)
            ],
            dim=0,
        )
        v = torch.cat(
            [
                v[
                    :,
                    :,
                    i * self.dilation : (i + 1) * self.dilation
                    + (self.dilation // 2) * 2,
                ]
                for i in range(num_block)
            ],
            dim=0,
        )
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat(
            [
                padding_mask[
                    :,
                    :,
                    i * self.dilation : (i + 1) * self.dilation
                    + (self.dilation // 2) * 2,
                ]
                for i in range(num_block)
            ],
            dim=0,
        )
        final_mask = (
            self.window_mask.to(padding_mask.device).repeat(B * num_block, 1, 1)
            * padding_mask
        )

        output = self.dot(q, k, v, final_mask)
        output: torch.Tensor = self.conv_out(F.relu(output))

        output = rearrange(
            output, "(b n) c d -> b c (n d)", n=num_block, d=self.dilation
        )
        output = output[:, :, 0:T]
        return output * mask[:, 0:1, :]

    def forward(self, x1: Tensor, x2: Tensor | None, mask: Tensor) -> Tensor:
        """
        Parameters:
            x1: (batch size, feature map dimension, number of frames)
            x2: (batch size, feature map dimension, number of frames) or None
            mask: (batch size, number of action classes, number of frames)
        Returns:
            output: (batch size, feature map dimension, number of frames)
        """
        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == "decoder":
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == "normal_att":
            return self._normal_att(query, key, value, mask)
        elif self.att_type == "block_att":
            return self._block_wise_att(query, key, value, mask)
        elif self.att_type == "sliding_att":
            return self._sliding_window_att(query, key, value, mask)
        else:
            raise ValueError("Invalid attention type")

    def dot(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters:
            q: (feature dimension, feature map dimension, feature dimension)
            k: (feature dimension, feature map dimension, feature dimension)
            v: (feature dimension, feature map dimension, feature dimension)
            mask: (feature dimension, feature dimension, feature dimension)
        Returns:
            output: (feature dimension, feature map dimension, feature dimension)
        """
        _, c1, _ = q.shape
        _, c2, _ = k.shape

        assert c1 == c2

        energy = torch.bmm(q.permute(0, 2, 1), k)
        attention = energy / math.sqrt(c2)
        attention = attention + torch.log(mask + 1e-6)
        attention: torch.Tensor = self.softmax(attention)
        attention = attention * mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(v, attention)
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        r1: int,
        r2: int,
        r3: int,
        dilation: int,
        stage: str,
        att_type: str,
        num_head: int,
    ):
        """
        Parameters:
            q_dim: dimension of query
            k_dim: dimension of key
            v_dim: dimension of value
            r1: expansion rate for query
            r2: expansion rate for key
            r3: expansion rate for value
            dilation: dilation rate
            stage: stage of the model, it can be 'encoder' or 'decoder'
            att_type: type of attention, it can be 'normal_att', 'block_att', 'sliding_att'
            num_head: number of heads
        """
        super().__init__()
        assert v_dim % num_head == 0
        assert stage in ["encoder", "decoder"]
        assert att_type in ["normal_att", "block_att", "sliding_att"]
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [
                deepcopy(
                    AttentionLayer(
                        q_dim, k_dim, v_dim, r1, r2, r3, dilation, stage, att_type
                    )
                )
                for i in range(num_head)
            ]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.dilation = dilation
        self.stage = stage
        self.att_type = att_type

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters:
            x1: (batch size, feature map dimension, number of frames)
            x2: (batch size, feature map dimension, number of frames)
            mask: (batch size, number of action classes, number of frames)
        Returns:
            out: (batch size, feature map dimension, number of frames)
        """
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class AttentionModule(nn.Module):
    def __init__(
        self,
        dilation: int,
        in_channels: int,
        out_channels: int,
        r1: int,
        r2: int,
        att_type: str,
        stage: str,
        alpha: float,
    ):
        """
        Parameters:
            dilation: dilation rate
            in_channels: number of input channels
            out_channels: number of output channels
            r1: expansion rate for query and key
            r2: expansion rate for value
            att_type: type of attention, it can be 'normal_att', 'block_att', 'sliding_att'
            stage: stage of the model, it can be 'encoder' or 'decoder'
            alpha: hyperparameter for residual connection in encoder's attention module
        """
        super().__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttentionLayer(
            in_channels,
            in_channels,
            out_channels,
            r1,
            r1,
            r2,
            dilation,
            att_type,
            stage,
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        self.dilation = dilation
        self.stage = stage

    def forward(self, x: Tensor, f: Tensor | None, mask: Tensor) -> Tensor:
        """
        Parameters:
            x: (batch size, feature map dimension, number of frames)
            f: (batch size, feature map dimension, number of frames) or None
            mask: (batch size, number of action classes, number of frames)
        Returns:
            out: (batch size, number of action classes, number of frames)
        """
        # dilation convolution + ReLU
        # out: (batch size, feature map dimension, number of frames)
        out: Tensor = self.feed_forward(x)

        # residual connection + self-attention with instance normalization
        # out: (batch size, feature map dimension, number of frames)
        out = self.alpha * self.att_layer(self.norm(out), f, mask) + out

        # adjust output dimension
        # out: (batch size, number of action classes, number of frames)
        out = self.conv_1x1(out)

        # dropout
        # out: (batch size, number of action classes, number of frames)
        out = self.dropout(out)

        return (x + out) * mask[:, 0:1, :]


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        r1: int,
        r2: int,
        num_f_maps: int,
        input_dim: int,
        num_classes: int,
        channel_masking_rate: float,
        att_type: str,
        alpha: float,
    ):
        """
        Parameters:
            num_layers: number of layers
            r1: expansion rate for query and key
            r2: expansion rate for value
            num_f_maps: dimension of feature map in each layer
            input_dim: dimension of input
            num_classes: number of action classes
            channel_masking_rate: rate of dropout for channel masking
            att_type: type of attention, it can be 'normal_att', 'block_att', 'sliding_att'
            alpha: hyperparameter for residual connection in encoder's attention module
        Returns:
            out: (batch size, number of action classes, number of frames)
            feature: (batch size, feature map dimension, number of frames)
        """
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.layers = nn.ModuleList(
            [
                AttentionModule(
                    2**i, num_f_maps, num_f_maps, r1, r2, att_type, "encoder", alpha
                )
                for i in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters:
            x: (batch size, feature dimension, number of frames)
            mask: (batch size, number of action classes, number of frames)
        Returns:
            out: (batch size, number of action classes, number of frames)
            feature: (batch size, feature map dimension, number of frames)
        """
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        # adjust input dimension
        # feature: (batch size, feature map dimension, number of frames)
        feature: Tensor = self.conv_1x1(x)

        # encoder block
        # feature: (batch size, feature map dimension, number of frames)
        # outputs: (batch size, number of action classes, number of frames)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        # adjust output dimension
        # out: (batch size, number of action classes, number of frames)
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(
        self,
        index: int,
        num_layers: int,
        r1: int,
        r2: int,
        num_f_maps: int,
        input_dim: int,
        num_classes: int,
        att_type: str,
        alpha: float,
    ):
        """
        Parameters:
            index: index of decoder
            num_layers: number of layers
            r1: expansion rate for query and key
            r2: expansion rate for value
            num_f_maps: dimension of feature map in each layer
            input_dim: dimension of input
            num_classes: number of action classes
            att_type: type of attention, it can be 'normal_att', 'block_att', 'sliding_att'
            alpha: hyperparameter for residual connection in encoder's attention module
        """
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.layers = nn.ModuleList(
            [
                AttentionModule(
                    2**i, num_f_maps, num_f_maps, r1, r2, att_type, "decoder", alpha
                )
                for i in range(num_layers)
            ]
        )
        self.index = index

    def forward(self, x: Tensor, f: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters:
            x: (batch size, number of action classes, number of frames)
            f: (batch size, feature map dimension, number of frames)
            mask: (batch size, number of action classes, number of frames)
        """
        # adjust input dimension
        # feature: (batch size, feature map dimension, number of frames)
        feature: Tensor = self.conv_1x1(x)

        # decoder block
        # feature: (batch size, feature map dimension, number of frames)
        for layer in self.layers:
            feature = layer(feature, f, mask)

        # adjust output dimension
        # out: (batch size, number of action classes, number of frames)
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class ASFormer(nn.Module):
    def __init__(
        self,
        num_decoders: int,
        num_layers: int,
        r1: int,
        r2: int,
        num_f_maps: int,
        input_dim: int,
        num_classes: int,
        channel_masking_rate: float = 0.3,
        att_type: str = "sliding_att",
        alpha: float = 1,
        p: float = 3,
    ):
        """
        Parameters:
            num_decoders: number of decoders
            num_layers: number of layers
            r1: expansion rate for query and key
            r2: expansion rate for value
            num_f_maps: dimension of feature map in each layer
            input_dim: dimension of input
            num_classes: number of action classes
            channel_masking_rate: rate of dropout for channel masking
            att_type: type of attention, it can be 'normal_att', 'block_att', 'sliding_att'
            alpha: hyperparameter for residual connection in encoder's attention module
            p: hyperparameter for exponential decrease of alpha in decoder's attention module
        """
        super().__init__()
        self.encoder = Encoder(
            num_layers,
            r1,
            r2,
            num_f_maps,
            input_dim,
            num_classes,
            channel_masking_rate,
            att_type,
            alpha,
        )
        self.decoders = nn.ModuleList(
            [
                deepcopy(
                    Decoder(
                        i,
                        num_layers,
                        r1,
                        r2,
                        num_f_maps,
                        num_classes,
                        num_classes,
                        att_type=att_type,
                        alpha=math.exp(-p * i),
                    )
                )
                for i in range(num_decoders)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters:
            x: (batch size, feature dimension, number of frames)
            mask: (batch size, number of action classes, number of frames)
        Returns:
            outputs: (1 + num_decoders, batch size, number of action classes, number of frames)
        """
        # encode input
        # out: (batch size, number of action classes, number of frames)
        # feature: (batch size, feature map dimension, number of frames)
        # outputs: (1, batch size, number of action classes, number of frames)
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        # decode encoded input
        # out: (batch size, number of action classes, number of frames)
        # feature: (batch size, feature map dimension, number of frames)
        # outputs: (1 + num_decoders, batch size, number of action classes, number of frames)
        for decoder in self.decoders:
            out = F.softmax(out, dim=1) * mask[:, 0:1, :]
            feature = feature * mask[:, 0:1, :]
            out, feature = decoder(out, feature, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs
