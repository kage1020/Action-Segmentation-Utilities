"""Contains a PyTorch definition for Gated Separable 3D network (S3D-G)
with a text module for computing joint text-video embedding from raw text
and video input. The following code will enable you to load the HowTo100M
pretrained S3D Text-Video model from:
  A. Miech, J.-B. Alayrac, L. Smaira, I. Laptev, J. Sivic and A. Zisserman,
  End-to-End Learning of Visual Representations from Uncurated Instructional Videos.
  https://arxiv.org/abs/1912.06430.

S3D-G was proposed by:
  S. Xie, C. Sun, J. Huang, Z. Tu and K. Murphy,
  Rethinking Spatiotemporal Feature Learning For Video Understanding.
  https://arxiv.org/abs/1712.04851.
  Tensorflow code: https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py

The S3D architecture was slightly modified with a space to depth trick for TPU
optimization.
"""

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class SentenceEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int = 2048,
        word_embedding_dim: int = 300,
        num_embeddings: int = 66250,
        max_words: int = 16,
        token_to_word_path: str = "dict.npy",
    ):
        super().__init__()
        self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embedding_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def __zero_pad_tensor_token(self, x: Tensor, size: int) -> Tensor:
        if len(x) >= size:
            return x[:size]
        else:
            zero = torch.zeros(size - len(x)).long()
            return torch.cat((x, zero), dim=0)

    def __split_text(self, sentence) -> list[str]:
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def __words_to_token(self, words: list[str]) -> Tensor:
        words = [
            self.word_to_token[word] for word in words if word in self.word_to_token
        ]
        if words:
            return self.__zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
        else:
            return torch.zeros(self.max_words).long()

    def __words_to_ids(self, x):
        split_x = [self.__words_to_token(self.__split_text(sent.lower())) for sent in x]
        return torch.stack(split_x, dim=0)

    def forward(self, x: list[str]):
        out = self.__words_to_ids(x)
        out = out.to(self.word_embd.weight.device)
        out = self.word_embd(out)
        out = F.relu(self.fc1(out))
        out = torch.max(out, dim=1)[0]
        out = self.fc2(out)
        return {"text_embedding": out}


class SelfGating(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Feature gating as used in S3D-G."""
        spatiotemporal_average = torch.mean(x, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * x


class InceptionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_0_0a_dim: int,
        output_1_0a_dim: int,
        output_1_0b_dim: int,
        output_2_0a_dim: int,
        output_2_0b_dim: int,
        output_3_0b_dim: int,
        gating: bool = True,
    ):
        super().__init__()
        self.conv_b0 = STConv3D(input_dim, output_0_0a_dim, (1, 1, 1))
        self.conv_b1_a = STConv3D(input_dim, output_1_0a_dim, (1, 1, 1))
        self.conv_b1_b = STConv3D(
            output_1_0a_dim,
            output_1_0b_dim,
            (3, 3, 3),
            padding=(1, 1, 1),
            separable=True,
        )
        self.conv_b2_a = STConv3D(input_dim, output_2_0a_dim, (1, 1, 1))
        self.conv_b2_b = STConv3D(
            output_2_0a_dim,
            output_2_0b_dim,
            (3, 3, 3),
            padding=(1, 1, 1),
            separable=True,
        )
        self.maxpool_b3 = nn.MaxPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, output_3_0b_dim, (1, 1, 1))
        self.gating = gating
        self.output_dim = (
            output_0_0a_dim + output_1_0b_dim + output_2_0b_dim + output_3_0b_dim
        )
        if gating:
            self.gating_b0 = SelfGating(output_0_0a_dim)
            self.gating_b1 = SelfGating(output_1_0b_dim)
            self.gating_b2 = SelfGating(output_2_0b_dim)
            self.gating_b3 = SelfGating(output_3_0b_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Inception block"""
        b0 = self.conv_b0(x)
        b1 = self.conv_b1_a(x)
        b1 = self.conv_b1_b(b1)
        b2 = self.conv_b2_a(x)
        b2 = self.conv_b2_b(b2)
        b3 = self.maxpool_b3(x)
        b3 = self.conv_b3_b(b3)
        if self.gating:
            b0 = self.gating_b0(b0)
            b1 = self.gating_b1(b1)
            b2 = self.gating_b2(b2)
            b3 = self.gating_b3(b3)
        return torch.cat((b0, b1, b2, b3), dim=1)


class MaxPool3dTFPadding(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] | None = None,
        padding: str = "SAME",
    ):
        super().__init__()
        if padding == "SAME" and stride is not None:
            padding_shape = self.__get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = nn.ConstantPad3d(padding_shape, 0)
        self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def __get_padding_shape(
        self, filter_shape: tuple[int, int, int], stride: tuple[int, int, int]
    ):
        padding_shape = []
        for filter_dim, stride_val in zip(filter_shape, stride):
            pad_along = max(filter_dim - stride_val, 0)
            pad_top = pad_along // 2
            pad_bottom = pad_along - pad_top
            padding_shape.append(pad_top)
            padding_shape.append(pad_bottom)
        depth_top = padding_shape.pop(0)
        depth_bottom = padding_shape.pop(0)
        padding_shape.append(depth_top)
        padding_shape.append(depth_bottom)
        return tuple(padding_shape)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        out = self.pool(x)
        return out


class STConv3D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
        separable: bool = False,
    ):
        super().__init__()
        assert len(kernel_size) == 3

        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
            temporal_kernel_size = (kernel_size[0], 1, 1)
            spatial_stride = stride
            temporal_stride = (stride[0], 1, 1)
            spatial_padding = (0, padding[1], padding[2])
            temporal_padding = (padding[0], 0, 0)
        if separable:
            self.conv1 = nn.Conv3d(
                input_dim,
                output_dim,
                kernel_size=spatial_kernel_size,
                stride=spatial_stride,
                padding=spatial_padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(
                output_dim,
                output_dim,
                kernel_size=temporal_kernel_size,
                stride=temporal_stride,
                padding=temporal_padding,
                bias=False,
            )
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


class S3D(nn.Module):
    def __init__(
        self,
        num_classes: int = 512,
        gate: bool = True,
        space_to_depth: bool = True,
        dict_path: str = "dict.npy",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gate = gate
        self.space_to_depth = space_to_depth

        if space_to_depth:
            self.conv1 = STConv3D(24, 64, (2, 4, 4), (1, 1, 1), (1, 2, 2), False)
        else:
            self.conv1 = STConv3D(3, 64, (3, 7, 7), (1, 2, 2), (1, 3, 3), False)
        self.conv_2b = STConv3D(64, 64, (1, 1, 1), separable=False)
        self.conv_2c = STConv3D(64, 192, (3, 3, 3), padding=(1, 1, 1), separable=True)
        self.maxpool_2a = MaxPool3dTFPadding((1, 3, 3), (1, 2, 2))
        self.maxpool_3a = MaxPool3dTFPadding((1, 3, 3), (1, 2, 2))
        self.mixed_3b = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = InceptionBlock(
            self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64
        )
        self.maxpool_4a = MaxPool3dTFPadding((3, 3, 3), (2, 2, 2))
        self.mixed_4b = InceptionBlock(
            self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64
        )
        self.mixed_4c = InceptionBlock(
            self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64
        )
        self.mixed_4d = InceptionBlock(
            self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64
        )
        self.mixed_4e = InceptionBlock(
            self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64
        )
        self.mixed_4f = InceptionBlock(
            self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128
        )
        self.maxpool_5a = MaxPool3dTFPadding((2, 2, 2), (2, 2, 2))
        self.maxpool_5a_2x2 = MaxPool3dTFPadding((2, 2, 2), (2, 2, 2))
        self.mixed_5b = InceptionBlock(
            self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128
        )
        self.mixed_5c = InceptionBlock(
            self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128
        )
        self.fc = nn.Linear(self.mixed_5c.output_dim, num_classes)
        self.gating = SelfGating(192)
        self.text_module = SentenceEmbedding(num_classes, token_to_word_path=dict_path)

    def __space_to_depth(self, x: Tensor) -> Tensor:
        """3D space to depth trick for TPU optimization."""
        # B, C, T, H, W = x.shape
        x = rearrange(
            x,
            "b c (t1 t2) (h1 h2) (w1 w2) -> b (t2 h2 w2 c) t1 h1 w1",
            t2=2,
            h2=2,
            w2=2,
        )
        return x

    def forward(self, x: Tensor):
        """Defines the S3DG base architecture."""
        if self.space_to_depth:
            x = self.__space_to_depth(x)
        out = self.conv1(x)
        if self.space_to_depth:
            # we need to replicate 'SAME' tensorflow padding
            out = out[:, :, 1:, 1:, 1:]
        out = self.maxpool_2a(out)
        out = self.conv_2b(out)
        out = self.conv_2c(out)
        if self.gate:
            out = self.gating(out)
        out = self.maxpool_3a(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxpool_4a(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxpool_5a(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = torch.mean(out, dim=[2, 3, 4])
        return {"video_embedding": self.fc(out), "mixed_5c": out}

    def extract_features(self, x: Tensor):
        """Defines the S3DG base architecture."""
        if self.space_to_depth:
            x = self.__space_to_depth(x)
        out = self.conv1(x)
        if self.space_to_depth:
            # we need to replicate 'SAME' tensorflow padding
            out = out[:, :, 1:, 1:, 1:]
        out = self.maxpool_2a(out)
        out = self.conv_2b(out)
        out = self.conv_2c(out)
        if self.gate:
            out = self.gating(out)
        out = self.maxpool_3a(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxpool_4a(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxpool_5a(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = torch.mean(out, dim=[2, 3, 4])

        return out

    def extract_text_features(self, x: list[str]) -> Tensor:
        return self.text_module(x)["text_embedding"]
