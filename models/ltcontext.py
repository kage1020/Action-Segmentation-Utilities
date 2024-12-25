from functools import partial
import math
import torch
import torch.nn as nn
from torch.nn.init import constant_
import torch.nn.functional as F
from einops import rearrange

from trainer.ltcontext import LTContextConfig

from torch import Tensor


class ScaledDotProduct(nn.Module):
    """
    This code is inspired from xformers lib
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.attn_drop = nn.Dropout(dropout, inplace=False)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Tensor | None,
        position_bias: Tensor | None = None,
        dropout: nn.Module | None = None,
    ) -> tuple[Tensor, Tensor]:
        # TODO assume we have (N, S, hs) instead of (B, nh, S, hs), with N = B x nh
        # this is needed due to limitations in sparse_bmm for now

        # Self-attend: (N, S, hs) x (N, hs, S) -> (N, S, S)
        q = q / math.sqrt(k.size(-1))

        att = q @ k.transpose(-2, -1)
        if att_mask is not None:
            assert att_mask.dtype == torch.bool, "Mask should be boolean"
            if att_mask.ndim == 2:
                att_mask = att_mask.unsqueeze(0).expand(att.shape[0], -1, -1)
            att[~att_mask] = float("-inf")

        if position_bias is not None:
            att += position_bias

        out_mask = torch.ones_like(q)

        if att_mask is not None:
            # to avoid blocks that are all zero (i.e. softmax over rows with all zeros)
            if (att_mask.sum((-2, -1)) == 0).any():
                att[(att_mask.sum((-2, -1)) == 0)] = 0.00001
                out_mask[(att_mask.sum((-2, -1)) == 0), :, :] = 0

        att = torch.softmax(att, dim=att.ndim - 1)
        if dropout is not None:
            #  Optional dropout, could be part of the masking in the future
            att = dropout(att)
        # Get to the predicted values, for all heads
        y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
        y = y * out_mask
        return y, att

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Tensor | None = None,
        position_bias: Tensor | None = None,
        return_attn_matrix: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        :param q:
        :param k:
        :param v:
        :param att_mask:
        :param position_bias:
        :param return_attn_matrix:
        :return:
        """
        # Convenience, create an attention mask if a tensor was passed
        # Attend: (B x nh, S, hs) x (B x nh, hs, S) -> (B x nh, S, S)
        y, att_matrix = self.scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            att_mask=att_mask,
            dropout=self.attn_drop,
            position_bias=position_bias,
        )
        if return_attn_matrix:
            return y, att_matrix
        else:
            return y, None


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention. Based on the Attention is All You Need paper.
    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        use_conv1d_proj: bool,
        requires_input_projection: bool,
        use_separate_proj_weight: bool,
        dim_key: int,
        dim_value: int,
        out_proj: nn.Module | None = None,
        rel_pos_encoder: nn.Module | None = None,
    ):
        super().__init__()
        # Popular default is that all latent dimensions are the same
        dim_key, dim_value = map(lambda x: x if x else model_dim, (dim_key, dim_value))
        self.rel_pos_encoder = rel_pos_encoder
        self.num_heads = num_heads
        self.dim_k = dim_key // self.num_heads
        self.dim_value = dim_value
        self.dim_model = model_dim
        self.attention = ScaledDotProduct(dropout=dropout)
        self.requires_input_projection = requires_input_projection
        self.use_conv1d_proj = use_conv1d_proj
        if use_conv1d_proj:
            LinearProj = partial(nn.Conv1d, bias=bias, kernel_size=1)
        else:
            LinearProj = partial(nn.Linear, bias=bias)
        if use_separate_proj_weight:
            self.proj_q = LinearProj(model_dim, dim_key)
            self.proj_k = LinearProj(model_dim, dim_key)
            self.proj_v = LinearProj(model_dim, dim_value)
        else:
            # sharing weights
            assert dim_key == dim_value, (
                "To share qkv projection "
                "weights dimension of q, k, v should be the same"
            )
            self.proj_q = LinearProj(model_dim, dim_key)
            self.proj_v, self.proj_k = self.proj_q, self.proj_q
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.proj = out_proj if out_proj else LinearProj(dim_value, dim_value)
        if isinstance(self.proj, nn.Linear) and self.proj.bias is not None:
            constant_(self.proj.bias, 0.0)

    def _check(self, t, name):
        if self.use_conv1d_proj:
            d = t.shape[1]
        else:
            d = t.shape[2]
        assert (
            d % self.dim_k == 0
        ), f"the {name} embeddings need to be divisible by the number of heads"

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """
        :param tensor: [batch_size, seq_len, embed_dim]
        :return:
            [batch_size * num_heads, seq_len, embed_dim // num_heads]
        """
        assert len(tensor.shape) == 3, "Invalid shape for splitting heads"

        embed_dim = tensor.shape[2]
        new_embed_dim = embed_dim // self.num_heads

        tensor = rearrange(
            tensor, "b n (h d) -> b n h d", h=self.num_heads, d=new_embed_dim
        )

        # Transpose the matrix and flatten it so the outer-dimension will be the batch-size times number of heads
        tensor = (
            torch.transpose(tensor, 1, 2).flatten(start_dim=0, end_dim=1).contiguous()
        )
        return tensor

    def _combine_heads(self, tensor: Tensor, batch_size: int) -> Tensor:
        """
        :param tensor:  [batch_size * num_heads, seq_len, embed_dim // num_heads]
        :param batch_size:
        :return:
            [batch_size, seq_len, embed_dim]
        """
        assert len(tensor.shape) == 3, "Invalid shape to combine heads"

        tensor = tensor.unflatten(0, (batch_size, self.num_heads))
        tensor = torch.transpose(
            tensor, 1, 2
        )  # -> [batch_size, seq_len, num_heads, embed_dim // num_heads]
        seq_len = tensor.shape[1]
        embed_dim = tensor.shape[-1]

        # the new feature size, if we combine all the heads
        new_embed_dim = self.num_heads * embed_dim
        # Reshape the Tensor to remove the heads dimension and come back to a Rank-3 tensor
        tensor = torch.reshape(
            tensor, (batch_size, seq_len, new_embed_dim)
        ).contiguous()
        return tensor

    def forward(
        self,
        query: torch.Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        att_mask: Tensor | None = None,
    ) -> Tensor:
        """
        :param query: tensor w/ shape [batch_size, channels, n_queries]
        :param key: tensor w/ shape [batch_size, channels, n_keyval]
        :param value: tensor w/ shape [batch_size, channels, n_keyval] ,
                       number of key and values are always the same
        :param att_mask: tensor w/ shape [batch_size, 1, n_keyval]
        :return:
        """
        if key is None:
            key = query
        if value is None:
            value = query

        # Check the dimensions properly
        self._check(query, "query")
        self._check(value, "value")
        self._check(key, "key")

        bs, _, q_len = query.size()  # Batch x Sequence x Embedding (latent)

        # Calculate query, key, values for all heads in batch
        if self.requires_input_projection:
            q, k, v = self.proj_q(query), self.proj_k(key), self.proj_v(value)
        else:
            k, q, v = key, query, value

        if self.use_conv1d_proj:
            q = rearrange(q, "b d n -> b n d")
            k = rearrange(k, "b d n -> b n d")
            v = rearrange(v, "b d n -> b n d")

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        position_bias = None
        if isinstance(self.rel_pos_encoder, nn.Module):
            position_bias = self.rel_pos_encoder(q, k)

        if att_mask is not None:
            att_mask = att_mask.repeat(1, q_len, 1)
            att_mask = att_mask.repeat(self.num_heads, 1, 1)

        z, _ = self.attention(
            q=q,
            k=k,
            v=v,
            att_mask=att_mask,
            return_attn_matrix=True,
            position_bias=position_bias,
        )

        z = self._combine_heads(z, bs)
        if self.use_conv1d_proj:
            z = z.permute(0, 2, 1)

        output = self.dropout(self.proj(z))
        return output


class BaseAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        use_separate_proj_weight: bool,
        requires_input_projection: bool,
    ):
        """
        Base module for attention.
        """
        super().__init__()
        out_proj = nn.Sequential(
            nn.GELU(), nn.Conv1d(model_dim // 2, model_dim, kernel_size=1, bias=True)
        )
        self.attn = MultiHeadAttention(
            model_dim,
            num_heads=num_heads,
            dropout=dropout,
            out_proj=out_proj,
            bias=bias,
            use_conv1d_proj=True,
            use_separate_proj_weight=use_separate_proj_weight,
            requires_input_projection=requires_input_projection,
            dim_key=model_dim // 2,
            dim_value=model_dim // 2,
        )

    def convert_to_patches(
        self,
        x: Tensor,
        patch_size: int,
        masks: Tensor | None = None,
        overlapping_patches: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Reshape a tensor into 1d windows ( 1d patches) of 'window_size' size. It pads the tensor if it is needed.

        :param x: Tensor of shape [batch_size, hidden_dim, seq_len]
        :param patch_size: the size of each patch
        :param masks: Tensor of shape [batch_size, 1, seq_len]
        :param overlapping_patches: whether the patches are overlapped or not
        :return:
            patches: a Tensor of shape [batch_size, hidden_dim, num_patches, patch_size]
            padding_masks: a Tensor of shape [batch_size, 1, num_patches, patch_size]
        """
        _, _, seq_len = x.shape
        if seq_len % patch_size != 0:
            pad_size = patch_size - (seq_len % patch_size)
        else:
            pad_size = 0
        x, padded_masks = self.pad_sequence(x, pad_size, masks)

        if overlapping_patches:
            half_pad = (patch_size // 2, patch_size // 2)
            padded_x, padding_mask = self.pad_sequence(
                x, pad_size=half_pad, masks=padded_masks
            )
            patches = self.patchify(padded_x, 2 * patch_size, stride=patch_size)
            padding_mask = self.patchify(
                padding_mask, 2 * patch_size, stride=patch_size
            )
        else:
            patches = self.patchify(x, patch_size, stride=patch_size)
            padding_mask = self.patchify(padded_masks, patch_size, stride=patch_size)
        return patches, padding_mask

    def pad_sequence(
        self, x: Tensor, pad_size: int | tuple[int, int], masks: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
         Pad the sequence to have a length of as the given size
        :param x: a torch.Tensor of shape [batch_size, feat_dim, seq_len]
        :param pad_size: int or tuple of ints the amount that the sequence need to be padded
        :param masks: batch padding masks which corresponds
            to the padding done during batchification of sequences
        :return:
            padded_x: a torch.Tensor of shape [batch_size, feat_dim, size]
            padding_mask: a torch.Tensor of shape [batch_size, feat_dim, size]
        """
        bs, dim, seq_len = x.shape

        if isinstance(pad_size, int):
            pad_size = (0, pad_size)
        if masks is None:
            masks = torch.ones(bs, 1, seq_len).bool()
            masks = masks.to(x.device)
        if pad_size[-1] <= 0:
            return x, masks
        padded_x = F.pad(x, pad=pad_size, value=0.0)
        padding_mask = F.pad(masks, pad=pad_size, value=False)
        return padded_x, padding_mask

    def patchify(self, ts: Tensor, patch_size: int, stride: int) -> Tensor:
        """
        Convert a tensor into patches (windows) of size 'window_size' with overlap of 'stride'

        :param ts: a Tensor of shape [batch_size, hidden_dim, seq_len]
        :param patch_size: an integer of patch size
        :param stride: an integer of overlap between windows
        :return:
            Tensor of shape
            [batch_size, hidden_dim, num_windows, window_size]
        """
        patches = ts.unfold(2, patch_size, stride)  # [bs, d, nw, patch_size]
        return patches


class WindowedAttention(BaseAttention):
    def __init__(
        self,
        windowed_attn_w: int,
        model_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        use_separate_proj_weight: bool,
        requires_input_projection: bool,
    ):
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            use_separate_proj_weight=use_separate_proj_weight,
            requires_input_projection=requires_input_projection,
        )
        self.windowed_attn_w = windowed_attn_w

    def _reshape(
        self, x: Tensor, overlapping_patches: bool, masks: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param x:
        :param overlapping_patches:
        :param masks:
        :return:
            patches: a Tensor of shape [batch_size * num_patches, model_dim, patch_size]
            attn_mask: a Tensor of shape [batch_size * num_patches, patch_size]
        """
        patches, masks = self.convert_to_patches(
            x, self.windowed_attn_w, masks, overlapping_patches
        )
        # combine num_windows to the batch dimension
        patches = rearrange(patches, "b d n w -> (b n) d w")
        masks = rearrange(masks, "b d n w -> (b n) d w")
        return patches.contiguous(), masks.contiguous()

    def _undo_reshape(
        self, patches: Tensor, batch_size: int, orig_seq_len: int
    ) -> Tensor:
        num_patches = patches.shape[0] // batch_size
        x = rearrange(patches, "(b n) d w -> b d (n w)", b=batch_size, n=num_patches)
        x = x[:, :, :orig_seq_len]
        return x.contiguous()

    def _prep_qkv(
        self, qk: Tensor, v: Tensor | None, masks: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Prepare the query, key and value for attention
        by applying the transform_func.
        :param qk:
        :param v:
        :param masks:
        :return:
        """
        q, k = qk, qk
        q, _ = self._reshape(q, overlapping_patches=False)
        k, attn_mask = self._reshape(k, overlapping_patches=True, masks=masks)
        if v is not None:
            v, _ = self._reshape(v, overlapping_patches=True)
        else:
            v = k

        return q, k, v, attn_mask

    def forward(self, qk: Tensor, v: Tensor | None = None, masks: Tensor | None = None):
        """
        :param qk: Tensor with shape [batch_size, model_dim, seq_len]
        :param v: Tensor with shape [batch_size, model_dim, seq_len]
        :param masks: Tensor with shape [batch_size, 1, seq_len]
        :return:
        """
        batch_size, _, seq_len = qk.shape
        q, k, v, att_mask = self._prep_qkv(qk=qk, v=v, masks=masks)
        windowed_attn = self.attn(query=q, key=k, value=v, att_mask=att_mask)
        out = self._undo_reshape(windowed_attn, batch_size, seq_len)
        if masks is not None:
            out = out * masks
        return out


class LTContextAttention(BaseAttention):
    def __init__(
        self,
        long_term_attn_g: int,
        model_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        use_separate_proj_weight: bool,
        requires_input_projection: bool,
    ):
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            use_separate_proj_weight=use_separate_proj_weight,
            requires_input_projection=requires_input_projection,
        )
        self.long_term_attn_g = long_term_attn_g

    def _reshape(self, x: Tensor, masks: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        :param x:
        :param masks:
        :return:
            lt_patches: a Tensor of shape [batch_size * patch_size, model_dim, num_patches]
            attn_mask: a Tensor of shape [batch_size * patch_size, 1, num_patches]
        """
        patches, masks = self.convert_to_patches(x, self.long_term_attn_g, masks)

        # transpose and combine patch_size to the batch dimension which means tokens are taken with stride 'patch_size'
        lt_patches = rearrange(patches, "b d n w -> (b w) d n")
        masks = rearrange(masks, "b d n w -> (b w) d n")
        return lt_patches.contiguous(), masks.contiguous()

    def _undo_reshape(self, lt_patches, batch_size, orig_seq_len) -> Tensor:
        x = rearrange(
            lt_patches, "(b w) d n -> b d (w n)", b=batch_size, w=self.long_term_attn_g
        )
        x = x[:, :, :orig_seq_len]
        return x.contiguous()

    def _prep_qkv(
        self, qk: Tensor, v: Tensor | None, masks: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Prepare the query, key and value for attention by applying the transform_func.
        :param qk:
        :param v:
        :param masks:
        :return:
        """
        q, k = qk, qk
        q, _ = self._reshape(q)
        k, attn_mask = self._reshape(k, masks=masks)
        if v is not None:
            v, _ = self._reshape(v)
        else:
            v = k

        return q, k, v, attn_mask

    def forward(self, qk: Tensor, v: Tensor | None = None, masks: Tensor | None = None):
        """
        :param qk: Tensor with shape [batch_size, model_dim, seq_len]
        :param v: Tensor with shape [batch_size, model_dim, seq_len]
        :param masks: Tensor with shape [batch_size, 1, seq_len]
        :return:
        """
        batch_size, _, seq_len = qk.shape
        q, k, v, att_mask = self._prep_qkv(qk=qk, v=v, masks=masks)
        lt_attn = self.attn(query=q, key=k, value=v, att_mask=att_mask)
        out = self._undo_reshape(lt_attn, batch_size, seq_len)
        if masks is not None:
            out = out * masks
        return out


class DilatedConv(nn.Module):
    def __init__(self, n_channels: int, dilation: int, kernel_size: int):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=dilation,
            dilation=dilation,
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor, masks: Tensor) -> Tensor:
        """
        :param x:
        :param masks:
        :return:
        """
        return self.activation(self.dilated_conv(x)) * masks


class LTCBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dilation: int,
        windowed_attn_w: int,
        long_term_attn_g: int,
        bias: bool,
        use_instance_norm: bool,
        use_separate_proj_weight: bool,
        requires_input_projection: bool,
        dropout_prob: float,
        attention_dropout_prob: float,
    ):
        super().__init__()
        self.dilated_conv = DilatedConv(
            n_channels=model_dim, dilation=dilation, kernel_size=3
        )
        if use_instance_norm:
            self.instance_norm = nn.Identity()
        else:
            self.instance_norm = nn.InstanceNorm1d(model_dim)
        self.windowed_attn = WindowedAttention(
            windowed_attn_w=windowed_attn_w,
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=attention_dropout_prob,
            bias=bias,
            use_separate_proj_weight=use_separate_proj_weight,
            requires_input_projection=requires_input_projection,
        )
        self.ltc_attn = LTContextAttention(
            long_term_attn_g=long_term_attn_g,
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=attention_dropout_prob,
            bias=bias,
            use_separate_proj_weight=use_separate_proj_weight,
            requires_input_projection=requires_input_projection,
        )
        self.out_linear = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self, inputs: Tensor, masks: Tensor, prev_stage_feat: Tensor | None = None
    ) -> Tensor:
        """
        :param inputs:
        :param masks:
        :param prev_stage_feat:
        :return:
        """
        out = self.dilated_conv(inputs, masks)
        out = self.windowed_attn(self.instance_norm(out), prev_stage_feat, masks) + out
        out = self.ltc_attn(self.instance_norm(out), prev_stage_feat, masks) + out
        out = self.out_linear(out)
        out = self.dropout(out)
        out = out + inputs
        return out * masks


class LTCModule(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        num_heads: int,
        input_dim: int,
        model_dim: int,
        dilation_factor: int,
        windowed_attn_w: int,
        long_term_attn_g: int,
        bias: bool,
        use_instance_norm: bool,
        use_separate_proj_weight: bool,
        requires_input_projection: bool,
        dropout_prob: float,
        channel_dropout_prob: float,
        attention_dropout_prob: float,
    ):
        super().__init__()
        self.channel_dropout = nn.Dropout1d(channel_dropout_prob)
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1, bias=True)
        self.layers = nn.ModuleList(
            [
                LTCBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    dilation=dilation_factor**i,
                    windowed_attn_w=windowed_attn_w,
                    long_term_attn_g=long_term_attn_g,
                    bias=bias,
                    use_instance_norm=use_instance_norm,
                    use_separate_proj_weight=use_separate_proj_weight,
                    requires_input_projection=requires_input_projection,
                    dropout_prob=dropout_prob,
                    attention_dropout_prob=attention_dropout_prob,
                )
                for i in range(num_layers)
            ]
        )
        self.out_proj = nn.Conv1d(model_dim, num_classes, kernel_size=1, bias=True)

    def forward(
        self, inputs: Tensor, masks: Tensor, prev_stage_feat: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        :inputs: _description_
        :masks: _description_
        :prev_stage_feat (Tensor | None, optional): _description_. Defaults to None.
        """
        inputs = self.channel_dropout(inputs)
        feature = self.input_proj(inputs)
        for layer in self.layers:
            feature = layer(feature, masks, prev_stage_feat)
        out = self.out_proj(feature) * masks
        return out, feature


class LTC(nn.Module):
    def __init__(self, cfg: LTContextConfig):
        super().__init__()
        self.stage1 = LTCModule(
            num_layers=cfg.LTC.num_layers,
            num_classes=cfg.dataset.num_classes,
            num_heads=cfg.ATTENTION.num_attn_heads,
            input_dim=cfg.dataset.input_dim,
            model_dim=cfg.LTC.model_dim,
            dilation_factor=cfg.LTC.conv_dilation_factor,
            windowed_attn_w=cfg.LTC.windowed_attn_w,
            long_term_attn_g=cfg.LTC.long_term_attn_g,
            bias=True,
            use_instance_norm=cfg.LTC.use_instance_norm,
            use_separate_proj_weight=True,
            requires_input_projection=True,
            dropout_prob=cfg.LTC.dropout_prob,
            channel_dropout_prob=cfg.LTC.channel_masking_prob,
            attention_dropout_prob=cfg.ATTENTION.dropout,
        )
        self.dim_reduction = nn.Conv1d(
            in_channels=int(cfg.LTC.model_dim),
            out_channels=int(cfg.LTC.model_dim // cfg.LTC.dim_reduction),
            kernel_size=1,
            bias=True,
        )
        self.stages = nn.ModuleList(
            [
                LTCModule(
                    num_layers=cfg.LTC.num_layers,
                    num_classes=cfg.dataset.num_classes,
                    num_heads=cfg.ATTENTION.num_attn_heads,
                    input_dim=cfg.dataset.num_classes,
                    model_dim=int(cfg.LTC.model_dim // cfg.LTC.dim_reduction),
                    dilation_factor=cfg.LTC.conv_dilation_factor,
                    windowed_attn_w=cfg.LTC.windowed_attn_w,
                    long_term_attn_g=cfg.LTC.long_term_attn_g,
                    bias=True,
                    use_separate_proj_weight=True,
                    requires_input_projection=True,
                    use_instance_norm=cfg.LTC.use_instance_norm,
                    dropout_prob=cfg.LTC.dropout_prob,
                    channel_dropout_prob=cfg.LTC.channel_masking_prob,
                    attention_dropout_prob=cfg.ATTENTION.dropout,
                )
                for _ in range(cfg.LTC.num_stages)
            ]
        )

    def forward(self, inputs: Tensor, masks: Tensor) -> Tensor:
        """
        :param inputs: Tensor with shape [batch_size, input_dim_size, sequence_length]
        :param masks: Tensor with shape [batch_size, sequence_length, 1]
        :return: outputs: Tensor with shape [batch_size, num_classes, sequence_length]
        """
        out, feature = self.stage1(inputs, masks)
        output_list = [out]
        feature = self.dim_reduction(feature)
        for stage in self.stages:
            out, feature = stage(
                F.softmax(out, dim=1) * masks,
                prev_stage_feat=feature * masks,
                masks=masks,
            )
            output_list.append(out)
        logits = torch.stack(output_list, dim=0)
        return logits
