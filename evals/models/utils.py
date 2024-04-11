from __future__ import annotations

import einops as E
import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.vit_mae.modeling_vit_mae import (
    get_2d_sincos_pos_embed_from_grid,
)


def resize_pos_embed(
    pos_embed: torch.Tensor, hw: tuple[int, int], has_cls_token: bool = True
):
    """
    Resize positional embedding for arbitrary image resolution. Resizing is done
    via bicubic interpolation.

    Args:
        pos_embed: Positional embedding tensor of shape ``(n_patches, embed_dim)``.
        hw: Target height and width of the tensor after interpolation.
        has_cls_token: Whether ``pos_embed[0]`` is for the ``[cls]`` token.

    Returns:
        Tensor of shape ``(new_n_patches, embed_dim)`` of resized embedding.
        ``new_n_patches`` is ``new_height * new_width`` if ``has_cls`` is False,
        else ``1 + new_height * new_width``.
    """

    n_grid = pos_embed.shape[0] - 1 if has_cls_token else pos_embed.shape[0]

    # Do not resize if already in same shape.
    if n_grid == hw[0] * hw[1]:
        return pos_embed

    # Get original position embedding and extract ``[cls]`` token.
    if has_cls_token:
        cls_embed, pos_embed = pos_embed[[0]], pos_embed[1:]

    orig_dim = int(pos_embed.shape[0] ** 0.5)

    pos_embed = E.rearrange(pos_embed, "(h w) c -> 1 c h w", h=orig_dim)
    pos_embed = F.interpolate(
        pos_embed, hw, mode="bicubic", align_corners=False, antialias=True
    )
    pos_embed = E.rearrange(pos_embed, "1 c h w -> (h w) c")

    # Add embedding of ``[cls]`` token back after resizing.
    if has_cls_token:
        pos_embed = torch.cat([cls_embed, pos_embed], dim=0)

    return pos_embed


def center_padding(images, patch_size):
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    COPIED FROM TRANSFORMERS PACKAGE AND EDITED TO ALLOW FOR DIFFERENT WIDTH-HEIGHT
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or
        (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output
