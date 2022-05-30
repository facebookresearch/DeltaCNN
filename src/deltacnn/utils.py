# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.functional import conv2d


def tile(a, dim, n_tile):
    if type(dim) != torch.Tensor:
        dim = torch.tensor(dim)
    if type(n_tile) != torch.Tensor:
        n_tile = torch.tensor(n_tile)
    return torch.repeat_interleave(a, n_tile.to(a.device), dim.to(a.device))


def scale_conv_mask(inactive_mask, sparse_dilation, kernel_size, padding, stride, bias, tile_size=-1, tile_thresh=0.0, dilation=1):
    if sparse_dilation is None:
        mask = inactive_mask
        padding = (kernel_size - 1 - 2 * padding) // 2
        if padding > 0:
            mask = mask[:, :, padding:-padding, padding:-padding]
        if type(stride) == tuple:
            mask = mask[:, :, ::stride[0], ::stride[1]]
        else:
            mask = mask[:, :, ::stride, ::stride]
    elif sparse_dilation == "natural":
        float_mask = torch.zeros_like(inactive_mask[0:1, 0:1], dtype=torch.float)
        float_mask[inactive_mask[0:1, 0:1]] += 1.0
        filter = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float, device=inactive_mask.device)
        conv_mask = tile(conv2d(float_mask, filter, bias=None, dilation=dilation, stride=stride, padding=(padding, padding)), dim=1,
                         n_tile=inactive_mask.shape[1])
        mask = conv_mask > 0.0
    elif sparse_dilation == "tile":
        mask = scale_conv_mask(inactive_mask, None, kernel_size, padding, stride, bias, tile_thresh=tile_thresh)

        if tile_size < 0:
            tile_size = mask.shape[2] // (-tile_size)

        shape = mask.shape
        if (shape[-1] % tile_size + shape[-2] % tile_size) != 0:
            return scale_conv_mask(inactive_mask, sparse_dilation, kernel_size, padding, stride, bias, tile_size // 2, tile_thresh=tile_thresh)

        if tile_thresh <= 0.0:
            mask = mask.reshape(*mask.shape[:3], -1, tile_size)
            mask = torch.max(mask, dim=-1, keepdim=False)[0]
            mask = torch.transpose(mask, dim0=2, dim1=-1)
            mask = mask.reshape(*mask.shape[:3], -1, tile_size)
            mask = torch.max(mask, dim=-1, keepdim=False)[0]
            mask = torch.transpose(mask, dim0=2, dim1=-1)[:, :, :, None, :, None]
        else:
            mask = mask.float()
            mask = mask.reshape(*mask.shape[:3], -1, tile_size)
            mask = torch.sum(mask, dim=-1, keepdim=False)
            mask = torch.transpose(mask, dim0=2, dim1=-1)
            mask = mask.reshape(*mask.shape[:3], -1, tile_size)
            mask = torch.sum(mask, dim=-1, keepdim=False)
            mask = torch.transpose(mask, dim0=2, dim1=-1)[:, :, :, None, :, None]
            mask = mask > (tile_thresh * tile_size * tile_size)

        mask = tile(mask, dim=-3, n_tile=tile_size)
        mask = tile(mask, dim=-1, n_tile=tile_size)
        mask = mask.reshape(shape)
    elif sparse_dilation == "natural-tile":
        mask = scale_conv_mask(inactive_mask, "natural", kernel_size, padding, 1, bias, tile_thresh=tile_thresh)
        mask = scale_conv_mask(mask, "tile", kernel_size, padding, stride, bias, tile_thresh=tile_thresh)
    else:
        raise Exception(f"dilation mode {sparse_dilation} unknown")

    return mask

def inpaint_masked(t, mask, neighborhood=4, steps=1):
    assert neighborhood == 4 or neighborhood == 8
    in_mask_dilated = mask.clone()
    in_mask_dilated_float = in_mask_dilated.float()
    out = t.clone()
    out[mask] = 0.0
    for i in range(steps):
        neighbors = torch.zeros_like(out)
        neighbor_count = torch.zeros_like(out)
        neighbor_count[:, :, 1:, :] += (~in_mask_dilated_float[:, :, :-1, :])
        neighbors[:, :, 1:, :] += out[:, :, :-1, :]
        neighbor_count[:, :, :-1, :] += (~in_mask_dilated_float[:, :, 1:, :])
        neighbors[:, :, :-1, :] += out[:, :, 1:, :]
        neighbor_count[:, :, :, 1:] += (~in_mask_dilated_float[:, :, :, :-1])
        neighbors[:, :, :, 1:] += out[:, :, :, :-1]
        neighbor_count[:, :, :, :-1] += (~in_mask_dilated_float[:, :, :, 1:])
        neighbors[:, :, :, :-1] += out[:, :, :, 1:]

        if neighborhood == 8:
            neighbor_count[:, :, 1:, 1:] += (~in_mask_dilated_float[:, :, :-1, :-1])
            neighbors[:, :, 1:, 1:] += out[:, :, :-1, :-1]
            neighbor_count[:, :, :-1, 1:] += (~in_mask_dilated_float[:, :, 1:, :-1])
            neighbors[:, :, :-1, 1:] += out[:, :, 1:, :-1]
            neighbor_count[:, :, 1:, :-1] += (~in_mask_dilated_float[:, :, :-1, 1:])
            neighbors[:, :, 1:, :-1] += out[:, :, -1:, 1:]
            neighbor_count[:, :, :-1, :-1] += (~in_mask_dilated_float[:, :, 1:, 1:])
            neighbors[:, :, :-1, :-1] += out[:, :, 1:, 1:]

        neighbor_count[neighbor_count == 0] = 100000
        neighbors /= neighbor_count
        neighbor_count[neighbor_count == 100000] = 0
        out[in_mask_dilated] = neighbors[in_mask_dilated]
        in_mask_dilated[neighbor_count > 0] = False
        in_mask_dilated_float = in_mask_dilated.float()

    return out


def inpaint_binary_mask(mask, pixels=1):
    fmask = mask.float()
    filter = torch.ones((1, 1, pixels * 2 + 1, pixels * 2 + 1), device=mask.device, dtype=torch.float)
    inpainted_fmask = torch.conv2d(fmask[:, :1], filter, padding=pixels)
    inpainted_bmask = inpainted_fmask > 0

    if mask.shape[1] != inpainted_bmask.shape[1]:
        inpainted_bmask = tile(inpainted_bmask, dim=1, n_tile=mask.shape[1])

    return inpainted_bmask


def count_active_neighbors(mask, kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1), padding=(0, 0), exclude_self=False):
    float_mask = mask.float()
    kx, ky = kernel_size
    filter = torch.ones((1, 1, ky, kx), device=mask.device)
    if exclude_self:
        filter[:, :, (ky - 1) // 2, (ky - 1) // 2] = 0

    if len(padding) == 4:
        float_mask = torch.nn.functional.pad(float_mask, padding)
        padding = 0

    return torch.round(torch.conv2d(float_mask, filter, None, stride, padding, dilation)).to(torch.int32)


def count_inactive_neighbors(mask, kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1), padding=(0, 0), exclude_self=False):
    return count_active_neighbors(~mask, kernel_size, dilation, stride, padding, exclude_self)
