# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def convert_filter_out_channels_last(filter, transposed=False):
    if transposed:
        return torch.transpose(torch.transpose(filter, 2, 1), 3, 2).contiguous().clone()
    return torch.transpose(torch.transpose(torch.transpose(filter, 1, 0), 2, 1), 3, 2).contiguous().clone()

def convert_half_filter(x, pixel_wise=False, transposed=False):
    def add_available(a, b):
        c_in = b.shape[0]
        c_out = b.shape[-1]
        a[:c_in,:,:,:c_out].add_(b)
        
    if transposed:
        x = torch.transpose(x, 0, 1)

    c_in = x.shape[1]
    c_out = x.shape[0]

    # align to 64 bit
    c_out_new = c_out + (c_out % 2)
    c_in_new = c_in + (c_in % 2)
    result = torch.zeros((c_in_new, x.shape[-2], x.shape[-1], c_out_new), device=x.device, dtype=torch.half)

    x_out_last = convert_filter_out_channels_last(x)
    if pixel_wise:
        return x_out_last

    add_available(result[::2,:,:,::2], x_out_last[::2,:,:,::2])
    add_available(result[::2,:,:,1::2], x_out_last[1::2,:,:,1::2])
    add_available(result[1::2,:,:,::2], x_out_last[1::2,:,:,::2])
    add_available(result[1::2,:,:,1::2], x_out_last[::2,:,:,1::2])

    return result