# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from deltacnn.cuda import sparse_conv_bias_wrapper_masked, sparse_deconv_bias_wrapper_masked
from deltacnn.cuda import deltacnn_activate_truncate, deltacnn_prepare_diff_mask_wrapper
from deltacnn.cuda import sparse_add_tensors_wrapper, sparse_add_to_dense_tensor_wrapper, sparse_upsample_wrapper, sparse_concatenate_wrapper, sparse_mul_add_wrapper
from deltacnn.cuda import sparse_pooling_wrapper_masked
from deltacnn.cuda import deltacnn_init_performance_metrics, deltacnn_reset_performance_metrics, deltacnn_retrieve_metrics

def sparse_conv(x, filter, mask=None, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, c_out:int=None, create_out_mask=False, sub_tile_sparsity=True, out_mask=None, out_shape=None) -> torch.Tensor:
    # DeltaCNN currently only support zero padding
    pad_mode_int = 0
    
    out_b = x.shape[0]
    if out_shape is not None and len(out_shape) == 2:
        out_h, out_w = out_shape[0], out_shape[1]
    else:
        if len(padding) == 2:
            out_h = int((x.shape[2] + 2 * padding[0] - dilation[0] * (filter.shape[1]-1) - 1) // stride[0] + 1)
            out_w = int((x.shape[3] + 2 * padding[1] - dilation[1] * (filter.shape[2]-1) - 1) // stride[1] + 1)
        elif len(padding) == 4:
            out_h = int((x.shape[2] + padding[0] + padding[1] - dilation[0] * (filter.shape[1]-1) - 1) // stride[0] + 1)
            out_w = int((x.shape[3] + padding[2] + padding[3] - dilation[1] * (filter.shape[2]-1) - 1) // stride[1] + 1)
        else:
            raise "Padding must be iterable of size 2 or 4"
        if type(out_shape) == list:
            out_shape.extend([out_h, out_w])
        
    out_c = filter.shape[3] if c_out is None else c_out

    out = torch.empty((out_b, out_c, out_h, out_w), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    if mask is not None and out_mask is None:
        if create_out_mask:
            out_mask = torch.empty((out_b, 1, out_h, out_w), dtype=torch.int32, device=x.device, memory_format=torch.channels_last)
        else:
            out_mask = None
    sparse_conv_bias_wrapper_masked(x, filter, bias, out, mask, out_mask, stride, padding, dilation, groups, pad_mode_int, sub_tile_sparsity)

    if create_out_mask and mask is not None:
        return out, out_mask
    else:
        return out

def sparse_deconv(x, filter, mask=None, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, c_out:int=None, create_out_mask=False, sub_tile_sparsity=True, out_mask=None, out_shape=None) -> torch.Tensor:
    # DeltaCNN currently only support zero padding
    pad_mode_int = 0
    
    out_b = x.shape[0]
    if out_shape is not None and len(out_shape) == 2:
        out_h, out_w = out_shape[0], out_shape[1]
    else:
        if len(padding) == 2:
            out_h = int((x.shape[2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (filter.shape[1]-1) + 1)
            out_w = int((x.shape[3] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (filter.shape[2]-1) + 1)
        else:
            raise "Padding must be iterable of size 2"
        if type(out_shape) == list:
            out_shape.extend([out_h, out_w])
        
    out_c = filter.shape[3] if c_out is None else c_out

    # TODO remove the contiguous call here and try to work around it
    out = torch.empty((out_b, out_c, out_h, out_w), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    if mask is not None and out_mask is None:
        if create_out_mask:
            out_mask = torch.empty((out_b, 1, out_h, out_w), dtype=torch.int32, device=x.device, memory_format=torch.channels_last)
        else:
            out_mask = None

    sparse_deconv_bias_wrapper_masked(x, filter, bias, out, mask, out_mask, stride, padding, dilation, groups, pad_mode_int, sub_tile_sparsity)

    if create_out_mask and mask is not None:
        return out, out_mask
    else:
        return out

def sparse_pooling(x, prev_x, kernel_size, mask=None, stride=(1,1), padding=(0,0), dilation=(1,1), create_out_mask=True, sub_tile_sparsity=True, out_mask=None, pooling_mode_int=0, out_shape=None) -> torch.Tensor:
    # DeltaCNN currently only support zero padding
    pad_mode_int = 0
    out_b = x.shape[0]
    c = x.shape[1]
    if out_shape is not None and len(out_shape) == 2:
        out_h, out_w = out_shape[0], out_shape[1]
    else:
        if len(padding) == 2:
            out_h = int((x.shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0]-1) - 1) // stride[0] + 1)
            out_w = int((x.shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1]-1) - 1) // stride[1] + 1)
        elif len(padding) == 4:
            out_h = int((x.shape[2] + padding[0] + padding[1] - dilation[0] * (kernel_size[0]-1) - 1) // stride[0] + 1)
            out_w = int((x.shape[3] + padding[2] + padding[3] - dilation[1] * (kernel_size[1]-1) - 1) // stride[1] + 1)
        else:
            raise "Padding must be an iterable of size 2 or 4"
        if type(out_shape) == list:
            out_shape.extend([out_h, out_w])

    dtype_original = x.dtype

    if out_h > 1 or out_w > 1:
        out = torch.empty((out_b, c, out_h, out_w), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    else:
        dtype = torch.float32 if x.dtype == torch.float16 else x.dtype
        if pooling_mode_int == 0:
            out = torch.empty((out_b, c, out_h, out_w), dtype=dtype, device=x.device, memory_format=torch.channels_last)
            out[:] = -1e-10
        else:
            out = torch.zeros((out_b, c, out_h, out_w), dtype=dtype, device=x.device).contiguous(memory_format=torch.channels_last)

    if mask is not None and out_mask is None:
        if create_out_mask:
            if out_h > 1 or out_w > 1:
                out_mask = torch.empty((out_b, 1, out_h, out_w), dtype=torch.int32, device=x.device, memory_format=torch.channels_last)
            else:
                # this operation uses a special kernel which sets the update mask to one atomically. thus, it has to be set zero first
                out_mask = torch.zeros((out_b, 1, out_h, out_w), dtype=torch.int32, device=x.device).contiguous(memory_format=torch.channels_last)
        else:
            out_mask = None

    sparse_pooling_wrapper_masked(x, prev_x, out, mask, out_mask, kernel_size, stride, padding, dilation, pad_mode_int, pooling_mode_int, sub_tile_sparsity)

    if out.dtype != dtype_original:
        out = out.to(dtype=dtype_original)

    if create_out_mask and mask is not None:
        return out, out_mask
    else:
        return out


def sparse_activation(x, prev_x, truncated, mask, threshold, activation, truncation_mode):
    deltacnn_activate_truncate(x, prev_x, truncated, mask, threshold, activation, truncation_mode)


def sparsify(input, prev_in, delta, mask, threshold):
    deltacnn_prepare_diff_mask_wrapper(input, prev_in, delta, mask, threshold)


def sparse_add_tensors(a, b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, activation, dense_out):
    sparse_add_tensors_wrapper(a, b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, activation, dense_out)


def sparse_add_to_dense_tensor(a, b, mask_a, activation=0):
    sparse_add_to_dense_tensor_wrapper(a, b, mask_a, activation)


def sparse_mul_add(x, x_mask, out, out_mask, scale, bias):
    sparse_mul_add_wrapper(x, out, x_mask, out_mask, scale, bias)


def sparse_upsample(input, mask_in, scale, out=None):
    if out is None:
        is_channels_last = input.is_contiguous(memory_format=torch.channels_last)
        if is_channels_last:
            out = torch.empty((input.shape[0], input.shape[1], input.shape[2] * scale, input.shape[3] * scale), device=input.device, dtype=input.dtype, memory_format=torch.channels_last)
        else:
            out = torch.empty((input.shape[0], input.shape[1], input.shape[2] * scale, input.shape[3] * scale), device=input.device, dtype=input.dtype)

        mask_out = torch.empty((input.shape[0], 1, input.shape[2] * scale, input.shape[3] * scale), device=input.device, dtype=torch.int)
    else:
        out, mask_out = out

    sparse_upsample_wrapper(input, out, mask_in, mask_out, scale)

    return out, mask_out


def sparse_concatenate(a, b, out=None):
    a, mask_a = a
    b, mask_b = b
    if out is None:
        is_channels_last = a.is_contiguous(memory_format=torch.channels_last)
        if is_channels_last:
            out = torch.empty((a.shape[0], a.shape[1] + b.shape[1], a.shape[2], a.shape[3]), device=a.device, dtype=a.dtype, memory_format=torch.channels_last)
        else:
            out = torch.empty((a.shape[0], a.shape[1] + b.shape[1], a.shape[2], a.shape[3]), device=a.device, dtype=a.dtype)

        mask_out = torch.empty((a.shape[0], 1, a.shape[2], a.shape[3]), device=a.device, dtype=torch.int)
    else:
        out, mask_out = out

    sparse_concatenate_wrapper(a, b, out, mask_a, mask_b, mask_out)

    return out, mask_out


class DCPerformanceMetrics:
    def __init__(self, tiles, inputs, mode, flops, memtransfer, histogram, n_frames = 1):
        n_frames = max(n_frames, 1)
        self.tiles_active = tiles[0].cpu().item() // n_frames
        self.tiles_total = tiles[1].cpu().item() // n_frames
        self.tiles_ratio = self.tiles_active / max(self.tiles_total, 1)
        self.inputs_active = inputs[0].cpu().item() // n_frames
        self.inputs_total = inputs[1].cpu().item() // n_frames
        self.inputs_ratio = self.inputs_active / max(self.inputs_total, 1)
        self.mode_sparse = mode[0].cpu().item() // n_frames
        self.mode_dense = mode[1].cpu().item() // n_frames
        self.mode_ratio = self.mode_sparse / max((self.mode_sparse + self.mode_dense), 1)
        self.flops_actual = flops[0].cpu().item() // n_frames
        self.flops_theoretical = flops[1].cpu().item() // n_frames
        self.flops_dense = flops[2].cpu().item() // n_frames
        self.flops_ratio_actual = self.flops_actual / max(self.flops_dense, 1)
        self.flops_ratio_theoretical = self.flops_theoretical / max(self.flops_dense, 1)
        self.mem_read_actual = memtransfer[0].cpu().item() // n_frames
        self.mem_read_dense = memtransfer[1].cpu().item() // n_frames
        self.mem_read_ratio = self.mem_read_actual / max(self.mem_read_dense, 1)
        self.mem_write_actual = memtransfer[2].cpu().item() // n_frames
        self.mem_write_dense = memtransfer[3].cpu().item() // n_frames
        self.mem_write_ratio = self.mem_write_actual / max(self.mem_write_dense, 1)
        self.histogram = histogram.cpu().numpy()  // n_frames
        self.histogram_ratio = (histogram / histogram[1:].sum()).cpu().numpy()

    def to_dict(self):
        metrics_dict = {
            "tiles_active": self.tiles_active,
            "tiles_total": self.tiles_total,
            "tiles_ratio": self.tiles_ratio,
            "inputs_active": self.inputs_active,
            "inputs_total": self.inputs_total,
            "inputs_ratio": self.inputs_ratio,
            "mode_sparse": self.mode_sparse,
            "mode_dense": self.mode_dense,
            "mode_ratio": self.mode_ratio,
            "flops_actual": self.flops_actual,
            "flops_theoretical": self.flops_theoretical,
            "flops_dense": self.flops_dense,
            "flops_ratio_actual": self.flops_ratio_actual,
            "flops_ratio_theoretical": self.flops_ratio_theoretical,
            "mem_read_actual": self.mem_read_actual,
            "mem_read_dense": self.mem_read_dense,
            "mem_read_ratio": self.mem_read_ratio,
            "mem_write_actual": self.mem_write_actual,
            "mem_write_dense": self.mem_write_dense,
            "mem_write_ratio": self.mem_write_ratio,
            "histogram": [int(x) for x in self.histogram],
            "histogram_ratio": [float(x) for x in self.histogram_ratio],
        }
        return metrics_dict

    def save_json(self, path):
        import json
        metrics_dict = self.to_dict()
        with open(path, "w+") as f:
            json.dump(metrics_dict, f)

class DCPerformanceMetricsManager:
    _initialized = False
    
    @classmethod
    def init(cls):
        if cls._initialized:
            return True
        cls._initialized = deltacnn_init_performance_metrics()
        return cls._initialized

    @classmethod
    def reset(cls):
        if not cls.init():
            return
        deltacnn_reset_performance_metrics()

    @classmethod
    def get_metrics(cls, n_frames=1):
        if not cls.init():
            return None

        tiles, inputs, mode, flops, memtransfer, histogram = deltacnn_retrieve_metrics()
        metrics = DCPerformanceMetrics(tiles, inputs, mode, flops, memtransfer, histogram, n_frames=n_frames)
        return metrics
