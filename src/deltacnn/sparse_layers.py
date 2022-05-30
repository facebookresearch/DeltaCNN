# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_2_t
from collections import OrderedDict
from enum import Enum

from deltacnn.cuda_kernels import sparse_concatenate, sparse_conv, sparse_deconv, \
    sparse_activation, sparsify, sparse_add_tensors, sparse_upsample, sparse_pooling, sparse_add_to_dense_tensor, sparse_mul_add
from deltacnn.filter_conversion import convert_filter_out_channels_last, convert_half_filter

from .logging_layers import ComputationsLogger, InputLogger, PrevInputLogger, MultiplicationsLogger, OutputLogger


def convert_activation_string(name, **kwargs):
    if name is None or name == "" or name.lower() == "linear":
        return None, -1
    if name.lower() == "relu":
        return torch.nn.ReLU(), 1
    elif name.lower() == "relu6":
        return torch.nn.ReLU6(), 2
    elif name.lower() == "leaky":
        return torch.nn.LeakyReLU(kwargs.get('leaky_relu_negative_slope', 1e-2)), 3
    elif name.lower() == "sigmoid":
        return torch.nn.Sigmoid(), 4
    elif name.lower() == "swish":
        return (lambda x: torch.sigmoid(x).mul_(x)), 5
    else:
        raise Exception(f"Activation {name} not implemented")


def differentiable_threshold(input_activated, input, truncated, prev_input, threshold, mode="sigmoid_norm"):
    abs_active_input = input_activated.abs()
    _in_act = input_activated.clone()
    _in = input.clone()
    _trun = truncated.clone()
    _prev_in = prev_input.clone()

    if mode == "clamp":
        abs_active_input -= threshold

        mask = torch.max(abs_active_input, dim=1, keepdim=True)[0] > 0
        tiled_mask = torch.repeat_interleave(mask, repeats=input.shape[1], dim=1)

        _prev_in[tiled_mask] += input[tiled_mask] + truncated[tiled_mask]
        _trun[tiled_mask] = 0.0
        _trun[~tiled_mask] += input[~tiled_mask]
        _in_act[~tiled_mask] = 0.0
        return _in_act, _in, _trun, _prev_in, mask.int()

    elif mode == "sigmoid":
        soft_mask = abs_active_input - threshold
        soft_mask = torch.max(soft_mask, dim=1, keepdim=True)[0]
        soft_mask *= DCConv2d.delta_scale_sigmoid
        soft_mask = torch.sigmoid(soft_mask)
        soft_mask = torch.repeat_interleave(soft_mask, repeats=input.shape[1], dim=1)
        inv_soft_mask = 1.0 - soft_mask

        _prev_in += (_trun + _in) * soft_mask
        _trun *= inv_soft_mask
        _trun += _in * inv_soft_mask
        _in_act *= soft_mask
        return _in_act, _in, _trun, _prev_in, soft_mask

    elif mode == "sigmoid_norm":
        soft_mask = torch.norm(abs_active_input, dim=1, keepdim=True)
        soft_mask = soft_mask - threshold
        soft_mask *= DCConv2d.delta_scale_sigmoid
        soft_mask = torch.sigmoid(soft_mask)
        soft_mask = torch.repeat_interleave(soft_mask, repeats=input.shape[1], dim=1)
        inv_soft_mask = 1.0 - soft_mask

        _prev_in += (_trun + _in) * soft_mask
        _trun *= inv_soft_mask
        _trun += _in * inv_soft_mask
        _in_act *= soft_mask
        return _in_act, _in, _trun, _prev_in, soft_mask


class DCBackend(Enum):
    cudnn = 0
    deltacnn = 1
    delta_cudnn = 2

    @classmethod
    def parse_string(cls, x):
        x = x.lower()
        if x == "cudnn":
            return DCBackend.cudnn
        if x == "deltacnn":
            return DCBackend.deltacnn
        if x == "delta_cudnn":
            return DCBackend.delta_cudnn
        raise Exception(f"invalid backend {x}")


class DCTruncation(Enum):
    none = -1
    max = 0
    norm = 1
    rms = 2

    @classmethod
    def parse_string(cls, x):
        x = x.lower()
        if x == "none":
            return DCTruncation.none
        if x == "max":
            return DCTruncation.max
        if x == "norm":
            return DCTruncation.norm
        if x == "rms":
            return DCTruncation.rms
        raise Exception(f"invalid truncation mode {x}")


class DCThreshold:
    t_default = 0.0
    t = OrderedDict()
    path = None
    initialized = False
    is_parameterized = False
    t_parameters: torch.nn.ParameterDict = None
    __t_idx = {}
    scale = 1.0
    train_only_first_n = -1

    @classmethod
    def get(cls, key, default=0.0):
        if key not in cls.t:
            return default

        if not cls.is_parameterized:
            return cls.t[key]

        _key = cls.__t_idx[key]
        if cls.train_only_first_n > 0 and int(_key) >= cls.train_only_first_n:
            return cls.t[key]

        return cls.t_parameters[_key]

    @classmethod
    def set(cls, key, val):
        if key in cls.t:
            return

        cls.t[key] = val
        if cls.is_parameterized:
            _key = str(len(cls.__t_idx))
            cls.__t_idx[key] = _key
            cls.t_parameters[_key] = val

    @classmethod
    def load_thresholds(cls):
        if cls.initialized:
            return
        cls.initialized = True
        if cls.path is None:
            return

        import json
        with open(cls.path) as f:
            thresholds = json.load(f)

            if len(thresholds) != len(cls.t):
                print(f"WARNING: thresholds file contains different number of thresholds. file: {len(thresholds)} expected {len(cls.t)}")

            skip_thresholds = len(cls.t) - len(thresholds)
            for i, key in enumerate(cls.t.keys()):
                if i < skip_thresholds:
                    continue
                cls.t[key] = thresholds[str(i - skip_thresholds)] * cls.scale

    @classmethod
    def save_thresholds(cls, name):
        res = {}
        for i, val in enumerate(cls.t.values()):
            res[i] = val
        import json
        with open(f"thresholds_{name}.json", "w+") as f:
            json.dump(res, f)

    @classmethod
    def reset(cls):
        cls.t.clear()
        cls.initialized = False
        cls.path = None
        cls.is_parameterized = False
        cls.__t_idx = {}
        cls.t_parameters = None

    @classmethod
    def create_trainable_parameters(cls, device):
        t = torch.nn.ParameterDict()
        for i, key in enumerate(cls.t.keys()):
            _key = str(i)
            cls.__t_idx[key] = _key
            if cls.train_only_first_n > 0 and i >= cls.train_only_first_n:
                continue
            t[_key] = torch.nn.Parameter(torch.tensor(cls.t[key], device=device))

        cls.t_parameters = t
        cls.is_parameterized = True


class DCModule(nn.Module):
    temp_buffers = []

    def __init__(self):
        super().__init__()

    @classmethod
    def get_temp_buffer_numel(cls):
        return sum([x.numel() for x in cls.temp_buffers])

    def reset(self):
        DCModule.temp_buffers = []

    def process_filters(self):
        modules = list(self.modules())

        for mod in modules:
            if type(mod) in [DCConv2d, DCConvTranspose2d]:
                mod.process_filters()

    def reset_layers(self):
        def reset_recursive(mod):
            for module in mod.modules():
                if module == mod or module in checked_modules:
                    continue
                checked_modules.append(module)
                if issubclass(type(module), DCModule):
                    module.reset()
                reset_recursive(module)

        checked_modules = [self]
        reset_recursive(self)
        DCModule.temp_buffers.clear()


def to_tuple(*args) -> _size_2_t:
    result = []
    for x in args:
        if type(x) == int:
            result.append((x, x))
        elif len(x) == 1:
            result.append((x[0], x[0]))
        else:
            result.append(x)

    if len(result) == 1:
        return result[0]
    return result


class DCConv2d(nn.Conv2d, DCModule):
    backend = DCBackend.deltacnn
    conv_idx = -1
    out_masks = []
    use_logging = False
    n_sparse_inputs = 0
    n_dense_inputs = 0
    n_sparse_output = 0
    n_dense_output = 0
    store_out_masks = False
    delta_scale_sigmoid = 100
    flops_sum = 0
    measure_flops = False

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            delta_threshold: float = None,
            name: str = None,
            activation: str = None,
            dense_out=False,
            backend=None,
            use_logging=None,
            **kwargs
    ):
        kernel_size, stride, padding, dilation = to_tuple(kernel_size, stride, padding, dilation)

        super(DCConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.prev_in = None
        self.prev_out = None
        self.delta_threshold = delta_threshold if delta_threshold is not None else DCThreshold.t_default
        self.acc_err = None
        self.name = name
        self.activation = None
        self.activation_int = 0
        self.backend = backend if backend is not None else DCConv2d.backend
        self.activation, self.activation_int = convert_activation_string(activation, **kwargs)

        self.dense_out = dense_out
        self.use_logging = use_logging

        self.densify_layer: DCDensify = None
        self.activation_layer: DCActivation = None
        if self.dense_out:
            self.densify_layer = DCDensify(activation=activation, name=self.name)
        elif self.activation is not None:
            self.activation_layer = DCActivation(self.name, delta_threshold=delta_threshold, activation=activation)
        elif self.logging_enabled():
            self.activation_layer = DCActivation(self.name, inplace=False, delta_threshold=-1, activation=None, truncation_mode=DCTruncation.none)

        if self.logging_enabled():
            self.computation_logger = ComputationsLogger(name=self.name)
            self.multiplications_logger = MultiplicationsLogger(name=self.name)
            self.input_logger = InputLogger(name=self.name)
            self.prev_in_logger = PrevInputLogger(name=self.name)
            self.output_logger = OutputLogger(name=self.name)

        self.conv_idx = -1
        self.dense_in = False
        self.mask = None
        self.frame_idx = -1
        self.out_truncated = None
        self.out_mask = None
        self.out_shape = []
        self.kwargs = kwargs
        self.in_shape = None
        assert kwargs.get('leaky_relu_negative_slope', 0.1) == 0.1, "Leaky ReLU implementation uses hard coded negative slope of 0.1"

    def logging_enabled(self):
        return self.use_logging if self.use_logging is not None else DCConv2d.use_logging

    def init_first_iter(self, input):
        self.conv_idx = DCConv2d.conv_idx
        self.dense_in = type(input) == Tensor
        self.in_shape = input.shape if type(input) == Tensor else input[0].shape

        if self.logging_enabled() and not self.computation_logger.added_id:
            conv_info = f"k={self.kernel_size[0]}x{self.kernel_size[1]} s={self.stride[0]}x{self.stride[1]}"
            conv_info += f" d={self.dilation[0]}x{self.dilation[1]}"
            if len(self.padding) == 4:
                conv_info += f" p={self.padding[0]}x{self.padding[1]}x{self.padding[2]}x{self.padding[3]}"
            else:
                conv_info += f" p={self.padding[0]}x{self.padding[1]}"
            conv_info += f"{' dw' if self.groups == self.in_channels == self.out_channels else ''}"
            self.computation_logger.name = f"{DCConv2d.conv_idx} {self.computation_logger.name} {conv_info}"
            self.multiplications_logger.name = f"{DCConv2d.conv_idx} {self.multiplications_logger.name} {conv_info}"
            self.input_logger.name = f"{DCConv2d.conv_idx} {self.input_logger.name} {conv_info}"
            self.prev_in_logger.name = f"{DCConv2d.conv_idx} {self.prev_in_logger.name} {conv_info}"
            self.output_logger.name = f"{DCConv2d.conv_idx} {self.output_logger.name} {conv_info}"
            self.computation_logger.added_id = True
            self.multiplications_logger.added_id = True
            self.input_logger.added_id = True
            self.prev_in_logger.added_id = True

        # TODO fix tuned thresholds so that we can remove self.dense_out here. does not make sense to use at all
        if (self.activation is not None or self.dense_out or (self.dense_in and not (self.backend == DCBackend.delta_cudnn))) \
                and not self.backend == DCBackend.cudnn:
            DCThreshold.set(self, self.delta_threshold)

        if self.dense_in:
            DCConv2d.n_dense_inputs += 1
        else:
            DCConv2d.n_sparse_inputs += 1
        if self.dense_out:
            DCConv2d.n_dense_output += 1
        else:
            DCConv2d.n_sparse_output += 1

        # print(f"k={self.kernel_size[0]}x{self.kernel_size[1]}, c={self.in_channels}x{self.out_channels} s={self.stride[0]}x{self.stride[1]} g={self.groups}")

    def __repr__(self):
        return f"MyConv2d{self.conv_idx} cin={self.in_channels} cout={self.out_channels}"

    def get_delta_and_mask(self, input, first_iter, dense=False, set_zero=False):
        if first_iter:
            x = input
            self.stores_prev_in = True
            mask = torch.ones_like(x[:, :1], dtype=torch.int)
            self.prev_in = x.clone()
            DCModule.temp_buffers.append(self.prev_in)
        else:
            if self.conv_idx == 0:
                # dilate mask
                x = input - self.prev_in
                mask = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
                pool_size = 15
                threshold = DCThreshold.get(self)

                mask = torch.max_pool2d(mask, (pool_size, pool_size), stride=(1, 1),
                                        padding=((pool_size - 1) // 2, (pool_size - 1) // 2)) > threshold

                tiled_mask = torch.repeat_interleave(mask, dim=1, repeats=x.shape[1])

                self.prev_in[tiled_mask] = input[tiled_mask]
                mask = mask.int()
                if dense and set_zero:
                    x[~tiled_mask] = 0.0
            else:
                if dense and set_zero:
                    x = torch.zeros_like(input)
                else:
                    x = torch.empty_like(input)

                mask = torch.empty_like(input[:, :1], dtype=torch.int)
                threshold = DCThreshold.get(self)

                if dense and not set_zero:
                    delta = x - self.prev_in

                sparsify(input, self.prev_in, x, mask, threshold)
                if dense and not set_zero:
                    inv_tiled_mask = torch.repeat_interleave(mask == 0, dim=1, repeats=x.shape[1])
                    x[inv_tiled_mask] = delta[inv_tiled_mask]

        return x, mask

    def forward(self, input):
        DCConv2d.conv_idx += 1
        self.frame_idx += 1
        first_iter = self.conv_idx == -1

        # if not first_iter and self.conv_idx == 0:
        #     print(
        #         f"sparse in: {DCConv2d.n_sparse_inputs} dense in: {DCConv2d.n_dense_inputs} "
        #         f"sparse_out: {DCConv2d.n_sparse_output} dense out: {DCConv2d.n_dense_output}")

        if first_iter:
            self.init_first_iter(input)

        if self.backend == DCBackend.cudnn:
            assert (len(self.padding) == 2)
            return self._forward_cudnn(input, first_iter)
        elif self.backend == DCBackend.delta_cudnn:
            assert (len(self.padding) == 2)
            return self._forward_delta_cudnn(input, first_iter)
        else:
            if self.logging_enabled():
                return self._forward_delta_conv_debug(input, first_iter)
            else:
                return self._forward_delta_conv(input, first_iter)

    def _forward_cudnn(self, input, first_iter):
        if self.logging_enabled():
            self.input_logger(input)
        out = super(DCConv2d, self).forward(input)

        if self.logging_enabled():
            self.prev_in_logger(input)
            self.output_logger(out)

        if self.activation is not None:
            out = self.activation(out)
        
        if DCConv2d.measure_flops:
            n_updated = out[:,:1].numel()
            DCConv2d.flops_sum += n_updated * self.weight.numel()

        return out

    def _forward_delta_cudnn(self, input, first_iter):
        if self.logging_enabled():
            self.input_logger(input)

        b = self.bias
        if not first_iter:
            self.bias = None
        out = super(DCConv2d, self).forward(input)

        out_mask = None

        if self.logging_enabled():
            threshold = 0.01 if self not in DCThreshold.t else DCThreshold.get(self)
            if out_mask is None:
                out_mask = torch.max(torch.abs(out), dim=1, keepdim=True)[0] > threshold
            self.computation_logger(out_mask)
            self.multiplications_logger(out_mask != 0, self.weight, in_mask=None, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride,
                                        padding=self.padding)

        self.bias = b

        if self.activation is not None:
            if first_iter:
                self.prev_out = out.clone()
                if self in DCThreshold.t:
                    self.out_truncated = torch.zeros_like(out)

                if self.logging_enabled():
                    self.output_logger(self.prev_out)

                if self.activation is not None:
                    out = self.activation(out)
                return out

            out_absolute = self.prev_out + out + self.out_truncated
            out_before_activ = out.clone()
            if self.dense_out:
                self.prev_out = self.prev_out + out_before_activ
                if self.logging_enabled():
                    self.output_logger(self.prev_out)
                return self.activation(out_absolute)
            else:
                out = self.activation(out_absolute) - self.activation(self.prev_out)

            if self in DCThreshold.t and not first_iter:
                threshold = DCThreshold.get(self)
                out, out_before_activ, self.out_truncated, self.prev_out, out_mask = \
                    differentiable_threshold(out, out_before_activ, self.out_truncated, self.prev_out, threshold)

                if DCConv2d.store_out_masks:
                    DCConv2d.out_masks.append(out_mask)

        elif self.dense_out:
            out_copy = out
            if self.prev_out is None:
                self.prev_out = torch.zeros_like(out_copy)
            out = self.prev_out + out_copy
            self.prev_out += out_copy

        if self.logging_enabled():
            if self.prev_out is None:
                self.prev_out = out.clone()
            elif self.activation is None and not self.dense_out:
                self.prev_out += out
            self.output_logger(self.prev_out)

        return out

    def _forward_delta_conv_debug(self, input, first_iter):
        if self.dense_in:
            x, mask = self.get_delta_and_mask(input, first_iter)
        else:
            x, mask = input
            if self.logging_enabled():
                if self.prev_in is None:
                    self.prev_in = x.clone()
                    tiled_mask = torch.repeat_interleave(mask != 0, dim=1, repeats=x.shape[1])
                    x[~tiled_mask] = 0.0
                else:
                    tiled_mask = torch.repeat_interleave(mask != 0, dim=1, repeats=x.shape[1])
                    self.prev_in[tiled_mask] += x[tiled_mask]

        if self.logging_enabled():
            self.input_logger(input if type(input) == Tensor else input[0])
            if self.prev_in is not None:
                self.prev_in_logger(self.prev_in)

        # TODO decide when to buffer out_mask
        if self.out_mask is not None and x.shape[-1] == self.out_mask.shape[-1] and x.shape[-2] == self.out_mask.shape[-2]:
            self.out_mask = None
            self.out_shape = []

        out = sparse_conv(
            x=x,
            filter=self.weight,
            mask=mask,
            bias=self.bias if self.frame_idx <= 0 else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            c_out=self.out_channels,
            create_out_mask=True,
            out_mask=self.out_mask,
            out_shape=self.out_shape
        )
        

        if self.dense_out:
            out = self.densify_layer(out)
            if self.logging_enabled():
                self.output_logger(self.densify_layer.prev_out)
            return out
        else:
            if type(out) == Tensor:
                out_mask = None
            else:
                out, out_mask = out

            if self.logging_enabled() and out_mask is not None:
                self.computation_logger(out_mask)
                self.multiplications_logger(out_mask != 0, self.weight, mask != 0, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride,
                                            padding=self.padding)

            if self.activation is not None:
                out = self.activation_layer((out, out_mask))
                if self.logging_enabled():
                    self.output_logger(self.activation_layer.prev_out)
                    self.truncated_logger(self.activation_layer.out_truncated)
                return out
            elif self.logging_enabled():
                _ = self.activation_layer((out, out_mask))
                self.output_logger(self.activation_layer.prev_out)

            return out, out_mask

    def _forward_delta_conv(self, input, first_iter):
        x, mask = input

        # TODO decide when to buffer out_mask
        if x.shape[-1] != self.in_shape[-1] or x.shape[-2] != self.in_shape[-2]:
            self.out_mask = None
            self.in_shape = x.shape
            self.out_shape = []

        out = sparse_conv(
            x=x,
            filter=self.weight,
            mask=mask,
            bias=self.bias if self.frame_idx <= 0 else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            c_out=self.out_channels,
            create_out_mask=True,
            out_mask=self.out_mask,
            out_shape=self.out_shape
        )

        if self.dense_out:
            out = self.densify_layer(out)
            return out
        else:
            if type(out) == Tensor:
                out_mask = None
            else:
                out, out_mask = out

            if self.activation is not None:
                out = self.activation_layer((out, out_mask))
                return out

            return out, out_mask

    def reset(self):
        super().reset()
        self.frame_idx = -1
        self.conv_idx = -1
        self.prev_out = None
        self.prev_in = None
        self.out_truncated = None
        self.out_mask = None
        self.out_shape = []
        DCConv2d.conv_idx = -1
        DCConv2d.n_dense_output, DCConv2d.n_sparse_output = 0, 0
        DCConv2d.n_dense_inputs, DCConv2d.n_sparse_inputs = 0, 0

    def process_filters_half(self):
        self.orig_weights = self.weight.data.clone()
        result = super().half()
        if self.backend == DCBackend.deltacnn:
            pixel_wise = self.groups == self.out_channels and self.groups == self.in_channels
            result.weight.data = convert_half_filter(result.weight.data, pixel_wise=pixel_wise)

        return result

    def process_filters_single(self):
        self.orig_weights = self.weight.data.clone()
        if self.backend == DCBackend.deltacnn:
            self.weight.data = convert_filter_out_channels_last(self.weight.data)

        return self

    def process_filters(self):
        if self.weight.data.dtype == torch.float16:
            self.process_filters_half()
        else:
            self.process_filters_single()


class DCConvTranspose2d(nn.ConvTranspose2d, DCModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            output_padding: _size_2_t = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            delta_threshold: float = None,
            name: str = None,
            activation: str = None,
            store_prev_in=False,
            dense_out=False,
            backend=None,
            use_logging=None,
            **kwargs
    ):
        kernel_size, stride, padding, dilation = to_tuple(kernel_size, stride, padding, dilation)

        super(DCConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.prev_in = None
        self.prev_out = None
        self.delta_threshold = delta_threshold if delta_threshold is not None else DCThreshold.t_default
        self.acc_err = None
        self.name = name
        self.activation = None
        self.activation_int = 0
        self.backend = backend if backend is not None else DCConv2d.backend
        self.activation, self.activation_int = convert_activation_string(activation, **kwargs)

        self.stores_prev_in = store_prev_in
        self.dense_out = dense_out
        self.use_logging = use_logging

        self.densify_layer: DCDensify = None
        self.activation_layer: DCActivation = None
        if self.dense_out:
            self.densify_layer = DCDensify(activation=activation, name=self.name)
        elif self.activation is not None:
            self.activation_layer = DCActivation(self.name, delta_threshold=delta_threshold, activation=activation)
        elif self.logging_enabled():
            self.activation_layer = DCActivation(self.name, inplace=False, delta_threshold=-1, activation=None, truncation_mode=DCTruncation.none)

        if self.logging_enabled():
            self.computation_logger = ComputationsLogger(name=self.name)
            self.multiplications_logger = MultiplicationsLogger(name=self.name)
            self.input_logger = InputLogger(name=self.name)
            self.prev_in_logger = PrevInputLogger(name=self.name)
            self.output_logger = OutputLogger(name=self.name)
        self.conv_idx = -1
        self.dense_in = False
        self.mask = None
        self.frame_idx = -1
        self.out_truncated = None
        self.out_mask = None
        self.out_shape = []
        self.kwargs = kwargs
        self.in_shape = None
        assert kwargs.get('leaky_relu_negative_slope', 0.1) == 0.1, "Leaky ReLU implementation uses hard coded negative slope of 0.1"


    def logging_enabled(self):
        return self.use_logging if self.use_logging is not None else DCConv2d.use_logging

    def init_first_iter(self, input):
        self.conv_idx = DCConv2d.conv_idx
        self.dense_in = type(input) == Tensor
        self.in_shape = input.shape if type(input) == Tensor else input[0].shape

        if self.logging_enabled() and not self.computation_logger.added_id:
            conv_info = f"k={self.kernel_size[0]}x{self.kernel_size[1]} s={self.stride[0]}x{self.stride[1]}"
            conv_info += f" d={self.dilation[0]}x{self.dilation[1]}"
            if len(self.padding) == 4:
                conv_info += f" p={self.padding[0]}x{self.padding[1]}x{self.padding[2]}x{self.padding[3]}"
            else:
                conv_info += f" p={self.padding[0]}x{self.padding[1]}"
            conv_info += f"{' dw' if self.groups == self.in_channels == self.out_channels else ''}"
            self.computation_logger.name = f"{DCConv2d.conv_idx} {self.computation_logger.name} {conv_info}"
            self.multiplications_logger.name = f"{DCConv2d.conv_idx} {self.multiplications_logger.name} {conv_info}"
            self.input_logger.name = f"{DCConv2d.conv_idx} {self.input_logger.name} {conv_info}"
            self.prev_in_logger.name = f"{DCConv2d.conv_idx} {self.prev_in_logger.name} {conv_info}"
            self.output_logger.name = f"{DCConv2d.conv_idx} {self.output_logger.name} {conv_info}"
            self.computation_logger.added_id = True
            self.multiplications_logger.added_id = True
            self.input_logger.added_id = True
            self.prev_in_logger.added_id = True

        # TODO fix tuned thresholds so that we can remove self.dense_out here. does not make sense to use at all
        if (self.activation is not None or self.dense_out or (self.dense_in and not (self.backend == DCBackend.delta_cudnn))) \
                and not self.backend == DCBackend.cudnn:
           DCThreshold.set(self, self.delta_threshold)

        if self.dense_in:
            DCConv2d.n_dense_inputs += 1
        else:
            DCConv2d.n_sparse_inputs += 1
        if self.dense_out:
            DCConv2d.n_dense_output += 1
        else:
            DCConv2d.n_sparse_output += 1
        # print(f"k={self.kernel_size[0]}x{self.kernel_size[1]}, c={self.in_channels}x{self.out_channels} s={self.stride[0]}x{self.stride[1]} g={self.groups}")

    def __repr__(self):
        return f"MyConv2d{self.conv_idx} cin={self.in_channels} cout={self.out_channels}"

    def get_delta_and_mask(self, input, first_iter, dense=False, set_zero=False):
        if first_iter:
            x = input
            self.stores_prev_in = True
            mask = torch.ones_like(x[:, :1], dtype=torch.int)
            self.prev_in = x.clone()
            DCModule.temp_buffers.append(self.prev_in)
        else:
            if self.conv_idx == 0:
                # dilate mask
                x = input - self.prev_in
                mask = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
                pool_size = 15
                threshold = DCThreshold.get(self)

                mask = torch.max_pool2d(mask, (pool_size, pool_size), stride=(1, 1),
                                        padding=((pool_size - 1) // 2, (pool_size - 1) // 2)) > threshold

                tiled_mask = torch.repeat_interleave(mask, dim=1, repeats=x.shape[1])

                self.prev_in[tiled_mask] = input[tiled_mask]
                mask = mask.int()
                if dense and set_zero:
                    x[~tiled_mask] = 0.0
            else:
                if dense and set_zero:
                    x = torch.zeros_like(input)
                else:
                    x = torch.empty_like(input)

                mask = torch.empty_like(input[:, :1], dtype=torch.int)
                threshold = DCThreshold.get(self)

                if dense and not set_zero:
                    delta = x - self.prev_in

                sparsify(input, self.prev_in, x, mask, threshold)
                if dense and not set_zero:
                    inv_tiled_mask = torch.repeat_interleave(mask == 0, dim=1, repeats=x.shape[1])
                    x[inv_tiled_mask] = delta[inv_tiled_mask]

        return x, mask

    def forward(self, input):
        DCConv2d.conv_idx += 1
        self.frame_idx += 1
        first_iter = self.conv_idx == -1

        # if not first_iter and self.conv_idx == 0:
        #     print(
        #         f"sparse in: {DCConv2d.n_sparse_inputs} dense in: {DCConv2d.n_dense_inputs} "
        #         f"sparse_out: {DCConv2d.n_sparse_output} dense out: {DCConv2d.n_dense_output}")

        if first_iter:
            self.init_first_iter(input)

        assert(self.backend == DCBackend.cudnn or self.backend == DCBackend.deltacnn)

        if self.backend == DCBackend.cudnn:
            assert (len(self.padding) == 2)
            return self._forward_cudnn(input, first_iter)
        elif self.backend == DCBackend.delta_cudnn:
            assert (len(self.padding) == 2)
            return self._forward_delta_cudnn(input, first_iter)
        else:
            if self.logging_enabled():
                return self._forward_delta_conv_debug(input, first_iter)
            else:
                return self._forward_delta_conv(input, first_iter)

    def _forward_cudnn(self, input, first_iter):
        if self.logging_enabled():
            self.input_logger(input)
        out = super(DCConvTranspose2d, self).forward(input)

        if self.logging_enabled():
            self.prev_in_logger(input)
            self.output_logger(out)

        if self.activation is not None:
            out = self.activation(out)
        
        if DCConv2d.measure_flops:
            n_updated = input[:,:1].numel()
            DCConv2d.flops_sum += n_updated * self.weight.numel()

        return out

    def _forward_delta_cudnn(self, input, first_iter):
        if self.logging_enabled():
            self.input_logger(input)

        b = self.bias
        if not first_iter:
            self.bias = None
        out = super(DCConv2d, self).forward(input)

        out_mask = None

        if self.logging_enabled():
            threshold = 0.01 if self not in DCThreshold.t else DCThreshold.get(self)
            if out_mask is None:
                out_mask = torch.max(torch.abs(out), dim=1, keepdim=True)[0] > threshold
            self.computation_logger(out_mask)
            self.multiplications_logger(out_mask != 0, self.weight, in_mask=None, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride,
                                        padding=self.padding)

        self.bias = b

        if self.activation is not None:
            if first_iter:
                self.prev_out = out.clone()
                if self in DCThreshold.t:
                    self.out_truncated = torch.zeros_like(out)

                if self.logging_enabled():
                    self.output_logger(self.prev_out)

                if self.activation is not None:
                    out = self.activation(out)
                return out

            out_absolute = self.prev_out + out + self.out_truncated
            out_before_activ = out.clone()
            if self.dense_out:
                self.prev_out = self.prev_out + out_before_activ
                if self.logging_enabled():
                    self.output_logger(self.prev_out)
                return self.activation(out_absolute)
            else:
                out = self.activation(out_absolute) - self.activation(self.prev_out)

            if self in DCThreshold.t and not first_iter:
                threshold = DCThreshold.get(self)
                out, out_before_activ, self.out_truncated, self.prev_out, out_mask = \
                    differentiable_threshold(out, out_before_activ, self.out_truncated, self.prev_out, threshold)

                if DCConv2d.store_out_masks:
                    DCConv2d.out_masks.append(out_mask)
        elif self.dense_out:
            out_copy = out
            if self.prev_out is None:
                self.prev_out = torch.zeros_like(out_copy)
            out = self.prev_out + out_copy
            self.prev_out += out_copy

        if self.logging_enabled():
            if self.prev_out is None:
                self.prev_out = out.clone()
            elif self.activation is None and not self.dense_out:
                self.prev_out += out
            self.output_logger(self.prev_out)

        return out

    def _forward_delta_conv_debug(self, input, first_iter):
        if self.dense_in:
            x, mask = self.get_delta_and_mask(input, first_iter)
        else:
            x, mask = input
            if self.logging_enabled():
                if self.prev_in is None:
                    self.prev_in = x.clone()
                    tiled_mask = torch.repeat_interleave(mask != 0, dim=1, repeats=x.shape[1])
                    x[~tiled_mask] = 0.0
                else:
                    tiled_mask = torch.repeat_interleave(mask != 0, dim=1, repeats=x.shape[1])
                    self.prev_in[tiled_mask] += x[tiled_mask]

        if self.logging_enabled():
            self.input_logger(input if type(input) == Tensor else input[0])
            if self.prev_in is not None:
                self.prev_in_logger(self.prev_in)

        # TODO decide when to buffer out_mask
        if self.out_mask is not None and x.shape[-1] == self.out_mask.shape[-1] and x.shape[-2] == self.out_mask.shape[-2]:
            self.out_mask = None
            self.out_shape = []
            
        out = sparse_deconv(
            x=x,
            filter=self.weight,
            mask=mask,
            bias=self.bias if self.frame_idx <= 0 else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            c_out=self.out_channels,
            create_out_mask=True,
            out_mask=self.out_mask,
            out_shape=self.out_shape
        )
        

        if self.dense_out:
            out = self.densify_layer(out)
            if self.logging_enabled():
                self.output_logger(self.densify_layer.prev_out)
            return out
        else:
            if type(out) == Tensor:
                out_mask = None
            else:
                out, out_mask = out

            if self.logging_enabled() and out_mask is not None:
                self.computation_logger(out_mask)
                self.multiplications_logger(out_mask != 0, self.weight, mask != 0, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride,
                                            padding=self.padding)

            if self.activation is not None:
                out = self.activation_layer((out, out_mask))
                if self.logging_enabled():
                    self.output_logger(self.activation_layer.prev_out)
                    self.truncated_logger(self.activation_layer.out_truncated)
                return out
            elif self.logging_enabled():
                _ = self.activation_layer((out, out_mask))
                self.output_logger(self.activation_layer.prev_out)

            return out, out_mask

    def _forward_delta_conv(self, input, first_iter):
        x, mask = input

        # TODO decide when to buffer out_mask
        if x.shape[-1] != self.in_shape[-1] or x.shape[-2] != self.in_shape[-2]:
            self.out_mask = None
            self.in_shape = x.shape
            self.out_shape = []

        out = sparse_deconv(
            x=x,
            filter=self.weight,
            mask=mask,
            bias=self.bias if self.frame_idx <= 0 else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            c_out=self.out_channels,
            create_out_mask=True,
            out_mask=self.out_mask,
            out_shape=self.out_shape
        )
        
        if self.dense_out:
            out = self.densify_layer(out)
            return out
        else:
            if type(out) == Tensor:
                out_mask = None
            else:
                out, out_mask = out

            if self.activation is not None:
                out = self.activation_layer((out, out_mask))
                return out

            return out, out_mask

    def reset(self):
        super().reset()
        self.frame_idx = -1
        self.conv_idx = -1
        self.prev_out = None
        self.prev_in = None
        self.out_truncated = None
        self.out_mask = None
        self.out_shape = []
        DCConv2d.conv_idx = -1
        DCConv2d.n_dense_output, DCConv2d.n_sparse_output = 0, 0
        DCConv2d.n_dense_inputs, DCConv2d.n_sparse_inputs = 0, 0

    def process_filters_half(self):
        self.orig_weights = self.weight.data.clone()
        result = super().half()
        if self.backend == DCBackend.deltacnn:
            pixel_wise = self.groups == self.out_channels and self.groups == self.in_channels
            result.weight.data = convert_half_filter(result.weight.data, pixel_wise=pixel_wise, transposed=True)

        return result

    def process_filters_single(self):
        self.orig_weights = self.weight.data.clone()
        if self.backend == DCBackend.deltacnn:
            self.weight.data = convert_filter_out_channels_last(self.weight.data, transposed=True)

        return self

    def process_filters(self):
        if self.weight.data.dtype == torch.float16:
            self.process_filters_half()
        else:
            self.process_filters_single()


class DCAdd(DCModule):
    idx = -1

    def __init__(self, activation: str = None, dense_out=False, weight_a=None, weight_b=None, name=""):
        super(DCAdd, self).__init__()
        self.name = name
        self.dense_out = dense_out
        self.activation, self.activation_int = convert_activation_string(activation)
        self.idx = -1
        self.prev_out = None
        self.mask_out = None
        self.weight_a = 1.0 if weight_a is None else weight_a
        self.weight_b = 1.0 if weight_b is None else weight_b
        self.first_iter = True

    def forward(self, a, b, weight_a=None, weight_b=None):
        w_a = self.weight_a if weight_a is None else weight_a
        w_b = self.weight_b if weight_b is None else weight_b

        if type(a) == torch.Tensor:
            if DCConv2d.backend == DCBackend.delta_cudnn:
                c = w_a * a + w_b * b
                if self.activation is not None:
                    if self.first_iter:
                        self.first_iter = False
                        self.prev_out = c.clone()
                        return self.activation(c)

                    out_absolute = self.prev_out + c
                    out = self.activation(out_absolute) - self.activation(self.prev_out)
                    self.prev_out = self.prev_out + c
                    c = out
                return c
            else:
                c = w_a * a + w_b * b
                if self.activation is not None:
                    return self.activation(c)
                return c

        val_a, mask_a = a
        val_b, mask_b = b

        if self.first_iter:
            self.first_iter = False
            DCAdd.idx += 1
            self.idx = DCAdd.idx
            if self.dense_out or self.activation is not None:
                self.prev_out = torch.zeros_like(val_a)
                DCModule.temp_buffers.append(self.prev_out)

            self.mask_out = torch.empty_like(val_a[:, :1], dtype=torch.int)

        use_python_implementation = False

        if use_python_implementation:
            mask_out = mask_a + mask_b
            out = torch.zeros_like(val_a)
            tiled_mask_a = torch.repeat_interleave(mask_a != 0, dim=1, repeats=out.shape[1])
            out[tiled_mask_a] += val_a[tiled_mask_a] * w_a
            tiled_mask_b = torch.repeat_interleave(mask_b != 0, dim=1, repeats=out.shape[1])
            out[tiled_mask_b] += val_b[tiled_mask_b] * w_b

            tiled_merged_mask = torch.repeat_interleave(mask_out != 0, dim=1, repeats=out.shape[1])

            if self.activation is not None:
                prev_out_active = self.activation(self.prev_out)

            if self.dense_out or self.activation is not None:
                self.prev_out[tiled_merged_mask] += out[tiled_merged_mask]

            if self.activation is not None:
                out_active = self.activation(self.prev_out)
                if not self.dense_out:
                    out[tiled_merged_mask] = (out_active - prev_out_active)[tiled_merged_mask]

            if self.dense_out:
                if self.activation is None:
                    return self.prev_out.clone()
                else:
                    return out_active
        else:
            out = torch.empty_like(val_a)
            mask_out = self.mask_out if self.mask_out.shape == mask_a.shape else torch.empty_like(mask_a)
            sparse_add_tensors(val_a, val_b, self.prev_out, out, mask_a, mask_b, mask_out, w_a, w_b, self.activation_int, self.dense_out)

            if self.dense_out:
                return out

        return out, mask_out

    def reset(self):
        super().reset()
        self.idx = -1
        self.mask_out = None
        self.prev_out = None
        self.first_iter = True
        DCAdd.idx = -1


class DCConcatenate(DCModule):
    def __init__(self,  name=""):
        super(DCConcatenate, self).__init__()
        self.name = name

    def forward(self, a, b):
        if type(a) == torch.Tensor:
            return torch.cat((a, b), dim=1)
            
        return sparse_concatenate(a, b)


class DCSparsify(DCModule):
    idx = -1

    def __init__(self, name="", delta_threshold=None, dilation=-1):
        super(DCSparsify, self).__init__()
        self.name = name
        self.prev_in = None
        self.idx = -1
        self.delta_threshold = delta_threshold if delta_threshold is not None else DCThreshold.t_default
        self.dilation = dilation
        self.frame_idx = -1
        self.mask = None

    def forward(self, input):
        if DCConv2d.backend != DCBackend.deltacnn and DCConv2d.backend != DCBackend.delta_cudnn:
            return input

        first_iter = self.idx == -1
        self.frame_idx += 1

        if first_iter:
            if self not in DCThreshold.t:
                DCThreshold.set(self, self.delta_threshold)
            DCSparsify.idx += 1
            self.idx = DCSparsify.idx
            self.prev_in = input.clone()
            DCModule.temp_buffers.append(self.prev_in)
            mask = torch.ones_like(input[:, :1], dtype=torch.int)

            if DCConv2d.backend == DCBackend.delta_cudnn:
                return input

            return input, mask

        threshold = DCThreshold.get(self)

        if self.dilation > 1:
            x = input - self.prev_in
            mask = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
            pool_size = self.dilation

            mask = torch.max_pool2d(mask, (pool_size, pool_size), stride=(1, 1),
                                    padding=((pool_size - 1) // 2, (pool_size - 1) // 2)) > threshold

            tiled_mask = torch.repeat_interleave(mask, dim=1, repeats=x.shape[1])

            self.prev_in[tiled_mask] = input[tiled_mask]
            mask = mask.int()

            if DCConv2d.backend == DCBackend.delta_cudnn:
                x[~tiled_mask] = 0
                return x
        else:
            if DCConv2d.backend == DCBackend.delta_cudnn:
                x = input - self.prev_in
                mask = torch.max(torch.abs(x), dim=1, keepdim=True)[0] > threshold
                tiled_mask = torch.repeat_interleave(self.mask, dim=1, repeats=x.shape[1])
                self.prev_in[tiled_mask] = input
                x[~tiled_mask] = 0
                return x

            if self.mask is None:
                self.mask = torch.empty_like(input[:, :1], dtype=torch.int)
            x = torch.empty_like(input)
            sparsify(input, self.prev_in, x, self.mask, threshold)
            return x, self.mask

        return x, mask

    def reset(self):
        super().reset()
        self.idx = -1
        self.frame_idx = -1
        self.prev_in = None
        self.prev_out = None
        self.mask = None
        DCSparsify.idx = -1


class DCDensify(DCModule):
    idx = -1

    def __init__(self, activation: str = None, name="", clone_out=True):
        super(DCDensify, self).__init__()
        self.name = name
        self.activation, self.activation_int = convert_activation_string(activation)
        self.prev_out = None
        self.idx = -1
        self.clone_out = clone_out

    def forward(self, input):
        if type(input) == torch.Tensor and DCConv2d.backend != DCBackend.delta_cudnn:
            return input

        first_iter = self.prev_out is None
        if DCConv2d.backend == DCBackend.delta_cudnn:
            if first_iter:
                self.prev_out = torch.zeros_like(input)
            out_absolute = self.prev_out + input
            self.prev_out += input

            if self.activation is not None:
                out_absolute = self.activation(out_absolute)

            return out_absolute

        input, mask = input
        if first_iter:
            DCDensify.idx += 1
            self.idx = DCDensify.idx
            self.prev_out = torch.zeros_like(input)
            DCModule.temp_buffers.append(self.prev_out)

        use_python = False
        if use_python:
            tiled_mask = torch.repeat_interleave(mask != 0, dim=1, repeats=input.shape[1])
            self.prev_out[tiled_mask] += input[tiled_mask]

            out = self.prev_out.clone()

            if self.activation is not None:
                out = self.activation(out)

            return out
        else:
            sparse_activation(input, self.prev_out, None, mask, -1, -1, DCTruncation.none.value)
            out = self.prev_out
            if self.clone_out:
                out = out.clone()
                
            if self.activation is not None:
                return self.activation(out)
            return out

    def reset(self):
        super().reset()
        self.prev_out = None


class DCUpsamplingNearest2d(DCModule):
    idx = -1

    def __init__(self, scale_factor=2, name=""):
        super(DCUpsamplingNearest2d, self).__init__()
        self.name = name
        self.scale_factor = int(scale_factor)
        self.idx = -1
        self.out = None
        self.torch_upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)

    def forward(self, input):
        if type(input) == torch.Tensor:
            return self.torch_upsample(input)

        val, mask = input

        self.out = sparse_upsample(val, mask, self.scale_factor, self.out)

        return self.out

    def reset(self):
        super().reset()
        self.out = None


class DCActivation(DCModule):
    idx = -1
    truncation_default = DCTruncation.max

    def __init__(self, name="", inplace=True, delta_threshold=None, activation="relu", truncation_mode:DCTruncation=None):
        super(DCActivation, self).__init__()
        self.name = name
        self.idx = -1
        self.prev_out = None
        self.inplace = inplace
        self.out_truncated = None
        self.activation, self.activation_int = convert_activation_string(activation)
        self.delta_threshold = delta_threshold if delta_threshold is not None else DCThreshold.t_default
        self.first_iter = True
        self.truncation_mode = truncation_mode if truncation_mode is not None else DCActivation.truncation_default

    def forward(self, input):
        if type(input) == torch.Tensor and DCConv2d.backend != DCBackend.delta_cudnn:
            return self.activation(input)

        if DCConv2d.backend == DCBackend.delta_cudnn:
            val = input
        else:
            val, mask = input

        if self.first_iter:
            self.first_iter = False
            if self not in DCThreshold.t:
                DCThreshold.set(self, self.delta_threshold)
            if self.idx < 0:
                DCActivation.idx += 1
                self.idx = DCActivation.idx
            self.prev_out = val.clone()
            self.out_truncated = torch.zeros_like(val)
            DCModule.temp_buffers.append(self.prev_out)
            DCModule.temp_buffers.append(self.out_truncated)
            val = self.activation(val)
            if DCConv2d.backend == DCBackend.delta_cudnn:
                return val
            return val, mask

        threshold = DCThreshold.get(self)

        if DCConv2d.backend == DCBackend.delta_cudnn:
            val_abs = self.prev_out + input + self.out_truncated
            val_activated = self.activation(val_abs) - self.activation(self.prev_out)
            val_activated, val, self.out_truncated, self.prev_out, _ = \
                differentiable_threshold(val_activated, val, self.out_truncated, self.prev_out, threshold)
            return val_activated

        if not self.inplace:
            val = val.clone()
            mask = mask.clone()
        sparse_activation(val, self.prev_out, self.out_truncated, mask, threshold, self.activation_int, self.truncation_mode.value)

        return val, mask

    def reset(self):
        super().reset()
        self.idx = -1
        self.prev_out = None
        self.out_truncated = None
        self.first_iter = True
        DCActivation.idx = -1


class DCMaxPooling(nn.MaxPool2d, DCModule):
    def __init__(self,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = (1, 1),
                 padding: _size_2_t = (0, 0),
                 dilation: _size_2_t = (1, 1)
                 ):
        kernel_size, stride, padding, dilation = to_tuple(kernel_size, stride, padding, dilation)
        super().__init__(kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.first_iter = True
        self.out_shape = []
        self.prev_in = None

    def forward(self, input) -> Tensor:
        if type(input) == Tensor:
            old_padding = self.padding
            if len(self.padding) == 4:
                input = torch.nn.functional.pad(input, self.padding)
                self.padding = 0

            if DCConv2d.backend == DCBackend.delta_cudnn:
                if self.first_iter:
                    self.first_iter = False
                    self.prev_in = torch.zeros_like(input)
                input_abs = input + self.prev_in
                out = super().forward(input_abs) - super().forward(self.prev_in)
                self.prev_in += input
            else:
                out = super().forward(input)
            self.padding = old_padding
            return out
        else:
            x, mask = input
            if self.first_iter:
                self.first_iter = False
                self.prev_in = torch.zeros_like(x)
                DCModule.temp_buffers.append(self.prev_in)

            out = sparse_pooling(x, self.prev_in, self.kernel_size, mask, self.stride, self.padding, self.dilation, create_out_mask=True,
                                 sub_tile_sparsity=True, pooling_mode_int=0, out_shape=self.out_shape)
            sparse_add_to_dense_tensor(x, self.prev_in, mask)
            return out

    def reset(self):
        super().reset()
        self.first_iter = True
        self.prev_in = None


class DCAdaptiveAveragePooling(nn.AdaptiveAvgPool2d, DCModule):
    def __init__(self,
                 output_size: _size_2_t = None
                 ):
        output_size = to_tuple(output_size)
        super().__init__(output_size=output_size)
        self.first_iter = True
        self.out_shape = output_size
        self.prev_in = None

    def get_target_size(self, input):
        # other modes are not implemented yet
        assert (self.output_size[0] == 1 and self.output_size[1] == 1)
        k = input.shape[2:]
        s = (1, 1)
        p = (0, 0)
        d = (1, 1)
        return k, s, p, d

    def forward(self, input) -> Tensor:
        if type(input) == Tensor:
            out = super().forward(input)
            return out
        else:
            x, mask = input
            if self.first_iter:
                self.first_iter = False

            kernel_size, stride, padding, dilation = self.get_target_size(x)
            out = sparse_pooling(x, None, kernel_size, mask, stride, padding, dilation, create_out_mask=True, sub_tile_sparsity=True,
                                 pooling_mode_int=1, out_shape=self.out_shape)
            return out

    def reset(self):
        super().reset()
        self.first_iter = True
        self.prev_in = None


class DCBatchNorm2d(nn.BatchNorm2d, DCModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            inplace=True
    ):
        super(DCBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.first_iter = True
        self.scale = None
        self.offset = None
        self.inplace = inplace

    def convert_to_scale_offset(self, input):
        bn_scale = self.weight * torch.rsqrt(self.running_var + self.eps)

        self.scale = bn_scale
        self.offset = -self.running_mean * bn_scale + self.bias

        self.scale = self.scale[None, :, None, None]
        self.offset = self.offset[None, :, None, None]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        use_offset = False
        mask = None
        use_pytorch_implementation = False

        if type(input) == tuple:
            input, mask = input
        else:
            use_offset = True
            use_pytorch_implementation = True

        if self.first_iter:
            self.first_iter = False
            use_offset = True
            self.convert_to_scale_offset(input)

        out = input
        out_mask = mask

        if use_pytorch_implementation:
            if self.inplace:
                out.mul_(self.scale)
            else:
                out = input * self.scale
            if use_offset:
                out.add_(self.offset)
        else:
            bias = None
            if use_offset:
                bias = self.offset
            if self.inplace:
                sparse_mul_add(out, out_mask, out, out_mask, self.scale, bias)
            else:
                out = torch.empty_like(input)
                out_mask = mask.clone()
                sparse_mul_add(input, mask, out, out_mask, self.scale, bias)

        if mask is not None:
            out = (out, out_mask)

        return out

    def reset(self):
        super().reset()
        self.first_iter = True
