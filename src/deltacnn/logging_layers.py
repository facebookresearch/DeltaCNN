# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from .utils import count_active_neighbors
from torch.nn.functional import interpolate


class LoggingController:
    img_epochs = 10

    def __init__(self, model, writer: SummaryWriter):
        self.model = model
        self.writer = writer
        self._all_logging_layers = None

    def log_densities(self, epoch, reset=False, disable=False, prefix=""):
        for layer in self.get_all_logging_layers().get(DensityLogger, []):
            self.writer.add_scalar(f"{prefix}densities/{layer.name}", layer.get(), epoch)
            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

    def _log_pixel_masks_impl(self, channels, name, epoch, individual_channels=False, prefix=""):
        def to_img(x):
            return (x * 255).to(torch.uint8)

        self.writer.add_image(f"{prefix}pixel-mask-acc/{name}/", to_img(torch.mean(channels, dim=0, keepdim=True)), epoch)

        if individual_channels:
            for ch_idx in range(len(channels)):
                self.writer.add_image(f"{prefix}pixel-mask-channels/{name}/{ch_idx}/", to_img(channels[None, ch_idx]), epoch)

    def log_pixel_masks(self, epoch, reset=False, disable=False):
        for layer in self.get_all_logging_layers().get(PixelMaskLogger, []):
            self._log_pixel_masks_impl(layer.get(), layer.name, epoch)

            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

    def log_diff_in_histogram(self, epoch, reset=False, disable=False):
        total_histogram = []
        for layer in self.get_all_logging_layers().get(DifferentialPixelMaskLogger, []):
            histrogram = layer.get_histogram(tile_size=(6, 6))

            bins = range(0, 65)

            if histrogram is not None and len(histrogram) > 0:
                self.writer.add_histogram(f"tile-density/{layer.name}", histrogram, epoch, bins=bins)
                histogram_non_empty = histrogram[histrogram > 0]
                if len(histogram_non_empty > 0):
                    self.writer.add_histogram(f"tile-density-non-empty/{layer.name}", histogram_non_empty, epoch, bins=bins)
                    total_histogram.extend(list(histrogram))

            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

        if len(total_histogram) > 0:
            t_total_histogram = torch.tensor(total_histogram)
            self.writer.add_histogram(f"tile-density-total/all", t_total_histogram, epoch, bins=bins)
            t_total_histogram_non_empty = t_total_histogram[t_total_histogram > 0]
            if len(t_total_histogram_non_empty) > 0:
                self.writer.add_histogram(f"tile-density-total/non-empty", t_total_histogram_non_empty, epoch, bins=bins)

    def image_logger(self, epoch, reset, disable, path, cls):
        def to_img(x):
            return (x * 255).to(torch.uint8)

        for layer in self.get_all_logging_layers().get(cls, []):
            if epoch % LoggingController.img_epochs == (LoggingController.img_epochs - 1):
                val = layer.get()
                if val is not None:
                    self.writer.add_image(f"{path}/{layer.name}/", to_img(torch.mean(val[None], dim=0, keepdim=True)), epoch)
            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

    def log_prev_input(self, epoch, reset=False, disable=False):
        self.image_logger(epoch, reset, disable, "prev_input", PrevInputLogger)

    def log_input(self, epoch, reset=False, disable=False):
        self.image_logger(epoch, reset, disable, "input", InputLogger)

    def log_output(self, epoch, reset=False, disable=False):
        self.image_logger(epoch, reset, disable, "output", OutputLogger)

    def log_computations(self, epoch, reset=False, disable=False):
        def to_img(x):
            return (x * 255).to(torch.uint8)

        all_masks = []
        max_res = [0,0]

        logged_total_computations = False
        for layer in self.get_all_logging_layers().get(ComputationsLogger, []):
            if not logged_total_computations:
                self.writer.add_scalar(f"computations", layer.get(), epoch)
                logged_total_computations = True

            mask = layer.get_mask()
            if mask is not None and epoch % LoggingController.img_epochs == (LoggingController.img_epochs - 1):
                self.writer.add_image(f"update-mask/{layer.name}/", to_img(torch.mean(mask[None], dim=0, keepdim=True)), epoch)
                all_masks.append(mask)
                if mask.shape[0] > max_res[0] or mask.shape[1] > max_res[1]:
                    max_res = mask.shape

            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

        if len(all_masks) > 0:
            for i in range(len(all_masks)):
                all_masks[i] = interpolate(all_masks[i][None, None], size=(max_res[-2], max_res[-1]), mode="nearest")
            mean_mask = to_img(torch.mean(torch.cat(all_masks, dim=1),dim=1))
            self.writer.add_image("update-density", mean_mask, epoch)

    def log_multiplications(self, epoch, reset=False, disable=False):
        logged_global = False
        for layer in self.get_all_logging_layers().get(MultiplicationsLogger, []):
            if not logged_global:
                self.writer.add_scalar(f"multiplications/rel", layer.get_global_rel(), epoch)
                self.writer.add_scalar(f"multiplications/abs", layer.get_global_abs(), epoch)
                logged_global = True

            self.writer.add_scalar(f"mul-per-layer-rel/{layer.name}", layer.get_self_rel(), epoch)
            self.writer.add_scalar(f"mul-per-layer-abs/{layer.name}", layer.get_self_abs(), epoch)

            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

    def log_prev_input_bandwidth(self, epoch, reset=False, disable=False):
        for layer in self.get_all_logging_layers().get(PrevInBandwidthLogger, []):
            self.writer.add_scalar(f"prev_in_bandwidth", layer.get(), epoch)

            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()
            break

    def log_prev_in_sizes(self, epoch, reset=False, disable=False):
        logged_global = False
        for layer in self.get_all_logging_layers().get(PrevInputSizeLogger, []):
            if not logged_global:
                self.writer.add_scalar(f"prev-in-size-global/", layer.get_global(), epoch)
                logged_global = True

            self.writer.add_scalar(f"prev-in-size/{layer.name}/", layer.get(), epoch)

            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

    def log_gradient(self, epoch, reset=False, disable=False):
        for layer in self.get_all_logging_layers().get(GradientLogger, []):
            result = layer.get()
            if result is None:
                continue
            self.writer.add_image(f"gradient", result, epoch)

            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

    def log_combined_loggers(self, epoch, reset=False, disable=False):
        logged_total_diff_density = False

        for layer in self.get_all_logging_layers().get(Loggers, []):
            outputs = layer.get()

            for log_name, log_out in outputs.items():
                if "DiffMask" in log_name and epoch % LoggingController.img_epochs == (LoggingController.img_epochs - 1):
                    self._log_pixel_masks_impl(log_out, layer.name, epoch, prefix="diff-")
                elif "Mask" in log_name and epoch % LoggingController.img_epochs == (LoggingController.img_epochs - 1):
                    self._log_pixel_masks_impl(log_out, layer.name, epoch)
                elif "DiffDensity" in log_name:
                    self.writer.add_scalar(f"diff-densities/{layer.name}", log_out, epoch)
                    if not logged_total_diff_density:
                        self.writer.add_scalar(f"diff_densities_sum", layer.logs[log_name].get_sum(), epoch)
                        logged_total_diff_density = True
                elif "Density" in log_name:
                    self.writer.add_scalar(f"densities/{layer.name}", log_out, epoch)

            if reset:
                layer.reset()
            if disable:
                layer.disable_logging()

    def get_all_logging_layers(self):
        # if list was already computed, return a copy of it to save time
        if self._all_logging_layers is not None:
            return {**self._all_logging_layers}

        layers = {}

        def recursive_get_layers(module):
            for layer in module.modules():
                cls = type(layer)

                is_logging_layer = issubclass(cls, LoggingLayer)
                if is_logging_layer and cls not in layers:
                    layers[cls] = []
                if is_logging_layer and layer in layers[cls]:
                    continue
                elif is_logging_layer:
                    layers[cls].append(layer)
                elif layer != module:
                    recursive_get_layers(layer)

        recursive_get_layers(self.model)
        self._all_logging_layers = {**layers}
        return layers

    def reset_loggers(self, enable=False, disable=False):
        for layers in self.get_all_logging_layers().values():
            for layer in layers:
                layer.reset()
                if enable:
                    layer.enable_logging()
                if disable:
                    layer.disable_logging()

    def reset_loggers_history(self):
        for layers in self.get_all_logging_layers().values():
            for layer in layers:
                layer.reset_history()

    def write_logs(self, epoch, reset=False, disable=False):
        self.log_densities(epoch, reset, disable)
        self.log_pixel_masks(epoch, reset, disable)
        self.log_computations(epoch, reset, disable)
        self.log_multiplications(epoch, reset, disable)
        self.log_prev_input_bandwidth(epoch, reset, disable)
        self.log_input(epoch, reset, disable)
        self.log_output(epoch, reset, disable)
        self.log_prev_input(epoch, reset, disable)
        self.log_prev_in_sizes(epoch, reset, disable)
        self.log_diff_in_histogram(epoch, reset, disable)
        # self.log_combined_loggers(epoch, reset, disable)


class LoggingLayer(nn.Module):
    id = 1

    def __init__(self, name="", enabled=False):
        super(LoggingLayer, self).__init__()
        self.name = name
        self.enabled = enabled
        self.added_id = False

    def enable_logging(self):
        self.enabled = True

    def disable_logging(self):
        self.enabled = False

    def get(self):
        pass

    def reset(self):
        pass

    def reset_history(self):
        pass


class DensityLogger(LoggingLayer):
    def __init__(self, threshold=0.0, **kwargs):
        super(DensityLogger, self).__init__(**kwargs)
        self.active_pixels = 0
        self.total_pixels = 0
        self.threshold = threshold

    def forward(self, x):
        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        if self.enabled:
            self.active_pixels += torch.sum(torch.abs(x) > self.threshold)
            self.total_pixels += x.numel()
        return x

    def get(self):
        return float(self.active_pixels) / float(self.total_pixels)

    def reset(self):
        self.active_pixels = 0
        self.total_pixels = 0


class DifferentialDensityLogger(LoggingLayer):
    sum_active_pixels = 0
    sum_total_pixels = 0

    def __init__(self, threshold=0.05, **kwargs):
        super(DifferentialDensityLogger, self).__init__(**kwargs)
        self.active_pixels = 0
        self.total_pixels = 0
        self.prev_frame = None
        self.threshold = threshold

    def forward(self, x):
        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        if self.enabled:
            if self.prev_frame is None:
                diff = torch.ones_like(x)
            else:
                diff = torch.abs(self.prev_frame - x) > self.threshold
            self.active_pixels += torch.sum(diff != 0.0)
            self.total_pixels += diff.numel()
            self.prev_frame = x.detach()
            DifferentialDensityLogger.sum_active_pixels += torch.sum(diff != 0.0)
            DifferentialDensityLogger.sum_total_pixels += diff.numel()
        return x

    def get(self):
        return float(self.active_pixels) / float(self.total_pixels)

    def get_sum(self):
        if DifferentialDensityLogger.sum_total_pixels == 0:
            return 0
        return float(DifferentialDensityLogger.sum_active_pixels) / float(DifferentialDensityLogger.sum_total_pixels)

    def reset(self):
        self.active_pixels = 0
        self.total_pixels = 0
        DifferentialDensityLogger.sum_active_pixels = 0
        DifferentialDensityLogger.sum_total_pixels = 0

    def reset_history(self):
        self.prev_frame = None


class PixelMaskLogger(LoggingLayer):
    def __init__(self, threshold=0.0, **kwargs):
        super(PixelMaskLogger, self).__init__(**kwargs)
        self.channels = None
        self.n_samples = 0
        self.threshold = threshold

    def reset(self):
        self.channels = None
        self.n_samples = 0

    def forward(self, x):
        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        if self.enabled:
            if self.channels is None:
                self.channels = torch.zeros_like(x[0])
            self.channels += torch.sum(torch.abs(x) > self.threshold, dim=0)
            self.n_samples += len(x)

        return x

    def get(self):
        return self.channels / self.n_samples


class DifferentialPixelMaskLogger(LoggingLayer):
    def __init__(self, threshold=0.05, **kwargs):
        super(DifferentialPixelMaskLogger, self).__init__(**kwargs)
        self.channels = None
        self.n_samples = 0
        self.prev_frame = None
        self.threshold = threshold

    def reset(self):
        self.channels = None
        self.n_samples = 0

    def reset_history(self):
        self.prev_frame = None

    def forward(self, x, use_direct=True):
        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        if self.enabled:
            if self.channels is None:
                self.channels = torch.zeros_like(x[0]).int()

            if use_direct:
                diff = x
            else:
                if self.prev_frame is None:
                    diff = torch.ones_like(x)
                else:
                    diff = torch.abs(self.prev_frame - x) > self.threshold
            self.channels += torch.sum(diff, dim=0)
            self.n_samples += len(x)
            if not use_direct:
                self.prev_frame = x.detach()

        return x

    def get(self):
        return self.channels / self.n_samples

    def get_histogram(self, tile_size=(6, 6)):
        # densities = torch.zeros((tile_size[-2] + 2) * (tile_size[-1] + 2) + 1)
        # tiles = ((self.channels.shape[-2] + tile_size[-2] - 1) // tile_size[-2]) * ((self.channels.shape[-1] + tile_size[-1] - 1) // tile_size[-1])
        densities = []

        if self.channels is None:
            return densities

        for y_off in range(0, self.channels.shape[-2], tile_size[-2]):
            for x_off in range(0, self.channels.shape[-1], tile_size[-1]):
                y_start = max(y_off - 1, 0)
                x_start = max(x_off - 1, 0)
                y_end = min(y_off + 1 + tile_size[0], self.channels.shape[-2])
                x_end = min(x_off + 1 + tile_size[1], self.channels.shape[-1])
                in_tile = self.channels[0, y_start:y_end, x_start:x_end]
                density = torch.sum(in_tile > 0)
                densities.append(density)

        return torch.tensor(densities)

    def get_mask(self):
        return self.channels


class ComputationsLogger(LoggingLayer):
    n_computed = 0
    n_samples = 0

    def __init__(self, **kwargs):
        super(ComputationsLogger, self).__init__(**kwargs)
        self.computed_mask = None
        self.n_samples = 0

    def reset(self):
        ComputationsLogger.n_computed = 0
        ComputationsLogger.n_samples = 0
        self.computed_mask = None
        self.n_samples = 0

    def forward(self, x):
        if not self.enabled:
            return x

        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        ComputationsLogger.n_computed += torch.sum(x)
        ComputationsLogger.n_samples += x.numel()

        self.n_samples += x.shape[0] * x.shape[1]
        comp_mask = torch.sum(torch.sum(x.clone().float(), dim=0), dim=0)
        if self.computed_mask is None or self.computed_mask.shape != x.shape[2:]:
            self.computed_mask = comp_mask
        else:
            self.computed_mask += comp_mask

        return x

    def get(self):
        if self.n_samples == 0:
            return 0
        return ComputationsLogger.n_computed / ComputationsLogger.n_samples

    def get_mask(self):
        if self.computed_mask is None:
            return None
        return self.computed_mask / self.n_samples


class MultiplicationsLogger(LoggingLayer):
    n_computed = 0
    n_samples = 0

    def __init__(self, **kwargs):
        super(MultiplicationsLogger, self).__init__(**kwargs)
        self.n_computed = 0
        self.n_samples = 0

    def reset(self):
        self.n_computed = 0
        self.n_samples = 0
        MultiplicationsLogger.n_computed = 0
        MultiplicationsLogger.n_samples = 0

    def forward(self, x, filter, in_mask=None, kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        if self.enabled:
            if in_mask is None:
                computed = torch.sum(x[:, 0]) * filter.numel()
                self.n_computed += computed
                MultiplicationsLogger.n_computed += computed
            else:
                n_filter = (filter[:, 0, 0, :] if filter.shape[1] == filter.shape[2] else filter[:, :, 0, 0]).numel()
                if filter.shape[2] == 1:
                    computed = torch.sum(x[:, 0]) * n_filter
                    self.n_computed += computed
                    MultiplicationsLogger.n_computed += computed
                else:
                    active_neighbors = count_active_neighbors(in_mask[:, :1], kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)
                    active_neighbors = active_neighbors[:, 0][x[:, 0]]

                    # computed = torch.sum(active_neighbors) * filter[::groups, :, 0, 0].numel()
                    computed = torch.sum(active_neighbors) * n_filter
                    self.n_computed += computed
                    MultiplicationsLogger.n_computed += computed
            samples = x[:, 0].numel() * filter.numel()
            self.n_samples += samples
            MultiplicationsLogger.n_samples += samples

        return x

    def get_self_rel(self):
        if self.n_samples > 0:
            return self.n_computed / self.n_samples
        return 1

    def get_self_abs(self):
        return self.n_computed

    def get_global_rel(self):
        if MultiplicationsLogger.n_samples > 0:
            return MultiplicationsLogger.n_computed / MultiplicationsLogger.n_samples
        else:
            return 1

    def get_global_abs(self):
        return MultiplicationsLogger.n_computed


class PrevInBandwidthLogger(LoggingLayer):
    n_read = 0

    def __init__(self, **kwargs):
        super(PrevInBandwidthLogger, self).__init__(**kwargs)

    def reset(self):
        PrevInBandwidthLogger.n_read = 0

    def forward(self, mask: torch.Tensor, dilation: str, tile_size: int = 0):
        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        if self.enabled:
            if dilation == "tile":
                tile_size = tile_size if tile_size > 0 else mask.shape[2] // (-tile_size)
                mask = mask[:, :, ::tile_size, ::tile_size]
                inv_long_mask = (~mask).to(torch.long)
                inactive_neighbors = torch.zeros_like(inv_long_mask)
                inactive_neighbors[:, :, :-1] += inv_long_mask[:, :, 1:]
                inactive_neighbors[:, :, 1:] += inv_long_mask[:, :, :-1]
                inactive_neighbors[:, :, :, :-1] += inv_long_mask[:, :, :, 1:]
                inactive_neighbors[:, :, :, 1:] += inv_long_mask[:, :, :, :-1]
                inactive_neighbors[~mask] = 0
                n_read = torch.sum(inactive_neighbors).item() * tile_size
            else:
                inv_long_mask = (~mask).to(torch.long)
                inactive_neighbors = torch.zeros_like(inv_long_mask)
                inactive_neighbors[:, :, :-1] += inv_long_mask[:, :, 1:]
                inactive_neighbors[:, :, 1:] += inv_long_mask[:, :, :-1]
                inactive_neighbors[:, :, :, :-1] += inv_long_mask[:, :, :, 1:]
                inactive_neighbors[:, :, :, 1:] += inv_long_mask[:, :, :, :-1]
                inactive_neighbors[:, :, :-1, :-1] += inv_long_mask[:, :, 1:, 1:]
                inactive_neighbors[:, :, 1:, :-1] += inv_long_mask[:, :, :-1, 1:]
                inactive_neighbors[:, :, :-1, 1:] += inv_long_mask[:, :, 1:, :-1]
                inactive_neighbors[:, :, 1:, 1:] += inv_long_mask[:, :, :-1, :-1]
                inactive_neighbors[~mask] = 0
                n_read = torch.sum(inactive_neighbors).item()

            PrevInBandwidthLogger.n_read += n_read

        return mask

    def get(self):
        return PrevInBandwidthLogger.n_read


class PrevInputSizeLogger(LoggingLayer):
    n_vals = 0

    def __init__(self, **kwargs):
        super(PrevInputSizeLogger, self).__init__(**kwargs)
        self.n_vals = 0

    def reset(self):
        PrevInputSizeLogger.n_vals = 0
        self.n_vals = 0

    def forward(self, x):
        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        if x is None:
            return x

        PrevInputSizeLogger.n_vals += x[0].numel()
        self.n_vals += x[0].numel()

        return x

    def get_global(self):
        return PrevInputSizeLogger.n_vals

    def get(self):
        return self.n_vals


class PrevInputLogger(LoggingLayer):
    def __init__(self, **kwargs):
        super(PrevInputLogger, self).__init__(**kwargs)
        self.prev_input = None

    def reset(self):
        self.prev_input = None

    def forward(self, x):
        if not self.enabled:
            return x

        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            LoggingLayer.id += 1
            self.added_id = True

        if x is None:
            return x

        prev_input = torch.sum(torch.sum(x.clone().float(), dim=0), dim=0)
        if self.prev_input is None or self.prev_input.shape != prev_input.shape:
            self.prev_input = prev_input
        else:
            self.prev_input += prev_input

        return x

    def get(self):
        if self.prev_input is None:
            return None
        return self.prev_input / torch.max(self.prev_input)


class InputLogger(PrevInputLogger):
    def __init__(self, **kwargs):
        super(InputLogger, self).__init__(**kwargs)


class OutputLogger(PrevInputLogger):
    def __init__(self, **kwargs):
        super(OutputLogger, self).__init__(**kwargs)


class RemoveDifferentialThreshold(LoggingLayer):
    def __init__(self, threshold=0.05, name="", enable_training=False, enabled=True, **kwargs):
        super(RemoveDifferentialThreshold, self).__init__(name=name, enabled=enabled)
        self.threshold = threshold
        self.prev_frame = None
        self.enable_training = enable_training

    def forward(self, x):
        if self.enabled and (torch.is_grad_enabled() and self.enable_training or not torch.is_grad_enabled()):
            if self.prev_frame is None:
                self.prev_frame = x.detach()
                return x
            mask = torch.abs(x - self.prev_frame) < self.threshold
            x[mask] = self.prev_frame[mask]
            self.prev_frame[~mask] = x[~mask].detach()

        return x

    def disable_logging(self):
        pass

    def enable_logging(self):
        pass

    def reset_history(self):
        self.prev_frame = None


class RemoveConsistentPixels(LoggingLayer):
    def __init__(self, threshold=0.05, name="", enable_training=False, enabled=True, **kwargs):
        super(RemoveConsistentPixels, self).__init__(name=name, enabled=enabled)
        self.threshold = threshold
        self.prev_frame = None
        self.enable_training = enable_training

    def forward(self, x):
        if self.enabled and (torch.is_grad_enabled() and self.enable_training or not torch.is_grad_enabled()):
            if self.prev_frame is None:
                self.prev_frame = x.detach()
                return x
            mask = torch.abs(x - self.prev_frame) < self.threshold
            x[mask] = 0.0
            self.prev_frame[~mask] = x[~mask].detach()

        return x

    def disable_logging(self):
        pass

    def enable_logging(self):
        pass

    def reset_history(self):
        self.prev_frame = None


class GradientLogFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, log_out):
        ctx.log_out = log_out
        return X

    @staticmethod
    def backward(ctx, grad):
        ctx.log_out['loss'] = grad
        return grad, None


class GradientLogger(LoggingLayer):
    def __init__(self, **kwargs):
        super(GradientLogger, self).__init__(**kwargs)
        self.fn = GradientLogFunction()
        self.loss = {'loss': None}

    def reset(self):
        self.loss['loss'] = None

    def forward(self, x):
        self.fn(x, self.loss)
        return x

    def get(self):
        return self.loss['loss']


class Loggers(LoggingLayer):
    threshold = 0.0

    def __init__(self, name="", modes=None, enabled=False, threshold=None, **kwargs):
        super(Loggers, self).__init__(name, enabled, **kwargs)
        self.threshold = threshold if threshold is not None else Loggers.threshold

        if modes is None:
            modes = ["DiffMask", "DiffDensity", "Density", "Mask"]
        self.logs = {}
        if "RemoveDiff" in modes:
            self.logs["RemoveDiff"] = RemoveDifferentialThreshold(enable_training=False, enabled=True, threshold=self.threshold)
        if "RemoveConsistent" in modes:
            self.logs["RemoveConsistent"] = RemoveConsistentPixels(enable_training=False, enabled=True, threshold=self.threshold)
        if "Density" in modes:
            self.logs["Density"] = DensityLogger(name=name, enabled=enabled, threshold=self.threshold, **kwargs)
        if "Mask" in modes:
            self.logs["Mask"] = PixelMaskLogger(name=name, enabled=enabled, threshold=self.threshold, **kwargs)
        if "DiffDensity" in modes:
            self.logs["DiffDensity"] = DifferentialDensityLogger(name=name, enabled=enabled, threshold=self.threshold, **kwargs)
        if "DiffMask" in modes:
            self.logs["DiffMask"] = DifferentialPixelMaskLogger(name=name, enabled=enabled, threshold=self.threshold, **kwargs)

        self.active_layers = ["ReLU", "Input"]
        self.always_use_removediff = False

    def reset(self):
        for name, log in self.logs.items():
            if issubclass(type(log), LoggingLayer):
                log.reset()

    def reset_history(self):
        for name, log in self.logs.items():
            if issubclass(type(log), LoggingLayer):
                log.reset_history()

    def _is_active(self):
        if self.active_layers is None:
            return True

        for layer in self.active_layers:
            if layer in self.name:
                return True

        return False

    def forward(self, x):
        if not self.added_id:
            self.name = f"{LoggingLayer.id} {self.name}"
            for name, log in list(self.logs.items()):
                self.logs[f"{LoggingLayer.id} {name}"] = log
                self.logs.pop(name)
            self.added_id = True

            for name, log in self.logs.items():
                log.name = name
                log.added_id = True

            LoggingLayer.id += 1

        if not self._is_active():
            if self.always_use_removediff:
                for name, log in self.logs.items():
                    if "RemoveDiff" in name:
                        x = log(x)
            return x

        for name, log in self.logs.items():
            if self.enabled:
                log.enable_logging()
            else:
                log.disable_logging()
            x = log(x)

        return x

    def get(self):
        res = {}

        if not self._is_active():
            return res

        for name, log in self.logs.items():
            if issubclass(type(log), LoggingLayer):
                res[name] = log.get()

        return res