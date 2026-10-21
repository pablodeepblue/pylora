"""A simple implementation of LoRA (Low-Rank Adaptation) in PyTorch."""

from abc import abstractmethod
from itertools import repeat
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    """Shared state for LoRA layers: rank, scaling, dropout and merge flags."""

    def __init__(
        self,
        rank: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.merged = False
        self.merge_weights = merge_weights
        self.scaling = self.lora_alpha / self.rank
        self.lora_dropout_func = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def merge(self, merge: bool = True):
        ...


# ---------------------------------------------------------------------------
# Frozen layers
# ---------------------------------------------------------------------------

# TODO: maybe replace nn.Linear with FrozenLinear for consistent implementations?
class FrozenLinear(nn.Linear):
    """nn.Linear with frozen weight and bias."""

    def __init__(self, *args, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    @staticmethod
    def _from_linear(module: nn.Linear, **kwargs):
        if isinstance(module, nn.Linear):
            out_size, in_size = module.weight.shape
            lora_module = FrozenLinear(
                in_size,
                out_size,
                bias=module.bias is not None,
                **kwargs
            )
            lora_module.weight.data = module.weight.data
            if module.bias is not None:
                lora_module.bias.data = module.bias.data
            return lora_module
        else:
            raise ValueError(
                "_from_linear classmethod supports only objects "
                f"of torch.nn.Linear class, but {str(type(module))} is given."
            )


class FrozenConv2d(nn.Conv2d):
    """nn.Conv2d with frozen weight and bias."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    @staticmethod
    def _from_conv2d(module: nn.Conv2d, **kwargs):
        if isinstance(module, nn.Conv2d):
            out_channels, in_channels, kh, kw = module.weight.shape

            frozen_module = FrozenConv2d(
                in_channels,
                out_channels,
                kernel_size=(kh, kw),
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
                **kwargs
            )

            frozen_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                frozen_module.bias.data = module.bias.data.clone()

            return frozen_module
        else:
            raise ValueError(
                "_from_conv2d supports only torch.nn.Conv2d, "
                f"but got {type(module)}."
            )


# ---------------------------------------------------------------------------
# LoRA layers
# ---------------------------------------------------------------------------

class LoRAConv2d(nn.Conv2d, LoRALayer):
    """nn.Conv2d with a trainable low-rank update on top of frozen weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size,
        rank: int,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        device=None,
        dtype=None
    ):
        assert rank > 0, "LoRA rank must be greater than 0"
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.Conv2d.__init__(
            self,
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs
        )
        # Freezing the pre-trained weight matrix and bias (following peft example)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        out_channels, in_channels, kH, kW = self.weight.size()

        LoRALayer.__init__(self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        # Actual trainable parameters
        flattened_in = in_channels * kH * kW

        self.lora_A = nn.Parameter(
            torch.zeros(rank, flattened_in, **factory_kwargs),
            requires_grad=True,
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_channels, rank, **factory_kwargs),
            requires_grad=True,
        )

        # Parameter-count difference vs. the full weight (shown in repr)
        self.size_diff = torch.numel(self.lora_A) + torch.numel(self.lora_B) - torch.numel(self.weight)
        self.reset_parameters()
        self.merged = False

    def _lora_weight(self):
        """
        Returns LoRA weight update reshaped to Conv2d weight shape.
        """
        delta = self.lora_B @ self.lora_A
        return delta.view_as(self.weight) * self.scaling

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lora_A"):
            nn.init.uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                self.weight.data -= self._lora_weight()
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                self.weight.data += self._lora_weight()
                self.merged = True

    @torch.jit.export
    def merge(self, merge: bool = True):
        with torch.no_grad():
            if merge and not self.merged:
                self.weight.add_(self._lora_weight())
                self.merged = True
            if not merge and self.merged:
                self.weight.sub_(self._lora_weight())
                self.merged = False

    def forward(self, x: torch.Tensor):
        if not self.merged:
            weight = self.weight + self._lora_weight()
            return F.conv2d(
                x,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"bias={self.bias is not None}, "
            f"rank={self.rank}, "
            f"size_diff={self.size_diff}"
        )

    def state_dict(self, *args, keep_weights=False, keep_bias=False, **kwargs):
        dest = super().state_dict(*args, **kwargs)
        if not keep_weights or not keep_bias:
            for k in list(dest.keys()):
                if not keep_weights and "weight" in k:
                    del dest[k]
                elif not keep_bias and "bias" in k:
                    del dest[k]
        return dest

    @staticmethod
    def _from_conv2d(
        module: nn.Conv2d,
        rank: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        share_mem: bool = False,
        device=None,
        dtype=None,
    ):
        if not isinstance(module, nn.Conv2d):
            raise ValueError(
                "_from_conv2d method supports only objects "
                f"of torch.nn.Conv2d class, but {type(module)} is given."
            )

        if rank == 0:
            raise ValueError('Rank 0 not supported')

        device = module.weight.device
        dtype = module.weight.dtype

        lora_module = LoRAConv2d(
            in_features=module.in_channels,
            out_features=module.out_channels,
            kernel_size=module.kernel_size,
            rank=rank,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            device=device,
            dtype=dtype,
        )

        if share_mem:
            lora_module.weight = module.weight
            lora_module.bias = module.bias
        else:
            lora_module.weight.data.copy_(module.weight.data)

            if module.bias is not None:
                lora_module.bias.data.copy_(module.bias.data)

        return lora_module


class LoRALinear(nn.Linear, LoRALayer):
    """nn.Linear with a trainable low-rank update on top of frozen weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        device=None,
        dtype=None,
    ):
        assert rank > 0, "LoRA rank must be greater than 0"
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias, **factory_kwargs)
        # Freezing the pre-trained weight matrix and bias (following peft example)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        LoRALayer.__init__(self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        # Actual trainable parameters
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features, **factory_kwargs),
            requires_grad=True,
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, **factory_kwargs),
            requires_grad=True,
        )

        # Parameter-count difference vs. the full weight (shown in repr)
        self.size_diff = torch.numel(self.lora_A) + torch.numel(self.lora_B) - torch.numel(self.weight)
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lora_A"):
            nn.init.uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                self.weight.data -= self.lora_B @ self.lora_A * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                self.weight.data += self.lora_B @ self.lora_A * self.scaling
                self.merged = True

    def _lora_weight(self):
        return self.lora_B @ self.lora_A * self.scaling

    @torch.jit.export
    def merge(self, merge: bool = True):
        with torch.no_grad():
            if merge and not self.merged:
                self.weight.add_(self._lora_weight())
                self.merged = True
            if not merge and self.merged:
                self.weight.sub_(self._lora_weight())
                self.merged = False

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        if self.merged:
            return result
        lora_x = self.lora_dropout_func(x)
        lora_h = F.linear(lora_x, self.lora_A) * self.scaling
        result += F.linear(lora_h, self.lora_B)
        return result

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}, size_diff={self.size_diff}"

    def state_dict(self, *args, keep_weights=False, keep_bias=False, **kwargs):
        dest = super().state_dict(*args, **kwargs)
        if not keep_weights or not keep_bias:
            for k in list(dest.keys()):
                if not keep_weights and "weight" in k:
                    del dest[k]
                elif not keep_bias and "bias" in k:
                    del dest[k]
        return dest

    @staticmethod
    def _from_linear(
        module: nn.Linear,
        rank: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        share_mem: bool = False,
        device=None,
        dtype=None,
    ):
        if isinstance(module, nn.Linear):
            if rank == 0:
                return FrozenLinear._from_linear(module)
            out_size, in_size = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype
            lora_module = LoRALinear(
                in_size,
                out_size,
                rank=rank,
                bias=module.bias is not None,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
                device=device,
                dtype=dtype,
            )
            if share_mem:
                lora_module.weight = module.weight
                lora_module.bias = module.bias
            else:
                lora_module.weight.data = module.weight.data
                if module.bias is not None:
                    lora_module.bias.data = module.bias.data
            return lora_module
        else:
            raise ValueError(
                "_from_linear method supports only objects "
                f"of torch.nn.Linear class, but {str(type(module))} is given."
            )


# ---------------------------------------------------------------------------
# Swapping between several LoRAs
# ---------------------------------------------------------------------------

class MultiLora(nn.Module):
    """Wraps a Linear/Conv2d and swaps between several LoRA updates in place."""

    def __init__(self, base_module):
        super().__init__()
        assert isinstance(base_module, (nn.Linear, nn.Conv2d)), \
            f"Only created for these 2 types for now, not {type(base_module)}"
        self.base_module = base_module
        self.loras: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        self.current_index: int = -1

    @torch.jit.unused
    def _apply(self, fn):
        super()._apply(fn)
        self.loras = [
            (fn(lora_A), fn(lora_B), scaling)
            for (lora_A, lora_B, scaling) in self.loras
        ]
        return self

    @torch.jit.unused
    def append(self, lora: "LoRALayer"):
        self.loras.append((lora.lora_A, lora.lora_B, lora.scaling))

    def _lora_weight(self, triple: Tuple[torch.Tensor, torch.Tensor, float]):
        lora_A, lora_B, scaling = triple
        return (lora_B @ lora_A).view_as(self.base_module.weight) * scaling

    @torch.jit.export
    def swap(self, index: int):
        assert -1 <= index < len(self.loras), f"index {index} out of range"
        if index == self.current_index:
            return
        with torch.no_grad():
            weight: Optional[torch.Tensor] = None
            for i, lora in enumerate(self.loras):
                if i == self.current_index:
                    self.base_module.weight -= self._lora_weight(lora)
                if i == index:
                    weight = self._lora_weight(lora)
            if weight is not None:
                self.base_module.weight += weight
        self.current_index = index

    def forward(self, input_tensor):
        return self.base_module.forward(input_tensor)


class SequentialMultiLora(nn.Sequential):
    """nn.Sequential whose Linear/Conv2d layers are wrapped in MultiLora."""

    def __init__(self, sequence):
        super().__init__(*[MultiLora(module) if isinstance(module, (nn.Linear, nn.Conv2d)) else module for module in sequence])

    @torch.jit.unused
    def append(self, lora_sequence):
        multilora_layers = [m for m in self]
        lora_list = list(lora_sequence)
        for base, lora in zip(multilora_layers, lora_list):
            if isinstance(base, MultiLora):
                base.append(lora)

    @torch.jit.export
    def swap(self, index: int):
        for base in self:
            if hasattr(base, "loras"):
                base.swap(index)


# ---------------------------------------------------------------------------
# Model conversion helpers
# ---------------------------------------------------------------------------

def _from_sequential(model: nn.Sequential, ranks=None, targets=None):
    modules = []
    check = targets is None
    if ranks is None:
        ranks = iter([])
    elif isinstance(ranks, int):
        ranks = repeat(ranks)
    else:
        ranks = iter(ranks)
    for i, layer in enumerate(model):
        if (check or i in targets) and isinstance(layer, nn.Linear):
            rank = next(ranks, 1)
            modules.append(LoRALinear._from_linear(layer, rank))
        elif (check or i in targets) and isinstance(layer, nn.Conv2d):
            rank = next(ranks, 1)
            modules.append(LoRAConv2d._from_conv2d(layer, rank))
        elif isinstance(layer, nn.Linear):
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            modules.append(FrozenLinear._from_linear(layer))
        elif isinstance(layer, nn.Conv2d):
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            modules.append(FrozenConv2d._from_conv2d(layer))
        else:
            modules.append(layer)
    lora_model = nn.Sequential(*modules)
    return lora_model


def _merge_seq(model, merge: bool = True):
    if isinstance(model, LoRALinear):
        model.merge(merge)
        return
    if isinstance(model, nn.Sequential):
        for layer in model:
            if hasattr(layer, LoRALinear):
                layer.merge(merge)


class Constructed_LoRAs(nn.Module):
    """Holds several LoRA models that share the base model's parameters."""

    def __init__(self, base_model: nn.Sequential, loras_models: List[nn.Sequential]):
        super().__init__()

        base_params = dict(base_model.named_parameters())
        base_buffers = dict(base_model.named_buffers())
        self.loras = nn.ModuleList()

        for lora_model in loras_models:
            for name, param in lora_model.named_parameters():
                if name in base_params:
                    parent_name, param_name = self._get_parent_and_name(lora_model, name)
                    setattr(parent_name, param_name, base_params[name])
            for name, buf in lora_model.named_buffers():
                if name in base_buffers:
                    parent_name, buf_name = self._get_parent_and_name(lora_model, name)
                    setattr(parent_name, buf_name, base_buffers[name])

            self.loras.append(lora_model)

        self.index: int = 0
        for layer in self.loras[0]:
            if isinstance(layer, LoRALinear):
                layer.merge()

    def _get_parent_and_name(self, model, target_name):
        """Helper to find the parent module of a nested parameter name."""
        names = target_name.split(".")
        curr = model
        for i in range(len(names) - 1):
            curr = getattr(curr, names[i])
        return curr, names[-1]

    @torch.jit.export
    def swap(self, inp: int):
        assert 0 <= inp < len(self.loras), f"index {inp} out of range"
        for i, model in enumerate(self.loras):
            if i == self.index:
                for layer in model:
                    if isinstance(layer, LoRALinear):
                        layer.merge(False)
        self.index = inp
        for i, model in enumerate(self.loras):
            if i == self.index:
                for layer in model:
                    if isinstance(layer, LoRALinear):
                        layer.merge(True)

    def forward(self, inp):
        for i, model in enumerate(self.loras):
            if i == self.index:
                return model.forward(inp)
        raise RuntimeError("active LoRA index is out of range")
