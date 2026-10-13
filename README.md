<h1 align="center">pylora</h1>

<p align="center">
  <b>A small, readable implementation of LoRA (Low-Rank Adaptation) in PyTorch.</b><br>
  Freeze a pretrained layer, train a tiny low-rank update, merge it away for inference, or swap between several adapters on the fly.
</p>

<p align="center">
  <img alt="language: Python" src="https://img.shields.io/badge/language-Python-blue">
  <img alt="built with: PyTorch" src="https://img.shields.io/badge/built%20with-PyTorch-ee4c2c">
  <img alt="license: GPL-2.0" src="https://img.shields.io/badge/license-GPL--2.0-green">
</p>

---
## Acknowledgement

This work is based on [PyTorch Adventures](https://github.com/priyammaz/PyTorch-Adventures) and [LoRA](https://github.com/microsoft/LoRA/tree/main) repos implementation.
This a clean version of the one used in a [Legged Gym EX fork](https://github.com/pghezzi/Legged_Gym_EX) I used for a paper in initial testing.

---

## What is LoRA?

Instead of fine-tuning a full weight matrix `W`, LoRA freezes it and learns a low-rank update:

```
W' = W + (alpha / rank) · B · A
```

`A` has shape `(rank, in_features)` and `B` has shape `(out_features, rank)`. With a small `rank`, that is a tiny fraction of the parameters of `W`. `B` starts at zero, so a fresh adapter leaves the model's output unchanged.

## Features

- **Drop-in layers:** `LoRALinear` and `LoRAConv2d` subclass `nn.Linear` and `nn.Conv2d`
- **Frozen base weights:** only `lora_A` and `lora_B` are trainable
- **Convert existing layers or whole `nn.Sequential` models** to LoRA in one call
- **Merge and unmerge** the adapter into the base weights for zero-overhead inference
- **Adapter-only checkpoints:** `state_dict()` on a LoRA layer skips the base weight and bias by default
- **Multi-LoRA swapping:** keep several adapters on one base model and switch between them in place
- **TorchScript-friendly:** `Constructed_LoRAs` can be scripted, saved and reloaded with `torch.jit`
- **One file, no dependencies** beyond PyTorch

## Installation

Install straight from GitHub:

```sh
pip install git+https://github.com/pablodeepblue/pylora.git
```

Or, since the library is a single file, copy `pylora.py` into your project.

Requires [PyTorch](https://pytorch.org/get-started/locally/). If you need a specific CUDA build, install PyTorch first, then pylora.

## Quick start

### Create a LoRA layer

```python
import torch
from pylora import LoRALinear

layer = LoRALinear(
    in_features=128,
    out_features=64,
    rank=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

y = layer(torch.randn(4, 128))

# Only the adapter is trainable
trainable = [p for p in layer.parameters() if p.requires_grad]   # lora_A, lora_B
optimizer = torch.optim.AdamW(trainable, lr=1e-3)
```

`LoRAConv2d(in_features, out_features, kernel_size, rank, ...)` works the same way for convolutions.

### Convert an existing layer

```python
import torch.nn as nn
from pylora import LoRALinear, LoRAConv2d

lora_fc = LoRALinear._from_linear(nn.Linear(512, 256), rank=8, lora_alpha=16)
lora_conv = LoRAConv2d._from_conv2d(nn.Conv2d(3, 16, 3), rank=4)
```

The converted layer starts with the original layer's weights, which stay frozen. Pass `share_mem=True` to reuse the original weight and bias `Parameter` objects directly.

### Convert a whole `nn.Sequential`

```python
from pylora import _from_sequential

model = nn.Sequential(nn.Linear(100, 20), nn.ReLU(), nn.Linear(20, 70), nn.ReLU())

lora_model = _from_sequential(model, ranks=8)          # same rank for every layer
lora_model = _from_sequential(model, ranks=[10, 3])    # one rank per layer, in order
lora_model = _from_sequential(model, ranks=8, targets=[0])  # only layer 0 gets LoRA
```

Layers that are not LoRA targets are replaced with frozen equivalents. If you run out of ranks in a list (or pass none), the default rank is `1`.

### Merge for inference

With `merge_weights=True` (the default), switching modes does the merging for you:

```python
lora_model.eval()    # adapter merged into the base weights, no extra compute
lora_model.train()   # adapter unmerged again, ready for more training
```

You can also merge manually:

```python
layer.merge(True)    # fold the adapter into the weight
layer.merge(False)   # take it back out
```

### Save only the adapter

```python
torch.save(layer.state_dict(), "adapter.pt")            # lora_A and lora_B only
layer.state_dict(keep_weights=True, keep_bias=True)     # include the base weight and bias too
```

## Swapping between several adapters

Keep one base model and flip between adapters without reloading anything.

### `Constructed_LoRAs`

Holds several LoRA models that share the base model's parameters. `swap(i)` unmerges the current adapter and merges adapter `i`. It works under TorchScript:

```python
from pylora import Constructed_LoRAs, _from_sequential

base = nn.Sequential(nn.Linear(100, 20), nn.ReLU(), nn.Linear(20, 70), nn.ReLU())
adapter_a = _from_sequential(base, [10, 3])
adapter_b = _from_sequential(base, [10, 3])

factory = Constructed_LoRAs(base, [adapter_a, adapter_b])

scripted = torch.jit.script(factory)
torch.jit.save(scripted, "lora_factory.pt")

loaded = torch.jit.load("lora_factory.pt")
x = torch.randn(1, 100)

loaded.swap(0)
out_a = loaded(x)
loaded.swap(1)
out_b = loaded(x)
```

A complete runnable version is in [`examples/demo.py`](examples/demo.py). From the repo root, run it with `PYTHONPATH=. python examples/demo.py` (or just `python examples/demo.py` after installing the package).

### `MultiLora` and `SequentialMultiLora`

`MultiLora` wraps a single `nn.Linear` or `nn.Conv2d` and applies a chosen adapter directly to the base weight. `SequentialMultiLora` does the same for every layer of a sequential model:

```python
from pylora import SequentialMultiLora

multi = SequentialMultiLora(base)
multi.append(_from_sequential(base, 8))   # adapter 0
multi.append(_from_sequential(base, 8))   # adapter 1

multi.swap(0)    # use adapter 0
multi.swap(1)    # switch to adapter 1
multi.swap(-1)   # back to the plain base weights
```

## API overview

| Name | What it is |
| --- | --- |
| `LoRALinear` | `nn.Linear` with a trainable low-rank update |
| `LoRAConv2d` | `nn.Conv2d` with a trainable low-rank update |
| `FrozenLinear`, `FrozenConv2d` | Plain layers with frozen weight and bias, used for non-target layers |
| `LoRALinear._from_linear`, `LoRAConv2d._from_conv2d` | Convert an existing layer to LoRA |
| `_from_sequential` | Convert an `nn.Sequential` to LoRA, with per-layer `ranks` and optional `targets` |
| `Constructed_LoRAs` | Several LoRA models sharing one base, with `swap(i)` (TorchScript-friendly) |
| `MultiLora`, `SequentialMultiLora` | Swap adapters directly in the base weights with `swap(i)` |

Common layer arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `rank` | required | Rank of the update. Must be greater than 0 when constructing a layer directly |
| `lora_alpha` | `1` | Scaling numerator; the update is scaled by `lora_alpha / rank` |
| `lora_dropout` | `0.0` | Dropout applied to the input of the LoRA branch |
| `merge_weights` | `True` | Merge on `.eval()` and unmerge on `.train()` |
| `share_mem` | `False` | In `LoRALinear._from_linear` and `LoRAConv2d._from_conv2d`, reuse the original weight and bias `Parameter` objects |

## Notes and current limitations

- The conversion helpers (`_from_linear`, `_from_conv2d`, `_from_sequential`) are underscore-prefixed for now, but they are the intended way to convert existing models.
- Converting an `nn.Linear` assumes it has a bias term.
- `lora_dropout` is applied by `LoRALinear`. `LoRAConv2d` accepts the argument but does not use it yet.
- `Constructed_LoRAs.swap` merges and unmerges `LoRALinear` layers only.
- `MultiLora` and `SequentialMultiLora` support `nn.Linear` and `nn.Conv2d` only.
- There is no automated test suite yet.

## Contributing

Issues and pull requests are welcome. The whole library lives in [`pylora.py`](pylora.py).

## License

Released under the [GNU General Public License v2.0](LICENSE).
