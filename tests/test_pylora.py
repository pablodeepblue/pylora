import copy

import pytest
import torch
from torch import nn

from pylora import (
    Constructed_LoRAs,
    FrozenConv2d,
    FrozenLinear,
    LoRAConv2d,
    LoRALinear,
    MultiLora,
    SequentialMultiLora,
    _from_sequential,
)


torch.manual_seed(0)


def set_adapter(layer, value=0.25):
    """Give a LoRA layer a deterministic, non-zero update."""
    with torch.no_grad():
        layer.lora_A.fill_(value)
        layer.lora_B.fill_(value)


def test_linear_has_frozen_base_and_trainable_adapter():
    layer = LoRALinear(4, 3, rank=2, lora_alpha=4)
    inputs = torch.randn(2, 4)

    assert not layer.weight.requires_grad
    assert not layer.bias.requires_grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad
    assert layer.scaling == 2
    assert torch.allclose(layer(inputs), nn.functional.linear(inputs, layer.weight, layer.bias))


def test_linear_zero_adapter_matches_base_and_dropout_is_applied():
    layer = LoRALinear(4, 3, rank=2, lora_dropout=0.5)
    inputs = torch.randn(8, 4)

    assert torch.allclose(layer(inputs), nn.functional.linear(inputs, layer.weight, layer.bias))

    set_adapter(layer)
    layer.train()
    train_output = layer(inputs)
    layer.eval()
    eval_output = layer(inputs)
    assert not torch.allclose(train_output, eval_output)


@pytest.mark.parametrize("layer_type", ["linear", "conv"])
def test_merge_unmerge_and_mode_transitions_preserve_outputs(layer_type):
    if layer_type == "linear":
        layer, inputs = LoRALinear(4, 3, rank=2, lora_alpha=2), torch.randn(2, 4)
    else:
        layer, inputs = LoRAConv2d(2, 3, 3, rank=2, padding=1, lora_alpha=2), torch.randn(2, 2, 5, 5)
    set_adapter(layer)
    base_weight = layer.weight.detach().clone()
    unmerged_output = layer(inputs)

    layer.merge()
    assert layer.merged
    assert torch.allclose(layer(inputs), unmerged_output)
    layer.merge()  # Merging twice must not apply the update twice.
    assert torch.allclose(layer(inputs), unmerged_output)

    layer.train()
    assert not layer.merged
    assert torch.allclose(layer.weight, base_weight)
    assert torch.allclose(layer(inputs), unmerged_output)
    layer.eval()
    assert layer.merged
    assert torch.allclose(layer(inputs), unmerged_output)


def test_conversion_preserves_output_and_can_share_parameters():
    linear = nn.Linear(4, 3)
    converted = LoRALinear._from_linear(linear, rank=2)
    shared = LoRALinear._from_linear(linear, rank=2, share_mem=True)
    inputs = torch.randn(2, 4)

    assert torch.allclose(converted(inputs), linear(inputs))
    assert shared.weight is linear.weight
    assert shared.bias is linear.bias

    conv = nn.Conv2d(2, 3, 3, stride=2, padding=1, bias=False)
    converted_conv = LoRAConv2d._from_conv2d(conv, rank=2)
    conv_inputs = torch.randn(2, 2, 7, 7)
    assert torch.allclose(converted_conv(conv_inputs), conv(conv_inputs))


def test_adapter_state_dict_excludes_base_unless_requested():
    layer = LoRALinear(4, 3, rank=2)

    assert set(layer.state_dict()) == {"lora_A", "lora_B"}
    assert set(layer.state_dict(keep_weights=True, keep_bias=True)) == {
        "weight", "bias", "lora_A", "lora_B"
    }


def test_sequential_conversion_targets_and_freezes_other_layers():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Conv2d(1, 2, 3))
    converted = _from_sequential(model, ranks=[2, 3], targets=[0, 2])

    assert isinstance(converted[0], LoRALinear)
    assert isinstance(converted[2], LoRAConv2d)
    assert converted[0].rank == 2
    assert converted[2].rank == 3

    only_first = _from_sequential(model, ranks=2, targets=[0])
    assert isinstance(only_first[0], LoRALinear)
    assert isinstance(only_first[2], FrozenConv2d)
    assert not only_first[2].weight.requires_grad


def test_rank_zero_linear_conversion_returns_frozen_layer():
    source = nn.Linear(4, 3, bias=False)
    converted = LoRALinear._from_linear(source, rank=0)
    assert isinstance(converted, FrozenLinear)
    assert not converted.weight.requires_grad
    assert converted.bias is None
    inputs = torch.randn(2, 4)
    assert torch.allclose(converted(inputs), source(inputs))


def test_multilora_swaps_between_adapter_updates_and_base():
    base = nn.Linear(4, 3, bias=False)
    wrapper = MultiLora(base)
    first = LoRALinear._from_linear(copy.deepcopy(base), rank=1)
    second = LoRALinear._from_linear(copy.deepcopy(base), rank=1)
    set_adapter(first, 0.1)
    set_adapter(second, 0.2)
    wrapper.append(first)
    wrapper.append(second)
    inputs = torch.randn(2, 4)
    plain = base(inputs).clone()

    wrapper.swap(0)
    first_output = wrapper(inputs)
    wrapper.swap(1)
    second_output = wrapper(inputs)
    wrapper.swap(-1)

    assert not torch.allclose(first_output, second_output)
    assert torch.allclose(wrapper(inputs), plain)


def test_sequential_multilora_swaps_every_eligible_layer():
    base = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    adapters = _from_sequential(copy.deepcopy(base), ranks=1)
    for layer in adapters:
        if isinstance(layer, LoRALinear):
            set_adapter(layer)
    multi = SequentialMultiLora(base)
    inputs = torch.randn(2, 4)
    plain = multi(inputs).clone()
    multi.append(adapters)
    multi.swap(0)

    assert not torch.allclose(multi(inputs), plain)
    multi.swap(-1)
    assert torch.allclose(multi(inputs), plain)


def test_constructed_loras_can_swap_and_be_torchscripted(tmp_path):
    base = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    first = _from_sequential(copy.deepcopy(base), ranks=1)
    second = _from_sequential(copy.deepcopy(base), ranks=1)
    for layer in first:
        if isinstance(layer, LoRALinear):
            set_adapter(layer, 0.1)
    for layer in second:
        if isinstance(layer, LoRALinear):
            set_adapter(layer, 0.2)
    factory = Constructed_LoRAs(base, [first, second])
    inputs = torch.randn(2, 4)

    factory.swap(0)
    first_output = factory(inputs)
    factory.swap(1)
    second_output = factory(inputs)
    assert not torch.allclose(first_output, second_output)

    scripted = torch.jit.script(factory)
    output_path = tmp_path / "factory.pt"
    torch.jit.save(scripted, output_path)
    loaded = torch.jit.load(output_path)
    loaded.swap(0)
    loaded_first = loaded(inputs)
    loaded.swap(1)
    loaded_second = loaded(inputs)
    loaded.swap(0)

    assert not torch.allclose(loaded_first, loaded_second)
    assert torch.allclose(loaded(inputs), loaded_first)
