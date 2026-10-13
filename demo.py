"""Demo: build several LoRA variants of a base model, script them, save, reload and swap."""

import torch
import torch.nn as nn

from pylora import Constructed_LoRAs, _from_sequential


def main():
    base_model = nn.Sequential(
        nn.Linear(100, 20), nn.ReLU(), nn.Linear(20, 70), nn.ReLU()
    )

    base_model_2 = nn.Sequential(
        nn.Linear(100, 20), nn.ReLU(), nn.Linear(20, 70), nn.ReLU()
    )

    lora_model = _from_sequential(base_model, [10, 3])
    lora_model_2 = _from_sequential(base_model_2, [10, 3])
    lora_model_3 = _from_sequential(base_model_2, [10, 3])

    lora_model_2.load_state_dict(base_model.state_dict(), strict=False)
    lora_model_2.load_state_dict(lora_model.state_dict(), strict=False)

    d = base_model.state_dict()
    d.update(lora_model.state_dict())

    lora_model_3.load_state_dict(d)

    lora_model_4 = _from_sequential(base_model_2, [10, 3])

    factory = Constructed_LoRAs(base_model, [lora_model, lora_model_4, lora_model_2])

    print("Saving TorchScript model...")
    scripted_model = torch.jit.script(factory)
    torch.jit.save(scripted_model, "lora_factory.pt")

    print("Loading TorchScript model...")
    loaded_model = torch.jit.load("lora_factory.pt")

    x = torch.randn(1, 100)

    loaded_model.swap(0)
    out_a = loaded_model(x)
    print(out_a)

    loaded_model.swap(1)
    out_b = loaded_model(x)
    print(out_b)

    loaded_model.swap(0)
    out_c = loaded_model(x)
    print(out_c)

    # Verification
    is_same = torch.allclose(out_a, out_c)
    print(f"Consistency Check (Swap 0 vs Repeat Swap 0): {'PASSED' if is_same else 'FAILED'}")


if __name__ == "__main__":
    main()
