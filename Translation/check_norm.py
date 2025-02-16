import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Define the checkpoint directories with a descriptive key for each model
model_dirs = {
    "t5-small": "t5-small",
    "full_fine_tune": "./t5_checkpoints/t5-translation-checkpoints-lr_5e-05_bs_16/checkpoint-8000",
    "lora_fine_tune": "./lora_checkpoints/lora-t5-translation-checkpoints_0.0001_dropout_0/checkpoint-8750"
}

# Set the device (using "mps" if available; otherwise "cpu")
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Create an empty list to store results
results = []

for model_name, model_path in model_dirs.items():
    print(f"Loading model: {model_name} from {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    if model_name == "lora_fine_tune":
        # For LoRA fine-tuned model, iterate over modules.
        for module_name, module in model.named_modules():
            # Skip modules whose name contains "base_layer" or the low-rank factors themselves.
            if "base_layer" in module_name or "lora_A" in module_name or "lora_B" in module_name:
                continue

            # Only process modules that have a base weight.
            if hasattr(module, "weight"):
                # If the module is LoRA-adapted, record effective weight norm.
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    # Compute BA from lora_A and lora_B.
                    if isinstance(module.lora_A, torch.nn.ModuleDict) and isinstance(module.lora_B, torch.nn.ModuleDict):
                        BA = 0
                        # Iterate over each key in the ModuleDict and accumulate BA.
                        for key in module.lora_A:
                            lora_A_weight = module.lora_A[key].weight
                            lora_B_weight = module.lora_B[key].weight
                            BA_component = torch.matmul(lora_B_weight, lora_A_weight)
                            BA = BA + BA_component
                    else:
                        BA = torch.matmul(module.lora_B, module.lora_A)
                    
                    # Compute the effective weight (base weight + BA)
                    effective_weight = module.weight + BA
                    effective_norm = effective_weight.data.norm(p=2).item()
                    results.append({
                        "model": model_name,
                        "layer": module_name,
                        "norm_type": "effective_weight (weight + BA)",
                        "l2_norm": effective_norm
                    })
                else:
                    # For modules without LoRA updates, record the base weight norm.
                    weight_norm = module.weight.data.norm(p=2).item()
                    results.append({
                        "model": model_name,
                        "layer": module_name,
                        "norm_type": "weight",
                        "l2_norm": weight_norm
                    })
    else:
        # For non-LoRA models, iterate over all parameters.
        for param_name, param in model.named_parameters():
            # Skip any parameter name that contains "base_layer"
            if "base_layer" in param_name:
                continue
            l2_norm = param.data.norm(p=2).item()
            results.append({
                "model": model_name,
                "parameter": param_name,
                "norm_type": "parameter",
                "l2_norm": l2_norm
            })
    
    print(f"Finished processing {model_name}.\n")

# Create a DataFrame from the collected results and save to CSV.
df = pd.DataFrame(results)
csv_path = "./results/model_parameter_norms.csv"
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
