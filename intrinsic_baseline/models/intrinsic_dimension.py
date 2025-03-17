# models/intrinsic_dimension.py
import torch
import torch.nn as nn
import numpy as np
from typing import Set
from .projection import fastfood_vars
from .global_intrinsic_linear import GlobalIntrinsicLinear
from typing import Dict, Any

# def replace_linear_with_global(model: nn.Module, intrinsic_parameter: nn.Parameter, 
#                                trainable_layers="ALL", train_mode: str = "attention_matrices"):
#     """
#     Replace specified nn.Linear layers in the transformer encoder with GlobalIntrinsicLinear wrappers.
    
#     Args:
#         model (nn.Module): The pre-trained model.
#         intrinsic_parameter (nn.Parameter): The global intrinsic parameter to be shared.
#         trainable_layers (str or list): If "ALL", replace all encoder layers; 
#                                         otherwise a list of layer indices (e.g. [7,8,9]) to replace.
#         train_mode (str): Which parts to replace. Options:
#                           - "attention_matrices": replace attention-related linear layers.
#                           - "ffn_matrices": replace feed-forward network linear layers.
#                           - "all": replace both.
    
#     Returns:
#         model: Modified model with specified layers replaced.
#     """
#     # Identify the transformer backbone (e.g., "bert")
#     transformer_backbone = None
#     for attr in ["bert", "roberta", "distilbert", "electra", "xlm"]:
#         if hasattr(model, attr):
#             transformer_backbone = getattr(model, attr)
#             break
#     if transformer_backbone is None:
#         raise ValueError("No supported transformer backbone found in the model.")

#     # Locate the encoder layers.
#     if hasattr(transformer_backbone, "encoder"):
#         layers = transformer_backbone.encoder.layer
#     elif hasattr(transformer_backbone, "transformer"):
#         layers = transformer_backbone.transformer.layer
#     else:
#         raise ValueError("Unable to find encoder layers in the transformer backbone.")

#     # Determine which layer indices to replace.
#     if trainable_layers == "ALL":
#         indices = list(range(len(layers)))
#     else:
#         indices = trainable_layers  # assume it's already a list of indices

#     # For each specified encoder layer, replace its linear layers.
#     for idx in indices:
#         layer = layers[idx]
#         if train_mode in ["all", "attention_matrices"]:
#             # Replace attention-related linear layers:
#             for attr in ["query", "key", "value"]:
#                 linear_layer = getattr(layer.attention.self, attr)
#                 num_elements = int(np.prod(linear_layer.weight.shape))
#                 proj_params = fastfood_vars(num_elements, device=linear_layer.weight.device)
#                 setattr(layer.attention.self, attr, GlobalIntrinsicLinear(linear_layer, proj_params, intrinsic_parameter))
#                 print(f"Replaced layer {idx} attention.self.{attr} successfully.")
#             # Replace attention output linear layer.
#             linear_layer = layer.attention.output.dense
#             num_elements = int(np.prod(linear_layer.weight.shape))
#             proj_params = fastfood_vars(num_elements, device=linear_layer.weight.device)
#             setattr(layer.attention.output, "dense", GlobalIntrinsicLinear(linear_layer, proj_params, intrinsic_parameter))
#             print(f"Replaced layer {idx} attention.output.dense successfully.")

#         if train_mode in ["all", "ffn_matrices"]:
#             # Replace feed-forward network (FFN) linear layers:
#             linear_layer = layer.intermediate.dense
#             num_elements = int(np.prod(linear_layer.weight.shape))
#             proj_params = fastfood_vars(num_elements, device=linear_layer.weight.device)
#             setattr(layer.intermediate, "dense", GlobalIntrinsicLinear(linear_layer, proj_params, intrinsic_parameter))
#             print(f"Replaced layer {idx} intermediate.dense successfully.")

#             linear_layer = layer.output.dense
#             num_elements = int(np.prod(linear_layer.weight.shape))
#             proj_params = fastfood_vars(num_elements, device=linear_layer.weight.device)
#             setattr(layer.output, "dense", GlobalIntrinsicLinear(linear_layer, proj_params, intrinsic_parameter))
#             print(f"Replaced layer {idx} output.dense successfully.")
#     return model

def replace_linear_with_global(module: nn.Module, intrinsic_parameter: nn.Parameter, str_filter: Set[str] = None):
    """
    Recursively replace adjustable nn.Linear layers with GlobalIntrinsicLinear wrappers.
    Only replace layers whose names match any string in str_filter (if provided).
    If str_filter is empty, replace all nn.Linear layers.
    """
    # # Freeze all parameters in the model.
    # for param in module.parameters():
    #     param.requires_grad = False

    for name, child in module.named_children():
        print(name)
        if isinstance(child, nn.Linear):
            # Check if we should replace this layer.
            if str_filter and not any(s in name for s in str_filter):
                continue
            # Compute number of elements in weight.
            num_elements = int(np.prod(child.weight.shape))
            # Generate projection parameters for this layer.
            proj_params = fastfood_vars(num_elements, device=child.weight.device)
            # Replace the linear layer with GlobalIntrinsicLinear.
            setattr(module, name, GlobalIntrinsicLinear(child, proj_params, intrinsic_parameter))
            print("--------------------replace successfully--------------------")
        else:
            # Recursively process child modules.
            replace_linear_with_global(child, intrinsic_parameter, str_filter)
    return module

def intrinsic_dimension(module: nn.Module, intrinsic_dimension: int, output_dir: str, 
                        training_config: Dict[str, Any], projection: str = "global", device: str = "cuda") -> nn.Module:
    """
    Applies the global intrinsic dimension transformation to all adjustable parameters in the model
    by replacing linear layers with a global intrinsic linear wrapper.
    """
    # Freeze all parameters in the model.
    for param in module.parameters():
        param.requires_grad = False

    # Create the global intrinsic parameter (trainable)
    intrinsic_parameter = nn.Parameter(torch.zeros(intrinsic_dimension, device=device), requires_grad=True)
    module.register_parameter("intrinsic_parameter", intrinsic_parameter)
    # Replace all nn.Linear layers (or those matching str_filter) with the wrapper.
    str_filter = training_config['str_filter']
    module = replace_linear_with_global(module, intrinsic_parameter, str_filter)
    # trainable_layers = training_config["trainable_layers"]
    # train_mode = training_config["train_mode"]
    # module = replace_linear_with_global(module, intrinsic_parameter, trainable_layers, train_mode)
    return module