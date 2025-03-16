# models/intrinsic_dimension.py
import torch
import torch.nn as nn
import numpy as np
from typing import Set
from .projection import fastfood_vars
from .global_intrinsic_linear import GlobalIntrinsicLinear

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
        # else:
        #     # Recursively process child modules.
        #     replace_linear_with_global(child, intrinsic_parameter, str_filter)
    return module

def intrinsic_dimension(module: nn.Module, intrinsic_dimension: int, output_dir: str, 
                        str_filter: Set[str] = None, projection: str = "global", device: str = "cuda") -> nn.Module:
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
    module = replace_linear_with_global(module, intrinsic_parameter, str_filter)
    return module