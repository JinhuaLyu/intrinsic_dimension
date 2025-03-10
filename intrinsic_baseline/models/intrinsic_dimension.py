# models/intrinsic_dimension.py
import torch
import torch.nn as nn
import numpy as np
from typing import Set
from .projection import fastfood_vars, fastfood_torched

class GlobalIntrinsicDimensionHook:
    """
    A hook that applies a global intrinsic dimension projection to all adjustable parameters of a module.
    
    For each adjustable parameter, a random projection is generated using FastFood,
    and the update is computed as: update = Projection(intrinsic_parameter)
    The updated parameter is then: new_value = initial_value + update.
    """
    def __init__(self, module: nn.Module, intrinsic_dim: int, projection: str = "fastfood", 
                 device: str = "cuda", str_filter: Set[str] = None):
        self.module = module
        self.intrinsic_dim = intrinsic_dim
        self.projection = projection
        self.device = torch.device(device)
        self.str_filter = str_filter if str_filter is not None else set()
        # Global low-dimensional parameter vector, ensure it is trainable.
        self.intrinsic_parameter = nn.Parameter(torch.zeros(intrinsic_dim, device=self.device), requires_grad=True)
        module.register_parameter("intrinsic_parameter", self.intrinsic_parameter)
        
        # Dictionaries to store initial parameter values and projection parameters.
        self.initial_values = {}
        self.proj_params = {}
        self.param_list = []
        for name, param in module.named_parameters():
            # Skip the intrinsic parameter and only select adjustable parameters.
            if param.requires_grad and "intrinsic_parameter" not in name and (not self.str_filter or any(f in name for f in self.str_filter)):
                self.initial_values[name] = param.clone().detach().to(self.device)
                num_elements = int(np.prod(param.shape))
                self.param_list.append((name, param, param.shape, num_elements))
                # Use fastfood_vars to generate projection parameters (a tuple) for this parameter.
                self.proj_params[name] = fastfood_vars(num_elements, device=self.device)
                # Freeze the original parameter so that only intrinsic_parameter is trainable.
                param.requires_grad = False

    def hook(self, module, inputs):
        # For each adjustable parameter, compute its update using the global intrinsic parameter.
        for name, param, shape, num_elements in self.param_list:
            # Retrieve the LL value from the corresponding projection parameters.
            _, _, _, _, LL = self.proj_params[name]
            # Adjust the global intrinsic_parameter to the required length LL.
            theta_adjusted = self.intrinsic_parameter
            if theta_adjusted.numel() > LL:
                theta_adjusted = theta_adjusted[:LL]
            elif theta_adjusted.numel() < LL:
                theta_adjusted = torch.nn.functional.pad(theta_adjusted, (0, LL - theta_adjusted.numel()), value=0.0)
            # Compute the update vector using the adjusted intrinsic parameter.
            update_vector = fastfood_torched(theta_adjusted, num_elements, self.proj_params[name])
            update = update_vector.view(shape)
            new_value = self.initial_values[name] + update
            self._set_param_by_name(module, name, new_value)
        return

    def _set_param_by_name(self, module: nn.Module, param_name: str, new_value: torch.Tensor):
        # Update the data in-place to preserve the parameter's registration.
        attrs = param_name.split('.')
        obj = module
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        getattr(obj, attrs[-1]).data.copy_(new_value)

    def register(self):
        self.module.register_forward_pre_hook(self.hook)

def intrinsic_dimension(module: nn.Module, intrinsic_dimension: int, output_dir: str, 
                        str_filter: Set[str] = None, projection: str = "fastfood", device: str = "cuda") -> nn.Module:
    hook = GlobalIntrinsicDimensionHook(module, intrinsic_dimension, projection=projection, device=device, str_filter=str_filter)
    hook.register()
    return module