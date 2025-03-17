# models/global_intrinsic_linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .projection import fastfood_vars, fastfood_torched

class GlobalIntrinsicLinear(nn.Module):
    """
    A wrapper for a linear layer that computes its effective weight as:
        W_eff = W_0 + update,
    where update is computed from a global intrinsic parameter using a differentiable projection.
    """
    def __init__(self, original_layer: nn.Linear, proj_params, intrinsic_parameter: torch.Tensor):
        super().__init__()
        # Store the frozen initial weight and bias.
        self.register_buffer("W_0", original_layer.weight.clone().detach())
        if original_layer.bias is not None:
            self.register_buffer("b", original_layer.bias.clone().detach())
        else:
            self.b = None
        self.proj_params = proj_params  # Projection parameters tuple for this layer.
        self.intrinsic_parameter = intrinsic_parameter  # Global intrinsic parameter (trainable)
        self.out_features = original_layer.out_features
        self.in_features = original_layer.in_features
        self.num_elements = self.out_features * self.in_features
        for name, param in self.named_parameters():
            if name != "intrinsic_parameter":
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine the required length (LL) from proj_params.
        _, _, _, _, LL = self.proj_params
        # Adjust the global intrinsic parameter length to LL.
        theta = self.intrinsic_parameter
        if theta.numel() > LL:
            theta_adjusted = theta[:LL]
        elif theta.numel() < LL:
            theta_adjusted = F.pad(theta, (0, LL - theta.numel()), value=0.0)
        else:
            theta_adjusted = theta

        # # Force a dummy operation to keep the gradient connected.
        # theta_adjusted = theta_adjusted + 0.0
        # theta_adjusted = theta_adjusted.contiguous()

        # Compute update vector: shape = (num_elements,)
        update_vector = fastfood_torched(theta_adjusted, self.num_elements, self.proj_params)
        update = update_vector.view(self.out_features, self.in_features)
        # Compute effective weight.
        W_eff = self.W_0 + update
        # Compute output: x @ W_eff^T + bias.
        output = torch.matmul(x, W_eff.T)
        if self.b is not None:
            output += self.b
        return output