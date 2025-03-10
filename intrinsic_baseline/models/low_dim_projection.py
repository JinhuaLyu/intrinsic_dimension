# models/low_dim_projection.py
import torch
import torch.nn as nn
from .projection import fastfood_vars, fastfood_torched  # Import fastfood functions from projection.py
import numpy as np

class LowDimWeightWrapper(nn.Module):
    """
    Replace the original linear layer's weight matrix with a low-dimensional trainable parameterization.
    Two projection methods are supported:
      - "linear": W = W_0 + einsum("mnd,d->mn", M, theta)
      - "fastfood": W = W_0 + fastfood(theta) reshaped to (out_features, in_features)
    """
    def __init__(self, original_layer: nn.Linear, d: int, seed: int, projection: str = "linear"):
        super().__init__()
        if not isinstance(original_layer, nn.Linear):
            raise ValueError("LowDimWeightWrapper only supports nn.Linear layers.")

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.d = d
        self.projection = projection

        # Freeze the original weight matrix
        self.register_buffer("W_0", original_layer.weight.clone().detach())

        if self.projection == "linear":
            # Fixed random mapping matrix M of shape (out_features, in_features, d)
            current_rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            self.register_buffer("M", torch.randn(self.out_features, self.in_features, d, device=original_layer.weight.device))
            torch.set_rng_state(current_rng_state)
        elif self.projection == "fastfood":
            # Compute desired high-dim output size: D = out_features * in_features
            DD = self.out_features * self.in_features
            # Call fastfood_vars only once and store the whole tuple in self.proj_params
            self.proj_params = fastfood_vars(DD, device=original_layer.weight.device)
        else:
            raise ValueError("Unsupported projection method. Use 'linear' or 'fastfood'.")

        # Trainable low-dimensional parameter theta, allocated on the same device as the original layer
        self.theta = nn.Parameter(torch.zeros(d, device=original_layer.weight.device))

        # Keep bias unchanged (if exists)
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.clone().detach(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection == "linear":
            # Traditional linear projection: W_new = W_0 + einsum("mnd,d->mn", M, theta)
            W_update = torch.einsum("mnd,d->mn", self.M, self.theta)
        elif self.projection == "fastfood":
            # FastFood projection: project theta from low-dim to high-dim update vector
            DD = self.out_features * self.in_features
            # fastfood_torched returns a vector of shape (DD,)
            W_update_vector = fastfood_torched(self.theta, DD, self.proj_params)
            # Reshape update vector to matrix shape (out_features, in_features)
            W_update = W_update_vector.view(self.out_features, self.in_features)
        else:
            raise ValueError("Unsupported projection method in forward.")

        # Compute the new weight matrix: W_new = W_0 + W_update
        W_new = self.W_0 + W_update

        # Perform linear transformation using updated weight matrix
        output = torch.matmul(x, W_new.T)
        if self.bias is not None:
            output += self.bias
        return output