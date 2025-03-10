# models/projection.py
import torch
import torch.nn.functional as F
import numpy as np

def fastfood_vars(DD: int, device="cpu"):
    """
    Returns parameters for fastfood transform.
    DD: desired high-dimensional output size.
    The returned parameters are allocated on the specified device.
    """
    # Calculate the next power of 2 for DD.
    ll = int(np.ceil(np.log2(DD)))
    LL = 2 ** ll

    # Generate binary scaling vector B with values in {-1, +1}
    BB = torch.randint(0, 2, (LL,), device=device, dtype=torch.float32)
    BB = BB * 2 - 1

    # Generate a random permutation Pi of indices [0, LL)
    Pi = torch.randperm(LL, device=device)

    # Generate Gaussian scaling vector G ~ N(0, 1)
    GG = torch.randn(LL, device=device)
    divisor = torch.sqrt(LL * torch.sum(GG ** 2))
    return (BB, Pi, GG, divisor, LL)

def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Fast Walsh-Hadamard Transform of a 1D tensor x.
    x must have a length that is a power of 2.
    This vectorized implementation uses reshape and concatenation to avoid Python loops.
    All operations run on the same device as x.
    """
    n = x.shape[0]
    assert (n & (n - 1) == 0), "Length of x must be a power of 2"
    H = 1
    # Ensure x is a 2D tensor with shape (1, n) for vectorized processing
    y = x.unsqueeze(0)
    while H < n:
        # Reshape y to (batch, n//(2*H), 2*H)
        y = y.view(-1, n // (2 * H), 2 * H)
        a = y[:, :, :H]
        b = y[:, :, H:]
        # Compute sum and difference in a vectorized manner
        y = torch.cat([a + b, a - b], dim=2)
        y = y.view(-1, n)
        H *= 2
    return y.squeeze(0)

def fastfood_torched(theta: torch.Tensor, DD: int, proj_params: tuple) -> torch.Tensor:
    """
    Applies FastFood transform to the low-dimensional parameter vector theta.
    theta: low-dimensional parameter vector of shape (d,)
    DD: desired high-dimensional size (an integer)
    proj_params: tuple returned by fastfood_vars
    Returns a vector of shape (DD,).
    
    All operations are performed on the device of theta.
    """
    d = theta.numel()
    BB, Pi, GG, divisor, LL = proj_params

    # If theta is shorter than LL, pad it with zeros
    if d < LL:
        theta_padded = F.pad(theta, (0, LL - d), value=0.0)
    else:
        theta_padded = theta

    # Multiply element-wise by binary vector B
    v = theta_padded * BB

    # Apply the vectorized Walsh-Hadamard transform
    v = hadamard_transform(v)

    # Permute the result using Pi
    v = v[Pi]

    # Multiply element-wise by Gaussian vector G
    v = v * GG

    # Apply the Walsh-Hadamard transform again
    v = hadamard_transform(v)

    # Take the first DD elements and normalize
    ret = v[:DD] / (divisor * np.sqrt(float(DD) / LL))
    return ret