import torch

def merge_tensors(method: str, v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    if method == "lerp":
        return merge_tensors_lerp(v0, v1, t)
    elif method == "slerp":
        return merge_tensors_slerp(v0, v1, t)

def merge_tensors_lerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    
    result = ((1 - t) * v0) + (t * v1)
    
    return result

def merge_tensors_slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, dot_threshold: float = 0.9995, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two tensors or linear interpolation if they are one-dimensional.
       Full credit to https://github.com/cg123/mergekit for the original code."""

    # We LERP single dimensional tensors
    if v0.dim() == 1 and v1.dim() == 1:
        return merge_tensors_lerp(v0, v1, t)

    # Make copies of the original tensors to use for interpolation
    v0_copy = v0.clone()
    v1_copy = v1.clone()

    # Normalize the original tensors for angle computation
    v0 = safe_normalize(v0, eps)
    v1 = safe_normalize(v1, eps)

    # Compute the cosine of the angle between the normalized vectors.
    dot = (v0 * v1).sum()

    # If the inputs are too close, linearly interpolate using the original tensors.
    if abs(dot) > dot_threshold:
        return merge_tensors_lerp(v0_copy, v1_copy, t)

    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    # Use the weights with the original tensors (not normalized) for the final result
    result = s0 * v0_copy + s1 * v1_copy

    return result

def safe_normalize(tensor: torch.Tensor, eps: float):
    norm = tensor.norm()
    if norm > eps:
        return tensor / norm
    return tensor
