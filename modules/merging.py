import torch

def merge_tensors(method: str, v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    if method == "lerp":
        return merge_tensors_lerp(v0, v1, t)
    elif method == "slerp":
        return merge_tensors_slerp(v0, v1, t)
    elif method == "slice":
        return merge_tensors_slice(v0, v1, t)
    elif method == "cyclic":
        return merge_tensors_cyclic(v0, v1, t)
    elif method == "gradient":
        return merge_tensors_gradient(v0, v1, t)

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

# MODEL 1 > 10% blend > MODEL 2
def merge_tensors_slice(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    # We're only working on the second dimension here
    if v0.dim() == 2:
        # Calculate the slice indices for each tensor
        slice_index_0 = int(v0.shape[1] * (1 - t))
        slice_index_1 = v1.shape[1] - slice_index_0
    
        blend_slice_size = int(v0.shape[1] * 0.05)
        blend_slice_0 = v0.narrow(1, slice_index_0 - blend_slice_size, blend_slice_size * 2)
        blend_slice_1 = v1.narrow(1, slice_index_0 - blend_slice_size, blend_slice_size * 2)
        blended_slice = blend_slice_0

        # Apply gradient blending
        for i in range(blend_slice_size * 2):
            blend_ratio = i / (blend_slice_size * 2)
            blended_slice[:, i] = (blend_slice_1[:, i] * blend_ratio) + (blend_slice_0[:, i] * (1 - blend_ratio))
    
        slice_index_0 = slice_index_0 - blend_slice_size
        slice_index_1 = slice_index_0 + blend_slice_size + blend_slice_size
    
        # Perform slicing
        slice_0 = v0.narrow(1, 0, slice_index_0)
        slice_1 = v1.narrow(1, slice_index_1, v1.shape[1] - slice_index_1)
    
        # Concatenate the slices
        result = torch.cat([slice_0, blended_slice, slice_1], dim=1)
    
        return result
    else:
        return v0

# MODEL 1 > 10% blend > 10% of MODEL 2 > 10% blend > MODEL 1, with varying starting positions as defined by t
def merge_tensors_cyclic(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    # We're only working on the second dimension here
    if v0.dim() == 2:
        blend_slice_size = int(v0.shape[1] * 0.05) # Blending zone is eventually multiplied by two due to overlap
        v1_slice_size = int(v0.shape[1] * 0.10) # 10% of Model 2, accounting for the 5% blend zone on both sides. So kinda 15%.

        slice_index_0 = int(v0.shape[1] * (1 - t)) - blend_slice_size # Model 1, first slice length

        # First MODEL 1 > MODEL 2 blend
        # -----------------------
        blend_slice_0_0 = v0.narrow(1, slice_index_0, blend_slice_size * 2)
        blend_slice_0_1 = v1.narrow(1, slice_index_0, blend_slice_size * 2)
        blended_slice_0 = blend_slice_0_0

        # Apply gradient blending
        for i in range(blend_slice_size * 2):
            blend_ratio = i / (blend_slice_size * 2)
            blended_slice_0[:, i] = (blend_slice_0_0[:, i] * (1 - blend_ratio)) + (blend_slice_0_1[:, i] * blend_ratio)

        # Second MODEL 2 > MODEL 1 blend
        # -----------------------
        blend_slice_1_0 = v0.narrow(1, slice_index_0 + (blend_slice_size * 2) + v1_slice_size, blend_slice_size * 2)
        blend_slice_1_1 = v1.narrow(1, slice_index_0 + (blend_slice_size * 2) + v1_slice_size, blend_slice_size * 2)
        blended_slice_1 = blend_slice_1_0

        # Apply gradient blending
        for i in range(blend_slice_size * 2):
            blend_ratio = i / (blend_slice_size * 2)
            blended_slice_1[:, i] = (blend_slice_1_1[:, i] * (1 - blend_ratio)) + (blend_slice_1_0[:, i] * blend_ratio)

        # Time to out main candidates into various pieces
        m1len_0   = slice_index_0
        m2start   = slice_index_0 + (blend_slice_size * 2)
        m1start_1 = m2start + v1_slice_size + (blend_slice_size * 2)
        m2end_1   = v1.shape[1] - m1start_1

        # print(f"M1 0-{m1len_0} > B1 {m1len_0}-{m1len_0+(blend_slice_size * 2)} > M2 {m2start}-{m2start+v1_slice_size} > B2 {m2start+v1_slice_size}-{m1start_1} > M1 {m1start_1}-{v1.shape[1]}")
        
        slice_0_0 = v0.narrow(1, 0, m1len_0) # Model 1, first piece
        slice_1_0 = v1.narrow(1, m2start, v1_slice_size) # Model 2 slice
        slice_0_1 = v0.narrow(1, m1start_1, m2end_1) # Model 1, second piece
    
        # Concatenate the slices
        result = torch.cat([slice_0_0, blended_slice_0, slice_1_0, blended_slice_1, slice_0_1], dim=1)
    
        return result
    else:
        return v0

# Model 1 > Model 2 > Model 1, with t defining the peak of the gradient along the tensor's width
def merge_tensors_gradient(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    if v0.dim() == 2:
        total_length = v0.shape[1]
        peak = int(total_length * (1 - t))

        # Create an index array
        indices = torch.arange(total_length).float()

        # Vectorized computation of blend ratios
        blend_ratios = torch.zeros_like(indices)
        blend_ratios[:peak] = (indices[:peak] / peak) * 0.9  # Scale to max 0.9 for v1
        blend_ratios[peak:] = torch.flip(indices[:total_length - peak], dims=[0]) / (total_length - peak) * 0.9  # Scale to max 0.9 for v1

        # Ensure that v0 still has influence
        v0_ratios = 1 - blend_ratios

        # Vectorized blending of the tensors
        result = (v1 * blend_ratios.unsqueeze(0)) + (v0 * v0_ratios.unsqueeze(0))

        return result
    else:
        return v0

def safe_normalize(tensor: torch.Tensor, eps: float):
    norm = tensor.norm()
    if norm > eps:
        return tensor / norm
    return tensor

def merge_header_tensors(model1, model2, method, v0, v1, t) -> torch.Tensor:
    # TLDR - We reshape model 2's tensors to match model 1's
    model1bos  = model1.config.bos_token_id
    model1eos  = model1.config.eos_token_id
    model1size = v0.shape[0]

    model2bos  = model2.config.bos_token_id
    model2eos  = model2.config.eos_token_id
    model2size = v1.shape[0]

    # If model 2 has a smaller vocab, expand it
    if model1size > model2size:
        # Calculate the difference in size
        size_diff = model1size - model2size
        # Copy the additional entries from v0 to v1
        v1 = torch.cat([v1, v0[-size_diff:]], dim=0)

    # Swap special tokens if needed
    if model1bos != model2bos: 
        v1[model1bos] = v1[model2bos]
        v1[model2bos] = v0[model1bos]
    if model1eos != model2eos: 
        v1[model1eos] = v1[model2eos]
        v1[model2eos] = v0[model1eos]

    # If model 1 is smaller then 2, truncate
    # We do this after swapping tokens around
    if model1size < model2size:
        v1 = v1[:model1size]

    return merge_tensors_lerp(v0, v1, t)