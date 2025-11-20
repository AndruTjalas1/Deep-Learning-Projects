"""Device management for GPU/CPU training."""

import torch


def get_device(use_gpu: bool = True) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        use_gpu: Whether to use GPU if available
        
    Returns:
        torch.device: The device to use (cuda, mps, or cpu)
    """
    if not use_gpu:
        return torch.device("cpu")
    
    # Check for NVIDIA CUDA (Windows/Linux)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    # Check for Apple Metal Performance Shaders (Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
        return device
    
    # Fall back to CPU
    device = torch.device("cpu")
    print("Using CPU (no GPU detected)")
    return device


def get_device_info() -> dict:
    """Get information about available devices."""
    info = {
        "cpu": True,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_capability"] = torch.cuda.get_device_capability(0)
    
    if torch.backends.mps.is_available():
        info["mps_device"] = "Apple Metal Performance Shaders"
    
    return info
