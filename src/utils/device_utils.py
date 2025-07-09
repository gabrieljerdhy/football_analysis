"""
Device detection and configuration utilities for GPU acceleration.
"""

import os
import torch
from typing import Union, Optional


def check_gpu_availability() -> dict:
    """
    Check GPU availability and return detailed information.
    
    Returns:
        dict: Dictionary containing GPU availability information
    """
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "current_device": None,
        "gpu_names": [],
        "memory_info": []
    }
    
    if torch.cuda.is_available():
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["current_device"] = torch.cuda.current_device()
        
        for i in range(gpu_info["device_count"]):
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append(f"cuda:{i}")
            gpu_info["gpu_names"].append(props.name)
            
            # Get memory information
            if torch.cuda.is_available():
                try:
                    torch.cuda.set_device(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    cached_memory = torch.cuda.memory_reserved(i)
                    free_memory = total_memory - allocated_memory
                    
                    gpu_info["memory_info"].append({
                        "device": f"cuda:{i}",
                        "total_mb": total_memory / (1024**2),
                        "allocated_mb": allocated_memory / (1024**2),
                        "cached_mb": cached_memory / (1024**2),
                        "free_mb": free_memory / (1024**2),
                        "utilization_percent": (allocated_memory / total_memory) * 100
                    })
                except Exception as e:
                    gpu_info["memory_info"].append({
                        "device": f"cuda:{i}",
                        "error": str(e)
                    })
    
    return gpu_info


def get_optimal_device(device: Optional[Union[str, torch.device]] = None, verbose: bool = True) -> torch.device:
    """
    Get the optimal device for processing based on availability and user preference.
    
    Args:
        device (str | torch.device, optional): Preferred device. Options:
            - None or "auto": Auto-select best available device
            - "cpu": Force CPU usage
            - "cuda" or "gpu": Use first available GPU
            - "cuda:0", "cuda:1", etc.: Use specific GPU
        verbose (bool): Whether to print device selection information
        
    Returns:
        torch.device: Selected device
    """
    # Handle explicit device specification
    if device is not None:
        if isinstance(device, torch.device):
            if verbose:
                print(f"🔧 Using explicitly specified device: {device}")
            return device
        
        device_str = str(device).lower().strip()
        
        # Force CPU
        if device_str == "cpu":
            if verbose:
                print("🔧 Using CPU (explicitly requested)")
            return torch.device("cpu")
        
        # Force GPU (first available)
        if device_str in ["cuda", "gpu"]:
            if torch.cuda.is_available():
                selected_device = torch.device("cuda:0")
                if verbose:
                    gpu_info = check_gpu_availability()
                    gpu_name = gpu_info["gpu_names"][0] if gpu_info["gpu_names"] else "Unknown"
                    print(f"🚀 Using GPU: {gpu_name} (cuda:0)")
                return selected_device
            else:
                if verbose:
                    print("⚠️ GPU requested but CUDA not available, falling back to CPU")
                return torch.device("cpu")
        
        # Specific GPU device
        if device_str.startswith("cuda:"):
            try:
                device_index = int(device_str.split(":")[1])
                if torch.cuda.is_available() and device_index < torch.cuda.device_count():
                    selected_device = torch.device(device_str)
                    if verbose:
                        gpu_info = check_gpu_availability()
                        gpu_name = gpu_info["gpu_names"][device_index] if device_index < len(gpu_info["gpu_names"]) else "Unknown"
                        print(f"🚀 Using GPU: {gpu_name} ({device_str})")
                    return selected_device
                else:
                    if verbose:
                        print(f"⚠️ {device_str} not available, falling back to CPU")
                    return torch.device("cpu")
            except (ValueError, IndexError):
                if verbose:
                    print(f"⚠️ Invalid device specification '{device_str}', falling back to CPU")
                return torch.device("cpu")
    
    # Auto-select best available device
    if torch.cuda.is_available():
        gpu_info = check_gpu_availability()
        
        # Select GPU with most free memory
        best_device_idx = 0
        max_free_memory = 0
        
        for i, mem_info in enumerate(gpu_info["memory_info"]):
            if "free_mb" in mem_info and mem_info["free_mb"] > max_free_memory:
                max_free_memory = mem_info["free_mb"]
                best_device_idx = i
        
        selected_device = torch.device(f"cuda:{best_device_idx}")
        
        if verbose:
            gpu_name = gpu_info["gpu_names"][best_device_idx] if best_device_idx < len(gpu_info["gpu_names"]) else "Unknown"
            free_gb = max_free_memory / 1024
            print(f"🚀 Auto-selected GPU: {gpu_name} (cuda:{best_device_idx}) - {free_gb:.1f}GB free")
        
        return selected_device
    else:
        if verbose:
            print("💻 No GPU available, using CPU")
        return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """
    Print detailed information about the selected device.
    
    Args:
        device (torch.device): Device to print information about
    """
    print(f"\n🔧 Device Configuration:")
    print(f"   Selected device: {device}")
    
    if device.type == "cuda":
        gpu_info = check_gpu_availability()
        if gpu_info["cuda_available"]:
            device_idx = device.index if device.index is not None else 0
            if device_idx < len(gpu_info["gpu_names"]):
                gpu_name = gpu_info["gpu_names"][device_idx]
                print(f"   GPU Name: {gpu_name}")
                
                if device_idx < len(gpu_info["memory_info"]):
                    mem_info = gpu_info["memory_info"][device_idx]
                    if "total_mb" in mem_info:
                        total_gb = mem_info["total_mb"] / 1024
                        free_gb = mem_info["free_mb"] / 1024
                        print(f"   GPU Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
    else:
        print(f"   Using CPU with {torch.get_num_threads()} threads")
    
    print()


def configure_device_environment(device: torch.device) -> None:
    """
    Configure environment variables and settings for optimal device usage.
    
    Args:
        device (torch.device): Device to configure for
    """
    if device.type == "cuda":
        # Set CUDA device if specific GPU is selected
        if device.index is not None:
            torch.cuda.set_device(device.index)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
        
        # Enable optimizations for GPU
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            
        # Clear GPU cache to start fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Optimize CPU usage
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
