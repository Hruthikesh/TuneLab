from typing import Any, Dict, Optional


def get_gpu_memory_stats() -> Dict[str, Any]:
    """Captures current and peak GPU memory allocation in MB if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "peak_allocated_mb": 0.0,
                "peak_reserved_mb": 0.0,
            }
        
        alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        res = torch.cuda.memory_reserved() / (1024 * 1024)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)

        return {
            "cuda_available": True,
            "allocated_mb": round(alloc, 2),
            "reserved_mb": round(res, 2),
            "peak_allocated_mb": round(peak_alloc, 2),
            "peak_reserved_mb": round(peak_res, 2),
        }
    except ImportError:
        return {
            "cuda_available": False,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }


def reset_peak_memory_stats() -> None:
    """Resets peak memory tracking stats in PyTorch CUDA if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
