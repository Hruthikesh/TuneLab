from dataclasses import dataclass
import os
import platform
import shutil
from typing import Any, Dict, Optional


@dataclass
class HardwareProfile:
    device: str
    cuda_available: bool
    gpu_count: int
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    torch_version: str
    transformers_version: str
    peft_version: str
    bitsandbytes_available: bool
    os_info: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "cpu_count": self.cpu_count,
            "ram_total_gb": round(self.ram_total_gb, 2),
            "ram_available_gb": round(self.ram_available_gb, 2),
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "peft_version": self.peft_version,
            "bitsandbytes_available": self.bitsandbytes_available,
            "os_info": self.os_info,
            "status": self.status,
        }


def detect_hardware() -> HardwareProfile:
    """Inspects the execution environment and returns a structured hardware profile."""
    # CPU & RAM inspection
    cpu_count = os.cpu_count() or 1
    total_ram = 0.0
    avail_ram = 0.0

    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("MemTotal:"):
                    total_ram = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    avail_ram = int(line.split()[1]) / (1024 * 1024)
        except Exception:
            pass

    # PyTorch and CUDA check
    torch_version = "NOT_INSTALLED"
    cuda_available = False
    gpu_count = 0
    gpu_name = None
    gpu_vram_gb = None

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except ImportError:
        pass

    # Transformers check
    transformers_version = "NOT_INSTALLED"
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        pass

    # PEFT check
    peft_version = "NOT_INSTALLED"
    try:
        import peft
        peft_version = peft.__version__
    except ImportError:
        pass

    # Bitsandbytes check
    bnb_available = False
    try:
        import bitsandbytes
        bnb_available = True
    except ImportError:
        pass

    device = "cuda" if cuda_available else "cpu"
    status = "READY" if cuda_available else "TRAINING_PENDING_GPU"

    return HardwareProfile(
        device=device,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        cpu_count=cpu_count,
        ram_total_gb=total_ram,
        ram_available_gb=avail_ram,
        torch_version=torch_version,
        transformers_version=transformers_version,
        peft_version=peft_version,
        bitsandbytes_available=bnb_available,
        os_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
        status=status,
    )
