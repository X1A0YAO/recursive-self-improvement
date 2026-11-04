"""Phase 0 环境检查脚本：GPU/CUDA/PyTorch/磁盘/网络一览。"""

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> None:
    info: dict = {}
    info["generated_at"] = datetime.now(timezone.utc).isoformat()
    info["python"] = sys.version
    info["executable"] = sys.executable
    info["platform"] = platform.platform()

    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    info["gpu_count"] = torch.cuda.device_count()
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "index": i,
                "name": p.name,
                "total_memory_gb": round(p.total_memory / 1024**3, 2),
                "multi_processor_count": p.multi_processor_count,
                "capability": f"{p.major}.{p.minor}",
            }
        )
    info["gpus"] = gpus

    total, used, free = shutil.disk_usage(Path.cwd().drive + "\\")
    info["disk_free_gb"] = round(free / 1024**3, 1)
    info["disk_total_gb"] = round(total / 1024**3, 1)

    print(json.dumps(info, indent=2, ensure_ascii=False))
    out = Path("results/environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
