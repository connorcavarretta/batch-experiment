
import csv, os, json, random
from typing import Dict, Any, Optional
import numpy as np
import torch

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model) -> float:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6

def try_flops(model, input_shape, device: str = "cpu") -> Optional[float]:
    # Estimate FLOPs (GMACs) using thop if installed; else return None
    try:
        from thop import profile
        dummy = torch.randn(*input_shape, device=device)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        return macs / 1e9
    except Exception:
        return None

def append_row(csv_path: str, row: Dict[str, Any]):
    header_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not header_exists:
            writer.writeheader()
        writer.writerow(row)

def write_json(path: str, payload: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
