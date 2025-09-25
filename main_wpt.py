# main_wpt_convnext_adapt.py
import os, math, argparse, copy, time, json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

from argparse import BooleanOptionalAction
from metrics_utils import seed_everything

from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup,
)

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm

# ==============================
# EMA
# ==============================
class ModelEma:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.data.mul_(d).add_(msd[k].data, alpha=1.0 - d)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd, strict=False)

# ==============================
# WPT helpers
# ==============================
import numpy as np
import pywt

class WaveletPacketTransform(torch.nn.Module):
    """
    Apply 2D Wavelet Packet Transform to each channel (RGB).
    output='ll'    -> keep LL band at given level -> channels stay C (e.g., 3)
    output='concat'-> concat all 4**level subbands per channel -> channels become C * 4**level
    Input:  CHW float tensor in [0,1]
    Output: CHW float tensor
    """
    def __init__(self, wavelet: str = "db2", level: int = 2, mode: str = "periodization",
                 output: Literal["ll", "concat"] = "ll"):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.output = output

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("WPT expects CHW tensor")
        C, H, W = x.shape
        x_np = x.detach().cpu().numpy().astype(np.float32)
        out_bands = []
        for c in range(C):
            wp = pywt.WaveletPacket2D(x_np[c], wavelet=self.wavelet, mode=self.mode, maxlevel=self.level)
            if self.output == "ll":
                node = wp["a" * self.level]
                out_bands.append(node.data.astype(np.float32)[None, ...])  # (1,h,w)
            else:  # concat
                grid = wp.get_level(self.level, order="freq")
                nodes = [n for row in grid for n in row]
                subbands = [n.data.astype(np.float32)[None, ...] for n in nodes]  # each (1,h,w)
                out_bands.extend(subbands)
        y = np.concatenate(out_bands, axis=0)  # (C' , h, w)
        return torch.from_numpy(y)

class PerImageStandardize(torch.nn.Module):
    """Zero-mean, unit-std over (C,H,W)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std  = x.std().clamp_min(self.eps)
        return (x - mean) / std

# ==============================
# Transforms (augments -> tensor -> erasing -> WPT -> standardize)
# ==============================
def build_transforms(
    img_size: int,
    randaug_mag=9,
    re_prob=0.25,
    wpt_wavelet="db2",
    wpt_level=2,
    wpt_output="ll"
):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=randaug_mag),
        transforms.ToTensor(),                      # -> CHW in [0,1]
        transforms.RandomErasing(p=re_prob, inplace=True),
        WaveletPacketTransform(wavelet=wpt_wavelet, level=wpt_level, output=wpt_output),
        PerImageStandardize(),
        # NOTE: no resize-back and no RGB mean/std here by design.
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        WaveletPacketTransform(wavelet=wpt_wavelet, level=wpt_level, output=wpt_output),
        PerImageStandardize(),
    ])
    return train_tfms, val_tfms

# ==============================
# Metrics
# ==============================
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.size(0)).item())
    return res

def _count_params_m(model) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def _try_flops_g(model, in_shape, device="cpu"):
    try:
        from thop import profile
        import torch
        dummy = torch.randn(*in_shape, device=device)
        macs, _ = profile(model.to(device), inputs=(dummy,),verbose=False)
        return macs / 1e9
    except Exception:
        return None

# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Imagenet directory")
    ap.add_argument("--out", default="./runs/convnext_tiny_wpt_noresize")
    ap.add_argument("--resume", default="")

    # Training lengths | batch | workers
    ap.add_argument("--epochs",        type=int, default=300)
    ap.add_argument("--warmup_epochs", type=int, default=20)
    ap.add_argument("--batch_size",    type=int, default=64)
    ap.add_argument("--workers",       type=int, default=8)
    ap.add_argument("--img_size",      type=int, default=224)

    # Optimizer | Scheduler
    ap.add_argument("--base_lr",      type=float, default=4e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)

    # Data Augmentations
    ap.add_argument("--mixup_alpha",    type=float, default=0.8)
    ap.add_argument("--cutmix_alpha",   type=float, default=1.0)
    ap.add_argument("--label_smoothing",type=float, default=0.1)
    ap.add_argument("--randaug_mag",    type=int,   default=9)
    ap.add_argument("--re_prob",        type=float, default=0.25)

    # EMA + precision
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--precision", default="fp16", choices=["no", "fp16", "bf16"])

    # WPT options
    ap.add_argument("--wpt_level",   type=int, default=2, help="WPT level (spatial /2**level)")
    ap.add_argument("--wpt_wavelet", type=str, default="db2", help="e.g., db2, coif1, sym4")
    ap.add_argument("--wpt_output",  type=str, default="ll", choices=["ll","concat"],
                    help="ll: 3 channels; concat: 3*(4**level) channels")
    
    # Addtional Options
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    seed_everything(args.seed)

    # Make Directories
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tb_dir = os.path.join(args.out, "tb"); Path(tb_dir).mkdir(parents=True, exist_ok=True)

    # Accelerator and logging
    accelerator = Accelerator(mixed_precision=args.precision)
    cudnn.benchmark = True
    writer = SummaryWriter(tb_dir) if accelerator.is_main_process else None

    # Build transforms (no resize back after WPT)
    train_tfms, val_tfms = build_transforms(
        args.img_size,
        randaug_mag=args.randaug_mag,
        re_prob=args.re_prob,
        wpt_wavelet=args.wpt_wavelet,
        wpt_level=args.wpt_level,
        wpt_output=args.wpt_output,
    )

    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(args.data, "val"),   transform=val_tfms)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True
    )

    # Determine channels after WPT
    in_chans_after_wpt = 3 if args.wpt_output == "ll" else 3 * (4 ** args.wpt_level)


    # Instantiate Model
    from convnext_modified import ConvNeXt
    model = ConvNeXt(in_chans=in_chans_after_wpt)
   
    # Flop and Param count
    if args.wpt_output == "ll":
        C = 3
        H = W = args.img_size // (2**args.wpt_level)
    else: # concat
        C = 3 * (4**args.wpt_level)
        H = W = args.img_size // (2**args.wpt_level)

    params_m = _count_params_m(model)
    flops_g = _try_flops_g(model, in_shape=(1,C,H,W), device="cpu")

    # Save a small JSON file
    stats = {
        "params_m": round(params_m, 4),
        "flops_g": None if flops_g is None else round(float(flops_g), 4),
        "profile_input_shape": [1, C, H, W],
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out, "model_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Mixup | Cutmix
    mixup_fn = Mixup(
        mixup_alpha     = args.mixup_alpha,
        cutmix_alpha    = args.cutmix_alpha,
        label_smoothing = 0.0,
        num_classes     = num_classes,
        prob = 1.0, switch_prob = 0.5, mode = "batch"
    )

    # Optimizer and Scheduler
    world_size   = accelerator.num_processes
    global_batch = args.batch_size * world_size
    scaled_lr    = args.base_lr * (global_batch / 4096)

    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = args.warmup_epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Losses
    loss_train_soft = SoftTargetCrossEntropy()
    loss_eval_hard  = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # EMA
    ema = ModelEma(model, decay=args.ema_decay)

    # Prepare
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    ema.ema.to(accelerator.device)

    # (Optional) resume
    start_epoch, best_top1 = 1, 0.0
    extras_root = os.path.join(args.out, "extras.pt")
    if args.resume:
        accelerator.print(f"Resuming from {args.resume}")
        accelerator.load_state(args.resume)
        extras_file = os.path.join(args.resume, "extras.pt")
        if not os.path.exists(extras_file) and os.path.exists(extras_root):
            extras_file = extras_root
        if os.path.exists(extras_file):
            pack = torch.load(extras_file, map_location="cpu")
            start_epoch = int(pack.get("epoch", 1))
            best_top1 = float(pack.get("best_top1", 0.0))
            ema_sd = pack.get("ema_state_dict", None)
            if ema_sd is not None:
                ema.load_state_dict(ema_sd)
            accelerator.print(f"Resumed at epoch {start_epoch}, best_top1={best_top1:.2f}")
        else:
            accelerator.print("Warning: extras.pt not found.")

    # Train
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False) if accelerator.is_main_process else train_loader

        for step, (x, y) in enumerate(pbar):
            x, y_soft = mixup_fn(x, y)
            with accelerator.autocast():
                logits = model(x)
                loss = loss_train_soft(logits, y_soft)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            ema.update(accelerator.unwrap_model(model))
            running += loss.item()

            if accelerator.is_main_process and (step + 1) % 50 == 0:
                writer.add_scalar("train/loss", running / (step + 1), (epoch - 1) * steps_per_epoch + step + 1)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], (epoch - 1) * steps_per_epoch + step + 1)

        epoch_time = time.time() - t0

        # Validate with EMA weights
        model.eval()
        raw_sd = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
        accelerator.unwrap_model(model).load_state_dict(ema.state_dict(), strict=False)

        val_loss, val_top1, val_top5 = 0.0, 0.0, 0.0
        with torch.no_grad(), accelerator.autocast():
            for x, y in val_loader:
                logits = model(x)
                loss = loss_eval_hard(logits, y)
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                val_loss += loss.item(); val_top1 += top1; val_top5 += top5

        n = len(val_loader)
        val_loss /= n; val_top1 /= n; val_top5 /= n

        # Restore raw weights
        accelerator.unwrap_model(model).load_state_dict(raw_sd, strict=False)

        if accelerator.is_main_process:
            writer.add_scalar("val/loss_ema", val_loss, epoch)
            writer.add_scalar("val/top1_ema", val_top1, epoch)
            writer.add_scalar("val/top5_ema", val_top5, epoch)
            writer.add_scalar("time/epoch_sec", epoch_time, epoch)
            print(f"[VAL] epoch {epoch}: loss={val_loss:.4f} top1={val_top1:.2f} top5={val_top5:.2f} ({epoch_time:.1f}s)")

        # Checkpoint
        # ckpt_dir = os.path.join(args.out, f"ckpt_epoch_{epoch:04d}")
        # if accelerator.is_main_process:
        #     Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        # accelerator.save_state(ckpt_dir)

        if accelerator.is_main_process:
            pack = {"epoch": epoch + 1, "best_top1": max(best_top1, val_top1), "ema_state_dict": ema.state_dict()}
            #torch.save(pack, os.path.join(ckpt_dir, "extras.pt"))
            torch.save(pack, extras_root)
            if val_top1 > best_top1:
                best_top1 = val_top1
                torch.save({"model_ema": ema.state_dict(), "top1": val_top1, "epoch": epoch},
                           os.path.join(args.out, "best_ema.pth"))

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()
