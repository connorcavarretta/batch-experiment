
import os, json, argparse, subprocess, datetime, shlex, sys, time, pathlib
from metrics_utils import append_row, write_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def build_cmd(main_py, data_dir, out_dir, wavelet_cfg, common):
    launch = f"accelerate launch --multi-gpu --num_processes=3"
    parts = [
        launch,
        shlex.quote(main_py),
        f"--data {shlex.quote(data_dir)}",
        f"--out {shlex.quote(out_dir)}",
        f"--epochs {common['epochs']}",
        f"--warmup_epochs {common['warmup_epochs']}",
        f"--batch_size {common['batch_size']}",
        f"--workers {common['workers']}",
        f"--img_size {common['img_size']}",
        f"--base_lr {common['base_lr']}",
        f"--weight_decay {common['weight_decay']}",
        f"--randaug_mag {common['randaug_mag']}",
        f"--re_prob {common['re_prob']}",
        f"--precision {common['precision']}",
        f"--seed {common['seed']}",
    ]

    parts += [
        f"--wpt_level {wavelet_cfg.get('wpt_level', 2)}",
        f"--wpt_wavelet {shlex.quote(str(wavelet_cfg.get('wpt_wavelet', 'db2')))}",
        f"--wpt_output {shlex.quote(wavelet_cfg.get('wpt_output','ll'))}",
    ]

    return " ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to ImageNet root (train/ val/)")
    ap.add_argument("--main_py", default=os.path.join(THIS_DIR, "main_wpt.py"))
    ap.add_argument("--config", default=os.path.join(THIS_DIR, "wavelets.json"))
    ap.add_argument("--out_root", default=os.path.join(THIS_DIR, "batch_runs"))
    ap.add_argument("--csv", default=os.path.join(THIS_DIR, "results.csv"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--base_lr", type=float, default=4e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--randaug_mag", type=int, default=9)
    ap.add_argument("--re_prob", type=float, default=0.25)
    ap.add_argument("--precision", default="fp16", choices=["no","fp16","bf16"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        waves = json.load(f)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = os.path.join(args.out_root, ts)
    pathlib.Path(run_root).mkdir(parents=True, exist_ok=True)

    common = dict(
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size, workers=args.workers,
        img_size=args.img_size, base_lr=args.base_lr,
        weight_decay=args.weight_decay, randaug_mag=args.randaug_mag,
        re_prob=args.re_prob, precision=args.precision, seed=args.seed
    )

    summary_index = []

    for w in waves:
        label = w["name"]
        out_dir = os.path.join(run_root, label)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        cmd = build_cmd(args.main_py, args.data, out_dir, w, common)
        print(f"\n=== Running: {label}\n{cmd}\n")
        t0 = time.time()
        ret = subprocess.call(cmd, shell=True)
        elapsed = time.time() - t0

        record = {
            "label": label,
            "wpt_wavelet": w.get("wpt_wavelet"),
            "wpt_level": w.get("wpt_level"),
            "wpt_output": w.get("wpt_output"),
            "seed": common["seed"],
            "out_dir": out_dir,
            "status_code": ret,
            "wall_time_sec": round(elapsed, 2),
        }

        import torch
        best_pth = os.path.join(out_dir, "best_ema.pth")
        extras_pt = os.path.join(out_dir, "extras.pt")
        try:
            if os.path.exists(best_pth):
                pack = torch.load(best_pth, map_location="cpu")
                record["best_top1"] = float(pack.get("top1", float("nan")))
                record["best_epoch"] = int(pack.get("epoch", -1))
            if os.path.exists(extras_pt):
                pack2 = torch.load(extras_pt, map_location="cpu")
                record["best_top1_ckpt"] = float(pack2.get("best_top1", float("nan")))
        except Exception as e:
            record["harvest_error"] = str(e)

        stats_json = os.path.join(out_dir, "model_stats.json")
        try:
            if os.path.exists(stats_json):
                with open(stats_json, "r") as f:
                    stats = json.load(f)
                record["params_m"]=stats.get("params_m")
                record["flops_g"] =stats.get("flops_g")
        except Exception as e:
            record["stats_error"] = str(e)

        append_row(args.csv, record)
        summary_index.append(record)

    write_json(os.path.join(run_root, "summary.json"), {"runs": summary_index, "config": waves, "common": common})
    print(f"\nDone. CSV: {args.csv}\nIndex: {os.path.join(run_root,'summary.json')}")

if __name__ == "__main__":
    main()
