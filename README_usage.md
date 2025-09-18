
# Batch Wavelet Ablation Runner (WPT in DataLoader)

Artifacts created here:
- `wavelets.json`: list of wavelets to test (includes a **control** without WPT)
- `run_batch.py`: orchestrates sequential runs, writing `results.csv` and a per-batch index
- `metrics_utils.py`: helpers (seed, FLOPs/params optional, CSV/JSON)

## Usage

1) Patch your training script to accept `--seed` and `--use_wpt/--no-use_wpt` (see ChatGPT for patch).
2) Edit `wavelets.json` as desired.
3) Run:

```bash
python /mnt/data/run_batch.py --data /PATH/TO/IMAGENET --main_py /mnt/data/main_wpt.py   --out_root /PATH/TO/OUT --epochs 50 --batch_size 256 --seed 1337
```
