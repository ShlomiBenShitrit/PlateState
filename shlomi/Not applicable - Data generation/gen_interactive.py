#!/usr/bin/env python
"""Interactive FLUX generator - load the model ONCE, regenerate many times.

For the prompt-tuning loop: the headless generate_images.py reloads the 24 GB model
on every launch (~6 min). This keeps the model resident in an interactive Python
session so each regenerate starts instantly.

Usage (note the -i so you land in an interactive prompt after loading):
    python -i shlomi/gen_interactive.py --gpu 0

Then, at the >>> prompt:
    run('finished_leftovers', 10)     # generate 10 images for one class (overwrites)
    run('empty', 10)
    run('full', 5)
    reload()                          # after `git pull` updated prompts.json / utils.py
    run('finished_leftovers', 10)     # regenerate with the new prompts - NO model reload

Images go to shlomi/data/synthetic_clean/{class}/ (same place make_grid.py reads), so
in another terminal:
    python shlomi/make_grid.py
shows the latest. Ctrl-D (or exit()) quits and frees the GPU.
"""
import argparse
import importlib
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=3.5)
ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                       # noqa: E402
from diffusers import FluxPipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                       # noqa: E402

print(f"[load] {args.model} on GPU {args.gpu} (fp16 + cpu offload) - one-time ~6 min ...",
      flush=True)
pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
prompts = utils.load_prompts()
print("[load] ready. Try:  run('finished_leftovers', 10)   then  reload()  after a git pull.",
      flush=True)


def reload():
    """Re-read prompts.json (and utils.py) after a `git pull` - keeps the model loaded."""
    global prompts
    importlib.reload(utils)
    prompts = utils.load_prompts()
    print(f"[reload] prompts refreshed: "
          f"{ {c: len(prompts[c]) for c in utils.CLASS_NAMES} }", flush=True)


def _seed_for(cls, p_i, k=0):
    return utils.SEED + utils.CLASS_TO_IDX[cls] * 1_000_000 + p_i * 100 + k


def run(cls, n=10):
    """Generate the first n prompts of a class and save (OVERWRITING) to synthetic_clean."""
    if cls not in prompts:
        print(f"unknown class {cls!r}; valid: {list(prompts)}")
        return
    out_dir = utils.class_dir(utils.CLEAN_DIR, cls)
    total = min(n, len(prompts[cls]))
    t0 = time.time()
    for p_i in range(total):
        gen = torch.Generator("cpu").manual_seed(_seed_for(cls, p_i))
        img = pipe(
            prompts[cls][p_i],
            height=args.size, width=args.size,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            max_sequence_length=512,
            generator=gen,
        ).images[0]
        img.save(out_dir / f"{cls}_{p_i:04d}_0.jpg", quality=95)
        done = p_i + 1
        print(f"  [{cls}] {done}/{total}  {(time.time() - t0) / done:.0f}s/img", flush=True)
    print(f"[done] {cls}: {total} images in {(time.time() - t0) / 60:.1f} min", flush=True)
