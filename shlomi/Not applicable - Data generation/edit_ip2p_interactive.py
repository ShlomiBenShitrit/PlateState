#!/usr/bin/env python
"""Interactive SDXL-InstructPix2Pix leftovers editor - load the model ONCE, tune & re-run.

Loads the model a single time, then you call run(...) as many times as you like with different
settings - no reload. Each run() PRINTS, per image, the source file and the exact instruction, and
saves a side-by-side SOURCE | EDITED comparison grid so you can see what produced what.

Usage (note -i so you land at an interactive prompt after loading):
    python -i shlomi/edit_ip2p_interactive.py --gpu 0 \
        --src shlomi/data/synthetic_clean/finished_leftovers

Then at >>>:
    run(3)                                    # 3 images, default settings
    run(3, image_guidance=1.2, guidance=10)   # eat down MORE (lower image_guidance) + stronger prompt
    run(3, image_guidance=2.0)                # change LESS (stay closer to the original plate)
    reload()                                  # re-read utils after a `git pull` (model stays loaded)

Dials:
    image_guidance (default 1.5): higher = stay closer to the original (less change / less eaten).
                                  lower  = change more (eats food down harder, but may drift).
    guidance       (default 7.0): text-prompt strength.

Outputs go to --out (default shlomi/data/_ip2p_test/) and the comparison PNG to shlomi/ip2p_compare.png.
"""
import argparse
import importlib
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--src", default=None, help="folder of source plate images")
ap.add_argument("--out", default=None, help="output folder (default: data/_ip2p_test)")
ap.add_argument("--size", type=int, default=512, help="final saved resolution")
ap.add_argument("--res", type=int, default=768, help="working resolution (SDXL-IP2P native = 768)")
ap.add_argument("--model", default="diffusers/sdxl-instructpix2pix-768")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                                                # noqa: E402
from diffusers import StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                                                # noqa: E402

SRC_DIR = (Path(args.src) if args.src else (utils.CLEAN_DIR / "full")).resolve()
OUT_DIR = (Path(args.out) if args.out else (utils.DATA_DIR / "_ip2p_test")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_PNG = Path(__file__).resolve().parent / "ip2p_compare.png"

print(f"[load] {args.model} on GPU {args.gpu} (fp16) - one-time ...", flush=True)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
    args.model, vae=vae, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print("[load] ready. Try:  run(3)   or  run(3, image_guidance=1.2, guidance=10)", flush=True)


def reload():
    """Re-import utils after a `git pull` (e.g. updated build_eat_instruction). Model stays loaded."""
    importlib.reload(utils)
    print("[reload] utils refreshed.", flush=True)


def run(n=3, image_guidance=1.5, guidance=7.0, steps=30, prompt=None):
    """Edit the first n source images; print src+instruction per image; save a comparison grid.

    prompt: if given, use this EXACT instruction for every image (great for testing a short
            imperative like 'Remove most of the food, leave only a few small scraps.'). If None,
            use the randomized utils.build_eat_instruction.
    """
    srcs = sorted(SRC_DIR.glob("*.jpg"))[:n]
    if not srcs:
        print(f"no .jpg sources in {SRC_DIR}")
        return
    print(f"\n=== run: n={len(srcs)}  image_guidance={image_guidance}  guidance={guidance}  "
          f"steps={steps}  prompt={'custom' if prompt else 'auto'} ===", flush=True)
    pairs = []
    t0 = time.time()
    for i, src in enumerate(srcs):
        seed = utils.SEED + i
        instr = prompt if prompt else utils.build_eat_instruction(random.Random(seed))
        image = Image.open(src).convert("RGB").resize((args.res, args.res), Image.LANCZOS)
        edited = pipe(
            prompt=instr,
            image=image,
            num_inference_steps=steps,
            image_guidance_scale=image_guidance,
            guidance_scale=guidance,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0].resize((args.size, args.size))
        fp = OUT_DIR / f"finished_leftovers_{i:04d}_0.jpg"
        edited.save(fp, quality=95)
        pairs.append((src, fp))
        print(f"[{i}] src = {src.name}")
        print(f"     instruction = {instr}", flush=True)

    # Side-by-side comparison grid: one row per image, [ source | edited ].
    sz = 360
    grid = Image.new("RGB", (2 * sz, len(pairs) * sz), "white")
    draw = ImageDraw.Draw(grid)
    for r, (src, fp) in enumerate(pairs):
        grid.paste(Image.open(src).convert("RGB").resize((sz, sz)), (0, r * sz))
        grid.paste(Image.open(fp).convert("RGB").resize((sz, sz)), (sz, r * sz))
        draw.text((4, r * sz + 4), f"[{r}] SOURCE", fill="red")
        draw.text((sz + 4, r * sz + 4), f"[{r}] EDITED", fill="red")
    grid.save(COMPARE_PNG)
    print(f"\nsaved {COMPARE_PNG}  |  {len(pairs)} pairs  |  {(time.time() - t0) / 60:.1f} min", flush=True)
