#!/usr/bin/env python
"""Make `finished_leftovers` by EDITING full-plate images with SDXL-InstructPix2Pix.

Why this and not Kontext: FLUX.1-Kontext (12B) forces 1024px and, with no FlashAttention on the
V100, OOMs even quantized. SDXL-InstructPix2Pix (~3.5B) is a V100-friendly instruction editor:
native 768, fits fp16 with no offload, no resolution-forcing / crop bug, ~10-20s/image.

Same idea as the Kontext script: take a full-plate (or already-too-full leftovers) image and instruct
the model to "eat down" the food to ~a quarter, keeping the same plate/table/lighting.

fp16 note: SDXL's default VAE overflows in fp16 (black images), so we load the fp16-fixed VAE
(madebyollin/sdxl-vae-fp16-fix).

Source: a folder of plate images (default: shlomi/data/synthetic_clean/full).
Output: shlomi/data/synthetic_clean/finished_leftovers/ (or --out).

Pilot (edit 3 images to a test folder):
    python shlomi/edit_leftovers_ip2p.py --gpu 0 \
        --src shlomi/data/synthetic_clean/finished_leftovers \
        --out shlomi/data/synthetic_clean/_ip2p_test --limit 3 --overwrite

Tuning knobs: --image-guidance (higher = stay closer to the original plate) and --guidance (text CFG).
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--src", default=None,
                help="folder of source plate images (default: synthetic_clean/full)")
ap.add_argument("--size", type=int, default=512, help="final saved resolution (dataset size)")
ap.add_argument("--res", type=int, default=768, help="working resolution (SDXL-IP2P native = 768)")
ap.add_argument("--steps", type=int, default=30, help="num_inference_steps")
ap.add_argument("--guidance", type=float, default=7.0, help="text guidance_scale (prompt strength)")
ap.add_argument("--image-guidance", type=float, default=1.5,
                help="image_guidance_scale: higher = stay closer to the original image (less change)")
ap.add_argument("--limit", type=int, default=0, help="cap how many source images to edit (0 = all)")
ap.add_argument("--out", default=None, help="output folder (default: synthetic_clean/finished_leftovers)")
ap.add_argument("--overwrite", action="store_true", help="re-edit even if the output already exists")
ap.add_argument("--model", default="diffusers/sdxl-instructpix2pix-768")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                                                # noqa: E402
from diffusers import StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                                                # noqa: E402

SRC_DIR = (Path(args.src) if args.src else (utils.CLEAN_DIR / "full")).resolve()
OUT_DIR = (Path(args.out) if args.out else (utils.CLEAN_DIR / "finished_leftovers")).resolve()
MANIFEST = OUT_DIR / "ip2p_manifest.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(utils.ROOT_DIR))
    except ValueError:
        return str(p)


def main() -> None:
    if not SRC_DIR.is_dir() or not any(SRC_DIR.glob("*.jpg")):
        sys.exit(f"No source images in {SRC_DIR}. Generate some plates there first.")

    srcs = sorted(SRC_DIR.glob("*.jpg"))
    if args.limit > 0:
        srcs = srcs[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading {args.model} on GPU {args.gpu} (fp16, no offload) ...")
    # SDXL's stock VAE overflows in fp16 -> black images; use the fp16-fixed VAE.
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        args.model, vae=vae, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    log(f"loaded. editing {len(srcs)} plates -> leftovers @ {args.res}px, {args.steps} steps "
        f"(img_guid={args.image_guidance}, guid={args.guidance}).")

    rows = []
    t0 = time.time()
    made = 0
    for i, src in enumerate(srcs):
        out_fp = OUT_DIR / f"finished_leftovers_{i:04d}_0.jpg"
        if out_fp.exists() and not args.overwrite:
            continue
        seed = utils.SEED + i
        rng = random.Random(seed)
        instruction = utils.build_eat_instruction(rng)
        image = Image.open(src).convert("RGB").resize((args.res, args.res), Image.LANCZOS)
        edited = pipe(
            prompt=instruction,
            image=image,
            num_inference_steps=args.steps,
            image_guidance_scale=args.image_guidance,
            guidance_scale=args.guidance,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        edited.resize((args.size, args.size)).save(out_fp, quality=95)
        made += 1
        rate = (time.time() - t0) / made
        log(f"{i + 1}/{len(srcs)}  {rate:.0f}s/img  ETA ~{rate * (len(srcs) - i - 1) / 60:.0f} min")
        rows.append([_rel(out_fp), _rel(src), seed, instruction])

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "source_image", "seed", "instruction"])
        w.writerows(rows)
    log(f"DONE. {made} edited | elapsed {(time.time() - t0) / 60:.0f} min | manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
