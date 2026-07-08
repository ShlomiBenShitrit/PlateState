#!/usr/bin/env python
"""Headless FLUX image generation for the shlomi/ pipeline.

Designed for long unattended runs (nohup / background). Resumable (skips images
already on disk) and reproducible (deterministic per-image seeds). Reads prompts
from shlomi/data/prompts.json via the local utils.py and writes undegraded images to a fresh
versioned folder shlomi/data/Diff_DataSet_vN/{class}/ (never overwriting old data) plus a manifest.csv.

Quick pilot (10 images per class = 50 total):
    nohup /home/benshise/my_new_project/.venv/bin/python shlomi/generate_images.py \
        --gpu 6 --per-class 10 > shlomi/gen.log 2>&1 &

Full run (all 300/class = 1500):  drop --per-class.

Watch:   tail -f shlomi/gen.log     (Ctrl+C stops watching, NOT the job)
Stop:    ps aux | grep generate_images   then   kill <pid>

Note: --size is the image RESOLUTION in pixels (default 512), NOT a count. 512 is
~4x faster than 1024 and step 03 degrades to ~128 anyway. FLUX.1-schnell is much
faster (no license, ~4 steps):  --model black-forest-labs/FLUX.1-schnell --steps 4 --guidance 0
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="6", help="CUDA device index to use")
ap.add_argument("--size", type=int, default=512,
                help="image RESOLUTION in pixels (not a count). 512 is ~4x faster than 1024")
ap.add_argument("--steps", type=int, default=20, help="num_inference_steps")
ap.add_argument("--guidance", type=float, default=3.5, help="guidance_scale (use 0 for FLUX.1-schnell)")
ap.add_argument("--images-per-prompt", type=int, default=1)
ap.add_argument("--per-class", type=int, default=0,
                help="cap prompts per class (0 = all 300). e.g. 10 for a quick 50-image pilot")
ap.add_argument("--skip-unclassified", action="store_true",
                help="don't generate the 'unclassified' class (it just borrows other prompts; "
                     "its degraded images are made later in step 03)")
ap.add_argument("--classes", nargs="+", default=None, metavar="CLASS",
                help="only generate these classes (e.g. --classes empty finished_leftovers). "
                     "Default: all. Order still follows CLASS_NAMES.")
ap.add_argument("--out", default=None,
                help="output base folder. Default: a NEW versioned folder shlomi/data/Diff_DataSet_vN "
                     "(auto-incremented) so existing data is never overwritten. Pass an explicit path "
                     "to resume into / add to a specific run.")
ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
args = ap.parse_args()

# Must be set BEFORE torch initialises CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                       # noqa: E402
from diffusers import FluxPipeline  # noqa: E402

# Import the local shlomi/utils.py (this file lives in shlomi/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                       # noqa: E402

# Output base: by default a FRESH versioned folder (Diff_DataSet_v1, _v2, ...) so an existing
# curated dataset is never overwritten; or a custom --out folder (e.g. to resume into a specific run).
OUT_DIR = Path(args.out).resolve() if args.out else utils.next_versioned_dir()
MANIFEST = OUT_DIR / "manifest.csv"


def _rel(p) -> str:
    """Path relative to the repo if possible, else absolute (robust for any --out)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(utils.ROOT_DIR))
    except ValueError:
        return str(p)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    prompts = utils.load_prompts()

    # Classes to generate this run: optional explicit --classes filter, and/or skip
    # the borrow-only 'unclassified'. Order always follows CLASS_NAMES.
    if args.classes:
        unknown = [c for c in args.classes if c not in utils.CLASS_NAMES]
        if unknown:
            sys.exit(f"Unknown class(es): {unknown}. Valid: {utils.CLASS_NAMES}")
    selected = set(args.classes) if args.classes else set(utils.CLASS_NAMES)
    classes = [c for c in utils.CLASS_NAMES
               if c in selected and not (args.skip_unclassified and c == utils.UNCLASSIFIED)]

    def plist(cls):
        """Prompts for a class, optionally capped by --per-class."""
        return prompts[cls] if args.per_class <= 0 else prompts[cls][:args.per_class]

    def seed_for(cls, p_i, k):
        """Deterministic per (class, prompt, image) so a pilot and a later full run agree."""
        return utils.SEED + utils.CLASS_TO_IDX[cls] * 1_000_000 + p_i * 100 + k

    total = sum(len(plist(c)) for c in classes) * args.images_per_prompt

    log(f"loading {args.model} on GPU {args.gpu} (fp16 + cpu offload) ...")
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    scope = f", {args.per_class}/class" if args.per_class > 0 else " (full set)"
    log(f"loaded. target = {total} images @ {args.size}px, {args.steps} steps{scope}.")

    def generate(prompt, seed):
        gen = torch.Generator("cpu").manual_seed(seed)
        return pipe(
            prompt,
            height=args.size, width=args.size,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            max_sequence_length=512,
            generator=gen,
        ).images[0]

    log("generating ... (one progress line per image below)")
    t0 = time.time()
    made = done = 0
    for cls in classes:
        out_dir = utils.class_dir(OUT_DIR, cls)
        for p_i, prompt in enumerate(plist(cls)):
            for k in range(args.images_per_prompt):
                done += 1
                fp = out_dir / f"{cls}_{p_i:04d}_{k}.jpg"
                if fp.exists():
                    continue
                generate(prompt, seed_for(cls, p_i, k)).save(fp, quality=95)
                made += 1
                rate = (time.time() - t0) / made
                eta_min = rate * (total - done) / 60
                log(f"{done}/{total}  [{cls}]  {rate:.0f}s/img  ETA ~{eta_min:.0f} min")

    # Manifest: scan ALL class folders present in OUT_DIR (not just this run's classes),
    # so several runs into the same folder accumulate into one complete manifest.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "class", "seed", "model", "prompt"])
        for cls in utils.CLASS_NAMES:
            cdir = OUT_DIR / cls
            if not cdir.is_dir():
                continue
            for fp in sorted(cdir.glob(f"{cls}_*.jpg")):
                p_i, k = (int(x) for x in fp.stem.split("_")[-2:])
                prompt = prompts[cls][p_i] if p_i < len(prompts[cls]) else ""
                w.writerow([_rel(fp), cls, seed_for(cls, p_i, k), args.model, prompt])
                rows += 1

    log(f"DONE. {made} new images this run | {rows} total on disk | "
        f"elapsed {(time.time() - t0) / 60:.0f} min | manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
