#!/usr/bin/env python
"""Make `finished_leftovers` by EDITING full-plate images with FLUX.1 Kontext.

Text-to-image can't render "absence of food" - naming a dish makes FLUX draw a full
serving. Instruction-based image editing solves it directly: take a full-plate image
and tell the model to *subtract* food (simulate someone having eaten most of it).

Pipeline:
    full-plate image  --(Kontext: "the person ate most of it, leave ~a quarter")-->  leftovers

This also keeps the leftovers image on the SAME plate/table/lighting as a full plate,
so the only thing that differs is the food amount (the actual label) - no style shortcut.

Source images: any folder of full-plate photos (default: shlomi/data/synthetic_clean/full).
Output: shlomi/data/synthetic_clean/finished_leftovers/ (overwrites the text2img ones).

Speed: Kontext (12B) doesn't fit a 32GB V100, so the default --quant 8bit loads the transformer +
T5 quantized -> the whole model fits on ONE GPU with no CPU offload (minutes/img -> ~tens of s/img).

Prereqs:
    - Accept the license for black-forest-labs/FLUX.1-Kontext-dev on HF (gated); cached token works.
    - diffusers with FluxKontextPipeline (>=0.35) and:  pip install bitsandbytes

Pilot (edit 5 full plates, quantized):
    python shlomi/edit_leftovers_kontext.py --gpu 0 --limit 5 --quant 8bit
Full (edit all images in --src):  drop --limit.
Headless:  nohup ... > shlomi/gen_kontext.log 2>&1 &
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
                help="folder of full-plate source images (default: synthetic_clean/full)")
ap.add_argument("--size", type=int, default=512, help="edit resolution (match the dataset)")
ap.add_argument("--steps", type=int, default=28, help="num_inference_steps")
ap.add_argument("--guidance", type=float, default=2.5, help="Kontext guidance_scale (~2.5-4)")
ap.add_argument("--limit", type=int, default=0, help="cap how many source images to edit (0 = all)")
ap.add_argument("--out", default=None,
                help="output folder (default: synthetic_clean/finished_leftovers, in place). "
                     "Use a new folder to keep the originals for an A/B compare.")
ap.add_argument("--overwrite", action="store_true", help="re-edit even if the output already exists")
ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                help="bf16 avoids the fp16 NaN/black-image bug Kontext hits on V100 (default)")
ap.add_argument("--max-side", type=int, default=768,
                help="square working resolution. 768 is near Kontext's native 1024 (good framing) "
                     "and fits 8-bit on a 32GB V100; 1024 OOMs on V100 (no FlashAttention). We pass "
                     "this as explicit height/width with _auto_resize=False so Kontext SCALES rather "
                     "than crops (diffusers #12406). Output is still downscaled to --size on save.")
ap.add_argument("--offload", choices=["model", "sequential"], default="sequential",
                help="ONLY used with --quant none. model = keeps the 24GB transformer on GPU "
                     "(OOMs a 32GB V100); sequential = streams layers, fits but very slow")
ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="8bit",
                help="quantize the transformer + T5 so the whole model fits on one 32GB V100 with "
                     "NO offload (the big speed win). 8bit is the safe default on V100 (Volta); "
                     "4bit is smaller/faster but its Volta kernels are not guaranteed. "
                     "Needs: pip install bitsandbytes")
ap.add_argument("--model", default="black-forest-labs/FLUX.1-Kontext-dev")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                            # noqa: E402
from diffusers import FluxKontextPipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                            # noqa: E402

DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

SRC_DIR = (Path(args.src) if args.src else (utils.CLEAN_DIR / "full")).resolve()
OUT_DIR = (Path(args.out) if args.out else (utils.CLEAN_DIR / "finished_leftovers")).resolve()
MANIFEST = OUT_DIR / "kontext_manifest.csv"


def _rel(p) -> str:
    """Path relative to the repo if possible, else absolute (robust for any --out)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(utils.ROOT_DIR))
    except ValueError:
        return str(p)

# How much food the edit should leave (we want only ~a quarter / fifth).
EAT_AMOUNTS = [
    "only about a quarter of the food left",
    "roughly a fifth of the meal remaining",
    "just a few small scraps of food left",
    "only a small amount of food remaining in one corner",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_instruction(rng: random.Random) -> str:
    """A randomized 'eat-down' edit instruction (~1/2 cutlery, ~1/4 napkin)."""
    parts = [
        f"The person has eaten most of this meal. Edit the image so the same plate now has "
        f"{rng.choice(EAT_AMOUNTS)}:",
        "scattered messy scraps pushed to one side, smeared sauce, scattered crumbs and visible "
        "bite marks, while the rest of the plate is empty and bare.",
    ]
    parts.append("Leave a dirty fork or spoon resting in the plate."
                 if rng.random() < 0.5 else "No cutlery on the plate.")
    if rng.random() < 0.25:
        parts.append("Leave a crumpled, used paper napkin inside the plate.")
    parts.append("Keep the exact same plate, table surface, lighting and camera angle.")
    return " ".join(parts)


def load_pipeline():
    """Load FLUX.1-Kontext.

    With --quant 8bit/4bit the transformer and the T5 text-encoder are loaded quantized, so the
    whole pipeline fits on one 32GB V100 and runs FULLY on the GPU with no offload - the big speed
    win. With --quant none we fall back to the (slow) CPU-offload path for debugging.
    """
    if args.quant == "none":
        log(f"loading {args.model} on GPU {args.gpu} ({args.dtype}, {args.offload} offload) ...")
        pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=DTYPE)
        if args.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        return pipe

    log(f"loading {args.model} on GPU {args.gpu} ({args.dtype}, {args.quant} quantized, no offload) ...")
    from diffusers import FluxTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
    from transformers import T5EncoderModel
    from transformers import BitsAndBytesConfig as TransformersBnbConfig

    if args.quant == "8bit":
        diff_q = DiffusersBnbConfig(load_in_8bit=True)
        t5_q = TransformersBnbConfig(load_in_8bit=True)
    else:  # 4bit (NF4) - smaller/faster, but Volta (V100) kernels are not guaranteed
        diff_q = DiffusersBnbConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=DTYPE)
        t5_q = TransformersBnbConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=DTYPE)

    # The two big components, loaded quantized (they land on the GPU).
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model, subfolder="transformer", quantization_config=diff_q, torch_dtype=DTYPE)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.model, subfolder="text_encoder_2", quantization_config=t5_q, torch_dtype=DTYPE)

    pipe = FluxKontextPipeline.from_pretrained(
        args.model, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=DTYPE)
    # transformer + T5 are already on the GPU; move the small un-quantized parts (CLIP, VAE) there
    # too. Do NOT call pipe.to("cuda") - it raises on bnb-quantized modules.
    pipe.vae.to("cuda")
    pipe.text_encoder.to("cuda")
    return pipe


def main() -> None:
    if not SRC_DIR.is_dir() or not any(SRC_DIR.glob("*.jpg")):
        sys.exit(f"No source images in {SRC_DIR}. Generate some 'full' plates there first "
                 f"(e.g. generate_images.py --classes full).")

    srcs = sorted(SRC_DIR.glob("*.jpg"))
    if args.limit > 0:
        srcs = srcs[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline()
    pipe.set_progress_bar_config(disable=True)
    log(f"loaded. editing {len(srcs)} full plates -> leftovers @ {args.size}px, {args.steps} steps.")

    rows = []
    t0 = time.time()
    made = 0
    for i, src in enumerate(srcs):
        out_fp = OUT_DIR / f"finished_leftovers_{i:04d}_0.jpg"
        if out_fp.exists() and not args.overwrite:
            continue
        seed = utils.SEED + i
        rng = random.Random(seed)               # deterministic instruction per image
        instruction = build_instruction(rng)
        # Feed a square input at the working resolution and pass explicit height/width with
        # _auto_resize=False so Kontext SCALES (keeps the whole plate) instead of snapping to a
        # preferred resolution and cropping (diffusers #12406). Then downscale to args.size on save.
        image = Image.open(src).convert("RGB").resize((args.max_side, args.max_side), Image.LANCZOS)
        edited = pipe(
            image=image,
            prompt=instruction,
            height=args.max_side, width=args.max_side, _auto_resize=False,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        edited.resize((args.size, args.size)).save(out_fp, quality=95)
        made += 1
        rate = (time.time() - t0) / made
        log(f"{i + 1}/{len(srcs)}  {rate:.0f}s/img  ETA ~{rate * (len(srcs) - i - 1) / 60:.0f} min")
        rows.append([_rel(out_fp), _rel(src), seed, instruction])

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "source_full_image", "seed", "instruction"])
        w.writerows(rows)
    log(f"DONE. {made} edited | elapsed {(time.time() - t0) / 60:.0f} min | manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
