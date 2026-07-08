#!/usr/bin/env python
"""Headless Gemini (Nano Banana) image generation — A/B counterpart to generate_images.py.

Reads the SAME shlomi/data/prompts.json and writes to a SEPARATE folder
(shlomi/data/synthetic_clean_gemini/{class}/) so it never touches the FLUX images.
Filenames match the FLUX script ({class}_{p_i:04d}_{k}.jpg) so the same prompt index
lines up across both models for an apples-to-apples comparison.

Free path: a free Google AI Studio API key (NOT the Gemini Pro subscription, which gives
no API quota). Default model `gemini-2.5-flash-image` has a free tier (~500 img/day/key,
volatile). The newest `gemini-3.1-flash-image` (Nano Banana Pro) is better but PAID only.

Auth (never commit the key):
    export GEMINI_API_KEY=...        # from https://aistudio.google.com/apikey
Install:
    pip install google-genai pillow
Pilot (10/class = 50):
    python shlomi/generate_images_gemini.py --per-class 10
Watch in background:
    nohup python shlomi/generate_images_gemini.py --per-class 10 > shlomi/gen_gemini.log 2>&1 &
    tail -f shlomi/gen_gemini.log

NOTE: Gemini image models do NOT honor a fixed seed, so this set is not bit-reproducible
(unlike FLUX). We still log the prompt+model per image in the manifest.
"""
import argparse
import csv
import io
import os
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemini-2.5-flash-image",
                help="free tier: gemini-2.5-flash-image. Paid/better: gemini-3.1-flash-image")
ap.add_argument("--per-class", type=int, default=0,
                help="cap prompts per class (0 = all). e.g. 10 for the 50-image A/B pilot")
ap.add_argument("--images-per-prompt", type=int, default=1)
ap.add_argument("--out-name", default="synthetic_clean_gemini",
                help="output subfolder under shlomi/data/ (kept separate from the FLUX set)")
ap.add_argument("--sleep", type=float, default=1.0,
                help="seconds to pause between requests (stay under the per-minute rate limit)")
ap.add_argument("--max-retries", type=int, default=5,
                help="retries on rate-limit / transient errors (exponential backoff)")
args = ap.parse_args()

OUT_BASE = utils.DATA_DIR / args.out_name
MANIFEST = OUT_BASE / "manifest.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_image(response) -> Image.Image | None:
    """Pull the first inline image out of a generate_content response."""
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return Image.open(io.BytesIO(inline.data)).convert("RGB")
    return None


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey).")

    from google import genai  # imported here so the file parses without the SDK installed
    client = genai.Client(api_key=api_key)

    prompts = utils.load_prompts()

    def plist(cls):
        return prompts[cls] if args.per_class <= 0 else prompts[cls][:args.per_class]

    total = sum(len(plist(c)) for c in utils.CLASS_NAMES) * args.images_per_prompt
    scope = f", {args.per_class}/class" if args.per_class > 0 else " (full set)"
    log(f"model = {args.model} | target = {total} images{scope} -> {OUT_BASE}")

    def generate(prompt: str) -> Image.Image:
        last = None
        for attempt in range(args.max_retries):
            try:
                resp = client.models.generate_content(model=args.model, contents=prompt)
                img = extract_image(resp)
                if img is not None:
                    return img
                last = RuntimeError("no image part in response (possibly safety-blocked)")
            except Exception as e:  # noqa: BLE001 - includes rate-limit (429) and transient errors
                last = e
            wait = min(2 ** attempt, 30)
            log(f"  retry {attempt + 1}/{args.max_retries} in {wait}s ({type(last).__name__})")
            time.sleep(wait)
        raise RuntimeError(f"failed after {args.max_retries} retries: {last}")

    t0 = time.time()
    made = done = skipped = 0
    for cls in utils.CLASS_NAMES:
        out_dir = utils.class_dir(OUT_BASE, cls)
        for p_i, prompt in enumerate(plist(cls)):
            for k in range(args.images_per_prompt):
                done += 1
                fp = out_dir / f"{cls}_{p_i:04d}_{k}.jpg"
                if fp.exists():
                    skipped += 1
                    continue
                img = generate(prompt)
                img.save(fp, quality=95)
                made += 1
                rate = (time.time() - t0) / made
                eta_min = rate * (total - done) / 60
                log(f"{done}/{total}  [{cls}]  {rate:.0f}s/img  ETA ~{eta_min:.0f} min")
                time.sleep(args.sleep)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "class", "model", "prompt"])
        for cls in utils.CLASS_NAMES:
            for p_i, prompt in enumerate(plist(cls)):
                for k in range(args.images_per_prompt):
                    fp = OUT_BASE / cls / f"{cls}_{p_i:04d}_{k}.jpg"
                    if fp.exists():
                        w.writerow([str(fp.relative_to(utils.ROOT_DIR)), cls, args.model, prompt])
                        rows += 1

    log(f"DONE. {made} new | {skipped} skipped | {rows} on disk | "
        f"elapsed {(time.time() - t0) / 60:.0f} min | manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
