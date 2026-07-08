#!/usr/bin/env python
"""Preview the CCTV degradation (notebook 03) on a few images per class WITHOUT running the full
pipeline and WITHOUT saving anything to the dataset.

It reads the degradation code straight out of 03_degrade_and_augment.ipynb (the single source of
truth), degrades N images per class on the fly, and writes a grid to shlomi/degradation_preview.png
so you can judge the augmentation strength before the real run. Tune the ranges inside
03_degrade_and_augment.ipynb's `degrade_image`, re-run this, repeat until it looks right.

Usage:
    python shlomi/preview_degradation.py            # 10 per class
    python shlomi/preview_degradation.py 6          # 6 per class
    python shlomi/preview_degradation.py 8 3        # 8 images/class, 3 degraded variants each
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import utils  # noqa: E402

import matplotlib                  # noqa: E402
matplotlib.use("Agg")             # headless (no display): render straight to a file
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from PIL import Image             # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10        # images per class
VARIANTS = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # degraded versions shown per image

# --- Pull degrade_image() + its primitives out of the notebook and load them here ---
nb = json.loads((HERE / "03_degrade_and_augment.ipynb").read_text(encoding="utf-8"))
ns = {"utils": utils}
# the imports / flags the notebook's degradation cells rely on (defined in its config cell)
exec(
    "import os, random, hashlib\n"
    "from io import BytesIO\n"
    "import numpy as np\n"
    "from PIL import Image, ImageFilter, ImageEnhance\n"
    "import cv2\n"
    "USE_90_ROTATIONS = False\n",
    ns,
)
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "def add_blur" in src or "def degrade_image" in src:   # primitives cell + pipeline cell
        exec(src, ns)
degrade_image = ns["degrade_image"]

INPUT_DIR = utils.CLEAN_DIR
classes = sorted(d.name for d in INPUT_DIR.iterdir() if d.is_dir())
print("preview source :", INPUT_DIR)
print("classes        :", classes)
print(f"showing        : {N} images/class, original + {VARIANTS} degraded variant(s) each")

exts = (".jpg", ".jpeg", ".png")
rows_per_class = 1 + VARIANTS                      # 1 original row + VARIANTS degraded rows
total_rows = rows_per_class * len(classes)
fig, axes = plt.subplots(total_rows, N, figsize=(2 * N, 2 * total_rows), squeeze=False)

for c, cls in enumerate(classes):
    files = sorted(p for p in (INPUT_DIR / cls).glob("*") if p.suffix.lower() in exts)[:N]
    base = c * rows_per_class
    for i in range(N):
        for r in range(rows_per_class):
            axes[base + r][i].axis("off")
        if i >= len(files):
            continue
        img = Image.open(files[i]).convert("RGB")
        axes[base][i].imshow(img)                       # original
        for v in range(VARIANTS):
            axes[base + 1 + v][i].imshow(degrade_image(img.copy()))   # degraded (random each call)
    axes[base][0].set_title(f"{cls}  -  original (top) / degraded (below)", loc="left", fontsize=11)

fig.suptitle("Degradation preview  -  on the fly, nothing saved to the dataset", fontsize=13)
plt.tight_layout()
out = HERE / "degradation_preview.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved ->", out)
