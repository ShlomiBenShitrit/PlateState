#!/usr/bin/env python
"""Preview what 05's `train_transform` does - the LIVE, per-epoch augmentation that is applied to each
(already-degraded) image during training and is NOT saved to disk.

Reads `train_transform` straight out of 05_train_model.ipynb (single source of truth), applies it
several times to a few sample images per class, and writes a grid to shlomi/train_transform_preview.png.
Run it WITHOUT training, to judge/tune the augmentation.

Usage:
    python shlomi/preview_train_transform.py            # 3 images/class, 5 augmentations each
    python shlomi/preview_train_transform.py 4 6        # 4 images/class, 6 augmentations each

Note: in real training this is applied on top of the 03-degraded images; here it runs on the clean
source images so you can clearly see what the transform itself does.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import utils  # noqa: E402

import matplotlib                  # noqa: E402
matplotlib.use("Agg")             # headless: render straight to a file
import matplotlib.pyplot as plt   # noqa: E402
import torch                       # noqa: E402
from torchvision import transforms  # noqa: E402
from PIL import Image             # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3   # images per class
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5   # augmentations shown per image

# --- pull train_transform out of the notebook ---
nb = json.loads((HERE / "05_train_model.ipynb").read_text(encoding="utf-8"))
ns = {"transforms": transforms, "torch": torch}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "train_transform = transforms.Compose" in src:
        exec(src, ns)
        break
train_transform = ns["train_transform"]


def _unnorm(t):                       # invert Normalize([0.5]*3,[0.5]*3) -> [0,1] for display
    return (t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


INPUT_DIR = utils.CLEAN_DIR
classes = sorted(d.name for d in INPUT_DIR.iterdir() if d.is_dir())
print("source :", INPUT_DIR)
print("classes:", classes, f"| {N} imgs/class, {K} augmentations each")

exts = (".jpg", ".jpeg", ".png")
rows = N * len(classes)
fig, axes = plt.subplots(rows, K + 1, figsize=(2 * (K + 1), 2 * rows), squeeze=False)
ri = 0
for cls in classes:
    files = sorted(p for p in (INPUT_DIR / cls).glob("*") if p.suffix.lower() in exts)[:N]
    for f in files:
        img = Image.open(f).convert("RGB")
        axes[ri][0].imshow(img.resize((224, 224))); axes[ri][0].axis("off")
        axes[ri][0].set_title(f"{cls} (original)", fontsize=9)
        for k in range(K):
            axes[ri][k + 1].imshow(_unnorm(train_transform(img))); axes[ri][k + 1].axis("off")
            if ri == 0:
                axes[ri][k + 1].set_title(f"aug {k + 1}", fontsize=9)
        ri += 1

fig.suptitle("05 train_transform preview - live per-epoch augmentation (NOT saved to disk)", fontsize=12)
plt.tight_layout()
out = HERE / "train_transform_preview.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved ->", out)
