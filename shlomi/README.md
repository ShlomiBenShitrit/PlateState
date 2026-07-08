# shlomi/ — Plate Status Detection (FLUX pipeline)

A self-contained pipeline that decides whether a restaurant plate should be **cleared**, from a single
low-quality CCTV-style image. Training/eval data is **100% synthetic** (generated with FLUX.1-dev, then
degraded to look like security-camera footage). This is an independent fork kept separate from the
shared `code/` notebooks.

## Classes & decision rule (3 classes)
| Class | Meaning | Action |
|---|---|---|
| `clean` | pristine, unused plate — no food | **do not clear** |
| `finished` | meal is over — eaten bare / small leftovers / scraps | **clear** |
| `full` | a moderate-to-full serving of food | **do not clear** |

`clear = {finished}`, `do not clear = {clean, full}`. The boundary is **non-monotonic** (a `clean`
plate has the least on it yet is *do not clear*). Data is *generated* with a finer 5-class scheme and
consolidated to these 3 — see [`data/class_taxonomy.md`](data/class_taxonomy.md).

## Pipeline (run in order)
| Step | Notebook | Reads | Writes |
|---|---|---|---|
| 01 | `01_generate_prompts.ipynb` | — | `data/prompts.json` (attribute-based, identical nuisance distributions across classes) |
| 02 | `02_generate_images.ipynb` / `generate_images.py` | `prompts.json` | a fresh `data/Diff_DataSet_vN/{class}/` (FLUX.1-dev; never overwrites old data) |
| 03 | `03_degrade_and_augment.ipynb` | `data/Diff_DataSet/` | `data/synthetic_degraded/{class}/` — CCTV degradation (**novelty #1**) |
| 04 | `04_split_dataset.ipynb` | `Diff_DataSet/` + `synthetic_degraded/` | `data/splits/{train,val,test}/{class}/` (70/15/15) |
| 05 | `05_train_model.ipynb` | `data/splits/` | `results/` (model weights, plots, real-test CSV) |

- **`Diff_DataSet/`** is the curated **undegraded** image set that step 03 reads (`utils.CLEAN_DIR`).
  New generation runs go to versioned `Diff_DataSet_v1`, `_v2`, … so the curated set is never
  overwritten (`utils.next_versioned_dir()`).
- **03 is class-agnostic** — `degrade_image()` never sees the label, so no degradation can become a
  shortcut. Degradations (downscale, blur, noise, lighting, colour cast, vignette, JPEG, perspective)
  are applied **probabilistically** per image, 2–5 copies each, with deterministic seeds.
- **05** trains a **ResNet18 linear probe** (frozen backbone + new head), then an optional **partial
  fine-tune** (unfreeze `layer4` + head — full-backbone FT overfits the small synthetic set). Reports
  accuracy, macro-F1, confusion matrix, the binary clear/do-not-clear accuracy, a **CLIP zero-shot
  baseline**, and a **real-test per-image CSV**.

## Quick start (university GPU container)
```bash
# 1. get the code
cd ~/my_new_project/Plates && git clone https://github.com/AvitalSkop/genai-project.git
cd genai-project

# 2. dependencies (a dedicated venv keeps a delicate base env safe)
python -m venv ~/plate-train && source ~/plate-train/bin/activate
pip install -r requirements.txt          # if torch already works, drop the torch/torchvision pins

# 3. data — put your curated undegraded images here (NOT gitignored away on the box):
#    shlomi/data/Diff_DataSet/{clean,finished,full}/...

# 4. secrets (gitignored — never committed). W&B key + HF token for FLUX:
#    edit shlomi/secrets.env

# 5. run the training pipeline (03 -> 04 -> 05) headless on a free GPU:
nvidia-smi                               # pick a free GPU index
bash shlomi/run_in_container.sh 0
```
`run_in_container.sh` sets the GPU (`CUDA_VISIBLE_DEVICES`), loads `secrets.env`, points
`PLATE_DATA_ROOT` at the data, and executes the notebooks with `nbconvert`. You can also run the
notebooks interactively (VS Code / Jupyter).

## Tools
| File | Purpose |
|---|---|
| `generate_images.py` | Headless FLUX generation (used with `nohup`). Default output = next `Diff_DataSet_vN`; flags: `--gpu --per-class --classes --out --size --steps`. |
| `preview_degradation.py` | Preview step-03 degradation on N images/class **without** a full run → `degradation_preview.png`. |
| `preview_train_transform.py` | Preview step-05's live `train_transform` augmentation → `train_transform_preview.png`. |
| `run_in_container.sh` | One-shot: env + secrets + run 03→04→05 on a chosen GPU. |
| `make_grid.py` | Build image grids for visual QA. |

## Configuration & environment
- **`utils.py`** is the single source of truth: `SEED=42`, class names, the binary `to_binary()` rule,
  the prompt builder, and all paths. Import it from every notebook.
- **`PLATE_DATA_ROOT`** (env var) overrides the data root without editing any cell. Defaults to
  `shlomi/data/`. `RESULTS_DIR` is always `shlomi/results/` (not affected by it).
- **`secrets.env`** (gitignored): `WANDB_API_KEY` (shared team W&B), `HF_TOKEN` (for FLUX). The 05
  notebook logs to **W&B only if `WANDB_API_KEY` is set**, otherwise it runs without tracking.
- **Reproducibility:** fixed seeds everywhere; the degraded set and splits are deterministic.

## Outputs (`results/`, gitignored)
- `resnet18_frozen_best.pth` — best linear-probe weights.
- `resnet18_finetuned_best.pth` — best fine-tuned weights (kept separate from the probe).
- `real_test_predictions.csv` — per-image real-test report (true vs. predicted vs. clear-rule), UTF-8-BOM
  so Excel reads Hebrew filenames.

## Data layout
```
shlomi/
  data/
    prompts.json                 # tracked
    Diff_DataSet/{clean,finished,full}/      # curated undegraded set (step 03 input)
    Diff_DataSet_vN/...                       # versioned generation runs (never overwrite the above)
    synthetic_degraded/{class}/               # step 03 output
    splits/{train,val,test}/{class}/          # step 04 output
    real_test/{class}/                        # real images for the bonus real-test (calibration only)
  results/                                    # weights, plots, CSV (gitignored)
```
All image data is gitignored; only `prompts.json` and the code are tracked.

## Notes
- Real photos are a **bonus sanity check** only — training/eval is 100% synthetic (never train on real).
- `avital_reference/` holds Avital's raw notebooks for comparison (not part of the pipeline).
- Shared repo: pull before pushing to `main`. When working in a notebook on the container, **pull
  before you edit** (pulling after editing causes conflicts).
