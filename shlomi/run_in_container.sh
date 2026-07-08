#!/usr/bin/env bash
# One-shot pipeline runner for the university GPU container.
#
#   bash shlomi/run_in_container.sh [GPU_INDEX]
#
# It sets the environment (GPU, secrets, data location) and executes the notebooks
# headless, in order, with NO per-run cell editing. Run it from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."                      # -> repo root

# 1) Pick a GPU. On a shared box, choose a free one (check `nvidia-smi`). Default 0.
export CUDA_VISIBLE_DEVICES="${1:-0}"

# 2) Load secrets (W&B key + HF token). Never committed - see shlomi/secrets.env.
set -a
[ -f shlomi/secrets.env ] && . shlomi/secrets.env
set +a

# 3) Where the data lives: a folder that contains  synthetic_clean/{clean,finished,full}/.
#    Override by exporting PLATE_DATA_ROOT before calling; defaults to shlomi/data.
export PLATE_DATA_ROOT="${PLATE_DATA_ROOT:-$PWD/shlomi/data}"

echo "GPU=$CUDA_VISIBLE_DEVICES | DATA=$PLATE_DATA_ROOT | W&B=${WANDB_API_KEY:+on}${WANDB_API_KEY:-off}"

# 4) Execute the training pipeline in order (03 degrade -> 04 split -> 05 train).
#    (Step 02 / FLUX generation is run separately via generate_images.py.)
cd shlomi
for nb in 03_degrade_and_augment 04_split_dataset 05_train_model; do
    echo "=== running ${nb}.ipynb ==="
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 "${nb}.ipynb"
done
echo "PIPELINE DONE. Trained weights + plots are under shlomi/results/ ; metrics on W&B."
