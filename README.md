# EigenCache (ICLR Submission Package)

This repository contains the reproducible code for the EigenCache paper. It packages the accelerated sampling pipelines, multi-GPU launcher, and evaluation tooling used in our experiments while omitting proprietary checkpoints.

## Quick Start

1. **Create a virtual environment** (see `docs/ENVIRONMENT_SETUP.md`).
2. **Install dependencies:**
   ```bash
   pip install -e ".[all]"
   ```
3. **Download required models** into `models/` using `scripts/download_flux.py`, `scripts/hf_download.py`, or manual Hugging Face commands. The FLUX.1-dev weights are required for all experiments.
4. **Prepare prompts:** edit `data/prompt.txt` or provide your own prompt list.

## Running Samplers

- **Single GPU:**
  ```bash
  bash scripts/sample.sh --mode EigenCache --interval 5 --max_order 2 --hicache_scale 0.6
  ```
- **Multi GPU launcher (shards prompts across devices):**
  ```bash
  bash RUN/multi_gpu_launcher.sh --mode EigenCache --gpus 0,1 \
       --prompt-file data/prompt.txt --base-output-dir results/eigencache
  ```
  Additional options are documented in `RUN/multi_gpu_launcher.sh`.

## Evaluation Pipeline

Aggregate outputs can be scored with ImageReward, PSNR, SSIM, and CLIP metrics via:
```bash
bash evaluation/run_eval.sh \
    --gt results/taylor/interval_1/order_2 \
    --acc EigenCache=results/eigencache/run_xx
```
The script automatically activates the local environment, hashes logs under `evaluation/logs/`, and supports multiple `--acc` inputs.

## Repository Layout

- `scripts/sample.sh` – single-device sampling driver.
- `RUN/multi_gpu_launcher.sh` – orchestrates distributed sampling.
- `RUN/multi_gpu_launcher.py` – shards prompts and merges outputs.
- `evaluation/run_eval.sh` – wrapper for metric computation.
- `src/` – Python implementation of the EigenCache kernels and utilities.
- `models/` – placeholder for downloaded weights (kept empty in git).
- `results/` – default output directory for generated samples.

## Reproducing Paper Figures

Detailed experiment settings, ablation notes, and prompts are tracked under `evaluation/` and `docs/`. Please consult:
- `evaluation/ablation/ablation.md` for key command lines and metric tables.
- `docs/` for model-specific notes and extended methodology.

## License

This package reuses components distributed under the original FLUX license. Refer to `LICENSE` and `model_licenses/` for the complete terms.
