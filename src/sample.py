import os
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline
from tqdm import tqdm

from flux.sampling import get_noise, get_schedule, prepare, unpack, denoise_test_FLOPs
from flux.ideas import denoise_cache
from flux.eigencache import (
    KernelCalibrator,
    KernelStore,
    apply_kl_truncation,
    precompute_weight_tables,
)
from flux.taylor_utils import clear_collected_features, get_collected_features
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85  # NSFW score threshold


@dataclass
class SamplingOptions:
    prompts: list[str]  # List of prompts
    width: int  # Image width
    height: int  # Image height
    num_steps: int  # Number of sampling steps
    guidance: float  # Guidance value
    seed: int | None  # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int  # Batch size (batching of prompts)
    model_name: str  # Model name
    output_dir: str  # Output directory
    start_index: int  # Starting index offset for output numbering
    add_sampling_metadata: bool  # Whether to add metadata
    use_nsfw_filter: bool  # Whether to enable NSFW filter
    test_FLOPs: bool  # Whether in FLOPs test mode (no actual image generation)
    cache_mode: str  # Cache mode ('original', 'ToCa', 'Taylor', 'HiCache', 'Delta', 'collect')
    interval: int  # Cache period length
    max_order: int  # Maximum order of Taylor expansion
    first_enhance: int  # Initial enhancement steps
    hicache_scale: float  # HiCache scaling factor
    base_threshold: float  # Speca base threshold
    decay_rate: float  # Speca decay rate
    min_taylor_steps: int  # Speca minimum taylor steps
    max_taylor_steps: int  # Speca maximum taylor steps
    speca_error_metric: str  # Speca error metric
    speca_max_full_gap: int  # Speca max full gap
    speca_check_layers: list[int]  # Speca check layers
    speca_ema_alpha: float  # Speca EMA alpha
    # ClusCa parameters
    clusca_fresh_threshold: int  # ClusCa fresh threshold
    clusca_cluster_num: int  # Number of clusters for ClusCa
    clusca_cluster_method: str  # Clustering method (kmeans/kmeans++/random)
    clusca_k: int  # Number of selected fresh tokens per cluster
    clusca_propagation_ratio: float  # Propagation ratio for cluster updates
    # Kalman parameters
    kalman_sigma_candidates: list[float] | None  # Candidate sigmas for Kalman
    kalman_process_scale: float | None  # Kalman process scale
    kalman_measurement_scale: float | None  # Kalman measurement scale
    kalman_innovation_clip: float | None  # Kalman innovation clip
    # Feature collection parameters (enabled when cache_mode='collect')
    feature_layers: list[int]  # Target layers for feature collection
    feature_modules: list[str]  # Target modules for feature collection
    feature_streams: list[str]  # Target streams for feature collection
    skip_decoding: bool  # Skip VAE decoding (feature collection only)
    feature_output_dir: str  # Feature output directory


def main(opts: SamplingOptions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification", model="Falconsai/nsfw_image_detection", device=device
        )
    else:
        nsfw_classifier = None

    # Load model
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if opts.num_steps is None:
        opts.num_steps = 4 if model_name == "flux-schnell" else 50

    # Ensure width and height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    # Set output directory and index
    # In feature collection mode, save everything to feature_output_dir
    if opts.cache_mode == "collect":
        # Create a timestamp-based subdirectory for this run
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(opts.feature_output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Save images in the same run directory
        output_name = os.path.join(run_dir, "images", "img_{idx}.jpg")
        os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)

        # Save command info and metadata
        save_run_metadata(opts, run_dir)
        save_sampling_config(opts, run_dir)
    else:
        # Normal mode: use specified output_dir
        output_name = os.path.join(opts.output_dir, "img_{idx}.jpg")
        if not os.path.exists(opts.output_dir):
            os.makedirs(opts.output_dir)
        # Save a machine-readable config for reproducibility
        save_sampling_config(opts, opts.output_dir)

    idx = opts.start_index  # Image index offset for numbering

    # Initialize model components
    torch_device = device

    # Load T5 and CLIP models to GPU
    t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(torch_device)

    # Load model to GPU
    model = load_flow_model(model_name, device=torch_device)
    ae = load_ae(model_name, device=torch_device)

    eigencache_cfg = None
    if getattr(opts, "cache_method", "none") == "eigencache":
        if not getattr(opts, "eigencache_kernel_path", ""):
            raise ValueError("EigenCache inference requires --eigencache_kernel_path.")
        eigencache_cfg = {
            "kernel_path": opts.eigencache_kernel_path,
            "window": opts.eigencache_window_M,
            "lambda": opts.eigencache_lambda,
            "schedule": getattr(opts, "cache_schedule", "fixed"),
            "budget": opts.eigencache_budget_B,
            "var_tau": opts.eigencache_var_tau,
            "layer_weights": opts.eigencache_layer_weights,
        }

    # Set random seed
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    prompts = opts.prompts

    total_images = len(prompts) * opts.num_images_per_prompt

    progress_bar = tqdm(total=total_images, desc="Generating images")

    # Compute number of prompt batches
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            # Prepare random seed
            seed = base_seed + idx  # Assign a different seed for each image
            idx += num_prompts_in_batch  # Update image index

            # Prepare input
            batch_size = num_prompts_in_batch
            x = get_noise(
                batch_size,
                opts.height,
                opts.width,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # Prepare prompts
            # batch_prompts is a list containing the prompts in the current batch
            inp = prepare(t5, clip, x, prompt=batch_prompts)
            timesteps = get_schedule(
                opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell")
            )

            # Denoising
            with torch.no_grad():
                if opts.test_FLOPs:
                    x = denoise_test_FLOPs(
                        model,
                        **inp,
                        timesteps=timesteps,
                        guidance=opts.guidance,
                        cache_mode=opts.cache_mode,
                        base_threshold=opts.base_threshold,
                        decay_rate=opts.decay_rate,
                        min_taylor_steps=opts.min_taylor_steps,
                        max_taylor_steps=opts.max_taylor_steps,
                    )
                else:
                    # Configure feature collection (enabled when cache_mode='collect')
                    feature_collection_enabled = opts.cache_mode == "collect"
                    feature_config = None
                    if feature_collection_enabled:
                        feature_config = {
                            "target_layers": opts.feature_layers,
                            "target_modules": opts.feature_modules,
                            "target_streams": opts.feature_streams,
                        }

                    x = denoise_cache(
                        model,
                        **inp,
                        timesteps=timesteps,
                        guidance=opts.guidance,
                        cache_mode=opts.cache_mode,
                        cache_method=getattr(opts, "cache_method", "none"),
                        interval=opts.interval,
                        max_order=opts.max_order,
                        first_enhance=opts.first_enhance,
                        hicache_scale=opts.hicache_scale,
                        base_threshold=opts.base_threshold,
                        decay_rate=opts.decay_rate,
                        min_taylor_steps=opts.min_taylor_steps,
                        max_taylor_steps=opts.max_taylor_steps,
                        # ClusCa parameters
                        clusca_fresh_threshold=opts.clusca_fresh_threshold,
                        clusca_cluster_num=opts.clusca_cluster_num,
                        clusca_cluster_method=opts.clusca_cluster_method,
                        clusca_k=opts.clusca_k,
                        clusca_propagation_ratio=opts.clusca_propagation_ratio,
                        # Feature collection parameters
                        enable_feature_collection=feature_collection_enabled,
                        feature_collection_config=feature_config,
                        eigencache_config=eigencache_cfg,
                    )
                    # Note: Speca logging was removed with mode cleanup
                    # x = search_denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, interval=opts.interval, max_order=opts.max_order, first_enhance=opts.first_enhance)

                # Handle feature collection
                if feature_collection_enabled:
                    from flux.taylor_utils import get_collected_features

                    features, metadata = get_collected_features(model._last_cache_dic)
                    # Save feature data to the same run directory
                    current_run_dir = None
                    if opts.cache_mode == "collect":
                        # Extract run_dir from output_name path
                        current_run_dir = os.path.dirname(
                            os.path.dirname(output_name)
                        )  # Go up from /images/img_x.jpg
                    save_collected_features(
                        features, metadata, batch_prompts, opts, idx - num_prompts_in_batch, current_run_dir
                    )

                # Decode latent variables (skip if only collecting features)
                if not opts.skip_decoding:
                    x = unpack(x.float(), opts.height, opts.width)
                    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                        x = ae.decode(x)

            # Convert to PIL format and save (skip if only collecting features)
            if not opts.skip_decoding:
                x = x.clamp(-1, 1)
                x = embed_watermark(x.float())
                x = rearrange(x, "b c h w -> b h w c")

                for i in range(batch_size):
                    img_array = x[i]
                    img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().numpy().astype(np.uint8))

                    # Optional NSFW filtering
                    if opts.use_nsfw_filter:
                        nsfw_result = nsfw_classifier(img)
                        nsfw_score = next(
                            (res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0
                        )
                    else:
                        nsfw_score = 0.0  # If the filter is not enabled, assume safe

                    if nsfw_score < NSFW_THRESHOLD:
                        exif_data = Image.Exif()
                        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                        exif_data[ExifTags.Base.Model] = model_name
                        if opts.add_sampling_metadata:
                            exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                        # Save image
                        fn = output_name.format(idx=idx - num_prompts_in_batch + i)
                        img.save(fn, exif=exif_data, quality=95, subsampling=0)
                    else:
                        print("Generated image may contain inappropriate content, skipped.")

                    progress_bar.update(1)
            else:
                # If skipping decoding, still update progress bar
                for i in range(batch_size):
                    progress_bar.update(1)

    progress_bar.close()


def read_prompts(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def _parse_phase_boundaries(raw: str) -> list[int] | None:
    if not raw:
        return None
    values: list[int] = []
    for token in raw.split(","):
        piece = token.strip()
        if not piece:
            continue
        try:
            values.append(int(piece))
        except ValueError as exc:
            raise ValueError(f"鏃犳硶瑙ｆ瀽 eigencache_phase_boundaries 涓殑鏁存暟 '{piece}'") from exc
    return values or None


def _parse_phase_names(raw: str) -> list[str] | None:
    if not raw:
        return None
    names = [token.strip() for token in raw.split(",") if token.strip()]
    return names or None


def run_eigencache_calibration(args, prompts: list[str]):
    """
    Minimal offline calibration loop for EigenCache kernels.
    """
    if not args.eigencache_kernel_path:
        raise ValueError("Please provide --eigencache_kernel_path when --eigencache_calibrate is set.")
    if not prompts:
        raise ValueError("Calibration requires at least one prompt.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    num_steps = args.num_steps if args.num_steps is not None else (4 if model_name == "flux-schnell" else 50)
    width = 16 * (args.width // 16)
    height = 16 * (args.height // 16)

    torch_device = device
    t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(model_name, device=torch_device)

    base_seed = args.seed if args.seed is not None else torch.randint(0, 2**32, (1,)).item()
    phase_boundaries = _parse_phase_boundaries(args.eigencache_phase_boundaries)
    phase_names = _parse_phase_names(args.eigencache_phase_names)
    calibrator = KernelCalibrator(
        num_steps=num_steps,
        timesteps=None,
        phase_boundaries=phase_boundaries,
        phase_names=phase_names,
    )
    feature_config = {
        "target_layers": list(range(57)),
        "target_modules": ["any"],
        "target_streams": ["any"],
    }

    for run_idx in range(args.eigencache_calib_runs):
        prompt = prompts[run_idx % len(prompts)]
        calibrator.start_run()
        x = get_noise(
            1,
            height,
            width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=base_seed + run_idx,
        )
        inp = prepare(t5, clip, x, prompt=[prompt])
        timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell"))
        if calibrator.timesteps is None:
            calibrator.timesteps = timesteps[:-1]

        with torch.no_grad():
            denoise_cache(
                model,
                **inp,
                timesteps=timesteps,
                guidance=args.guidance,
                cache_mode="collect",
                interval=args.interval,
                max_order=args.max_order,
                first_enhance=args.first_enhance,
                hicache_scale=args.hicache_scale,
                enable_feature_collection=True,
                feature_collection_config=feature_config,
                cache_method="none",
                stream_calibrator=calibrator,
            )
        calibrator.finalize_run()

    payload = calibrator.export()
    if args.eigencache_kl_rank > 0:
        payload = apply_kl_truncation(payload, args.eigencache_kl_rank)
    if args.eigencache_precompute_weights:
        payload = precompute_weight_tables(payload, args.eigencache_window_M, args.eigencache_lambda)
    KernelStore.save(args.eigencache_kernel_path, payload)
    print(f"[EigenCache] Saved kernels to {args.eigencache_kernel_path}")


def save_run_metadata(opts: SamplingOptions, run_dir: str):
    """
    淇濆瓨杩愯鐨勫厓鏁版嵁鍜屽懡浠や俊鎭?

    Args:
        opts: 閲囨牱閫夐」
        run_dir: 杩愯鐩綍
    """
    import json
    import sys
    from datetime import datetime

    # Save command information
    command_info = {
        "timestamp": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "working_directory": os.getcwd(),
        "config": {
            "cache_mode": opts.cache_mode,
            "cache_method": getattr(opts, "cache_method", "none"),
            "cache_schedule": getattr(opts, "cache_schedule", "fixed"),
            "eigencache_phase_boundaries": getattr(opts, "eigencache_phase_boundaries", None),
            "eigencache_phase_names": getattr(opts, "eigencache_phase_names", None),
            "eigencache_precompute_weights": getattr(opts, "eigencache_precompute_weights", False),
            "eigencache_kl_rank": getattr(opts, "eigencache_kl_rank", 0),
            "feature_layers": opts.feature_layers,
            "feature_modules": opts.feature_modules,
            "feature_streams": opts.feature_streams,
            "model_name": opts.model_name,
            "width": opts.width,
            "height": opts.height,
            "num_steps": opts.num_steps,
            "guidance": opts.guidance,
            "seed": opts.seed,
            "num_images_per_prompt": opts.num_images_per_prompt,
            "batch_size": opts.batch_size,
            "interval": opts.interval if opts.cache_mode != "collect" else "auto (refresh each step)",
            "max_order": opts.max_order if opts.cache_mode != "collect" else 0,
            "first_enhance": opts.first_enhance if opts.cache_mode != "collect" else opts.num_steps,
            "hicache_scale": opts.hicache_scale if opts.cache_mode != "collect" else "unused",
            "skip_decoding": opts.skip_decoding,
            "actual_cache_behavior": (
                "collect mode: full compute every step (no cache)"
                if opts.cache_mode == "collect"
                else f"{opts.cache_mode} cache"
            ),
            "command_line_note": (
                "interval/max_order are overridden in collect mode"
                if opts.cache_mode == "collect"
                else None
            ),
            "eigencache_kernel": getattr(opts, "eigencache_kernel_path", None),
            "kalman": {
                "sigma_candidates": list(opts.kalman_sigma_candidates)
                if opts.kalman_sigma_candidates
                else None,
                "process_scale": opts.kalman_process_scale,
                "measurement_scale": opts.kalman_measurement_scale,
                "innovation_clip": opts.kalman_innovation_clip,
            }
            if opts.cache_mode == "Kalman-HiCache"
            else None,
        },
    }

    # Save command info as JSON
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(command_info, f, indent=2, ensure_ascii=False)

    # Save command as shell script for reproduction
    with open(os.path.join(run_dir, "reproduce_command.sh"), "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# 閲嶇幇姝ゆ杩愯鐨勫懡浠n")
        f.write("# 鐢熸垚鏃堕棿: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        # Add environment variables
        f.write("# 璁剧疆鐜鍙橀噺\n")
        f.write(
            'export FLUX_DEV="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"\n'
        )
        f.write(
            'export AE="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"\n\n'
        )

        # Add the original command
        f.write("# 杩愯鍛戒护\n")
        f.write(" ".join(sys.argv) + "\n")

    # Copy prompt file to run directory
    try:
        prompt_file_path = sys.argv[sys.argv.index("--prompt_file") + 1]
        if os.path.exists(prompt_file_path):
            import shutil

            prompt_file_name = os.path.basename(prompt_file_path)
            shutil.copy2(prompt_file_path, os.path.join(run_dir, f"prompts_{prompt_file_name}"))
    except (ValueError, IndexError):
        # If --prompt_file not found in sys.argv, skip copying
        pass

    print(f"馃搵 杩愯淇℃伅宸蹭繚瀛樺埌: {run_dir}")


def save_sampling_config(opts: SamplingOptions, out_dir: str):
    """Dump all key sampling parameters to a config.json in the output directory.

    This complements metadata and enables reproducible evaluation by capturing
    cache settings (interval, order, scales), sampler settings, and model info.
    """
    import json
    import sys
    from datetime import datetime

    os.makedirs(out_dir, exist_ok=True)

    # Build a structured config
    cfg = {
        "timestamp": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "working_directory": os.getcwd(),
        "mode": opts.cache_mode,
        "sampler": {
            "width": int(opts.width),
            "height": int(opts.height),
            "num_steps": int(opts.num_steps),
            "guidance": float(opts.guidance),
            "seed": int(opts.seed) if isinstance(opts.seed, int) else opts.seed,
            "num_images_per_prompt": int(opts.num_images_per_prompt),
            "batch_size": int(opts.batch_size),
            "start_index": int(opts.start_index),
        },
        "model": {
            "name": opts.model_name,
        },
        "cache": {
            "interval": int(opts.interval),
            "max_order": int(opts.max_order),
            "first_enhance": int(opts.first_enhance),
            "hicache_scale": float(opts.hicache_scale),
            "method": getattr(opts, "cache_method", "none"),
            "schedule": getattr(opts, "cache_schedule", "fixed"),
            "eigencache_kernel": getattr(opts, "eigencache_kernel_path", None),
            "phase_boundaries": getattr(opts, "eigencache_phase_boundaries", None),
            "phase_names": getattr(opts, "eigencache_phase_names", None),
            "precompute_weights": getattr(opts, "eigencache_precompute_weights", False),
            "kl_rank": getattr(opts, "eigencache_kl_rank", 0),
        },
        "speca": {
            "base_threshold": float(opts.base_threshold),
            "decay_rate": float(opts.decay_rate),
            "min_taylor_steps": int(opts.min_taylor_steps),
            "max_taylor_steps": int(opts.max_taylor_steps),
            "error_metric": opts.speca_error_metric,
            "max_full_gap": int(opts.speca_max_full_gap),
            "check_layers": list(opts.speca_check_layers),
            "ema_alpha": float(opts.speca_ema_alpha),
        },
        "clusca": {
            "fresh_threshold": int(opts.clusca_fresh_threshold),
            "cluster_num": int(opts.clusca_cluster_num),
            "cluster_method": opts.clusca_cluster_method,
            "k": int(opts.clusca_k),
            "propagation_ratio": float(opts.clusca_propagation_ratio),
        },
        "kalman": {
            "sigma_candidates": list(opts.kalman_sigma_candidates) if opts.kalman_sigma_candidates else None,
            "process_scale": float(opts.kalman_process_scale)
            if opts.kalman_process_scale is not None
            else None,
            "measurement_scale": float(opts.kalman_measurement_scale)
            if opts.kalman_measurement_scale is not None
            else None,
            "innovation_clip": float(opts.kalman_innovation_clip)
            if opts.kalman_innovation_clip is not None
            else None,
        },
        "prompts": {
            "count": len(opts.prompts),
        },
    }

    # Persist to JSON
    dst = os.path.join(out_dir, "config.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # Silent by default (avoid noisy logs under multi-gpu), caller can inspect file


def save_collected_features(features, metadata, prompts, opts: SamplingOptions, image_idx, run_dir=None):
    """Save collected features to disk."""
    import os
    import pickle
    from datetime import datetime

    if run_dir:
        output_base_dir = os.path.join(run_dir, "features")
    else:
        output_base_dir = opts.feature_output_dir

    os.makedirs(output_base_dir, exist_ok=True)

    if not features:
        return []

    saved_files = []
    print(f"[INFO] Saving features (sample {image_idx})...")

    for layer_idx, layer_data in features.items():
        base_path = os.path.join(output_base_dir, opts.model_name, f"layer_{layer_idx}")

        for module_name, module_features in layer_data.items():
            if module_name == "total":
                module_output_dir = base_path
            else:
                module_output_dir = os.path.join(base_path, f"module_{module_name}")

            os.makedirs(module_output_dir, exist_ok=True)

            filename = f"features_sample_{image_idx + 1:03d}.pkl"
            filepath = os.path.join(module_output_dir, filename)

            module_metadata = metadata.get(layer_idx, {}).get(module_name, [])

            data = {
                "features": module_features,
                "metadata": module_metadata,
                "prompts": prompts,
                "layer": layer_idx,
                "module": module_name,
                "feature_shape": str(module_features[0].shape) if module_features else "empty",
                "num_timesteps": len(module_features),
                "image_idx": image_idx,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "cache_mode": opts.cache_mode,
                    "interval": opts.interval if opts.cache_mode != "collect" else "auto (refresh each step)",
                    "max_order": opts.max_order if opts.cache_mode != "collect" else 0,
                    "first_enhance": opts.first_enhance
                    if opts.cache_mode != "collect"
                    else len(metadata.get(layer_idx, {}).get(module_name, [])),
                    "hicache_scale": opts.hicache_scale if opts.cache_mode != "collect" else "unused",
                    "actual_mode": "collect (original mode with feature capture)"
                    if opts.cache_mode == "collect"
                    else opts.cache_mode,
                    "feature_collection_note": "collect mode: full compute each step, no cache acceleration"
                    if opts.cache_mode == "collect"
                    else None,
                },
            }

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            saved_files.append(filepath)
            print(f"[INFO] Layer {layer_idx}, module {module_name} -> {filepath}")

    print(f"[INFO] Saved {len(saved_files)} feature files.")
    return saved_files


def save_speca_cache_logs(
    cache_dic, output_dir: str, run_index: int, prompts: list[str], opts: SamplingOptions, seed: int
):
    """Persist Speca/Hi-Speca scheduling logs for later analysis."""

    if not cache_dic or not cache_dic.get("speca_enabled", False):
        return

    import json
    from datetime import datetime

    output_dir = os.path.abspath(output_dir)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    events = cache_dic.get("speca_logs", [])
    error_events = cache_dic.get("speca_error_logs", [])
    summary = {
        "total_steps": len(events),
        "full_steps": sum(1 for e in events if e.get("decision") == "full"),
        "taylor_steps": sum(1 for e in events if e.get("decision") == "taylor_cache"),
        "corrective_actions": sum(1 for e in events if e.get("corrective_action")),
        "max_error": cache_dic.get("speca_error_state", {}).get("max_error"),
    }

    config = {
        "mode": cache_dic.get("mode"),
        "base_threshold": cache_dic.get("base_threshold"),
        "decay_rate": cache_dic.get("decay_rate"),
        "min_taylor_steps": cache_dic.get("min_taylor_steps"),
        "max_taylor_steps": cache_dic.get("max_taylor_steps"),
        "max_full_gap": cache_dic.get("max_full_gap"),
        "error_metric": cache_dic.get("error_metric"),
        "check_layers": cache_dic.get("check_layers"),
        "ema_alpha": cache_dic.get("ema_alpha"),
        "hicache_scale_factor": cache_dic.get("hicache_scale_factor"),
    }

    run_payload = {
        "timestamp": datetime.now().isoformat(),
        "run_index": run_index,
        "prompts": prompts,
        "seed": seed,
        "batch_size": len(prompts),
        "num_steps": opts.num_steps,
        "guidance": opts.guidance,
        "cache_mode": opts.cache_mode,
        "summary": summary,
        "config": config,
        "events": events,
        "error_events": error_events,
    }

    log_path = os.path.join(log_dir, f"speca_run_{run_index:04d}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, ensure_ascii=False, indent=2)


def app():
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt text file.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of prompts to use.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of sampling steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (prompt batching).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-schnell",
        choices=["flux-dev", "flux-schnell"],
        help="Model name.",
    )
    parser.add_argument("--output_dir", type=str, default="./samples", help="Directory to save images.")
    parser.add_argument(
        "--add_sampling_metadata", action="store_true", help="Whether to add prompt metadata to images."
    )
    parser.add_argument("--use_nsfw_filter", action="store_true", help="Enable NSFW filter.")
    parser.add_argument("--test_FLOPs", action="store_true", help="Test inference computation cost.")
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="original",
        choices=[
            "original",
            "ToCa",
            "Taylor",
            "Taylor-Scaled",
            "HiCache",
            "Delta",
            "collect",
            "ClusCa",
            "Hi-ClusCa",
        ],
        help="Cache mode for denoising.",
    )
    parser.add_argument("--interval", type=int, default=10, help="Cache period length.")
    parser.add_argument("--max_order", type=int, default=5, help="Maximum order of Taylor expansion.")
    parser.add_argument("--first_enhance", type=int, default=5, help="Initial enhancement steps.")
    parser.add_argument("--hicache_scale", type=float, default=1.0, help="HiCache scaling factor.")
    parser.add_argument(
        "--cache_method",
        type=str,
        default="none",
        choices=["none", "hicache", "eigencache"],
        help="Prediction method override (none keeps cache_mode default).",
    )
    parser.add_argument(
        "--schedule",
        dest="cache_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "greedy", "variance"],
        help="EigenCache scheduling policy.",
    )
    parser.add_argument(
        "--eigencache_kernel_path",
        type=str,
        default="",
        help="Path to EigenCache kernel file (required for eigencache inference).",
    )
    parser.add_argument(
        "--eigencache_calibrate",
        action="store_true",
        help="Run EigenCache kernel calibration and exit.",
    )
    parser.add_argument(
        "--eigencache_calib_prompts",
        type=str,
        default="",
        help="Optional prompt file for EigenCache calibration.",
    )
    parser.add_argument(
        "--eigencache_calib_runs",
        type=int,
        default=16,
        help="Number of calibration trajectories to record.",
    )
    parser.add_argument(
        "--eigencache_window_M",
        type=int,
        default=3,
        help="Number of anchor steps used by EigenCache.",
    )
    parser.add_argument(
        "--eigencache_lambda",
        type=float,
        default=1e-3,
        help="Jitter lambda for Kriging solves.",
    )
    parser.add_argument(
        "--eigencache_budget_B",
        type=int,
        default=8,
        help="Budget for greedy EigenCache scheduling.",
    )
    parser.add_argument(
        "--eigencache_var_tau",
        type=float,
        default=0.05,
        help="Variance threshold for EigenCache scheduling.",
    )
    parser.add_argument(
        "--eigencache_layer_weights",
        type=str,
        default="",
        help="Comma-separated layer weights for EigenCache variance schedule.",
    )
    parser.add_argument(
        "--eigencache_phase_boundaries",
        type=str,
        default="",
        help="Comma-separated step boundaries (include 0 and T) for phase-specific kernels.",
    )
    parser.add_argument(
        "--eigencache_phase_names",
        type=str,
        default="",
        help="Comma-separated phase names matching the segmented intervals.",
    )
    parser.add_argument(
        "--eigencache_precompute_weights",
        action="store_true",
        help="Precompute Kriging weight tables for contiguous anchor patterns.",
    )
    parser.add_argument(
        "--eigencache_kl_rank",
        type=int,
        default=0,
        help="Optional KL truncation rank for EigenCache kernels.",
    )
    parser.add_argument("--base_threshold", type=float, default=7.0, help=argparse.SUPPRESS)
    parser.add_argument("--decay_rate", type=float, default=0.3, help=argparse.SUPPRESS)
    parser.add_argument(
        "--min_taylor_steps",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max_taylor_steps",
        type=int,
        default=8,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--speca_error_metric",
        type=str,
        default="relative_l1",
        choices=["relative_l1", "cosine", "psnr"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--speca_max_full_gap",
        type=int,
        default=6,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--speca_check_layers",
        type=str,
        default="37",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--speca_ema_alpha",
        type=float,
        default=0.2,
        help=argparse.SUPPRESS,
    )
    # ClusCa arguments
    parser.add_argument(
        "--clusca_fresh_threshold",
        type=int,
        default=5,
        help="ClusCa fresh threshold.",
    )
    parser.add_argument(
        "--clusca_cluster_num",
        type=int,
        default=16,
        help="Number of clusters for ClusCa.",
    )
    parser.add_argument(
        "--clusca_cluster_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "kmeans++", "random"],
        help="Clustering method for ClusCa.",
    )
    parser.add_argument(
        "--clusca_k",
        type=int,
        default=1,
        help="Number of selected fresh tokens per cluster.",
    )
    parser.add_argument(
        "--clusca_propagation_ratio",
        type=float,
        default=0.005,
        help="Propagation ratio for cluster updates.",
    )
    # Kalman-HiCache arguments
    parser.add_argument(
        "--kalman_sigma_candidates",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--kalman_process_scale",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--kalman_measurement_scale",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--kalman_innovation_clip",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    # Feature collection arguments (enabled when cache_mode='collect')
    parser.add_argument(
        "--feature_layers",
        type=int,
        nargs="+",
        default=[14],
        help="Target layers for feature collection (supports multiple layers).",
    )
    parser.add_argument(
        "--feature_modules",
        type=str,
        nargs="+",
        default=["any"],
        help="Target modules for feature collection.",
    )
    parser.add_argument(
        "--feature_streams",
        type=str,
        nargs="+",
        default=["any"],
        help="Target streams for feature collection.",
    )
    parser.add_argument(
        "--skip_decoding", action="store_true", help="Skip VAE decoding (feature collection only)."
    )
    parser.add_argument(
        "--feature_output_dir", type=str, default="./features", help="Feature output directory."
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting index offset for img_*.jpg numbering."
    )

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]

    if args.eigencache_calibrate:
        calib_path = args.eigencache_calib_prompts or args.prompt_file
        calib_prompts = read_prompts(calib_path)
        run_eigencache_calibration(args, calib_prompts)
        return

    speca_layers = []
    for token in args.speca_check_layers.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "last":
            speca_layers.append(-1)
        else:
            speca_layers.append(int(token))

    kalman_sigma_candidates: list[float] | None = None
    raw_sigma = args.kalman_sigma_candidates.strip()
    if raw_sigma:
        kalman_sigma_candidates = []
        for piece in raw_sigma.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                kalman_sigma_candidates.append(float(piece))
            except ValueError as exc:  # Provide clearer error context
                raise ValueError(f"鏃犳硶瑙ｆ瀽 Kalman 蟽 鍊欓€夊€? '{piece}'") from exc
        if not kalman_sigma_candidates:
            kalman_sigma_candidates = None

    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        start_index=args.start_index,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        test_FLOPs=args.test_FLOPs,
        cache_mode=args.cache_mode,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        hicache_scale=args.hicache_scale,
        base_threshold=args.base_threshold,
        decay_rate=args.decay_rate,
        min_taylor_steps=args.min_taylor_steps,
        max_taylor_steps=args.max_taylor_steps,
        speca_error_metric=args.speca_error_metric,
        speca_max_full_gap=args.speca_max_full_gap,
        speca_check_layers=speca_layers,
        speca_ema_alpha=args.speca_ema_alpha,
        # ClusCa parameters
        clusca_fresh_threshold=args.clusca_fresh_threshold,
        clusca_cluster_num=args.clusca_cluster_num,
        clusca_cluster_method=args.clusca_cluster_method,
        clusca_k=args.clusca_k,
        clusca_propagation_ratio=args.clusca_propagation_ratio,
        kalman_sigma_candidates=kalman_sigma_candidates,
        kalman_process_scale=args.kalman_process_scale,
        kalman_measurement_scale=args.kalman_measurement_scale,
        kalman_innovation_clip=args.kalman_innovation_clip,
        # Feature collection parameters (enabled when cache_mode='collect')
        feature_layers=args.feature_layers,
        feature_modules=args.feature_modules,
        feature_streams=args.feature_streams,
        skip_decoding=args.skip_decoding,
        feature_output_dir=args.feature_output_dir,
    )

    # EigenCache-specific runtime options (kept outside dataclass init for compatibility)
    opts.cache_method = args.cache_method
    opts.cache_schedule = args.cache_schedule
    opts.eigencache_kernel_path = args.eigencache_kernel_path
    opts.eigencache_window_M = args.eigencache_window_M
    opts.eigencache_lambda = args.eigencache_lambda
    opts.eigencache_budget_B = args.eigencache_budget_B
    opts.eigencache_var_tau = args.eigencache_var_tau
    opts.eigencache_layer_weights = args.eigencache_layer_weights
    opts.eigencache_phase_boundaries = args.eigencache_phase_boundaries
    opts.eigencache_phase_names = args.eigencache_phase_names
    opts.eigencache_precompute_weights = args.eigencache_precompute_weights
    opts.eigencache_kl_rank = args.eigencache_kl_rank

    main(opts)


if __name__ == "__main__":
    app()





