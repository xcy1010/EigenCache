import torch
from torch import Tensor

from ..eigencache import EigenCacheAccelerator, KernelStore
from ..model import Flux
from ..modules.cache_functions import cache_init


def denoise_cache(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # cache mode
    cache_mode: str = "Taylor",
    cache_method: str = "none",
    # cache parameters
    interval: int = 6,
    max_order: int = 1,
    first_enhance: int = 3,
    hicache_scale: float = 0.5,
    base_threshold: float | None = None,
    decay_rate: float | None = None,
    min_taylor_steps: int | None = None,
    max_taylor_steps: int | None = None,
    # ClusCa parameters
    clusca_fresh_threshold: int | None = None,
    clusca_cluster_num: int | None = None,
    clusca_cluster_method: str | None = None,
    clusca_k: int | None = None,
    clusca_propagation_ratio: float | None = None,
    # 🔥 新增：特征收集参数
    enable_feature_collection: bool = False,
    feature_collection_config: dict = None,
    eigencache_config: dict | None = None,
    stream_calibrator=None,
):
    # Removed Speca mode support

    clusca_kwargs = None
    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        clusca_kwargs = {
            "clusca_fresh_threshold": clusca_fresh_threshold,
            "clusca_cluster_num": clusca_cluster_num,
            "clusca_cluster_method": clusca_cluster_method,
            "clusca_k": clusca_k,
            "clusca_propagation_ratio": clusca_propagation_ratio,
        }
        clusca_kwargs = {k: v for k, v in clusca_kwargs.items() if v is not None}

    kalman_kwargs = None
    # Removed Kalman-HiCache mode support

    # init cache with specified mode and parameters
    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        model_kwargs = clusca_kwargs
    else:
        model_kwargs = None
    cache_dic, current = cache_init(
        timesteps,
        model_kwargs=model_kwargs,
        mode=cache_mode,
        interval=interval,
        max_order=max_order,
        first_enhance=first_enhance,
        hicache_scale=hicache_scale,
    )

    if stream_calibrator is not None:
        cache_dic["calibrator"] = stream_calibrator
        cache_dic["stream_feature_only"] = True

    eigencache_config = eigencache_config or {}
    if cache_method == "hicache":
        cache_dic["prediction_mode"] = "hicache"
    elif cache_method == "eigencache":
        kernel_path = eigencache_config.get("kernel_path")
        if not kernel_path:
            raise ValueError("EigenCache requires --eigencache_kernel_path for inference.")
        payload = KernelStore.load(kernel_path)
        meta = payload.get("meta", {})
        expected_steps = len(timesteps) - 1
        kernel_steps = int(meta.get("num_steps", expected_steps))
        if kernel_steps != expected_steps:
            raise ValueError(
                f"Kernel calibrated for {kernel_steps} steps, but sampler uses {expected_steps} steps."
            )
        recorded = meta.get("timesteps")
        if recorded and len(recorded) == expected_steps:
            ref = torch.tensor(recorded, dtype=torch.float32)
            cur = torch.tensor(timesteps[:-1], dtype=torch.float32)
            if (ref - cur).abs().max().item() > 1e-4:
                raise ValueError("EigenCache calibration timesteps do not match current schedule.")
        if not cache_dic.get("taylor_cache", False):
            raise ValueError("EigenCache requires a Taylor cache mode (taylor_cache=True).")
        accelerator = EigenCacheAccelerator(
            payload["kernels"],
            meta,
            window=eigencache_config.get("window", 3),
            jitter=eigencache_config.get("lambda", 1e-3),
            schedule=eigencache_config.get("schedule", "fixed"),
            budget=eigencache_config.get("budget", 0),
            var_tau=eigencache_config.get("var_tau", 0.05),
            layer_weights=eigencache_config.get("layer_weights"),
        )
        cache_dic["prediction_mode"] = "eigencache"
        cache_dic["eigencache_accel"] = accelerator
        cache_dic["eigencache_variance_trace"] = accelerator.variance_trace

    # 🔥 新增：配置特征收集
    if enable_feature_collection:
        cache_dic["enable_feature_collection"] = True
        cache_dic["feature_collection_config"] = feature_collection_config or {
            "target_layer": 14,
            "target_module": "total",
            "target_stream": "single_stream",
        }

    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    current["step"] = 0
    current["num_steps"] = len(timesteps) - 1
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        current["t"] = t_curr
        # print(t_curr)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            cache_dic=cache_dic,
            current=current,
            guidance=guidance_vec,
        )
        # print(img.shape)
        img = img + (t_prev - t_curr) * pred
        current["step"] += 1

    # 🔥 新增：将cache_dic存储到模型中以便后续访问
    model._last_cache_dic = cache_dic

    return img
