from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import torch


def _format_key(stream: str, layer: int, module: str) -> str:
    return f"{stream}:{layer}:{module}"


class KernelStore:
    """
    Utility helpers for persisting EigenCache kernels.
    """

    @staticmethod
    def save(path: str | Path, payload: dict) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)

    @staticmethod
    def load(path: str | Path) -> dict:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Kernel file not found: {target}")
        data = torch.load(target, map_location="cpu")
        if "kernels" not in data or "meta" not in data:
            raise ValueError(f"Malformed EigenCache kernel file: {target}")
        return data


class KernelCalibrator:
    """
    Training-free kernel estimator. Aggregates cosine similarity Gram matrices per layer/module.
    """

    def __init__(
        self,
        num_steps: int,
        *,
        max_elements: int = 8192,
        eps: float = 1e-6,
        compression: str = "token_mean",
        timesteps: Sequence[float] | None = None,
        phase_boundaries: Sequence[int] | None = None,
        phase_names: Sequence[str] | None = None,
    ):
        self.num_steps = int(num_steps)
        self.max_elements = int(max_elements)
        self.eps = float(eps)
        self.compression = compression
        self.timesteps = list(timesteps) if timesteps is not None else None
        self.phase_slices = self._init_phases(phase_boundaries, phase_names)
        self._phase_lookup = {phase["name"]: phase for phase in self.phase_slices}
        self._gram_sums: dict[str, dict] = {}
        self._counts: dict[str, int] = {}
        self._phase_counts: dict[str, dict[str, int]] = {}
        self._metadata: dict[str, dict] = {}
        self._runs = 0
        # Streaming buffers for current trajectory (step-indexed vectors)
        self._current_vectors: dict[str, dict] = {}

    def _init_phases(self, boundaries, names):
        if not boundaries:
            return []
        try:
            boundaries = [int(b) for b in boundaries]
        except Exception:
            raise ValueError("Phase boundaries must be integers.")
        sanitized = []
        for b in boundaries:
            if 0 <= b <= self.num_steps:
                sanitized.append(b)
        sanitized = sorted(set(sanitized))
        if not sanitized or sanitized[0] != 0:
            sanitized = [0] + sanitized
        if sanitized[-1] != self.num_steps:
            sanitized.append(self.num_steps)
        phase_names = list(names) if names else []
        slices = []
        for idx in range(len(sanitized) - 1):
            start, end = sanitized[idx], sanitized[idx + 1]
            if end <= start:
                continue
            if idx < len(phase_names) and phase_names[idx]:
                name = phase_names[idx]
            else:
                name = f"phase_{idx}"
            slices.append({"name": name, "start": start, "end": end})
        return slices

    def ingest(self, features: Mapping, metadata: Mapping) -> None:
        """
        Consume one calibration run worth of cached trajectory features.
        """
        for layer, module_dict in features.items():
            meta_layer = metadata.get(layer, {})
            for module, tensors in module_dict.items():
                metas = meta_layer.get(module, [])
                gram, phase_grams, stream = self._build_gram_matrix(tensors, metas)
                if gram is None:
                    continue
                key = _format_key(stream, layer, module)
                if key not in self._gram_sums:
                    phase_zero = {
                        phase["name"]: torch.zeros(
                            (phase["end"] - phase["start"], phase["end"] - phase["start"]),
                            dtype=torch.float32,
                        )
                        for phase in self.phase_slices
                    }
                    self._gram_sums[key] = {
                        "global": torch.zeros(
                            (self.num_steps, self.num_steps), dtype=torch.float32
                        ),
                        "phases": phase_zero,
                    }
                    self._counts[key] = 0
                    self._phase_counts[key] = {phase["name"]: 0 for phase in self.phase_slices}
                    self._metadata[key] = {"stream": stream, "layer": layer, "module": module}
                self._gram_sums[key]["global"] += gram
                for name, mat in phase_grams.items():
                    if name in self._gram_sums[key]["phases"]:
                        self._gram_sums[key]["phases"][name] += mat
                        self._phase_counts[key][name] += 1
                self._counts[key] += 1

        self._runs += 1

    def _build_gram_matrix(self, tensors: Sequence[torch.Tensor], metas: Sequence[dict]):
        if not tensors or len(tensors) < self.num_steps:
            return None, None

        ordered = sorted(zip(metas, tensors), key=lambda item: item[0].get("step", 0))
        stream = ordered[0][0].get("stream", "double_stream") if ordered else "double_stream"
        step_vectors: dict[int, torch.Tensor] = {}
        for meta, tensor in ordered:
            step = int(meta.get("step", -1))
            if 0 <= step < self.num_steps and step not in step_vectors:
                step_vectors[step] = self._compress_feature(tensor)

        if len(step_vectors) < self.num_steps:
            return None, {}, stream

        vectors = [step_vectors[idx] for idx in range(self.num_steps)]
        matrix = torch.stack(vectors, dim=0)
        norms = matrix.norm(dim=1, keepdim=True).clamp_min(self.eps)
        normalized = matrix / norms
        gram = normalized @ normalized.T
        phase_grams = {}
        for phase in self.phase_slices:
            steps = list(range(phase["start"], phase["end"]))
            if any(step not in step_vectors for step in steps):
                continue
            phase_matrix = torch.stack([step_vectors[step] for step in steps], dim=0)
            p_norm = phase_matrix.norm(dim=1, keepdim=True).clamp_min(self.eps)
            phase_normed = phase_matrix / p_norm
            phase_grams[phase["name"]] = phase_normed @ phase_normed.T
        return gram, phase_grams, stream

    def _compress_feature(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a feature tensor to a manageable 1D vector.
        Strategy: mean over token dimension, flatten, and subsample evenly if needed.
        """
        vec = tensor.detach().to(torch.float32).cpu()
        if vec.ndim >= 3:
            vec = vec.mean(dim=1)
        vec = vec.reshape(-1)
        if vec.numel() > self.max_elements:
            step = max(1, vec.numel() // self.max_elements)
            vec = vec[::step][: self.max_elements]
        return vec

    # ---------- Streaming ingestion (no full trajectory kept in memory) ----------
    def start_run(self) -> None:
        """Reset per-run buffers before collecting streaming features."""
        self._current_vectors = {}

    def ingest_feature(
        self,
        stream: str,
        layer: int,
        module: str,
        step: int,
        tensor: torch.Tensor,
        *,
        timestep: float | None = None,
    ) -> None:
        """
        Collect a single feature for a given step without storing full trajectories.
        Only the compressed vector is kept until the run is finalized.
        """
        if step < 0 or step >= self.num_steps:
            return
        key = _format_key(stream, layer, module)
        entry = self._current_vectors.setdefault(
            key,
            {
                "vectors": [None for _ in range(self.num_steps)],
            },
        )
        # Skip duplicates for the same step
        if entry["vectors"][step] is None:
            entry["vectors"][step] = self._compress_feature(tensor)
            if key not in self._metadata:
                self._metadata[key] = {"stream": stream, "layer": layer, "module": module}

    def _accumulate_vectors(self, key: str, vectors: list[torch.Tensor]) -> None:
        if any(v is None for v in vectors):
            return
        matrix = torch.stack(vectors, dim=0)
        norms = matrix.norm(dim=1, keepdim=True).clamp_min(self.eps)
        normalized = matrix / norms
        gram = normalized @ normalized.T

        phase_grams: dict[str, torch.Tensor] = {}
        for phase in self.phase_slices:
            steps = list(range(phase["start"], phase["end"]))
            if any(vectors[s] is None for s in steps):
                continue
            phase_matrix = matrix[phase["start"] : phase["end"]]
            p_norm = phase_matrix.norm(dim=1, keepdim=True).clamp_min(self.eps)
            phase_normed = phase_matrix / p_norm
            phase_grams[phase["name"]] = phase_normed @ phase_normed.T

        if key not in self._gram_sums:
            phase_zero = {
                phase["name"]: torch.zeros(
                    (phase["end"] - phase["start"], phase["end"] - phase["start"]),
                    dtype=torch.float32,
                )
                for phase in self.phase_slices
            }
            self._gram_sums[key] = {
                "global": torch.zeros((self.num_steps, self.num_steps), dtype=torch.float32),
                "phases": phase_zero,
            }
            self._counts[key] = 0
            self._phase_counts[key] = {phase["name"]: 0 for phase in self.phase_slices}
            self._metadata.setdefault(key, {})

        self._gram_sums[key]["global"] += gram
        self._counts[key] += 1

        for name, mat in phase_grams.items():
            if name in self._gram_sums[key]["phases"]:
                self._gram_sums[key]["phases"][name] += mat
                self._phase_counts[key][name] += 1

    def finalize_run(self) -> None:
        """
        Convert buffered per-step vectors into Gram matrices and accumulate.
        Called once after a calibration trajectory finishes.
        """
        for key, payload in self._current_vectors.items():
            self._accumulate_vectors(key, payload["vectors"])
        if self._current_vectors:
            self._runs += 1
        self._current_vectors = {}

    def export(self) -> dict:
        kernels: dict[str, dict] = {}
        for key, gram in self._gram_sums.items():
            count = max(self._counts.get(key, 1), 1)
            kernels[key] = {
                "C": gram["global"] / count,
                "count": count,
                "meta": self._metadata.get(key, {}),
            }
            phase_payload = {}
            for name, tensor in gram["phases"].items():
                phase_count = self._phase_counts.get(key, {}).get(name, 0)
                if phase_count <= 0:
                    continue
                phase_info = self._phase_lookup.get(name)
                if not phase_info:
                    continue
                phase_payload[name] = {
                    "C": tensor / phase_count,
                    "count": phase_count,
                    "start": phase_info["start"],
                    "end": phase_info["end"],
                }
            if phase_payload:
                kernels[key]["phases"] = phase_payload

        payload = {
            "kernels": kernels,
            "meta": {
                "num_steps": self.num_steps,
                "compression": self.compression,
                "max_elements": self.max_elements,
                "runs": self._runs,
                "timesteps": self.timesteps,
                "keys": sorted(kernels.keys()),
                "phases": self.phase_slices,
            },
        }
        return payload

    def save(self, path: str | Path) -> None:
        KernelStore.save(path, self.export())


class KrigingPredictor:
    """
    Small helper handling the Kriging weight solve per layer.
    """

    def __init__(self, kernels: Mapping[str, Mapping], jitter: float):
        self._kernels: dict[str, dict] = {}
        for key, value in kernels.items():
            tensor = value.get("C") if isinstance(value, Mapping) else value
            entry: dict = {
                "global": torch.as_tensor(tensor, dtype=torch.float64) if tensor is not None else None,
                "phases": {},
                "kl": None,
                "weight_table": {"global": {}, "phases": {}},
            }
            if isinstance(value, Mapping):
                if "kl" in value:
                    entry["kl"] = {
                        "values": torch.as_tensor(value["kl"]["values"], dtype=torch.float64),
                        "vectors": torch.as_tensor(value["kl"]["vectors"], dtype=torch.float64),
                    }
                for name, info in (value.get("phases") or {}).items():
                    entry["phases"][name] = {
                        "C": torch.as_tensor(info["C"], dtype=torch.float64),
                        "start": int(info.get("start", 0)),
                        "end": int(info.get("end", info["C"].shape[0])),
                        "kl": None,
                    }
                    if "kl" in info:
                        entry["phases"][name]["kl"] = {
                            "values": torch.as_tensor(info["kl"]["values"], dtype=torch.float64),
                            "vectors": torch.as_tensor(info["kl"]["vectors"], dtype=torch.float64),
                        }
                if "weight_table" in value:
                    table = value["weight_table"]
                    if "global" in table:
                        entry["weight_table"]["global"] = {
                            k: {
                                "weights": torch.tensor(v["weights"], dtype=torch.float32),
                                "variance": float(v["variance"]),
                            }
                            for k, v in table["global"].items()
                        }
                    if "phases" in table:
                        entry["weight_table"]["phases"] = {
                            pname: {
                                k: {
                                    "weights": torch.tensor(v["weights"], dtype=torch.float32),
                                    "variance": float(v["variance"]),
                                }
                                for k, v in phase_table.items()
                            }
                            for pname, phase_table in table["phases"].items()
                        }
            self._kernels[key] = entry
        self.jitter = float(jitter)

    def _lookup_precomputed(self, entry: dict, step: int, anchors: Sequence[int], phase_name: str | None):
        table = entry.get("weight_table")
        if not table:
            return None
        key = (int(step), tuple(int(a) for a in anchors))
        if phase_name:
            info = table.get("phases", {}).get(phase_name, {}).get(key)
        else:
            info = table.get("global", {}).get(key)
        if not info:
            return None
        weights = torch.tensor(info["weights"], dtype=torch.float32)
        detail = {
            "step": int(step),
            "anchors": list(int(a) for a in anchors),
            "phase": phase_name,
            "variance": float(info["variance"]),
            "source": "precomputed",
        }
        return weights, float(info["variance"]), detail

    def solve(self, key: str, step: int, anchors: Sequence[int]):
        if not anchors:
            return None
        entry = self._kernels.get(key)
        if entry is None:
            return None

        phase_name = None
        kernel = entry["global"]
        kl_info = entry.get("kl")
        offset = 0
        filtered_anchors = list(anchors)

        for name, info in entry["phases"].items():
            start, end = info["start"], info["end"]
            if start <= step < end:
                if all(start <= a < end for a in filtered_anchors):
                    kernel = info.get("C")
                    kl_info = info.get("kl")
                    offset = start
                    phase_name = name
                break

        precomputed = self._lookup_precomputed(entry, step, filtered_anchors, phase_name)
        if precomputed is not None:
            return precomputed

        if kernel is None and not kl_info:
            return None

        local_step = step - offset
        if local_step < 0 or (kernel is not None and local_step >= kernel.shape[0]):
            return None

        local_idx = [a - offset for a in filtered_anchors]
        if not local_idx or any(i < 0 for i in local_idx):
            return None

        idx_tensor = torch.tensor(local_idx, dtype=torch.long)

        if kernel is not None:
            sub = kernel.index_select(0, idx_tensor).index_select(1, idx_tensor)
            diag = torch.eye(sub.shape[0], dtype=sub.dtype, device=sub.device)
            sub = sub + self.jitter * diag
            k = kernel.index_select(
                0, torch.tensor([local_step], dtype=torch.long, device=kernel.device)
            ).squeeze(0)
            k = k.index_select(0, idx_tensor)
            diag_value = kernel[local_step, local_step]
        elif kl_info:
            vectors = kl_info["vectors"]
            values = kl_info["values"].clamp_min(0)
            sqrt_vals = torch.sqrt(values)
            basis = vectors * sqrt_vals
            rows = basis.index_select(0, idx_tensor)
            sub = rows @ rows.T
            diag = torch.eye(sub.shape[0], dtype=sub.dtype, device=sub.device)
            sub = sub + self.jitter * diag
            basis_step = basis[local_step]
            k = basis_step @ rows.T
            diag_value = basis_step.dot(basis_step)
        else:
            return None

        try:
            chol = torch.linalg.cholesky(sub)
            weights = torch.cholesky_solve(k.unsqueeze(-1), chol).squeeze(-1)
        except RuntimeError:
            weights = torch.linalg.solve(sub, k)

        variance = diag_value - torch.dot(k, weights)
        weights = weights.to(torch.float32)
        variance_value = float(max(variance.item(), 0.0))
        detail = {
            "step": int(step),
            "anchors": list(int(a) for a in filtered_anchors),
            "phase": phase_name,
            "variance": variance_value,
            "source": "solve",
        }
        return weights, variance_value, detail


class BaseSchedule:
    def should_force_full(self, step: int, accelerator: "EigenCacheAccelerator", current=None) -> bool:
        return False


class FixedSchedule(BaseSchedule):
    pass


class GreedySchedule(BaseSchedule):
    def __init__(self, budget: int, selected: set[int]):
        self.budget = max(0, int(budget))
        self.selected = selected

    def should_force_full(self, step: int, accelerator: "EigenCacheAccelerator", current=None) -> bool:
        return step in self.selected


class VarianceSchedule(BaseSchedule):
    def __init__(self, tau: float):
        self.tau = float(tau)

    def should_force_full(self, step: int, accelerator: "EigenCacheAccelerator", current=None) -> bool:
        variance = accelerator.estimate_variance(step, current.get("t") if current else None)
        return variance is not None and variance > self.tau


@dataclass
class AnchorFeature:
    step: int
    tensor: torch.Tensor


class EigenCacheAccelerator:
    """
    Runtime integration for EigenCache prediction and adaptive scheduling.
    """

    def __init__(
        self,
        kernels: Mapping[str, Mapping],
        meta: Mapping,
        *,
        window: int = 3,
        jitter: float = 1e-3,
        schedule: str = "fixed",
        budget: int = 0,
        var_tau: float = 0.05,
        layer_weights: Mapping[str, float] | str | None = None,
    ):
        self.meta = dict(meta)
        self.num_steps = int(meta.get("num_steps", 0))
        self.window = max(1, int(window))
        self.predictor = KrigingPredictor(kernels, jitter)
        self.anchor_records: deque[tuple[int, float]] = deque()
        self.feature_bank: dict[str, OrderedDict[int, torch.Tensor]] = {}
        self.layer_weights = self._resolve_layer_weights(layer_weights, kernels.keys())
        self.schedule = self._build_schedule(schedule, budget, var_tau)
        self.variance_trace: list[dict] = []
        self._last_step_forced = -1

    def _build_schedule(self, schedule: str, budget: int, var_tau: float) -> BaseSchedule:
        if schedule == "greedy":
            selected = self._precompute_greedy_set(budget, self.layer_weights)
            return GreedySchedule(budget, selected)
        if schedule == "variance":
            return VarianceSchedule(var_tau)
        return FixedSchedule()

    def _precompute_greedy_set(self, budget: int, weights: Mapping[str, float]) -> set[int]:
        if budget <= 0 or self.num_steps <= 0:
            return set()

        available = set(range(self.num_steps))
        selected: set[int] = {0}

        def score(indices: list[int]) -> float:
            if not indices:
                return 0.0
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            total = 0.0
            for key, entry in self.predictor._kernels.items():
                weight = float(weights.get(key, 0.0))
                if weight == 0.0:
                    continue
                tensor = entry["global"]
                if tensor is None:
                    continue
                sub = tensor.index_select(0, idx_tensor).index_select(1, idx_tensor)
                eye = torch.eye(sub.shape[0], dtype=sub.dtype)
                gram = eye + (1.0 / max(self.predictor.jitter, 1e-6)) * sub
                sign, val = torch.linalg.slogdet(gram)
                if float(sign) > 0:
                    total += weight * 0.5 * float(val)
            return total

        while len(selected) < min(budget, self.num_steps) and available:
            best_gain = -math.inf
            best = None
            current_score = score(sorted(selected))
            for candidate in available:
                if candidate in selected:
                    continue
                gain = score(sorted(selected | {candidate})) - current_score
                if gain > best_gain:
                    best_gain = gain
                    best = candidate
            if best is None:
                break
            selected.add(best)
            available.remove(best)
        return selected

    def _resolve_layer_weights(
        self, config: Mapping[str, float] | str | None, keys: Iterable[str]
    ) -> dict[str, float]:
        if isinstance(config, Mapping):
            return {k: float(v) for k, v in config.items()}
        if isinstance(config, str) and config.strip():
            weights: dict[str, float] = {}
            for token in config.split(","):
                if ":" not in token:
                    continue
                name, value = token.split(":", 1)
                try:
                    weights[name.strip()] = float(value)
                except ValueError:
                    continue
            if weights:
                return weights
        # Default: uniform weights over all available kernels
        keys = list(keys)
        if not keys:
            return {}
        uniform = 1.0 / len(keys)
        return {key: uniform for key in keys}

    def layer_key(self, current: Mapping) -> str:
        return _format_key(current["stream"], current["layer"], current["module"])

    def observe(self, current: Mapping, feature: torch.Tensor) -> None:
        if current.get("type") != "full":
            return
        step = int(current["step"])
        t_value = float(current.get("t", step))
        if not self.anchor_records or self.anchor_records[-1][0] != step:
            self.anchor_records.append((step, t_value))
            if len(self.anchor_records) > self.window:
                expired_step, _ = self.anchor_records.popleft()
                self._drop_step(expired_step)
        key = self.layer_key(current)
        bank = self.feature_bank.setdefault(key, OrderedDict())
        bank[step] = feature.detach().clone()
        while len(bank) > len(self.anchor_records):
            bank.popitem(last=False)

    def _drop_step(self, step: int) -> None:
        for bank in self.feature_bank.values():
            bank.pop(step, None)

    def _select_anchor_steps(self, target_step: int, target_time: float | None) -> list[int]:
        if not self.anchor_records:
            return []

        target_time = float(target_time) if target_time is not None else None
        candidates = list(self.anchor_records)

        # Anchors are collected while timesteps decrease, so larger t-values correspond
        # to "future" anchors in the original derivation. Prefer those with t >= current t.
        preferred: list[int] = []
        if target_time is not None:
            for step, t_value in reversed(candidates):
                if t_value >= target_time and step not in preferred:
                    preferred.append(step)
                    if len(preferred) == self.window:
                        break

        if len(preferred) < self.window:
            for step, _ in reversed(candidates):
                if step not in preferred:
                    preferred.append(step)
                    if len(preferred) == self.window:
                        break

        preferred.sort()
        return preferred

    def predict(self, current: Mapping, fallback: torch.Tensor) -> torch.Tensor:
        key = self.layer_key(current)
        step = int(current["step"])
        target_time = current.get("t")
        anchors = self._select_anchor_steps(step, target_time)
        bank = self.feature_bank.get(key)
        if not bank:
            return fallback
        anchor_feats: list[torch.Tensor] = []
        filtered_steps: list[int] = []
        for s in anchors:
            if s in bank:
                filtered_steps.append(s)
                anchor_feats.append(bank[s])
        if not anchor_feats:
            return fallback
        if step in filtered_steps:
            idx = filtered_steps.index(step)
            return anchor_feats[idx].to(device=fallback.device, dtype=fallback.dtype)

        solve_out = self.predictor.solve(key, step, filtered_steps)
        if solve_out is None:
            return fallback
        weights, _, detail = solve_out
        weights = weights.to(device=fallback.device, dtype=fallback.dtype)
        pred = torch.zeros_like(fallback)
        for weight, feat in zip(weights, anchor_feats):
            pred = pred + weight * feat.to(device=fallback.device, dtype=fallback.dtype)
        if detail:
            record = dict(detail)
            record["layer_key"] = key
            record["type"] = "prediction"
            self.variance_trace.append(record)
        return pred

    def should_force_full(self, current: Mapping) -> bool:
        if current.get("type") == "full":
            return False
        step = int(current["step"])
        return self.schedule.should_force_full(step, self, current)

    def estimate_variance(self, step: int, target_time: float | None = None) -> float | None:
        anchors = self._select_anchor_steps(step, target_time)
        if not anchors:
            return None
        total = 0.0
        weight_sum = 0.0
        for key, weight in self.layer_weights.items():
            solve_out = self.predictor.solve(key, step, anchors)
            if solve_out is None:
                continue
            _, variance, detail = solve_out
            total += float(weight) * variance
            weight_sum += abs(float(weight))
            if detail:
                entry = dict(detail)
                entry["layer_key"] = key
                entry["weight"] = float(weight)
                entry["type"] = "per_layer"
                self.variance_trace.append(entry)
        if weight_sum == 0.0:
            return None
        aggregate = total / weight_sum
        self.variance_trace.append(
            {
                "type": "aggregate",
                "step": step,
                "variance": aggregate,
                "anchors": [int(a) for a in anchors],
            }
        )
        return aggregate


def _solve_weights_from_kernel(kernel: torch.Tensor, step: int, anchors: Sequence[int], jitter: float):
    if kernel is None or not anchors:
        return None, None
    idx = torch.tensor(anchors, dtype=torch.long)
    sub = kernel.index_select(0, idx).index_select(1, idx)
    diag = torch.eye(sub.shape[0], dtype=sub.dtype, device=sub.device)
    sub = sub + jitter * diag
    k = kernel.index_select(0, torch.tensor([step], dtype=torch.long, device=kernel.device)).squeeze(0)
    k = k.index_select(0, idx)
    try:
        chol = torch.linalg.cholesky(sub)
        weights = torch.cholesky_solve(k.unsqueeze(-1), chol).squeeze(-1)
    except RuntimeError:
        weights = torch.linalg.solve(sub, k)
    variance = kernel[step, step] - torch.dot(k, weights)
    return weights, float(max(variance.item(), 0.0))


def _build_weight_table(kernel: torch.Tensor, window: int, jitter: float, offset: int = 0) -> dict:
    T = kernel.shape[0]
    table: dict = {}
    for step in range(T):
        for size in range(1, window + 1):
            start = step + 1
            end = start + size
            if end > T:
                continue
            anchors = list(range(start, end))
            weights, variance = _solve_weights_from_kernel(kernel, step, anchors, jitter)
            if weights is None:
                continue
            global_step = offset + step
            global_anchors = tuple(offset + idx for idx in anchors)
            table[(global_step, global_anchors)] = {
                "weights": weights.cpu().tolist(),
                "variance": variance,
            }
    return table


def precompute_weight_tables(payload: dict, window: int, jitter: float) -> dict:
    if not payload or window <= 0:
        return payload
    kernels = payload.get("kernels", {})
    for key, data in kernels.items():
        tensor = torch.as_tensor(data["C"], dtype=torch.float64)
        global_table = _build_weight_table(tensor, window, jitter, offset=0)
        phase_tables = {}
        for name, info in (data.get("phases") or {}).items():
            phase_tensor = torch.as_tensor(info["C"], dtype=torch.float64)
            start = int(info.get("start", 0))
            phase_tables[name] = _build_weight_table(phase_tensor, window, jitter, offset=start)
        tables = {}
        if global_table:
            tables["global"] = global_table
        if phase_tables:
            tables["phases"] = {name: tbl for name, tbl in phase_tables.items() if tbl}
        if tables:
            data["weight_table"] = tables
    payload.setdefault("meta", {})["precompute_window"] = window
    return payload


def _compute_kl(matrix: torch.Tensor, rank: int) -> dict | None:
    r = min(rank, matrix.shape[0])
    if r <= 0:
        return None
    vals, vecs = torch.linalg.eigh(matrix)
    idx = torch.argsort(vals, descending=True)[:r]
    selected_vals = vals[idx].clamp_min(0)
    selected_vecs = vecs[:, idx]
    return {
        "values": selected_vals.to(torch.float32),
        "vectors": selected_vecs.to(torch.float32),
    }


def apply_kl_truncation(payload: dict, rank: int) -> dict:
    if not payload or rank <= 0:
        return payload
    kernels = payload.get("kernels", {})
    for key, data in kernels.items():
        tensor = torch.as_tensor(data["C"], dtype=torch.float64)
        kl_global = _compute_kl(tensor, rank)
        if kl_global:
            data["kl"] = kl_global
        for name, info in (data.get("phases") or {}).items():
            phase_tensor = torch.as_tensor(info["C"], dtype=torch.float64)
            kl_phase = _compute_kl(phase_tensor, rank)
            if kl_phase:
                info["kl"] = kl_phase
    payload.setdefault("meta", {})["kl_rank"] = rank
    return payload

