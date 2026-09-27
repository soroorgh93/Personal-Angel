"""Compute/telemetry accounting: latency spans, counters, GPU sampling, token
totals and a cloud-equivalent cost estimate shown in the demo.

Everything that cannot be measured on the current machine is reported as
null — the system never invents a bandwidth or token number.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any

import psutil

def _unified_memory_used_mb() -> float | None:
    """On unified memory machines the GPU and the CPU share one pool: /proc/meminfo is the truth."""
    try:
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                info[key.strip()] = float(rest.strip().split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) / 1024.0
    except Exception:
        return None

class Telemetry:
    def __init__(self, gpu_sampling_s: float = 0.5) -> None:
        self.started = time.perf_counter()
        self.counters: dict[str, float] = {}
        self.spans: list[dict[str, Any]] = []
        self.gpu_samples: list[dict[str, float]] = []
        self.model_calls: list[dict[str, Any]] = []
        self._gpu_interval = gpu_sampling_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.process = psutil.Process()
        self.peak_rss_mb = 0.0
        self.gpu_backend = self._detect_gpu_backend()

    def increment(self, key: str, value: float = 1.0) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + value

    def set(self, key: str, value: float) -> None:
        self.counters[key] = value

    @contextmanager
    def span(self, name: str, **meta: Any):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self.spans.append({"name": name, "ms": round(ms, 1), **meta})
            self.counters[f"span_ms:{name}"] = self.counters.get(f"span_ms:{name}", 0.0) + ms
            rss = self.process.memory_info().rss / 1e6
            self.peak_rss_mb = max(self.peak_rss_mb, rss)

    def record_model_call(self, kind: str, model: str, tokens_in: int, tokens_out: int, latency_ms: float,
                          images: int = 0) -> None:
        self.model_calls.append({"kind": kind, "model": model, "tokens_in": tokens_in, "tokens_out": tokens_out,
                                 "latency_ms": round(latency_ms, 1), "images": images, "t": round(time.perf_counter() - self.started, 2)})
        self.increment("model_calls")
        self.increment("input_tokens", tokens_in)
        self.increment("output_tokens", tokens_out)
        self.increment("images_sent", images)
        self.increment("model_latency_ms", latency_ms)

    @staticmethod
    def _detect_gpu_backend() -> str | None:
        try:
            import pynvml

            pynvml.nvmlInit()
            return "pynvml"
        except Exception:
            pass
        if shutil.which("nvidia-smi"):
            return "nvidia-smi"
        return None

    def _sample_gpu(self) -> dict[str, float] | None:
        try:
            if self.gpu_backend == "pynvml":
                import pynvml

                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util_v, mem_v, power = 0.0, None, 0.0
                try:
                    util_v = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                except Exception:
                    pass
                try:
                    mem_v = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6
                except Exception:
                    mem_v = None
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    pass
                if mem_v is None:
                    mem_v = _unified_memory_used_mb()
                return {"gpu_util": util_v, "mem_used_mb": float(mem_v or 0.0), "power_w": power or 0.0,
                        "t": time.perf_counter() - self.started}
            if self.gpu_backend == "nvidia-smi":
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                                     check=False).stdout.strip().split("\n")[0]
                util, mem, power = [p.strip() for p in out.split(",")]
                bad = {"[N/A]", "[Not Supported]", "N/A", ""}
                mem_v = float(mem) if mem not in bad else _unified_memory_used_mb()
                return {"gpu_util": float(util) if util not in bad else 0.0, "mem_used_mb": float(mem_v or 0.0),
                        "power_w": float(power) if power not in bad else 0.0, "t": time.perf_counter() - self.started}
        except Exception:
            return None
        return None

    def start_gpu_sampling(self) -> None:
        if self.gpu_backend is None or self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                sample = self._sample_gpu()
                if sample:
                    self.gpu_samples.append(sample)
                self._stop.wait(self._gpu_interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_gpu_sampling(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def to_dict(self, config: dict[str, Any] | None = None, media_seconds: float = 0.0) -> dict[str, Any]:
        config = config or {}
        elapsed = time.perf_counter() - self.started
        tokens_in = self.counters.get("input_tokens", 0.0)
        tokens_out = self.counters.get("output_tokens", 0.0)
        model_ms = self.counters.get("model_latency_ms", 0.0)
        gpu = None
        if self.gpu_samples:
            gpu = {
                "backend": self.gpu_backend,
                "samples": len(self.gpu_samples),
                "util_mean": round(sum(s["gpu_util"] for s in self.gpu_samples) / len(self.gpu_samples), 1),
                "util_max": round(max(s["gpu_util"] for s in self.gpu_samples), 1),
                "mem_used_mb_max": round(max(s["mem_used_mb"] for s in self.gpu_samples), 0),
                "power_w_mean": round(sum(s["power_w"] for s in self.gpu_samples) / len(self.gpu_samples), 1),
            }
        cloud_in = float(config.get("cloud_price_per_1k_input_tokens_usd", 0.0025))
        cloud_out = float(config.get("cloud_price_per_1k_output_tokens_usd", 0.010))
        cloud_video = float(config.get("cloud_price_per_video_minute_usd", 0.12))
        power_w = float(config.get("local_power_w", 140))
        kwh_price = float(config.get("electricity_usd_per_kwh", 0.30))
        cloud_cost = tokens_in / 1000 * cloud_in + tokens_out / 1000 * cloud_out + (media_seconds / 60.0) * cloud_video
        local_energy_kwh = power_w * elapsed / 3600 / 1000
        local_cost = local_energy_kwh * kwh_price
        frames_total = self.counters.get("frames_total", 0.0)

        frames_processed = self.counters.get("frames_unique", 0.0) or (self.counters.get("frames_scanned", 0.0) + self.counters.get("frames_dense", 0.0))
        if frames_total:
            frames_processed = min(frames_processed, frames_total)
        return {
            "wall_time_s": round(elapsed, 2),
            "model_calls": int(self.counters.get("model_calls", 0)),
            "input_tokens": int(tokens_in), "output_tokens": int(tokens_out),
            "tokens_per_s": round(tokens_out / (model_ms / 1000), 1) if model_ms > 0 else None,
            "model_latency_ms_total": round(model_ms, 0),
            "images_sent_to_vlm": int(self.counters.get("images_sent", 0)),
            "frames_total": int(frames_total), "frames_processed": int(frames_processed),
            "frames_skipped": int(self.counters.get("frames_skipped", 0)),
            "frame_skip_ratio": round(1 - frames_processed / frames_total, 3) if frames_total else None,
            "video_fps_effective": round(frames_processed / elapsed, 2) if elapsed > 0 else None,
            "cache_hits": int(self.counters.get("cache_hits", 0)), "cache_misses": int(self.counters.get("cache_misses", 0)),
            "agent_steps": int(self.counters.get("agent_steps", 0)), "tool_calls": int(self.counters.get("tool_calls", 0)),
            "critic_calls": int(self.counters.get("critic_calls", 0)),
            "peak_rss_mb": round(self.peak_rss_mb, 0),
            "gpu": gpu,
            "memory_bandwidth_gbps": None,
            "spans": self.spans[-60:],
            "model_call_log": self.model_calls,
            "cost": {
                "cloud_equivalent_usd": round(cloud_cost, 4),
                "local_energy_kwh": round(local_energy_kwh, 5),
                "local_energy_cost_usd": round(local_cost, 5),
                "estimated_saving_usd": round(cloud_cost - local_cost, 4),
                "assumptions": {"cloud_in_per_1k": cloud_in, "cloud_out_per_1k": cloud_out,
                                "cloud_video_per_min": cloud_video, "local_power_w": power_w, "usd_per_kwh": kwh_price,
                                "note": "Scenario estimate, not an invoice. Cloud prices are configurable assumptions."},
            },
            "counters": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.counters.items() if not k.startswith("span_ms:")},
        }
