"""进程内指标注册表（轻量，无外部依赖）——可观测性最小集。

企业排障要回答的"钱花在哪、慢在哪、烂在哪"：
  - OCR 兜底率：auto 模式里百度被触发多少次（每次=一次付费调用）
  - embedding：token 量 / 耗时（本地或云端都是成本信号）
  - 问答：LLM 重试率（RAG evaluate 不过重试=double 成本）、端到端延迟
  - 端点：请求计数 / 错误数 / 延迟分位数

用法：
  from services.kb import metrics
  metrics.incr("ocr.baidu_fallback")
  with metrics.timed("kb.ask.latency_ms"):
      ...
  metrics.snapshot()  # 返回扁平 dict（/admin/metrics 暴露）

多 worker 部署时各进程独立计数（聚合交给外部监控，这里保证单进程内准确）。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class _Metrics:
    def __init__(self):
        self._counts: dict[str, int] = defaultdict(int)
        self._timings: dict[str, list[float]] = defaultdict(list)  # 保留最近 N 条算分位
        self._lock = threading.Lock()

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counts[name] += by

    def add(self, name: str, value: float) -> None:
        """记录一次观测值（延迟/耗时/token 量），保留最近 200 条算分布。"""
        with self._lock:
            self._timings[name].append(value)
            if len(self._timings[name]) > 200:
                self._timings[name] = self._timings[name][-200:]

    def timed(self, name: str):
        """上下文管理器：自动把耗时（秒）add 进 name。"""
        start = time.perf_counter()

        class _Ctx:
            def __enter__(self_inner):
                return self

            def __exit__(self_inner, *exc):
                self.add(name, (time.perf_counter() - start) * 1000.0)  # ms
                return False

        return _Ctx()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {}
            for k, v in sorted(self._counts.items()):
                out[k] = v
            for k, vals in sorted(self._timings.items()):
                if not vals:
                    continue
                s = sorted(vals)
                n = len(s)
                def pct(p: float) -> float:
                    return s[min(n - 1, int(p * n))]
                out[f"{k}.count"] = n
                out[f"{k}.avg_ms"] = round(sum(s) / n, 2)
                out[f"{k}.p50_ms"] = round(pct(0.50), 2)
                out[f"{k}.p95_ms"] = round(pct(0.95), 2)
                out[f"{k}.max_ms"] = round(s[-1], 2)
            return out

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._timings.clear()

    def to_prometheus(self) -> str:
        """导出 Prometheus 文本格式（供 /metrics 端点抓取）。

        counts → counter（单调递增）；timings 派生值（count/avg/p50/p95/max）→ gauge。
        """
        snap = self.snapshot()
        lines: list[str] = []
        timing_suffixes = (".count", ".avg_ms", ".p50_ms", ".p95_ms", ".max_ms")
        for k, v in sorted(snap.items()):
            metric_name = k.replace(".", "_")
            if any(k.endswith(s) for s in timing_suffixes):
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {v}")
            else:
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name} {v}")
        return "\n".join(lines) + ("\n" if lines else "")


# 进程内单例
global_metrics = _Metrics()
