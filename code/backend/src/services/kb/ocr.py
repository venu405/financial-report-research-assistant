"""图片 OCR 服务（RapidOCR 统一版，onnxruntime 引擎）。

把图片 / 扫描页识别成文字。懒加载单例：onnxruntime session 初始化开销大
（det/cls/rec 三个模型），只在真正处理图片时才创建，避免污染纯文本入库路径。

模型处理：首次 `RapidOCR()` 时若本地无模型，自动从 ModelScope 下载
（国内可达；本项目一贯偏好 ModelScope 源）。下载慢/失败可手动下载模型后，
用构造参数 `params` 里的模型路径指向本地文件。

识别模式（ocr_image 的 mode 参数）：
- "local"（默认）：纯本地 RapidOCR，失败抛 ValueError（原有语义，零联网零费用）
- "baidu"：百度智能云手写识别优先，百度空/失败回退本地（难图显式指定）
- "auto"：本地优先，识别为空或平均置信度 < KB_OCR_FALLBACK_CONFIDENCE（默认 0.6）
  时兜底到百度。置信度达标后还会做一次内容启发式（_text_looks_bad）：
  文本呈现强乱码特征（连续混用 ≥2 种中文标点如"？，。"堆叠）时也视为"效果差"
  兜底到百度。注意：手写潦草字本地置信度往往也很高（0.8+），且启发式只认强
  信号、宁缺毋滥，auto 仍可能覆盖不到极端潦草——对已知难图请用 mode="baidu"。
"""
from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

_engine: Any = None
_engine_lock = threading.Lock()

# ocr_image 接受的输入：文件路径 / 图片字节 / numpy 数组（ndarray 未列，避免硬性依赖）
ImageInput = Union[str, Path, bytes]

@dataclass(frozen=True)
class OCRLine:
    """一条 OCR 结果及其文字框坐标（像素）。"""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    score: float | None = None


def get_ocr_engine():
    """懒加载单例 OCR 引擎（RapidOCR，onnxruntime 后端）。线程安全：加锁防并发首唤双初始化。"""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:  # 双检锁：等待锁期间别的线程可能已初始化
            return _engine
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise ValueError(
                "图片 OCR 不可用：缺少 rapidocr 包。"
                "请 `pip install rapidocr onnxruntime`（见 requirements.txt）"
            ) from exc

        _engine = RapidOCR()
    return _engine


def ocr_image(image_input: ImageInput, *, mode: str = "local") -> str:
    """识别图片中的文字，按行返回（\\n 分隔）；无文字返回空串。

    接受：文件路径（str/Path）、图片字节（bytes）、numpy 数组（RGB/BGR）。

    mode 见模块 docstring：local（默认）/ baidu / auto。
    """
    if mode == "baidu":
        return _mode_baidu(image_input)
    if mode == "auto":
        return _mode_auto(image_input)
    return _mode_local(image_input)



def ocr_image_with_layout(image_input: ImageInput) -> list[OCRLine]:
    """本地 OCR 的版面结果；供扫描 PDF 去除页眉页脚、重建阅读顺序使用。"""
    try:
        result = get_ocr_engine()(image_input)
    except Exception as exc:
        raise ValueError(f"图片 OCR 识别失败: {exc}") from exc
    texts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None)
    texts = [] if texts is None else texts
    boxes = [] if boxes is None else boxes
    scores = [] if scores is None else scores
    lines: list[OCRLine] = []
    for index, text in enumerate(texts):
        if not str(text).strip() or index >= len(boxes):
            continue
        box = boxes[index]
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        score = float(scores[index]) if index < len(scores) else None
        lines.append(OCRLine(str(text).strip(), min(xs), min(ys), max(xs), max(ys), score))
    return sorted(lines, key=lambda line: (line.y0, line.x0))
def _run_local(image_input: ImageInput) -> tuple[str, Any]:
    """本地 RapidOCR 识别，返回 (text, scores)。scores 可能为 None。"""
    engine = get_ocr_engine()
    result = engine(image_input)
    txts = getattr(result, "txts", None)
    text = "\n".join(txts) if txts else ""
    return text, getattr(result, "scores", None)


def _mode_local(image_input: ImageInput) -> str:
    """纯本地：识别失败抛 ValueError（原有语义），空图返回空串。"""
    try:
        text, _ = _run_local(image_input)
    except Exception as exc:
        raise ValueError(f"图片 OCR 识别失败: {exc}") from exc
    return text


def _mode_baidu(image_input: ImageInput) -> str:
    """百度优先：百度有结果就用；未配置 key / 失败 / 空结果回退本地。"""
    try:
        baidu_text = _fallback_baidu(image_input)
        if baidu_text:
            return baidu_text
    except Exception as exc:  # 后备通道自身异常不得外泄
        logger.warning("百度 OCR 后备异常，回退本地: %s", exc)
    return _mode_local(image_input)


def _mode_auto(image_input: ImageInput) -> str:
    """本地优先 + 空/低置信兜底到百度（对印刷体零联网；手写潦草置信度高时覆盖不到）。"""
    local_text = ""
    local_error: Exception | None = None
    fallback_triggered = False
    try:
        local_text, scores = _run_local(image_input)
        if _result_acceptable(local_text, scores) and not _text_looks_bad(local_text):
            return local_text
        fallback_triggered = True
        logger.info("本地 OCR 结果不佳（%d 字，置信度或内容不达标），启用百度后备", len(local_text))
    except Exception as exc:
        local_error = exc
        fallback_triggered = True
        logger.warning("本地 OCR 识别失败，尝试百度后备: %s", exc)

    # 兜底率指标：本地不达标/失败时触发百度（配合 baidu_ocr 的 success/fallback 算付费率）
    if fallback_triggered:
        _incr("ocr.auto_fallback_triggered")

    try:
        baidu_text = _fallback_baidu(image_input)
    except Exception as exc:
        logger.warning("百度 OCR 后备异常，已降级: %s", exc)
        baidu_text = ""
    if baidu_text:
        return baidu_text

    # 百度不可用/没帮上忙：回退到本地结果
    if local_text:
        return local_text
    if local_error:
        raise ValueError(f"图片 OCR 识别失败: {local_error}") from local_error
    return ""


def _result_acceptable(text: str, scores: Any) -> bool:
    """本地结果是否直接可用：有文字且平均置信度达标。"""
    if not text.strip():
        return False
    if not scores:
        return True  # 无分数信息时视为可接受
    avg = sum(scores) / len(scores)
    return avg >= _fallback_threshold()


# 内容启发式用到的正则（模块级编译，避免每张图重建）
_PUNCT_RUN = r"[，。；：！？、]{2,}"


def _text_looks_bad(text: str) -> bool:
    """内容启发式：本地结果是否疑似潦草/乱码，从而触发百度兜底。

    只认一个"强信号"：连续 ≥3 个中文标点且含 ≥2 种不同标点（如"？，。"堆叠）。
    这是潦草/乱码的高特异信号——正常中文几乎不会连续堆叠多种标点。
    同款连写（"！！""？？"）是正常口语强调，不触发。

    实测：潦草图（含"？，。"）触发；"好的！！"、含日期/编号/英数的正常文档
    （"2026年8月17日 采购编号2026-001""iPhone 15 Pro 128GB 黑色"）、
    拼多多式英数截图、纯英文均不触发。

    设计取舍：只认强信号、宁缺毋滥——auto 误伤 = 白付一次百度调用（成本），
    漏报 = 维持本地结果（无成本）。局限：无混合标点的潦草（如只夹孤立数字）
    会漏报，请显式 ocr_mode=baidu。
    """
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False

    for run in re.findall(_PUNCT_RUN, compact):
        if len(set(run)) >= 2 and len(run) >= 3:
            return True
    return False


def _fallback_threshold() -> float:
    """判定"效果差"的平均置信度阈值，可用环境变量 KB_OCR_FALLBACK_CONFIDENCE 覆盖。"""
    raw = os.getenv("KB_OCR_FALLBACK_CONFIDENCE", "")
    try:
        return float(raw) if raw else 0.6
    except ValueError:
        return 0.6


def _fallback_baidu(image_input: ImageInput) -> str:
    """百度手写 OCR 后备。未配置 key / 调用失败 / 空结果都返回空串，绝不抛异常。"""
    try:
        from services.kb import baidu_ocr

        return baidu_ocr.ocr_image(image_input)
    except Exception as exc:  # import 失败等极端情况
        logger.warning("百度 OCR 后备不可用: %s", exc)
        return ""


def _incr(name: str) -> None:
    """进程内指标埋点（失败不影响主流程）。"""
    try:
        from services.kb.metrics import global_metrics

        global_metrics.incr(name)
    except Exception:
        pass
