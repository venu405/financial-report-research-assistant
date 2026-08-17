"""百度智能云 OCR 后备通道。

用途：本地 RapidOCR 对手写/潦草字识别效果差（或引擎不可用）时，调用
百度智能云"手写文字识别"接口兜底——该接口对手写体专项优化，准确率量级远高于
PP-OCR 系印刷体模型。

配置（.env 或环境变量，main.py 启动时已 load_dotenv）：
    BAIDU_OCR_API_KEY=...      # 百度智能云控制台 → 创建 OCR 应用后获取
    BAIDU_OCR_SECRET_KEY=...

未配置 key 时本模块静默不可用（ocr_image 返回 ""），不影响本地 OCR 主流程。
access_token 有效期 30 天，模块内做内存缓存 + 锁，过期自动续期。
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Union

import requests

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
_HANDWRITING_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting"

# access_token 缓存：30 天有效，提前 60s 视为过期主动续期
_token_cache: dict[str, object] = {"token": "", "expires_at": 0.0}
_token_lock = threading.Lock()

# P2-2：百度 OCR QPS 限流（进程内滑动窗口，1 秒窗口）。批量扫描入库时
# 防止打爆百度配额/烧钱。KB_BAIDU_OCR_QPS 默认 2，超限降级返回空串不阻塞。
_qps_hits: list[float] = []
_qps_lock = threading.Lock()


def _default_qps() -> int:
    raw = os.getenv("KB_BAIDU_OCR_QPS", "").strip()
    if not raw:
        return 2
    try:
        return int(raw)
    except ValueError:
        return 2


def _allow_ocr_call() -> bool:
    """滑动窗口 QPS 检查：1 秒内已达上限则拒绝（返回 False）。"""
    qps = _default_qps()
    if qps <= 0:
        return False
    now = time.time()
    with _qps_lock:
        _qps_hits[:] = [t for t in _qps_hits if now - t < 1.0]
        if len(_qps_hits) >= qps:
            return False
        _qps_hits.append(now)
        return True


def is_configured() -> bool:
    """是否配置了百度 key（决定后备通道是否启用）。"""
    return bool(os.getenv("BAIDU_OCR_API_KEY", "") and os.getenv("BAIDU_OCR_SECRET_KEY", ""))


def _to_base64(image_input: Union[str, Path, bytes, object]) -> str:
    """把 str/Path/bytes/ndarray 统一转成 base64 字符串（百度接口入参）。"""
    if isinstance(image_input, (str, Path)):
        raw = Path(image_input).read_bytes()
    elif isinstance(image_input, bytes):
        raw = image_input
    else:
        # ndarray 等：PIL 转 PNG 字节
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(image_input).convert("RGB").save(buf, format="PNG")
        raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii")


def get_access_token() -> str:
    """换取百度 OCR access_token（内存缓存 + 锁，30 天有效）。未配置 key 返回空串。"""
    api_key = os.getenv("BAIDU_OCR_API_KEY", "")
    secret_key = os.getenv("BAIDU_OCR_SECRET_KEY", "")
    if not api_key or not secret_key:
        return ""

    now = time.time()
    with _token_lock:
        token = _token_cache["token"]
        expires_at = _token_cache["expires_at"]
        if token and expires_at > now + 60:
            return token

        resp = requests.post(
            _TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": api_key,
                "client_secret": secret_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token", "")
        expires_in = int(data.get("expires_in", 2592000))  # 默认 30 天
        _token_cache["token"] = token
        _token_cache["expires_at"] = now + expires_in
        return token


def ocr_image(image_input: Union[str, Path, bytes, object]) -> str:
    """调用百度手写文字识别，返回按行 join 的文本；失败/未配置返回空串。

    不抛异常：所有错误都记日志并返回空，交由上层静默降级。
    QPS 超限时同样降级返回空串（防打爆百度配额）。
    """
    if not is_configured():
        logger.debug("百度 OCR 未配置 key，后备通道关闭")
        return ""

    if not _allow_ocr_call():
        _incr("ocr.baidu_qps_limited")
        logger.warning("百度 OCR 触发 QPS 限流，本次降级返回空串")
        return ""

    try:
        token = get_access_token()
        if not token:
            _incr("ocr.baidu_fallback")
            logger.warning("百度 OCR 获取 access_token 失败")
            return ""

        img_b64 = _to_base64(image_input)
        resp = requests.post(
            _HANDWRITING_URL,
            params={"access_token": token},
            data={"image": img_b64},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error_code" in data:
            _incr("ocr.baidu_fallback")
            logger.warning(
                "百度 OCR 返回错误: %s %s", data.get("error_code"), data.get("error_msg")
            )
            return ""

        _incr("ocr.baidu_success")
        words = [item.get("words", "") for item in data.get("words_result", [])]
        return "\n".join(w for w in words if w)
    except Exception as exc:  # 网络/超时/解析等一律静默降级
        _incr("ocr.baidu_fallback")
        logger.warning("百度 OCR 调用失败，已降级: %s", exc)
        return ""


def _incr(name: str) -> None:
    """进程内指标埋点（失败不影响主流程）。"""
    try:
        from services.kb.metrics import global_metrics

        global_metrics.incr(name)
    except Exception:
        pass
