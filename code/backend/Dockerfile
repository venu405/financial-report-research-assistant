FROM python:3.13-slim

WORKDIR /app

# 装依赖（独立层，利用缓存：代码改动不重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码
COPY src/ /app/src/
COPY pyproject.toml /app/

# 数据目录（chroma + checkpoint + users db，由 compose 挂卷持久化）
RUN mkdir -p /app/data/chroma_data

WORKDIR /app/src
EXPOSE 8000

# 启动后端（main.py 在 src/，WORKDIR 已切到 src）
# P2-3：WEB_CONCURRENCY 环境变量控制 worker 数（默认 1）。
# 注意：多 worker 下进程内缓存（BM25/指标）各自独立，SQLite 已用 WAL+busy_timeout 兜底。
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1}"]
