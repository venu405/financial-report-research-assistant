FROM python:3.13-slim

WORKDIR /app

# 装依赖（独立层，利用缓存：代码改动不重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码
COPY src/ /app/src/
COPY pyproject.toml /app/
COPY scripts/ /app/scripts/
COPY data/ /app/seed_data/

# 数据目录（chroma + checkpoint + users db，由 compose 挂卷持久化）
RUN mkdir -p /app/data/chroma_data

WORKDIR /app/src
EXPOSE 8000

# 启动后端（main.py 在 src/，WORKDIR 已切到 src）
# 单进程定位（P0）：SQLite 三连接 + BM25 进程内缓存撑不起多 worker——
# 多 worker 会 database is locked + 检索索引互相不可见。真实扩容需换
# PostgreSQL + 共享检索索引（另立专项），在此之前锁定单进程。
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
