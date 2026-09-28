# -*- coding: utf-8 -*-
# download_model.py —— 把需要的模型下载到本地 models/ 目录（走 hf-mirror 国内镜像）
#   1) BAAI/bge-m3             向量 embedding（检索召回）
#   2) BAAI/bge-reranker-v2-m3  CrossEncoder 重排（两阶段检索 v2）
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像，须在导入前设
# 规避 HF 走 Xet 后端经 hf-mirror 返回 401 的问题
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = [
    (
        "BAAI/bge-m3",
        os.path.join(HERE, "models", "bge-m3"),
        ["onnx/*", "imgs/*", "*.jpg", "*.webp", "colbert_linear.pt", "sparse_linear.pt"],
    ),
    (
        "BAAI/bge-reranker-v2-m3",
        os.path.join(HERE, "models", "bge-reranker-v2-m3"),
        ["onnx/*", "imgs/*", "*.jpg", "*.webp"],
    ),
]

for repo_id, target, ignore in JOBS:
    print(f"下载 {repo_id} -> {target}")
    snapshot_download(repo_id, local_dir=target, ignore_patterns=ignore)
    print(f"✅ {repo_id} 完成")

print("全部完成")
