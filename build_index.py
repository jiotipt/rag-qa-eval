# -*- coding: utf-8 -*-
# build_index.py —— 文档 → 切块 → 嵌入 → 写入 ChromaDB
import os
import shutil

# ★ 限制 CPU 线程数：必须在导入 torch / sentence-transformers 之前设置。
#   不设置时 PyTorch/OpenMP/MKL 默认占满所有物理核，会把整机 CPU 打满。
#   默认 4 线程，可用环境变量 EMB_NUM_THREADS 覆盖（如 8、14）。
_NUM_THREADS = os.environ.get("EMB_NUM_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _NUM_THREADS)

# ★ 国内镜像：必须在导入 huggingface/sentence-transformers 之前设置，否则下载会连不上 HF
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

KB_DIR = "kb_docs"
DB_DIR = "chroma_db"


def _is_redundant_english(source) -> bool:
    path = str(source or "")
    if os.path.basename(path) != "SKILL.md":
        return False
    return os.path.exists(os.path.join(os.path.dirname(path), "SKILL.cn.md"))


# 收集多种扩展名（.md / .yaml / .txt 都是文本）
docs = []
for pat in ("**/*.md", "**/*.yaml", "**/*.txt"):
    docs += DirectoryLoader(
        KB_DIR, glob=pat, loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, silent_errors=True,
    ).load()

# 语料治理：同一技能已有中文版 SKILL.cn.md 时丢弃近重复的英文 SKILL.md，降低跨语言重复噪声
docs = [d for d in docs if not _is_redundant_english(d.metadata.get("source", ""))]
print(f"加载文档：{len(docs)} 篇")

chunks = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=120
).split_documents(docs)
print(f"切分 chunk：{len(chunks)} 个")

# ★ embedding 模型：优先 EVAL_EMB_MODEL 环境变量；其次本地 models/bge-m3；都没有则按 HF 名称（走 mirror 自动下载）
_LOCAL_EMB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-m3")
EMB_MODEL = os.getenv("EVAL_EMB_MODEL") or (_LOCAL_EMB if os.path.isdir(_LOCAL_EMB) else "BAAI/bge-m3")
print("embedding 模型：", EMB_MODEL)

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    # show_progress=True 会显示 embedding 进度条；不能写进 encode_kwargs，
    # 否则与内部 show_progress_bar=... 冲突，报 multiple values。
    show_progress=True,
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

# 重建前清库：Chroma 是 get_or_create_collection + add_documents，
# 不清空会把同样的 chunk 反复追加（936 → 1872 → …）。
shutil.rmtree(DB_DIR, ignore_errors=True)
print("已清空旧索引：", DB_DIR)

Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
print("索引已写入", DB_DIR)
print(f"embedding 线程数上限：{_NUM_THREADS}（可用环境变量 EMB_NUM_THREADS 调整）")
