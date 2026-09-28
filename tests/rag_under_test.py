import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from eval_config import (
    DB_DIR,
    EMB_MODEL,
    EVAL_TRACK,
    RECALL_K,
    RERANK_ENABLED,
    RERANK_MODEL,
    TOP_K,
    TRACK,
)

_PROMPT_TEXT = (
    "你是严谨的知识库问答助手。请只用【上下文】中的信息回答【问题】。\n"
    "规则：\n"
    "1. 只要上下文中包含与问题相关的信息（即使不完整、或分散在多处），就整合后作答，不要轻易说“无法回答”。\n"
    "2. 回答要忠实于上下文，不要编造上下文中没有的具体细节。\n"
    "3. 回答要简洁聚焦：先直接给出核心结论，再用不超过 3 条要点补充；不要罗列与问题无关的内容。\n"
    "4. 只有当上下文与问题完全无关、完全找不到可用信息时，才回答“根据现有资料无法回答”。\n\n"
    "【上下文】\n{context}\n\n【问题】\n{question}"
)

_prompt = None
_llm = None
_parser = None
_retriever = None
_reranker = None


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def ensure_ready() -> None:
    global _prompt, _llm, _parser, _retriever, _reranker
    if _retriever is not None:
        return
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    recall_k = RECALL_K if RERANK_ENABLED else TOP_K
    _retriever = Chroma(
        persist_directory=DB_DIR, embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": recall_k})
    if RERANK_ENABLED:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)

    # 双轨（D4）：检索层完全一致，只切换「被测生成模型」——控制变量。
    if EVAL_TRACK == "deepseek" and not TRACK["api_key"]:
        raise RuntimeError(
            "EVAL_TRACK=deepseek 需要环境变量 DEEPSEEK_API_KEY（被测轨密钥）"
        )
    llm_kwargs = {
        "base_url": TRACK["base_url"],
        "api_key": TRACK["api_key"],
        "model": TRACK["model"],
        "temperature": 0,
    }
    # 本地思考模型必须显式传 reasoning_effort="none"，否则返回空回答
    if TRACK.get("reasoning_effort"):
        llm_kwargs["reasoning_effort"] = TRACK["reasoning_effort"]
    _llm = ChatOpenAI(**llm_kwargs)

    _prompt = ChatPromptTemplate.from_template(_PROMPT_TEXT)
    _parser = StrOutputParser()


def retrieve(question: str) -> list:
    ensure_ready()
    docs = _retriever.invoke(question)
    if _reranker is None:
        return docs
    scores = _reranker.predict([(question, d.page_content) for d in docs])
    ranked = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
    return [docs[i] for i in ranked[:TOP_K]]


def answer(question: str) -> dict:
    """运行被测 RAG：返回 answer / contexts / sources / latency_s。"""
    docs = retrieve(question)
    contexts = [d.page_content for d in docs]
    sources = [d.metadata.get("source", "") for d in docs]
    t0 = time.perf_counter()
    output = (_prompt | _llm | _parser).invoke(
        {"context": _format_docs(docs), "question": question}
    )
    return {
        "output": output,
        "contexts": contexts,
        "sources": sources,
        "latency_s": time.perf_counter() - t0,
    }
