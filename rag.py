# -*- coding: utf-8 -*-
# rag.py —— 本地 LM Studio 模型 + 检索
import os
# ★ 国内镜像：必须在导入 huggingface/sentence-transformers 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DB_DIR = "chroma_db"
MODEL_ID = "qwen3.6-35b-a3b-ud"

# ★ embedding 模型：与 build_index.py 保持一致（优先本地 models/bge-m3）
_LOCAL_EMB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-m3")
EMB_MODEL = _LOCAL_EMB if os.path.isdir(_LOCAL_EMB) else "BAAI/bge-m3"

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
retriever = Chroma(persist_directory=DB_DIR, embedding_function=embeddings) \
    .as_retriever(search_kwargs={"k": 3})

# ★ reasoning_effort:"none" 关思考，否则本地思考模型会返回空回答
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model=MODEL_ID,
    temperature=0,
    model_kwargs={"reasoning_effort": "none"},
)

prompt = ChatPromptTemplate.from_template(
    "只根据下面的上下文回答问题。若上下文里没有答案，就回答“根据现有资料无法回答”。\n\n"
    "上下文：\n{context}\n\n问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

if __name__ == "__main__":
    q = "会议纪要生成器支持的单场会议最长是多久？超时怎么办？"
    print("检索到片段：", len(retriever.invoke(q)))
    print("回答：", chain.invoke(q))
