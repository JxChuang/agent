from pathlib import Path
from typing import List

import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings


def _load_api_key_from_env() -> str | None:
    """优先从项目目录的 .env 文件读取智谱 API Key。"""
    candidate_files = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]

    for env_path in candidate_files:
        if env_path.exists():
            load_dotenv(env_path, override=False)

    return os.getenv("ZHIPUAI_API_KEY")


class ZhipuAIEmbeddings(Embeddings):
    """自定义智谱 AI Embedding 类"""

    def __init__(self, api_key: str = None, model: str = "embedding-3"):
        from zhipuai import ZhipuAI

        self.api_key = api_key or _load_api_key_from_env() or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ZHIPUAI_API_KEY 未设置，请在项目根目录或当前目录的 .env 文件中配置，或显式传入。"
            )

        self.client = ZhipuAI(api_key=self.api_key)
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成向量"""
        result = []
        for i in range(0, len(texts), 64):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts[i:i+64]
                )
                # 提取每个 item 中的 embedding 属性
                result.extend([item.embedding for item in response.data])
            except Exception as e:
                # 捕获智谱 API 额度/限流等异常，抛出清晰、可读性强的友好提示
                err_msg = str(e)
                if "APIReachLimitError" in err_msg or "limit" in err_msg.lower() or "quota" in err_msg.lower() or "429" in err_msg or "1113" in err_msg:
                    raise RuntimeError(
                        "智谱 AI 接口额度超限 (APIReachLimitError: 余额不足或无可用资源包，请充值)。请登录智谱开放平台检查账户余额。"
                    ) from None
                raise RuntimeError(f"智谱 AI 向量生成失败: {err_msg}") from None
        return result
    
    def embed_query(self, text: str) -> List[float]:
        """为查询文本生成向量"""
        # 直接调用 embed_documents 处理单条文本
        return self.embed_documents([text])[0]
