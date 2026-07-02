from pathlib import Path
from typing import List
import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings


def _load_api_key_from_env() -> str | None:
    """优先从项目目录的 .env 文件读取 DashScope (Qwen) API Key。"""
    candidate_files = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]

    for env_path in candidate_files:
        if env_path.exists():
            load_dotenv(env_path, override=False)

    return os.getenv("DASHSCOPE_API_KEY")


class QwenEmbeddings(Embeddings):
    """自定义通义千问 (Qwen) Embedding 类，基于阿里百炼 DashScope 的 OpenAI 兼容接口"""

    def __init__(
        self, 
        api_key: str = None, 
        model: str = "text-embedding-v3", 
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        from openai import OpenAI

        self.api_key = api_key or _load_api_key_from_env() or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置，请在项目根目录的 .env 文件中配置，或显式传入。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成向量"""
        result = []
        # 百炼 API 限制单次请求文本条数（一般 text-embedding-v3 限制单次不超过 25 条，此处按安全值 25 进行分批）
        for i in range(0, len(texts), 25):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts[i:i+25]
                )
                # 提取每个 item 中的 embedding 属性
                result.extend([item.embedding for item in response.data])
            except Exception as e:
                # 捕获 API 额度、限流、鉴权等常见异常，抛出友好提示
                err_msg = str(e)
                if "429" in err_msg or "limit" in err_msg.lower() or "quota" in err_msg.lower():
                    raise RuntimeError(
                        "通义千问 (Qwen) 接口额度超限或调用太快。请检查您的百炼账号余额，或缩减请求频率。"
                    ) from None
                elif "401" in err_msg or "api_key" in err_msg.lower():
                    raise RuntimeError(
                        "通义千问 (Qwen) 鉴权失败。请检查您的 DASHSCOPE_API_KEY 是否配置正确。"
                    ) from None
                raise RuntimeError(f"通义千问 (Qwen) 向量生成失败: {err_msg}") from None
        return result
    
    def embed_query(self, text: str) -> List[float]:
        """为查询文本生成向量"""
        return self.embed_documents([text])[0]
