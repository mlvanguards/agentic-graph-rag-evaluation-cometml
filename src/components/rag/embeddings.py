from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
import os

class Embedding:
    def __init__(self, api_key: Optional[str] = None):
        if not api_key and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key must be provided either directly or through environment variable")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        self.model = OpenAIEmbeddings(openai_api_key=self.api_key)


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")
