import json
from langchain_ollama import ChatOllama
from config import *

class RAGPipeline:
    def __init__(self):
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )

    def format_answer(self, payload):
        prompt = f"""
Retorne exatamente este JSON, sem explicações:

{json.dumps(payload, indent=2, ensure_ascii=False)}
"""
        response = self.llm.invoke(prompt)
        return response.content.strip()
