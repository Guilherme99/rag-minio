from langchain_ollama import ChatOllama
from config import *

class RAGPipeline:
    def __init__(self):
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2
        )

    def answer(self, context, question):
        prompt = f"""
Você é um especialista em análise de imagens.

Foram recuperados objetos semanticamente similares.

Use SOMENTE as informações abaixo para responder.

{context}

Pergunta do usuário:
{question}

Se houver múltiplos resultados, explique as diferenças.
Se não houver evidência suficiente, diga isso.
"""

        response = self.llm.invoke(prompt)
        return response.content
