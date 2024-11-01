from research_agent.components.rag.tool import RAG
from langchain.tools import BaseTool
from pydantic import PrivateAttr

class RAGTool(BaseTool):

    name: str = "RAG"
    description: str = "Use this tool to retrieve research papers and generate answers to general queries."
    _rag_service: RAG = PrivateAttr()

    def __init__(self, rag_service: RAG):
        super().__init__()
        self._rag_service = rag_service

    def _run(self, query: str) -> str:
        """Execute the RAG functionality."""
        try:
            response = self._rag_service.answer_question(query)
            return response['response']
        except Exception as e:
            return f"Error processing RAG query: {str(e)}"