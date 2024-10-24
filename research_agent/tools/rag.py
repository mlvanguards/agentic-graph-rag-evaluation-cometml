from research_agent.tools.base import BaseTool
from research_agent.components.rag.tool import RAGTool
from langchain.schema import AIMessage
from typing import Dict, Any

class RAG(BaseTool):
    def __init__(self, rag_service: RAGTool):
        super().__init__(
            name="RAG",
            description="Use this tool to retrieve research papers and generate answers to general queries."
        )
        self.rag_service = rag_service

    def execute(self, query: str) -> str:
        """
        Execute the RAG functionality.
        This implements the abstract method from BaseTool.
        """
        try:
            return self.rag_service.answer_question(query)
        except Exception as e:
            return f"Error processing RAG query: {str(e)}"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the tool callable for LangGraph compatibility"""
        try:
            messages = state.get("messages", [])
            if not messages:
                return {"messages": [AIMessage(content="No message to process")]}

            last_message = messages[-1]
            query = last_message.content if hasattr(last_message, 'content') else str(last_message)

            # Use the execute method for the actual functionality
            result = self.execute(query)
            return {"messages": [AIMessage(content=result)]}

        except Exception as e:
            return {"messages": [AIMessage(content=f"Error in RAG tool: {str(e)}")]}