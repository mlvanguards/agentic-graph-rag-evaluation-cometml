from research_agent.tools.base import BaseTool
from research_agent.components.paper.tool import PaperTool
from langchain.schema import AIMessage
from research_agent.utils.paper_id_extractor import PaperIdExtractor
from typing import Dict, Any

class PaperLookupTool(BaseTool):
    def __init__(self, paper_service: PaperTool):
        super().__init__(
            name="Paper Lookup",
            description="Use this tool to retrieve details about a specific paper by its ID."
        )
        self.paper_service = paper_service
        self.paper_id_extractor = PaperIdExtractor()

    def execute(self, content: str) -> str:
        """
        Execute the paper lookup functionality.
        This implements the abstract method from BaseTool.
        """
        try:
            paper_id = self.paper_id_extractor.extract(content)
            if not paper_id:
                return "No valid paper ID found in the message"

            paper_info = self.paper_service.find_paper_by_id(paper_id)
            if paper_info is None:
                return f"Paper with ID {paper_id} not found"

            return paper_info.to_string()
        except Exception as e:
            return f"Error looking up paper: {str(e)}"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the tool callable for LangGraph compatibility"""
        try:
            messages = state.get("messages", [])
            if not messages:
                return {"messages": [AIMessage(content="No message to process")]}

            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)

            # Use the execute method for the actual functionality
            result = self.execute(content)
            return {"messages": [AIMessage(content=result)]}

        except Exception as e:
            return {"messages": [AIMessage(content=f"Error in paper lookup tool: {str(e)}")]}
