from research_agent.components.paper.tool import PaperTool
from research_agent.utils.paper_id_extractor import PaperIdExtractor
from langchain.tools.base import BaseTool
from pydantic import PrivateAttr

class PaperLookupTool(BaseTool):
    name: str = "Paper Lookup"
    description: str = "Use this tool to retrieve details about a specific paper by its ID."
    _paper_service: PaperTool = PrivateAttr()
    _paper_id_extractor: PaperIdExtractor = PrivateAttr()

    def __init__(self, paper_service: PaperTool):
        super().__init__()
        self._paper_service = paper_service
        self._paper_id_extractor = PaperIdExtractor()

    def _run(self, query: str) -> str:
        """Execute the paper lookup functionality."""
        try:
            paper_id = self._paper_id_extractor.extract(query)
            if not paper_id:
                return "No valid paper ID found in the message."

            # Get paper info with metrics
            result = self._paper_service.find_paper_by_id(paper_id)

            # Return paper information if found
            return result["response"] if result["success"] else f"Paper with ID {paper_id} not found."

        except Exception as e:
            return f"Error looking up paper: {str(e)}"
