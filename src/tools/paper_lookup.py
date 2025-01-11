from src.components.paper.tool import PaperTool
from src.utils.paper_id_extractor import PaperIdExtractor
from langchain.tools.base import BaseTool
from pydantic import PrivateAttr
import time
import json

from src.components.evaluation.experiment_tracker import ExperimentTracker
from src.components.evaluation.experiment_tracker import MetricsCollector, MetricsData


class PaperLookupTool(BaseTool):
    name: str = "Paper Lookup"
    description: str = "Use this tool to retrieve details about a specific paper by its ID."
    _paper_service: PaperTool = PrivateAttr()
    _paper_id_extractor: PaperIdExtractor = PrivateAttr()
    _experiment_tracker: ExperimentTracker = PrivateAttr()
    _metrics_collector: MetricsCollector = PrivateAttr()

    def __init__(
            self,
            paper_service: PaperTool,
            experiment_tracker: ExperimentTracker,
            metrics_collector: MetricsCollector
    ):
        super().__init__()
        self._paper_service = paper_service
        self._paper_id_extractor = PaperIdExtractor()
        self._experiment_tracker = experiment_tracker
        self._metrics_collector = metrics_collector

    def _run(self, query: str) -> str:
        """Execute the paper lookup functionality."""
        # Start timing
        start_time = time.time()

        paper_id = None
        response_text = ""
        success = False
        error_msg = None

        try:
            paper_id = self._paper_id_extractor.extract(query)
            if not paper_id:
                response_text = "No valid paper ID found in the message."
                return json.dumps({
                    "ground_truth": "",
                    "tool_answer": response_text
                })

            # Get paper info with metrics
            result = self._paper_service.find_paper_by_id(paper_id)
            if result["success"]:
                # The “official text” from the DB
                ground_truth = result["response"]
                success = True
                tool_answer = (
                    f"Here is the paper with ID {paper_id}:\n\n"
                    f"{ground_truth}"
                )
            else:
                ground_truth = ""
                tool_answer = f"Paper with ID {paper_id} not found."
        except Exception as e:
            ground_truth = ""
            error_msg = str(e)
            tool_answer = f"Error looking up paper: {error_msg}"
            success = False

        # End timing
        end_time = time.time()
        processing_time = end_time - start_time

        # Build MetricsData
        metrics_data = MetricsData(
            processing_time=processing_time,
            query_length=len(query),
            response_length=len(response_text),
            success=success,
            token_count=self._metrics_collector.count_tokens(response_text),
            error=error_msg
        )

        # Log to CometML
        if paper_id:
            self._experiment_tracker.log_paper_lookup(paper_id, metrics_data)

        return json.dumps({
            "ground_truth": ground_truth,
            "tool_answer": tool_answer
        })
