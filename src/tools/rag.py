from langchain.tools import BaseTool
from pydantic import PrivateAttr
import time

from src.components.rag.tool import RAG
from src.components.evaluation.experiment_tracker import ExperimentTracker, MetricsCollector, MetricsData

class RAGTool(BaseTool):
    name: str = "RAG"
    description: str = "Use this tool to retrieve research papers and generate answers to general queries."
    _rag_service: RAG = PrivateAttr()
    _experiment_tracker: ExperimentTracker = PrivateAttr()
    _metrics_collector: MetricsCollector = PrivateAttr()

    def __init__(
        self,
        rag_service: RAG,
        experiment_tracker: ExperimentTracker,
        metrics_collector: MetricsCollector
    ):
        super().__init__()
        self._rag_service = rag_service
        self._experiment_tracker = experiment_tracker
        self._metrics_collector = metrics_collector

    def _run(self, query: str) -> str:
        """Execute the RAG functionality."""
        # Start timing
        start_time = time.time()

        response_text = ""
        success = False
        error_msg = None

        try:
            response = self._rag_service.answer_question(query)
            response_text = response["response"]
            success = True
        except Exception as e:
            error_msg = str(e)
            response_text = f"Error processing RAG query: {error_msg}"

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
        self._experiment_tracker.log_rag_query(metrics_data)
        return response_text
