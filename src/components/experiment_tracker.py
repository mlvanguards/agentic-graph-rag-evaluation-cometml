from comet_ml import Experiment
from typing import Dict, Any, Optional
import time
from dataclasses import dataclass
import tiktoken

@dataclass
class MetricsData:
    processing_time: float
    query_length: int
    response_length: int
    success: bool
    context_length: int = 0  # Added for RAG context size
    token_count: int = 0  # Added for response tokens
    error: Optional[str] = None


class MetricsCollector:
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_text_stats(self, text: str) -> Dict[str, int]:
        return {
            "char_length": len(text),
            "token_count": self.count_tokens(text),
            "word_count": len(text.split()),
            "line_count": len(text.splitlines())
        }

class ExperimentTracker:
    def __init__(self, api_key: str, project_name: str):
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name
        )
        self.start_time = time.time()
        self.query_count = 0
        self.error_count = 0

    def log_paper_lookup(self, paper_id: str, metrics: MetricsData):
        """Log metrics for paper lookups."""
        self.query_count += 1
        if not metrics.success:
            self.error_count += 1

        self.experiment.log_metrics({
            "paper_lookup_latency": metrics.processing_time,
            "paper_id_length": len(paper_id),
            "paper_lookup_year": int(paper_id.split('.')[0]) if '.' in paper_id else None,
            "paper_lookup_success": int(metrics.success),
            "paper_response_length": metrics.response_length,
            "paper_response_tokens": metrics.token_count,
            "cumulative_queries": self.query_count,
            "error_rate": self.error_count / self.query_count if self.query_count > 0 else 0
        })
        if metrics.error:
            self.experiment.log_parameter("error", metrics.error)

    def log_rag_query(self, metrics: MetricsData):
        """Log metrics for RAG queries."""
        self.query_count += 1
        if not metrics.success:
            self.error_count += 1

        self.experiment.log_metrics({
            "rag_query_length": metrics.query_length,
            "rag_response_length": metrics.response_length,
            "rag_processing_time": metrics.processing_time,
            "rag_success": int(metrics.success),
            "rag_context_length": metrics.context_length,
            "rag_response_tokens": metrics.token_count,
            "cumulative_queries": self.query_count,
            "error_rate": self.error_count / self.query_count if self.query_count > 0 else 0
        })

        # Log hourly aggregates
        hour = time.strftime("%Y-%m-%d-%H")
        self.experiment.log_metrics({
            f"hourly_queries_{hour}": 1,
            f"hourly_errors_{hour}": 0 if metrics.success else 1,
            f"hourly_avg_latency_{hour}": metrics.processing_time
        })

    def log_session_metrics(self):
        """Log overall session metrics."""
        session_duration = time.time() - self.start_time
        self.experiment.log_metrics({
            "session_duration": session_duration,
            "total_queries": self.query_count,
            "total_errors": self.error_count,
            "session_error_rate": self.error_count / self.query_count if self.query_count > 0 else 0,
            "queries_per_minute": (self.query_count * 60) / session_duration if session_duration > 0 else 0
        })

    def end_session(self, session_metrics: Dict[str, Any]):
        """Log final session metrics and end experiment."""
        self.log_session_metrics()
        for key, value in session_metrics.items():
            self.experiment.log_metric(f"session_{key}", value)
        self.experiment.end()