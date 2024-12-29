from src.components.database.neo4j_client import Neo4jClient
from src.components.paper.models import Paper
from typing import Dict, Any
from src.components.experiment_tracker import MetricsCollector
import logging
from neo4j.exceptions import AuthError, ServiceUnavailable
import time

class PaperTool:
    def __init__(self, db_client: Neo4jClient):
        self.db_client = db_client
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)

    def find_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        start_time = time.time()

        try:
            # Test connection before query
            self.logger.info(f"Testing connection before paper lookup for {paper_id}")
            with self.db_client.session() as test_session:
                test_session.run("RETURN 1").single()

            # Perform actual query
            self.logger.info(f"Executing paper lookup query for {paper_id}")
            with self.db_client.session() as session:
                result = session.run(self._get_paper_query(), paper_id=paper_id)
                record = result.single()

                if not record:
                    self.logger.info(f"No paper found with ID {paper_id}")
                    return {
                        "response": f"Paper with ID {paper_id} not found.",
                        "success": False,
                        "metrics": {
                            "error": "Paper not found",
                            "success": False,
                            "processing_time": time.time() - start_time,
                            "question_length": len(paper_id)
                        }
                    }

                # Create paper object
                paper = Paper.from_db_record(record)
                paper_text = paper.to_string()

                # Collect metrics
                paper_stats = self.metrics_collector.get_text_stats(paper_text)
                metrics = {
                    "success": True,
                    "processing_time": time.time() - start_time,
                    "response_length": len(paper_text),
                    "response_tokens": paper_stats["token_count"],
                    "word_count": paper_stats["word_count"],
                    "authors_count": len(record["authors"]),
                    "categories_count": len(record["categories"]),
                    "question_length": len(paper_id)
                }

                return {
                    "response": paper_text,
                    "success": True,
                    "metrics": metrics
                }

        except AuthError as e:
            error_msg = f"Authentication failed: {str(e)}"
            self.logger.error(f"Authentication error during paper lookup: {str(e)}")
            return self._create_error_response(error_msg, start_time, paper_id)
        except ServiceUnavailable as e:
            error_msg = f"Database service unavailable: {str(e)}"
            self.logger.error(f"Service unavailable during paper lookup: {str(e)}")
            return self._create_error_response(error_msg, start_time, paper_id)
        except Exception as e:
            error_msg = f"Error retrieving paper: {str(e)}"
            self.logger.error(f"Error during paper lookup: {str(e)}")
            return self._create_error_response(error_msg, start_time, paper_id)

    def _create_error_response(self, error_msg: str, start_time: float, paper_id: str) -> Dict[str, Any]:
        return {
            "response": error_msg,
            "success": False,
            "metrics": {
                "error": error_msg,
                "success": False,
                "processing_time": time.time() - start_time,
                "question_length": len(paper_id)
            }
        }

    def _get_paper_query(self) -> str:
        """Return the paper lookup query."""
        return """
        MATCH (p:Paper {id: $paper_id})
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
        RETURN p.id as id, p.title AS title, p.abstract AS abstract, 
               p.submit_date AS submit_date, p.update_date AS update_date, 
               collect(DISTINCT a.name) AS authors, 
               collect(DISTINCT c.name) AS categories
        """