from research_agent.components.database.neo4j_client import Neo4jClient
from research_agent.components.paper.models import Paper
from typing import Optional
from research_agent.components.experiment_tracker import MetricsCollector

class PaperTool:
    def __init__(self, db_client: Neo4jClient):
        self.db_client = db_client
        self.metrics_collector = MetricsCollector()


    def find_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        query = """
        MATCH (p:Paper {id: $paper_id})
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
        RETURN p.title AS title, p.abstract AS abstract, 
               p.submit_date AS submit_date, p.update_date AS update_date, 
               collect(DISTINCT a.name) AS authors, 
               collect(DISTINCT c.name) AS categories
        """
        try:
            with self.db_client.session() as session:
                result = session.run(query, paper_id=paper_id)
                record = result.single()
                if not record:
                    return None
                return Paper.from_db_record(record)
        except Exception as e:
            raise ValueError(f"Error retrieving paper: {str(e)}")
