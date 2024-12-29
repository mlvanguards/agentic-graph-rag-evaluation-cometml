from typing import List, Dict, Any
from neo4j import GraphDatabase

class OptimizedNeo4jIngestor:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")

    def ingest_batch(self, batch: List[Dict[str, Any]]):
        with self.driver.session() as session:
            session.execute_write(self._create_and_link_batch, batch)

    def _create_and_link_batch(self, tx, batch: List[Dict[str, Any]]):
        query = """
        UNWIND $batch AS paper
        MERGE (p:Paper {id: paper.id})
        SET p.title = paper.title, p.abstract = paper.abstract, 
            p.submit_date = paper.submit_date, p.update_date = paper.update_date
        WITH p, paper
        UNWIND paper.authors AS author_name
        MERGE (a:Author {name: author_name})
        MERGE (p)-[:AUTHORED_BY]->(a)
        WITH p, paper
        UNWIND paper.categories AS category_name
        MERGE (c:Category {name: category_name})
        MERGE (p)-[:BELONGS_TO]->(c)
        """
        tx.run(query, batch=batch)


def worker(uri: str, user: str, password: str, batch: List[Dict[str, Any]]):
    ingestor = OptimizedNeo4jIngestor(uri, user, password)
    try:
        ingestor.ingest_batch(batch)
    finally:
        ingestor.close()
