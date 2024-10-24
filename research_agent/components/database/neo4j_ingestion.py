import json
from typing import List, Dict, Any
from neo4j import GraphDatabase
import multiprocessing
import time
import os


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


def ingest_data_parallel(uri: str, user: str, password: str, data: List[Dict[str, Any]], batch_size: int = 1000,
                         num_processes: int = 4):
    total = len(data)
    batches = [data[i:i + batch_size] for i in range(0, total, batch_size)]

    with multiprocessing.Pool(num_processes) as pool:
        results = []
        for batch in batches:
            result = pool.apply_async(worker, (uri, user, password, batch))
            results.append(result)

        for i, result in enumerate(results):
            result.get()  # Wait for the batch to complete
            print(f"Ingested {min((i + 1) * batch_size, total)}/{total} papers")


def main():
    # Neo4j connection details
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    # Initialize Neo4jIngestor and create constraints
    ingestor = OptimizedNeo4jIngestor(uri, user, password)
    ingestor.create_constraints()
    ingestor.close()

    # Load processed data
    with open('/data/processed_arxiv_metadata.json', 'r') as f:
        processed_data = json.load(f)

    # Ingest data in parallel
    start_time = time.time()
    ingest_data_parallel(uri, user, password, processed_data, batch_size=1000, num_processes=4)
    end_time = time.time()

    print(f"Total ingestion time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()