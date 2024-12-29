from src.services.database.neo4j_client import Neo4jClient
from src.services.rag.embeddings import Embedding
import logging

class IndexingService:
    def __init__(
        self,
        db_client: Neo4jClient,
        embedding_service: Embedding,
        batch_size: int = 100
    ):
        self.db_client = db_client
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    def ensure_vector_index(self, index_name: str):
        with self.db_client.session() as session:
            if not self._vector_index_exists(session, index_name):
                self._create_vector_index(session, index_name)

    def _vector_index_exists(self, session, index_name: str) -> bool:
        query = """
        SHOW INDEXES
        YIELD name, type
        WHERE name = $index_name AND type = 'VECTOR'
        RETURN count(*) > 0 AS exists
        """
        result = session.run(query, index_name=index_name)
        return result.single()['exists']

    def _create_vector_index(self, session, index_name: str):
        session.run("""
        CALL db.index.vector.createNodeIndex(
          $index_name,
          'Paper',
          'embedding',
          1536,
          'cosine'
        )
        """, index_name=index_name)
        self.logger.info(f"Vector index '{index_name}' created.")
