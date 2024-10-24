from langchain_community.vectorstores import Neo4jVector
from research_agent.components.database.neo4j_client import Neo4jClient
from typing import List, Tuple, Optional

class VectorStore:
    def __init__(self, neo4j_client: Neo4jClient, embedding_model, index_name: str):
        self.client = neo4j_client
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self) -> Neo4jVector:
        return Neo4jVector(
            embedding=self.embedding_model,
            url=self.client.uri,
            username=self.client.user,
            password=self.client.password,
            index_name=self.index_name,
            node_label="Paper",
            text_node_property="abstract"
        )

    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc.page_content, score) for doc, score in results]
        except Exception as e:
            raise ValueError(f"Error performing similarity search: {str(e)}")
