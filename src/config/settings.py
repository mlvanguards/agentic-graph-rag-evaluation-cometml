from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    def __init__(self):
        load_dotenv()
        self.cometml_api_key = os.getenv("COMETML_API_KEY")
        self.project_name = "research-paper-rag"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
