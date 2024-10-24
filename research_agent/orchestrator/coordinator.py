import uuid
import time
import logging
from typing import Dict, Any
import warnings

from research_agent.config.settings import Settings
from research_agent.core.graph import create_research_graph
from research_agent.components.database.neo4j_client import Neo4jClient
from research_agent.components.paper.tool import PaperTool
from research_agent.components.rag.tool import RAGTool
from research_agent.components.rag.embeddings import Embedding
from research_agent.components.database.vector_store import VectorStore
from research_agent.components.experiment_tracker import ExperimentTracker
from research_agent.tools.paper_lookup import PaperLookupTool
from research_agent.tools.rag import RAG
from research_agent.agents.research_assistant import ResearchAssistant
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Suppress HTTPX logging
logging.getLogger("httpx").setLevel(logging.WARNING)
# Suppress urllib3 logging if you're seeing those too
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class Coordinator:
    def __init__(self):
        self.settings = Settings()
        self.setup_warnings()
        self.experiment_tracker = self.setup_experiment_tracker()
        self.services = self.initialize_services()
        self.tools = self.initialize_tools()
        self.assistant = self.initialize_assistant()
        self.graph = self.setup_graph()

    @staticmethod
    def setup_warnings():
        """Configure warning filters"""
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def _validate_settings(self):
        if not self.settings.openai_api_key:
            raise ValueError("OpenAI API key not found in settings")
        if not self.settings.openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        os.environ["OPENAI_API_KEY"] = self.settings.openai_api_key

    def setup_experiment_tracker(self) -> ExperimentTracker:
        """Initialize the experiment tracker"""
        tracker = ExperimentTracker(
            api_key=self.settings.cometml_api_key,
            project_name=self.settings.project_name
        )
        tracker.experiment.add_tags(['v1', 'graph-rag', 'research-papers'])
        tracker.experiment.log_parameter("session_id", str(uuid.uuid4()))
        tracker.experiment.log_parameter("session_start", time.strftime("%Y-%m-%d %H:%M:%S"))
        return tracker

    def initialize_services(self) -> Dict[str, Any]:
        """Initialize all required components"""
        # Database client
        db_client = Neo4jClient(
            uri=self.settings.neo4j_uri,
            user=self.settings.neo4j_user,
            password=self.settings.neo4j_password
        )

        # Embedding service
        embedding_service = Embedding(api_key=self.settings.openai_api_key)

        # Vector store
        vector_store = VectorStore(
            neo4j_client=db_client,
            embedding_model=embedding_service.model,
            index_name="paper_vector_index"
        )

        # Paper service
        paper_service = PaperTool(db_client=db_client)

        # RAG service
        rag_service = RAGTool(
            vector_store=vector_store,
            openai_api_key=self.settings.openai_api_key
        )

        return {
            "db_client": db_client,
            "embedding_service": embedding_service,
            "vector_store": vector_store,
            "paper_service": paper_service,
            "rag_service": rag_service
        }

    def initialize_tools(self) -> Dict[str, Any]:
        """Initialize all tools"""
        paper_lookup_tool = PaperLookupTool(
            paper_service=self.services["paper_service"]
        )
        rag_tool = RAG(
            rag_service=self.services["rag_service"]
        )
        return {
            "paper_lookup": paper_lookup_tool,
            "rag": rag_tool
        }

    def initialize_assistant(self) -> ResearchAssistant:
        """Initialize the research assistant"""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful AI assistant for research papers. You can:
                1. Look up specific papers using their ID (e.g., "0704.2002")
                2. Answer general questions about research topics using the RAG system
                3. Provide detailed information about papers and research
                4. Reference our previous conversation as needed

                When referencing previous conversation, be specific about what was discussed.
                """
            ),
            ("placeholder", "{messages}")
        ])

        return ResearchAssistant(
            prompt=prompt,
            experiment_tracker=self.experiment_tracker,
            paper_service=self.services["paper_service"],
            rag_service=self.services["rag_service"]
        )

    def setup_graph(self) -> Any:
        """Set up the conversation graph"""
        return create_research_graph(
            assistant=self.assistant,
            rag_tool=self.tools["rag"],
            paper_lookup_tool=self.tools["paper_lookup"]
        )

    def process_message(self, message: str, state: Dict[str, Any]) -> None:
        """Process a single message and update state"""
        try:
            state["messages"].append(HumanMessage(content=message))

            # Directly call the assistant
            response = self.assistant(state)

            # Append assistant's messages to the state
            state["messages"].extend(response["messages"])

            # Print the assistant's response
            for msg in response["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"Assistant: {msg.content}")

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            print(f"An error occurred: {str(e)}")

    def run(self):
        """Run the main application loop"""
        try:
            print("Research Paper Assistant initialized. Type 'exit' to quit.")
            state = {
                "messages": [],
                "metrics": {},
                "conversation_history": []
            }

            while True:
                user_input = input("\nYour question: ").strip()

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    self.cleanup()
                    break

                self.process_message(user_input, state)

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")
            self.cleanup()
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print(f"\n\nAn error occurred: {str(e)}")
            self.cleanup()

    def cleanup(self):
        """Clean up resources and log final metrics"""
        try:
            # Close database connections
            self.services["db_client"].close()

            # Log final metrics
            if hasattr(self, 'session_metrics'):
                self.experiment_tracker.end_session(self.session_metrics)

            print("\nSession ended. Thank you for using the Research Paper Assistant!")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
