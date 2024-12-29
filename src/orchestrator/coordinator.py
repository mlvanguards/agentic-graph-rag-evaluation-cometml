import uuid
import time
import logging
from typing import Dict, Any

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from src.config.settings import Settings
from src.core.graph import create_research_graph
from src.components.database.neo4j_client import Neo4jClient
from src.components.paper.tool import PaperTool
from src.components.rag.tool import RAG
from src.components.rag.embeddings import Embedding
from src.components.database.vector_store import VectorStore
from src.components.experiment_tracker import ExperimentTracker, MetricsCollector
from src.tools.paper_lookup import PaperLookupTool
from src.tools.rag import RAGTool
from src.agents.research_assistant import ResearchAssistant
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
        self.experiment_tracker = self.setup_experiment_tracker()
        self.metrics_collector = MetricsCollector()
        self.services = self.initialize_services()
        self.tools = self.initialize_tools()
        self.assistant = self.initialize_assistant()
        self.graph = self.setup_graph()

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
        with db_client.session() as session:
            result = session.run("RETURN 1 as num").single()
            print(f"Initial connection test result: {result['num']}")
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
        rag_service = RAG(
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
            paper_service=self.services["paper_service"],
            experiment_tracker=self.experiment_tracker,
            metrics_collector=self.metrics_collector
        )
        rag_tool = RAGTool(
            rag_service=self.services["rag_service"],
            experiment_tracker=self.experiment_tracker,
            metrics_collector=self.metrics_collector
        )
        return {
            "paper_lookup": paper_lookup_tool,
            "rag": rag_tool
        }

    def initialize_assistant(self) -> ResearchAssistant:
        """Initialize the research assistant."""
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.settings.openai_api_key
        )
        tools = [self.tools["paper_lookup"], self.tools["rag"]]

        return ResearchAssistant(
            experiment_tracker=self.experiment_tracker,
            tools=tools,
            llm=llm
        )

    def setup_graph(self) -> Any:
        """Set up the conversation graph"""
        return create_research_graph(
            assistant=self.assistant,
            rag_tool=self.tools["rag"],
            paper_lookup_tool=self.tools["paper_lookup"]
        )

    def process_message(self, message: str, state: Dict[str, Any]) -> None:
        try:
            state["messages"].append(HumanMessage(content=message))

            # Log conversation metrics
            self.experiment_tracker.experiment.log_metrics({
                "conversation_turn": len(state["messages"]),
                "message_length": len(message)
            })

            # Directly call the assistant
            response = self.assistant(state)

            # Append assistant's messages to the state
            state["messages"].extend(response["messages"])

            # Log response metrics
            for msg in response["messages"]:
                if isinstance(msg, AIMessage):
                    self.experiment_tracker.experiment.log_metrics({
                        "response_length": len(msg.content)
                    })
                    print(f"Assistant: {msg.content}")

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            self.experiment_tracker.experiment.log_metric("errors", 1)
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
            # Calculate session duration
            if hasattr(self, 'experiment_tracker'):
                session_end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self.experiment_tracker.experiment.log_parameter("session_end", session_end_time)

                # Log final session metrics
                final_metrics = {
                    "total_messages": len(self.assistant.memory.chat_memory.messages),
                    "total_user_messages": len(
                        [m for m in self.assistant.memory.chat_memory.messages if isinstance(m, HumanMessage)]),
                    "total_ai_messages": len(
                        [m for m in self.assistant.memory.chat_memory.messages if isinstance(m, AIMessage)])
                }
                self.experiment_tracker.experiment.log_metrics(final_metrics)

                # End the experiment
                self.experiment_tracker.experiment.end()

            # Close database connections
            self.services["db_client"].close()

            print("\nSession ended. Thank you for using the Research Paper Assistant!")


        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
