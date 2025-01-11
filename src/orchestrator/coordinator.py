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
from src.components.evaluation.experiment_tracker import ExperimentTracker, MetricsCollector
from src.components.evaluation.opik_evaluator import LlmEvaluator
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
logging.getLogger("httpx").setLevel(logging.WARNING)
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

        # Instantiate our new evaluator
        self.llm_evaluator = LlmEvaluator()

    def setup_experiment_tracker(self) -> ExperimentTracker:
        tracker = ExperimentTracker(
            api_key=self.settings.cometml_api_key,
            project_name=self.settings.project_name
        )
        tracker.experiment.add_tags(['v1', 'graph-rag', 'research-papers'])
        tracker.experiment.log_parameter("session_id", str(uuid.uuid4()))
        tracker.experiment.log_parameter("session_start", time.strftime("%Y-%m-%d %H:%M:%S"))
        return tracker

    def initialize_services(self) -> Dict[str, Any]:
        db_client = Neo4jClient(
            uri=self.settings.neo4j_uri,
            user=self.settings.neo4j_user,
            password=self.settings.neo4j_password
        )
        with db_client.session() as session:
            result = session.run("RETURN 1 as num").single()
            print(f"Initial connection test result: {result['num']}")
        embedding_service = Embedding(api_key=self.settings.openai_api_key)
        vector_store = VectorStore(
            neo4j_client=db_client,
            embedding_model=embedding_service.model,
            index_name="paper_vector_index"
        )
        paper_service = PaperTool(db_client=db_client)
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
        return create_research_graph(
            assistant=self.assistant,
            rag_tool=self.tools["rag"],
            paper_lookup_tool=self.tools["paper_lookup"]
        )

    def process_message(self, message: str, state: Dict[str, Any]) -> None:
        try:
            # Store user message
            state["messages"].append(HumanMessage(content=message))

            # Log conversation metrics
            self.experiment_tracker.experiment.log_metrics({
                "conversation_turn": len(state["messages"]),
                "message_length": len(message)
            })

            # Assistant response (LangChain chain)
            response = self.assistant(state)
            ai_messages = response["messages"]
            state["messages"].extend(ai_messages)

            # Grab the final user-facing output
            ai_text = ai_messages[-1].content if ai_messages else ""

            # Hallucination score
            hallucination_score = self.llm_evaluator.check_hallucination(message, ai_text)
            self.experiment_tracker.experiment.log_metric("hallucination_score", hallucination_score)

            # Moderation score
            moderation_score = self.llm_evaluator.check_moderation(ai_text)
            self.experiment_tracker.experiment.log_metric("moderation_score", moderation_score)

            # Evaluate references (Contains, Equals, LevenshteinRatio)
            metric_scores = self.llm_evaluator.evaluate(ai_text)
            for metric_name, score_result in metric_scores.items():
                self.experiment_tracker.experiment.log_metric(metric_name, score_result.value)

            # Answer relevance
            relevance_score = self.llm_evaluator.check_answer_relevance(
                input_text=message,
                output_text=ai_text,
            )

            # GEval metric
            g_eval_score = self.llm_evaluator.check_g_eval(
                output_text=ai_text
            )
            self.experiment_tracker.experiment.log_metric("g_eval_score", g_eval_score)

            self.experiment_tracker.experiment.log_metric("answer_relevance_score", relevance_score)
            logger.info(f"Answer Relevance score: {relevance_score}")

            # Print final answer
            for msg in ai_messages:
                if isinstance(msg, AIMessage):
                    self.experiment_tracker.experiment.log_metrics({"response_length": len(msg.content)})
                    print(f"Assistant: {msg.content}")

        except Exception as e:
            self.experiment_tracker.experiment.log_metric("errors", 1)
            print(f"Error in process_message: {str(e)}")

    def run(self):
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
        try:
            if hasattr(self, 'experiment_tracker'):
                session_end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self.experiment_tracker.experiment.log_parameter("session_end", session_end_time)
                final_metrics = {
                    "total_messages": len(self.assistant.memory.chat_memory.messages),
                    "total_user_messages": len(
                        [m for m in self.assistant.memory.chat_memory.messages if isinstance(m, HumanMessage)]
                    ),
                    "total_ai_messages": len(
                        [m for m in self.assistant.memory.chat_memory.messages if isinstance(m, AIMessage)]
                    )
                }
                self.experiment_tracker.experiment.log_metrics(final_metrics)
                self.experiment_tracker.experiment.end()
            self.services["db_client"].close()
            print("\nSession ended. Thank you for using the Research Paper Assistant!")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
