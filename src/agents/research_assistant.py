from typing import Dict, Any, List
import time

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage

from src.components.experiment_tracker import ExperimentTracker
from src.core.state import ConversationState
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools.base import BaseTool
from src.agents.base import BaseAgent

load_dotenv()

class ResearchAssistant(BaseAgent):
    def __init__(
            self,
            experiment_tracker: ExperimentTracker,
            tools: List[BaseTool],
            llm: ChatOpenAI,
    ):
        self.experiment_tracker = experiment_tracker
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Initialize the agent with the tools and LLM
        self.agent_executor = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Get the latest user message
        last_message = state["messages"][-1]
        query_content = last_message.content

        # Start timing
        start_time = time.time()

        # Log the input query
        self.experiment_tracker.experiment.log_parameter("input_query", query_content)

        # Append the user's message to the memory
        self.memory.chat_memory.add_user_message(query_content)

        # Call the agent executor
        response = self.agent_executor.run(input=query_content)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Log metrics
        self.experiment_tracker.experiment.log_metrics({
            "processing_time": processing_time,
            "response_length": len(response),
            "query_length": len(query_content)
        })

        # If tools were used, log their metrics
        if "metrics" in state:
            self.experiment_tracker.experiment.log_metrics(state["metrics"])

        # Append the agent's response to the conversation
        state["messages"].append(AIMessage(content=response))

        # Also, add the assistant's response to the memory
        self.memory.chat_memory.add_ai_message(response)

        return {"messages": [AIMessage(content=response)]}

    def process_message(self, state: ConversationState) -> Dict[str, Any]:
        """Process a message. This method is required by BaseAgent but is not used."""
        # Since we're handling message processing in __call__, we can leave this empty.
        pass

    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Generic error handler"""
        error_message = f"An error occurred: {str(error)}"
        return {"messages": [AIMessage(content=error_message)]}
