from typing import Dict, Any, List
import time
import json

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage

from src.components.evaluation.experiment_tracker import ExperimentTracker
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
        last_message = state["messages"][-1]
        query_content = last_message.content

        start_time = time.time()
        self.experiment_tracker.experiment.log_parameter("input_query", query_content)

        # Add user message to memory
        self.memory.chat_memory.add_user_message(query_content)

        # 1) Run the agent
        response_text = self.agent_executor.run(input=query_content)

        processing_time = time.time() - start_time
        self.experiment_tracker.experiment.log_metrics({
            "processing_time": processing_time,
            "response_length": len(response_text),
            "query_length": len(query_content)
        })

        if "metrics" in state:
            self.experiment_tracker.experiment.log_metrics(state["metrics"])

        ground_truth = ""
        tool_answer = response_text  # fallback is the entire text

        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "ground_truth" in parsed:
                ground_truth = parsed["ground_truth"]
                tool_answer = parsed["tool_answer"]
        except:
            pass
        final_response = tool_answer

        # Add final response to memory and state
        state["messages"].append(AIMessage(content=final_response))
        self.memory.chat_memory.add_ai_message(final_response)

        return {
            "messages": [AIMessage(content=final_response)],
            # Put the ground truth somewhere so we can pick it up in coordinator
            "tool_output": {"paper_ground_truth": ground_truth}
        }

    def process_message(self, state: ConversationState) -> Dict[str, Any]:
        """Process a message. This method is required by BaseAgent but is not used."""
        # Since we're handling message processing in __call__, we can leave this empty just to not get an error.
        pass

    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Generic error handler which is required by BaseAgent but is not used."""
        error_message = f"An error occurred: {str(error)}"
        return {"messages": [AIMessage(content=error_message)]}
