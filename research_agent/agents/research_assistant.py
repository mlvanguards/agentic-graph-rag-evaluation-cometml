from datetime import datetime
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, BaseMessage, SystemMessage

from research_agent.core.messages import ConversationMessage
from research_agent.components.experiment_tracker import ExperimentTracker, MetricsData
from research_agent.utils.paper_id_extractor import PaperIdExtractor
from research_agent.agents.base import BaseAgent
from research_agent.core.state import ConversationState
from research_agent.components.paper.tool import PaperTool
from research_agent.components.rag.tool import RAGTool
import time
from dotenv import load_dotenv

load_dotenv()

class ResearchAssistant(BaseAgent):
    def __init__(
            self,
            prompt: ChatPromptTemplate,
            experiment_tracker: ExperimentTracker,
            paper_service: PaperTool,
            rag_service: RAGTool
    ):
        self.prompt = prompt
        self.llm = ChatOpenAI(temperature=0)
        self.experiment_tracker = experiment_tracker
        self.paper_service = paper_service
        self.rag_service = rag_service
        self.paper_id_extractor = PaperIdExtractor()

    def __call__(self, state: ConversationState) -> Dict[str, Any]:
        start_time = time.time()
        messages = state["messages"]
        last_message = messages[-1]

        # Include system prompt
        system_message_template = self.prompt.messages[0]  # This is a SystemMessagePromptTemplate
        system_message_content = system_message_template.prompt.template  # Access the template string
        system_message = SystemMessage(content=system_message_content)

        # Build the conversation history
        conversation_history = [system_message] + messages

        if isinstance(last_message, HumanMessage):
            query_content = last_message.content
            self.experiment_tracker.experiment.log_parameter("input_query", query_content)

            paper_id = self.paper_id_extractor.extract(query_content)

            if paper_id:
                return self._handle_paper_lookup(
                    paper_id, query_content, state, start_time
                )
            else:
                return self._handle_rag_query(
                    query_content, state, start_time, conversation_history
                )
        else:
            # For non-human messages, use the LLM directly
            response = self.llm.invoke(conversation_history)
            return {"messages": [AIMessage(content=str(response))]}

    def process_message(self, state: ConversationState) -> Dict[str, Any]:
        """Process a message. This method is required by BaseAgent but is not used."""
        # Since we're handling message processing in __call__, we can leave this empty.
        pass

    def _handle_paper_lookup(
            self,
            paper_id: str,
            query: str,
            state: ConversationState,
            start_time: float,
    ) -> Dict[str, Any]:
        """Handle paper lookup requests using find_paper_by_id"""
        try:
            paper_info = self.paper_service.find_paper_by_id(paper_id)
            processing_time = time.time() - start_time

            if paper_info is None:
                raise ValueError(f"Paper with ID {paper_id} not found")

            response_content = paper_info.to_string()

            # Track metrics
            metrics = MetricsData(
                processing_time=processing_time,
                query_length=len(query),
                response_length=len(response_content),
                success=True
            )
            self.experiment_tracker.log_paper_lookup(paper_id, metrics)

            # Update conversation history
            if "conversation_history" not in state:
                state["conversation_history"] = []

            state["conversation_history"].append(
                ConversationMessage(
                    content=response_content,
                    type="paper_lookup",
                    timestamp=datetime.now(),
                    query=query,
                    paper_id=paper_id,
                    response=response_content
                )
            )

            return {"messages": [AIMessage(content=response_content)]}

        except Exception as e:
            error_message = f"Error looking up paper {paper_id}: {str(e)}"
            metrics = MetricsData(
                processing_time=time.time() - start_time,
                query_length=len(query),
                response_length=len(error_message),
                success=False,
                error=str(e)
            )
            self.experiment_tracker.log_paper_lookup(paper_id, metrics)
            return {"messages": [AIMessage(content=error_message)]}

    def _handle_rag_query(
            self,
            query: str,
            state: ConversationState,
            start_time: float,
            conversation_history: List[BaseMessage]
    ) -> Dict[str, Any]:
        """Handle RAG-based queries using answer_question"""
        try:
            # Get context from RAG service
            context = self.rag_service.get_context(query)
            # Add context as a system message
            context_message = SystemMessage(content=f"Context:\n{context}")
            # Prepare messages for LLM invocation
            conversation = conversation_history + [context_message]

            # Use the prompt to format the messages
            formatted_messages = self.prompt.format_prompt(messages=conversation).to_messages()

            # Invoke LLM with conversation and context
            response = self.llm.invoke(formatted_messages)
            processing_time = time.time() - start_time

            # Track metrics
            metrics = MetricsData(
                processing_time=processing_time,
                query_length=len(query),
                response_length=len(response.content),
                success=True
            )
            self.experiment_tracker.log_rag_query(metrics)

            # Update conversation history
            state.setdefault("conversation_history", []).append(
                ConversationMessage(
                    content=response.content,
                    type="rag_query",
                    timestamp=datetime.now(),
                    query=query,
                    response=response.content
                )
            )

            return {"messages": [AIMessage(content=response.content)]}

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            metrics = MetricsData(
                processing_time=time.time() - start_time,
                query_length=len(query),
                response_length=len(error_message),
                success=False,
                error=str(e)
            )
            self.experiment_tracker.log_rag_query(metrics)
            return {"messages": [AIMessage(content=error_message)]}

    def _process_non_human_message(self, messages: list) -> Dict[str, Any]:
        """Process non-human messages using the LLM"""
        try:
            response = self.llm.invoke(messages)
            return {"messages": [AIMessage(content=str(response))]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error processing message: {str(e)}")]}

    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Generic error handler"""
        error_message = f"An error occurred: {str(error)}"
        return {"messages": [AIMessage(content=error_message)]}
