from typing import Annotated, TypedDict, Dict, List, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    metrics: dict
    conversation_history: List[Dict[str, Any]]
