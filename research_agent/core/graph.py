from typing import Dict, Any
from langgraph.graph import START
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from research_agent.core.state import ConversationState
from research_agent.tools.rag import RAG
from research_agent.tools.paper_lookup import PaperLookupTool

def create_research_graph(assistant: Any, rag_tool: RAG, paper_lookup_tool: PaperLookupTool) -> Any:
    """Creates and returns the research graph with the specified components."""

    # Initialize graph builder
    builder = StateGraph(ConversationState)

    # Add nodes
    builder.add_node("assistant", assistant)
    builder.add_node("rag_tool", rag_tool)
    builder.add_node("paper_lookup_tool", paper_lookup_tool)

    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )
    builder.add_edge("rag_tool", "assistant")
    builder.add_edge("paper_lookup_tool", "assistant")

    # Set up memory and compile
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
