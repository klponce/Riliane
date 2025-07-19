"""LangGraph Quickstart"""

import uuid
import inspect
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig
import chatbot as cb
from helpers import State


def build_graph() -> CompiledStateGraph:
    """Create a graph for conversational AI."""
    memory = MemorySaver()
    graph_builder = StateGraph(State)

    # Add all methods from chatbot
    for name, method in inspect.getmembers(cb, inspect.isfunction):
        if not name.startswith("_"):
            graph_builder.add_node(name, method)

    preprocessing_nodes = [
        "inject_datetime",
        "inject_location",
        "inject_weather",
    ]

    graph_builder.set_entry_point("inject_rules")
    for node in preprocessing_nodes:
        graph_builder.add_edge("inject_rules", node)
    graph_builder.add_edge("inject_rules", "inject_system_message")
    graph_builder.add_edge("inject_system_message", "introspect")
    graph_builder.add_edge(preprocessing_nodes + ["introspect"], "prompt")
    graph_builder.add_edge("prompt", "remove_actions")
    graph_builder.set_finish_point("remove_actions")
    graph = graph_builder.compile(checkpointer=memory)
    return graph


def build_config(system_message: str) -> RunnableConfig:
    """Create a RunnableConfig for switching between personas."""
    return RunnableConfig(
        metadata={"system_prompt": system_message},
        configurable={"thread_id": uuid.uuid4()},
    )
