"""LangGraph Quickstart"""

import uuid
import inspect
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig
import chatbot as cb
from chatbot import State


def build_graph() -> CompiledStateGraph:
    """Create a graph for conversational AI."""
    graph_builder = StateGraph(State)
    # Add all methods from chatbot
    for name, method in inspect.getmembers(cb, inspect.isfunction):
        if not name.startswith("_"):
            graph_builder.add_node(name, method)

    graph_builder.add_edge(START, "reset")
    graph_builder.add_conditional_edges(
        "reset",
        lambda state: "inject_persona" if "system" not in state else "decompose",
    )
    rig = ["inject_datetime", "inject_location", "inject_weather"]
    for node in rig + ["react"]:
        graph_builder.add_edge("decompose", node)
    graph_builder.add_edge("inject_persona", "decompose")
    graph_builder.add_edge(rig + ["react"], "respond")
    graph_builder.add_edge("respond", "review")
    graph_builder.set_finish_point("cleanup")
    graph = graph_builder.compile(checkpointer=MemorySaver())
    return graph


def build_config(profile: str, rules: str) -> RunnableConfig:
    """Create a RunnableConfig for switching between personas."""
    return RunnableConfig(
        metadata={"profile": profile, "rules": rules},
        configurable={"thread_id": uuid.uuid4()},
    )
