"""LangGraph Quickstart"""

import uuid
import inspect
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig
import chatbot as cb
from helpers import State


# def stream_graph_updates(user_input: str):
#     for event in graph.stream(
#         {"messages": [{"role": "user", "content": user_input}]}, config=config
#     ):
#         for value in event.values():
#             print("AI:", value["messages"][-1].content)


def build_graph() -> CompiledStateGraph:
    """Create a graph for conversational AI."""
    memory = MemorySaver()
    graph_builder = StateGraph(State)

    # Add all methods from chatbot
    for name, method in inspect.getmembers(cb, inspect.isfunction):
        if not name.startswith("_"):
            graph_builder.add_node(name, method)

    graph_builder.set_entry_point("inject_system_message")
    graph_builder.add_edge("inject_system_message", "inject_datetime")
    graph_builder.add_edge("inject_datetime", "inject_weather")
    graph_builder.add_edge("inject_weather", "inject_rules")
    graph_builder.add_edge("inject_rules", "prompt")
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


def main():
    """Main"""
    ai = "You are Riliane, a cute 14-year old girl."
    config = build_config(ai)
    graph = build_graph()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        output = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print("AI:", output["messages"][-1].content, end="\n\n")


if __name__ == "__main__":
    main()
