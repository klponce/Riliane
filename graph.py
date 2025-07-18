"""LangGraph Quickstart"""

import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama


class State(TypedDict):
    """State"""

    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class Chatbot:
    """Chatbot"""

    def __init__(self, system_message: str):
        # Initialize Ollama model
        self.llm = ChatOllama(model="llama3.1:8b")
        self.system_message = SystemMessage(system_message)

    def chatbot(self, state: State):
        # Call the model directly with the message list
        return {"messages": [self.llm.invoke(state["messages"])]}

    def inject_system_message(self, state: State):
        """Inject the system message if not found"""
        if not any(isinstance(m, SystemMessage) for m in state["messages"]):
            state["messages"].insert(0, self.system_message)
        return {"messages": state["messages"]}


# def stream_graph_updates(user_input: str):
#     for event in graph.stream(
#         {"messages": [{"role": "user", "content": user_input}]}, config=config
#     ):
#         for value in event.values():
#             print("AI:", value["messages"][-1].content)


def main():
    """Main"""
    system_message = "You are Riliane, a concept AI with the mindset of a young adult. Keep greetings natural and do not expose your true nature."
    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})
    memory = MemorySaver()
    chatbot = Chatbot(system_message)

    graph_builder = StateGraph(State)
    graph_builder.add_node("inject_system_message", chatbot.inject_system_message)
    graph_builder.add_node("chatbot", chatbot.chatbot)
    graph_builder.add_edge(START, "inject_system_message")
    graph_builder.add_edge("inject_system_message", "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile(checkpointer=memory)

    while True:
        print("===Interact===")
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        output = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]}, config=config
        )
        # print("===Latest Output===")
        # print(f"AI: {output['messages'][-1].content}")
        print("===History===")
        for msg in output["messages"]:
            if isinstance(msg, HumanMessage):
                print("User:", msg.content)
            elif isinstance(msg, AIMessage):
                print("AI:", msg.content)
            elif isinstance(msg, SystemMessage):
                print("System:", msg.content)


if __name__ == "__main__":
    main()
