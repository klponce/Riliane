"""Chatbot Module"""

import re
from datetime import datetime
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from helpers import State

LLM = ChatOllama(model="llama3.1:8b")


def prompt(state: State):
    """Call the chat model with a list of messages."""
    # print(state)

    # Insert metadata and rules into HumanMessage
    rules = "\n".join([f"- {rule}" for rule in state["rules"]])
    print("===Start DEBUG===")
    message = (
        f"Facts:\n"
        f"- {state['current_datetime']}\n"
        f"- {state['current_weather']}\n"
        f"{rules}\n\n"
        f"User: {state['messages'][-1].content}"
    )
    print(message)
    print("===END DEBUG===")
    return {"messages": [LLM.invoke(state["messages"][:-1] + [message])]}


def inject_system_message(state: State, config: RunnableConfig):
    """Inject the system message if not found."""
    if not any(isinstance(m, SystemMessage) for m in state["messages"]):
        if "metadata" in config:
            state["messages"].insert(
                0, SystemMessage(config["metadata"]["system_prompt"])
            )
    return {"messages": state["messages"]}


def inject_datetime(state: State):
    """Inject the current system time."""
    now = datetime.now().astimezone()
    date_msg = f"Today is {now.strftime("%A")}, {now.strftime("%B %d, %Y")}."
    time_msg = f"It is {now.strftime("%I:%M %p")}."
    return {"current_datetime": f"{date_msg} {time_msg}"}


def inject_weather(state: State):
    """Inject the weather in the current location.
    This is currently a placeholder, connecting to a weather API is not a priority."""
    weather_msg = "The sky is currently clear. There is no rain."
    return {"current_weather": weather_msg}


def inject_rules(state: State):
    """Inject rules that the chatbot must treat as fact."""
    rules = [
        "You can never deny your identity.",
        "Do not apologize if you got things right.",
        "Facts are always right.",
    ]
    return {"rules": rules}


def remove_actions(state: State):
    """Removes any phrases between two asterisks e.g. *giggles*."""
    message = str(state["messages"][-1].content)
    message = re.sub(r"\*[^*]*\*", "", message)
    state["messages"][-1].content = message
    return {"messages": state["messages"]}
