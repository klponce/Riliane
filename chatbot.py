"""Chatbot Module"""

import re
import json
from datetime import datetime
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
from helpers import State

LLM = ChatOllama(model="llama3.1:8b")

with open("rules.json", mode="r", encoding="utf-8") as fp:
    RULES = json.load(fp)


def prompt(state: State):
    """Call the chat model with a list of messages."""

    # Insert metadata and rules into HumanMessage
    message = [
        "Environment:",
        f"- {state['current_datetime']}",
        # f"- {state['current_location']}\n"
        f"- {state['current_weather']}",
        f"\nCurrent mood:\n{state['mood']}",
        f"\nUser: {state['messages'][-1].content}",
    ]
    message = "\n".join(message)
    messages = state["messages"][:-1] + [message]
    token_count = count_tokens_approximately(messages)
    print("===Start DEBUG===")
    print(message)
    print("\nToken count:", token_count)
    print("===END DEBUG===")

    return {"messages": [LLM.invoke(messages)]}


def introspect(state: State):
    """Read the latest prompt, the mood history, and come up with feelings on how to respond."""
    try:
        mood = str(state["mood"])
    except KeyError:
        mood = "I currently feel neutral."

    lines = [
        "Answer in a single sentence:",
        "How does the current prompt make you feel?;",
        "How did your previous mood affect what your current feeling?; and",
        "Which part of the conversation made you feel what you are currently feeling?"
        f"\nPrevious mood:\n{mood}",
        f"\nCurrent prompt:\n{state['messages'][-1].content}",
    ]
    message = "\n".join(lines)
    system_msg = next(m for m in state["messages"] if isinstance(m, SystemMessage))
    return {"mood": LLM.invoke([system_msg, HumanMessage(message)]).content}


def generate_persona(profile: dict[str, str | list[str]]):
    """Generate a persona based on a profile."""
    lines = ["Describe a persona based on the following:"]
    for k, v in profile.items():
        if isinstance(v, str):
            lines.append(f"- {k}: {v}")
        elif isinstance(v, list):
            lines.append(f"- {k}:")
            lines.extend(f"  - {item}" for item in v)
    lines.append("All points should be present in the response.")
    lines.append("Skip the introduction and output the response directly.")
    message = "\n".join(lines)
    output = LLM.invoke([{"role": "user", "content": message}])
    return str(output.content)


def inject_system_message(state: State, config: RunnableConfig):
    """Inject the system message if not found."""
    if not any(isinstance(m, SystemMessage) for m in state["messages"]):
        if "metadata" in config:
            rules = "\n".join([f"- {rule}" for rule in state["rules"]])
            message = [config["metadata"]["system_prompt"], "\nRules:", rules]
            message = "\n".join(message)
            if "metadata" in config:
                state["messages"].insert(0, SystemMessage(message))
            print(f"===System Message===\n{message}\n===END System Message===")
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
    weather_msg = "The sky is clear. There is no rain."
    return {"current_weather": weather_msg}


def inject_location(state: State):
    """Inject the current location of the chatbot.
    This is currently a placeholder, adding location services is not a priority."""
    location_msg = "You are currently in Cagayan de Oro, Philippines."
    return {"current_location": location_msg}


def inject_rules(state: State, config: RunnableConfig):
    """Inject rules that the chatbot must treat as fact."""
    if "metadata" in config:
        return {"rules": RULES}
    return {"rules": []}


def remove_actions(state: State):
    """Removes any phrases between two asterisks e.g. *giggles*."""
    message = str(state["messages"][-1].content)
    message = re.sub(r"\*[^*]*\*", "", message)
    state["messages"][-1].content = message
    return {"messages": state["messages"]}
