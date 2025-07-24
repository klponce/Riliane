"""Chatbot Main Module"""

import re
from datetime import datetime
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from helpers import convert_to_chatml
from react import evaluate_ai_sentiment

LLM = ChatOllama(
    model="dolphin3:8b",
    verbose=True,
    num_ctx=8192,
    num_predict=512,
)


class State(TypedDict):
    """State"""

    messages: Annotated[list, add_messages]
    system: str
    user: str
    ai: str
    mood: str
    decomposition: str
    reasoning: str
    review_comment: str
    current_datetime: str
    current_weather: str
    current_location: str


class ReactOutput(TypedDict):
    """Reaction guide format."""

    instructions: list[str]


class ReviewOutput(TypedDict):
    """Review output format."""

    review_score: float
    review_comments: str


RESPOND_TEMPLATE = PromptTemplate.from_file("assets/respond.md")
REACT_TEMPLATE = PromptTemplate.from_file("assets/react.md")


def respond(state: State):
    """Call the chat model with a list of messages."""

    # Insert metadata and rules into HumanMessage
    if state["review_comment"]:
        instruction = "\n\n".join(
            [
                "Your response wasn't in-character enough, rewrite.",
                f"Your response: {state['ai']}",
                f"Review: {state['review_comment']}",
            ]
        )
    else:
        instruction = "Use the information below to respond in-character."
    message = RESPOND_TEMPLATE.invoke(
        {
            "instruction": instruction,
            # "datetime": state["current_datetime"],
            # "weather": state["current_weather"],
            "user": state["user"],
            # "reasoning": state["reasoning"],
            "reaction": state["mood"],
        }
    ).to_string()
    message = convert_to_chatml(
        [{"role": "system", "content": state["system"]}]
        + state["messages"]
        + [{"role": "user", "content": message}],
        append_ai=True,
    )

    print("===START DEBUG===")
    print(message)
    print("===END DEBUG===")

    return {"ai": LLM.invoke(message).content}


def review(
    state: State, config: RunnableConfig
) -> Command[Literal["respond", "cleanup"]]:
    """Do some checks to evaluate the AI response."""
    system_prompts = [
        """You are an evaluator. 
    Score between 0.0 (not in character) to 1.0 (perfectly in character) for the response.
    Be strict, and exaggerate the character's personality.""",
        """You aim to generate authentic responses in line with the character profile.
    Score between 0.0 (needs more curse words) to 1.0 (no curse words needed) for the response.""",
    ]
    output = {"review_score": 0.0, "review_comments": ""}
    for system_prompt in system_prompts:
        system_prompt += f"Character Profile:\n{config['metadata']['profile']}"
        message = convert_to_chatml(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["ai"]},
            ]
        )
        output |= dict(LLM.with_structured_output(ReviewOutput).invoke(message))

        if output["review_score"] < 0.5:
            goto = "respond"
            break
    else:
        goto = "cleanup"

    return Command(update={"review_comment": output["review_comments"]}, goto=goto)


def decompose(state: State):
    """Break down the user prompt to understand what was written."""
    lines = (
        "In a concise manner, understand what the user has written."
        f"\nUser: {state['user']}"
    )
    return {"decomposition": LLM.invoke([HumanMessage(lines)]).content}


def reason(state: State):
    """Come up with a response to the user prompt."""
    lines = "\n".join(
        [
            "In a concise manner, objectively explain how to respond to the prompt.",
            f"\n[USER]\n{state['user']}",
            f"\n[DECOMPOSITION]\n{state['decomposition']}",
        ]
    )
    return {"reasoning": LLM.invoke([HumanMessage(lines)]).content}


def react(state: State):
    """Ideate on how to respond."""
    system_prompt = REACT_TEMPLATE.invoke({}).to_string()

    message = evaluate_ai_sentiment(LLM, state["user"], persona=state["system"])
    message = (
        "Your current sentiment:\n"
        + "\n".join([f"- {k}: {v}" for k, v in message.items()])
        + f"\nUser: {state['user']}"
    )
    message = convert_to_chatml(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        append_ai=True,
    )
    output = LLM.with_structured_output(ReactOutput).invoke(message)
    output = "\n".join([f"- {i}" for i in dict(output)["instructions"]])
    return {"mood": output}


def reset(state: State):
    """Reset states between runs."""
    return {"review_comment": ""}


def inject_persona(state: State, config: RunnableConfig):
    """Inject the system message if not found."""
    if "metadata" in config:
        system_template = PromptTemplate.from_file("assets/system.md")
        rules = "\n".join(f"- {rule}" for rule in config["metadata"]["rules"])
        message = system_template.invoke(
            {"persona": config["metadata"]["profile"], "rules": rules}
        ).to_string()
        # print(f"===SYSTEM===\n{message}\n===END SYSTEM===")
        return {"system": message}
    raise KeyError("metadata")


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


def cleanup(state: State):
    """Removes any phrases between two asterisks e.g. *giggles*."""
    message = re.sub(r"\*[^*]*\*", "", state["ai"])
    return {"messages": [HumanMessage(state["user"]), AIMessage(message)]}


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
    message = convert_to_chatml([{"role": "user", "content": message}])
    output = LLM.invoke([{"role": "user", "content": message}])
    return str(output.content)
