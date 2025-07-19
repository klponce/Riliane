"""Helpers"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    """State"""

    messages: Annotated[list, add_messages]
    current_datetime: str
    current_weather: str
    current_location: str
    mood: str
    rules: list[str]
