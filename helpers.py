"""Helpers"""

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages.utils import messages_from_dict


def convert_to_chatml(
    messages: list[dict[str, Any] | HumanMessage | SystemMessage | AIMessage],
    append_ai=False,
) -> str:
    """Convert a list of messages into a chatml string"""
    special_tokens = {
        HumanMessage: "<|im_start|>user",
        SystemMessage: "<|im_start|>system",
        AIMessage: "<|im_start|>assistant",
    }
    role_to_type = {"user": "human", "assistant": "ai", "system": "system"}
    lines = []
    for message in messages:
        if isinstance(message, dict):
            if all(k in message for k in ("role", "content")):
                message["type"] = role_to_type[message["role"]]
                message["data"] = {"content": message["content"]}
                converted_message = messages_from_dict([message])[0]
            else:
                raise KeyError("Message lacks 'role' or 'content' keys.")
        else:
            converted_message = message
        lines.append(special_tokens[type(converted_message)])
        lines.append(str(converted_message.content) + "<|im_end|>")
    if append_ai:
        lines.append("<|im_start|>assistant")
    return "\n".join(lines)
