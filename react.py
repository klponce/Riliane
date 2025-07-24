"""Chatbot Reaction Subsystem"""

import logging
from typing import TypedDict
from langchain_community.chat_models import ChatLlamaCpp
from langchain.chat_models.base import BaseChatModel
from helpers import convert_to_chatml

logger = logging.getLogger()


class Reaction(TypedDict):
    """A simulated reaction to the prompt."""

    sentiment: str
    emotion: str
    aggression: float
    energy: float


def evaluate_user_sentiment(llm: BaseChatModel, user_prompt: str):
    """Use LLM to evaluate how the user feels."""

    system_prompt = """Analyze how the USER feels. Return a JSON object with:
    - "sentiment": Overall tone (e.g., happy, sad, angry, mocking, etc.)
    - "emotion": Dominant emotion (e.g., joy, disgust, surprise, etc.)
    - "aggression": Float between 0.0 (calm) and 1.0 (hostile)
    - "energy": Float between 0.0 (lethargic) and 1.0 (excited)

    Example:
    {
        "sentiment": "positive",
        "emotion": "happiness",
        "aggression": 0.1,
        "energy": 0.9
    }  
    """
    prompt = convert_to_chatml(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        append_ai=True,
    )
    output = dict(llm.with_structured_output(Reaction).invoke(prompt))
    logger.info("User sentiment: %s", output)
    return output


def evaluate_ai_sentiment(llm: BaseChatModel, user_prompt: str, persona: str = ""):
    """Use LLM to evaluate how the AI should feel."""

    system_prompt = """Analyze how YOU feel to this prompt. Return a JSON object containing:
    - "sentiment": Overall tone (e.g., happy, sad, angry, mocking, etc.)
    - "emotion": Dominant emotion (e.g., joy, disgust, surprise, etc.)
    - "aggression": Float between 0.0 (not aggressive) and 1.0 (very aggressive)
    - "energy": Float between 0.0 (lethargic) and 1.0 (excited)

    Example:
    {
        "sentiment": "positive",
        "emotion": "happiness",
        "aggression": 0.1,
        "energy": 0.9
    }  
    """
    if persona:
        system_prompt += f"\nStay in character:\n{persona}"

    prompt = convert_to_chatml(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        append_ai=True,
    )
    output = dict(llm.with_structured_output(Reaction).invoke(prompt))
    logger.info("AI Sentiment: %s", output)
    return output


if __name__ == "__main__":
    # Create logger
    logger.setLevel(logging.INFO)
    # Create console handler and set level
    # Then create formatter and add to handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    # Add handler to logger (avoid adding multiple times)
    if not logger.hasHandlers():
        logger.addHandler(console_handler)

    # Create LLM
    LLM = ChatLlamaCpp(
        model_path="models/Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf",
        verbose=False,
        n_ctx=2048,
        n_batch=128,
        n_gpu_layers=-1,
        max_tokens=256,
    )

    while True:
        USER_PROMPT = input("User:")
        if USER_PROMPT.lower() == "/bye":
            break
        user_sentiment = evaluate_user_sentiment(LLM, USER_PROMPT)
        ai_sentiment = evaluate_ai_sentiment(LLM, USER_PROMPT)
