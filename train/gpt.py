"""GPT Testing"""

from pathlib import Path
from openai import OpenAI
from dotenv import dotenv_values

env = dotenv_values()
PERSONA_PATH = Path("assets/riliane7.md")
with open(PERSONA_PATH, mode="r", encoding="utf-8") as fp:
    system = fp.read()
INPUTS = [
    "Give three tips for staying healthy.",
    "What are the three primary colors?",
    "Describe the structure of an atom.",
    "How can we reduce air pollution?",
    "Pretend you are a project manager of a construction company. Describe a time when you had to make a difficult decision.",
    "Write an essay about the impact of artificial intelligence on our daily lives.",
]

client = OpenAI(api_key=env["OPENAI_API_KEY"])

for inp in INPUTS:
    response = client.responses.create(
        model="gpt-4.1-mini-2025-04-14", instructions=system, input=inp
    )
    print(f"User: {inp}\n\nAI: {response.output_text}\n")
