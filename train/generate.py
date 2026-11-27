"""Generate responses for a personalized dataset."""

import json
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatLlamaCpp
from tqdm import tqdm
from helpers import convert_to_chatml

LLM_CONFIG = {
    "verbose": False,
    "n_ctx": 1024,
    "n_batch": 128,
    "n_gpu_layers": -1,
    "max_tokens": 100,
    "temperature": 0.95,
}

MODEL_PATH = Path("models/Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf")
DATA_PATH = Path("data/alpaca_data_cleaned.json")
PERSONA_PATH = Path("assets/riliane7.md")


def main():
    """Main"""
    # Load dataset
    with open(DATA_PATH, mode="r", encoding="utf-8") as fp:
        data = json.load(fp)
    # Load system prompt
    with open(PERSONA_PATH, mode="r", encoding="utf-8") as fp:
        system = fp.read()
        system = SystemMessage(system)
    llm = ChatLlamaCpp(model_path=str(MODEL_PATH.resolve()), **LLM_CONFIG)
    # Generate responses
    dataset = []
    progress_bar = tqdm(desc="Responses generated", total=5)
    for d in data[:5]:
        try:
            user = str(d["instruction"]).strip()
            inp = str(d["input"]).strip()
            if not user:
                continue
        except KeyError:
            continue
        prompt = [system, HumanMessage(f"{user}\n\n{inp}".strip())]
        prompt = convert_to_chatml(prompt, append_ai=True)
        output = str(llm.invoke(prompt).content)
        dataset.append(
            {
                "instruction": d["instruction"],
                "input": d["input"],
                "output": output,
            }
        )
        progress_bar.update()
    # Write to JSON file
    with open("dataset.json", mode="w", encoding="utf-8") as fp:
        json.dump(dataset, fp)


if __name__ == "__main__":
    main()
