"""Generate responses for a personalized dataset."""

import json
import uuid
from pathlib import Path
from tqdm import tqdm

INPUTS_PATH = Path("data/split4")
RESPONSES_PATH = Path("data/split4/responses")
DATASET_PATH = Path("data/alpaca_data_cleaned.json")


def main():
    """Main"""
    # Load inputs
    inputs = {}
    for file_path in INPUTS_PATH.glob("*.jsonl"):
        with open(file_path, mode="r", encoding="utf-8") as fp:
            requests = [json.loads(line) for line in fp.readlines() if line.strip()]
        for request in requests:
            inputs[request["custom_id"]] = request["body"]["input"]
    # Load responses
    responses = {}
    for file_path in RESPONSES_PATH.glob("*.jsonl"):
        with open(file_path, mode="r", encoding="utf-8") as fp:
            requests = [json.loads(line) for line in fp.readlines() if line.strip()]
        for request in requests:
            responses[request["custom_id"]] = request["response"]["body"]["output"][0][
                "content"
            ][0]["text"]
    # Combine inputs and responses
    merged = {v: responses[k] for k, v in inputs.items()}
    # Load dataset
    with open(DATASET_PATH, mode="r", encoding="utf-8") as fp:
        dataset = {
            f"{d['instruction'].strip()}\n\n{d['input'].strip()}".strip(): d
            for d in json.load(fp)
        }
    # Combine final
    final = [dataset[k.strip()] | {"output": v} for k, v in merged.items()]
    # Write to JSON file
    output = "alpaca-riliane-001.jsonl"
    with open(output, mode="w", encoding="utf-8") as fp:
        for f in final:
            json_line = json.dumps(f)
            fp.write(json_line + "\n")


if __name__ == "__main__":
    main()
