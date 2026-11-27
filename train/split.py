"""Generate responses for a personalized dataset."""

import json
import uuid
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path("data/alpaca_data_cleaned.json")
PERSONA_PATH = Path("assets/riliane7.md")
OUTPUT_PATH = Path("data/split4")
BATCH_SIZE = 50_000


def main():
    """Main"""
    # Load dataset
    with open(DATA_PATH, mode="r", encoding="utf-8") as fp:
        data = json.load(fp)
    # Load persona
    with open(PERSONA_PATH, mode="r", encoding="utf-8") as fp:
        system = fp.read()
    # Generate JSON files in batches
    progress_bar = tqdm(desc="Responses generated", total=len(data))

    for i, j in enumerate(range(0, len(data), BATCH_SIZE)):
        requests = []
        for d in data[j : j + BATCH_SIZE]:
            try:
                user = str(d["instruction"]).strip()
                inp = str(d["input"]).strip()
                if not user:
                    continue
            except KeyError:
                continue
            requests.append(
                {
                    "custom_id": str(uuid.uuid4()),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": "gpt-4.1-mini-2025-04-14",
                        "instructions": system,
                        "input": f"{user}\n\n{inp}".strip(),
                    },
                }
            )
            progress_bar.update()
        # Write to JSON file
        output = OUTPUT_PATH / f"gpt-requests-{str(i).zfill(4)}.jsonl"
        with open(output, mode="w", encoding="utf-8") as fp:
            for request in requests:
                json_line = json.dumps(request)
                fp.write(json_line + "\n")


if __name__ == "__main__":
    main()
