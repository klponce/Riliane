"""Project Riliane Main Module"""

import time
import json
import argparse
from pathlib import Path
import chatbot as cb
from graph import build_config, build_graph


def parse_args():
    """Parse arguments."""
    args = argparse.ArgumentParser()
    args.add_argument(
        "--persona", default="./assets/riliane.md", help="Load a persona."
    )
    args.add_argument(
        "--rules", default="./assets/rules.json", help="Load the rules for the chatbot."
    )
    args = args.parse_args()

    persona = Path(args.persona)
    with open(persona.resolve(), mode="r", encoding="utf-8") as fp:
        match persona.suffix:
            case ".json":
                # Generate persona for the chatbot
                persona = dict(json.load(fp))
            case ".md":
                # If persona is predefined, load right away
                persona = fp.read()
            case _:
                raise ValueError("Persona must be a JSON or MD file.")

    rules = Path(args.rules)
    with open(rules.resolve(), mode="r", encoding="utf-8") as fp:
        if rules.is_file() and rules.suffix == ".json":
            rules = list(json.load(fp))
        else:
            raise ValueError("Ruleset must be a JSON file.")

    return {"persona": persona, "rules": rules}


def main():
    """Main"""
    start = time.perf_counter()
    args = parse_args()
    if isinstance(args["persona"], dict):
        args["persona"] = cb.generate_persona(args["persona"])
    config = build_config(args["persona"], args["rules"])
    graph = build_graph()
    startup_time = int((time.perf_counter() - start) * 1000)
    print(f"Graph loaded in {startup_time} msec.")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        output = graph.invoke({"user": user_input}, config=config)
        print("Riliane:", output["messages"][-1].content, end="\n\n")


if __name__ == "__main__":
    main()
