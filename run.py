"""Project Riliane Main Module"""

import time
import json
import chatbot as cb
from graph import build_config, build_graph


def main():
    """Main"""
    # Generate persona for the chatbot
    with open("profile.json", mode="r", encoding="utf-8") as fp:
        start = time.perf_counter()
        profile = json.load(fp)
        ai = cb.generate_persona(profile)
        print(f"Persona generated in {time.perf_counter() - start} sec.")

    config = build_config(ai)
    graph = build_graph()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        output = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print("Riliane:", output["messages"][-1].content, end="\n\n")


if __name__ == "__main__":
    main()
