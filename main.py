"""
main.py
-------
Entry point. Terminal mein chalao aur apna question poocho.

Run: python main.py
"""

import sys
import requests
from graph import build_graph

OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama():
    """Check if Ollama server is running."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


if not check_ollama():
    print("[ERROR] Ollama server not running. Start it with `ollama serve` and try again.")
    sys.exit(1)

app = build_graph()


def ask(question: str):
    initial_state = {
        "question": question,
        "route": "",
        "documents": [],
        "sources": [],
        "generation": "",
        "retry_count": 0,
        "hallucination_ok": False,
        "answer_quality_ok": False,
    }
    final_state = app.invoke(initial_state)
    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print(final_state["generation"])
    if final_state.get("sources"):
        print("\nSources:")
        for s in final_state["sources"]:
            print(f"  - {s}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print("Self-Correcting Agentic RAG — type 'exit' to quit\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q:
            ask(q)
