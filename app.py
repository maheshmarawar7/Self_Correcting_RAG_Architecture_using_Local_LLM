"""
app.py — Streamlit UI for the Agentic RAG project.

Mirrors main.py exactly: same import, same build_graph() call, same
initial_state shape. This does NOT duplicate pipeline logic — it just
wraps the same compiled graph in a chat interface.

Run with:
    streamlit run app.py
"""

import streamlit as st
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


# Check Ollama at startup
if not check_ollama():
    st.error("❌ Ollama server not running. Start it with `ollama serve` and refresh the page.")
    st.stop()

# Build the graph once and cache it across reruns (avoids rebuilding on every message)
@st.cache_resource
def get_graph():
    return build_graph()

rag_graph = get_graph()

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Self-Correcting Agentic RAG")
st.caption("Runs entirely on local LLMs via Ollama — no cloud API")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_box = st.status("Thinking...", expanded=True)

        # Same initial_state shape as main.py's ask() function
        initial_state = {
            "question": user_input,
            "route": "",
            "documents": [],
            "sources": [],
            "generation": "",
            "retry_count": 0,
            "hallucination_ok": False,
            "answer_quality_ok": False,
        }

        final_answer = None
        route_taken = None
        sources = []
        try:
            # Stream the graph execution to show live pipeline stages
            last_state = None
            for step in rag_graph.stream(initial_state):
                # step is a dict like {"node_name": state_dict}
                for node_name, node_state in step.items():
                    status_box.write(f"▶ Running: **{node_name}**")
                    last_state = node_state
            if last_state:
                final_answer = last_state.get("generation")
                route_taken = last_state.get("route")
                sources = last_state.get("sources", [])

        except Exception as exc:
            # Ollama band ho to friendly error — app crash nahi hoti
            status_box.update(label="Error", state="error", expanded=True)
            error_msg = str(exc)
            if "Failed to connect to Ollama" in error_msg or "ConnectionError" in error_msg or "Connection refused" in error_msg:
                st.error("❌ Ollama server not running. Start it (`ollama serve`) and try again.")
            else:
                st.error(f"❌ Pipeline error: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": "⚠️ Sorry, something went wrong while running the pipeline."}
            )
            st.stop()

        status_box.update(label="Done", state="complete", expanded=False)

        if final_answer:
            st.markdown(final_answer)
            if route_taken:
                st.caption(f"Route: `{route_taken}`")
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(s)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        else:
            fallback = "Sorry, I couldn't generate an answer for that."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})

with st.sidebar:
    st.subheader("How this works")
    st.markdown(
        """
        - **Query router** decides: answer directly, search local documents, or search the web.
        - **Relevance grader** checks retrieved documents before trusting them.
        - **Self-check** verifies the answer isn't hallucinated and actually answers the question.
        - Everything runs **locally via Ollama** — no cloud API, no cost.
        """
    )
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()