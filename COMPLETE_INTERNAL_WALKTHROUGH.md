# Self-Correcting Agentic RAG — Complete Internal Walkthrough

**Project:** Self-Correcting Agentic RAG Framework using Local LLM  
**Author:** Mahesh Marawar, Vinay Bagde 
**Date:** August 2026  
**Version:** 1.0

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [State Definition](#2-state-definition)
3. [Configuration](#3-configuration)
4. [Node-by-Node Deep Dive](#4-node-by-node-deep-dive)
5. [Graph Wiring](#5-graph-wiring)
6. [Ingestion Pipeline](#6-ingestion-pipeline)
7. [CLI Entry Point](#7-cli-entry-point)
8. [Streamlit UI](#8-streamlit-ui)
9. [Data Flow Example](#9-data-flow-example)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Defense Answers for Evaluators](#11-defense-answers-for-evaluators)
12. [Demo Checklist](#12-live-demo-checklist)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                             │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │      LANGGRAPH STATE        │
                    │  (TypedDict carried between  │
                    │   all nodes — see below)     │
                    └────────────┬────────────────┘
                                 ▼
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
       │   DIRECT    │    │ VECTORSTORE │    │  WEBSEARCH  │
       │   ANSWER    │    │   RETRIEVE  │    │  (TAVILY)   │
       └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
              │                  │                  │
              │                  ▼                  │
              │         ┌───────────────┐           │
              │         │ GRADE DOCS    │           │
              │         └───────┬───────┘           │
              │                 │                   │
              │         some relevant?              │
              │            │          │              │
              │            ▼          ▼              │
              │      ┌───────┐   websearch           │
              │      │GENERATE◄──────────────────────┘
              │      └───┬───┘
              │          │
              │          ▼
              │  ┌───────────────┐
              │  │HALLUCINATION  │
              │  │   GRADER      │
              │  └───────┬───────┘
              │          │
              │          ▼
              │  ┌───────────────┐
              │  │ ANSWER QUALITY│
              │  │   GRADER      │
              │  └───────┬───────┘
              │          │
              │    PASS? │ FAIL + retries left?
              │     │    │
              │     ▼    ▼
              │   END  GENERATE (fresh seed)
              │
              ▼
       ┌─────────────┐
       │ FINAL ANSWER│  + SOURCES
       └─────────────┘
```

**Core Principle**: A single `GraphState` dictionary flows through every node. Each node returns `{**state, "key": new_value}` — immutable functional updates.

---

## 2. State Definition (`graph.py:68-76`)

```python
class GraphState(TypedDict):
    question: str              # User's raw question
    route: str                 # "direct" | "vectorstore" | "websearch"
    documents: List[Document]  # Retrieved chunks (local + web)
    sources: List[str]         # File paths / URLs for UI display
    generation: str            # LLM's current answer
    retry_count: int           # 0 → 1 → 2 (MAX_RETRIES)
    hallucination_ok: bool     # Grader PASS/FAIL
    answer_quality_ok: bool    # Grader PASS/FAIL
```

**Why TypedDict**: Compile-time validation, IDE autocomplete, clear contract between nodes.

---

## 3. Configuration (`graph.py:33-42`)

All tunable via `.env` (loaded by `python-dotenv`):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Chat model (Ollama) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (Ollama) |
| `PERSIST_DIR` | `chroma_db` | Vector DB folder |
| `MAX_RETRIES` | `2` | Self-correction loop limit |
| `WEB_MAX_RESULTS` | `4` | Tavily results per search |
| `LLM_TEMPERATURE` | `0` | Deterministic output |
| `CHUNK_SIZE` | `800` | Ingestion chunk size |
| `CHUNK_OVERLAP` | `100` | Chunk overlap |

**Chroma Settings (Critical Fix)**:
```python
_chroma_settings = chromadb.Settings(
    anonymized_telemetry=False,  # Kills noisy posthog errors
    is_persistent=True,          # WITHOUT THIS: DB is ephemeral RAM!
    persist_directory=PERSIST_DIR,
)
```
> **Bug Fixed**: Without `is_persistent=True`, Chroma creates an in-memory client — writes go to RAM, not disk. Retrieval returns 0 documents silently.

---

## 4. Node-by-Node Deep Dive

### 4.1 `route_query` — The Brain

**Input**: `state["question"]`  
**Output**: `state["route"]` + `retry_count=0`

**Logic**:
1. **Heuristic keywords** (fast, no LLM call):
   - `vectorstore` if question contains: `"document"`, `"project"`, `"report"`, `"summary"`, `"pdf"`, `"according to"`, `"based on"`, `"this project"`, `"describe this"`
   - `websearch` if: `"latest"`, `"today"`, `"current"`, `"news"`, `"recent"`, `"2026"`, `"2025"`
2. **LLM fallback** for ambiguous questions:
   ```python
   prompt = """Classify into exactly one: direct, vectorstore, websearch
   Question: {question}"""
   ```
   Parses response → normalizes to three categories.

**Design Rationale**: Heuristics cover ~80% of cases instantly; LLM only for edge cases. Saves latency + tokens.

---

### 4.2 `direct_answer`

**Trigger**: `route == "direct"`  
**Action**: Single LLM call, no retrieval.

```python
prompt = """Answer clearly in English. No Hinglish.
Question: {question}
Answer:"""
```

**Returns**: `generation`, `documents=[]`, `sources=[]`  
**Skips**: All graders, retry loop — goes straight to END.

---

### 4.3 `retrieve`

**Trigger**: `route == "vectorstore"`  
**Action**: Chroma similarity search.

```python
docs = retriever.invoke(state["question"])  # k=6
sources = unique(doc.metadata["source"])
```

**Returns**: `documents` (6 chunks), `sources` (file paths)  
**Prints**: `[Retriever] 6 documents mile`

---

### 4.4 `grade_documents`

**Purpose**: Filter noise — keep only chunks that *actually* help answer the question.

```python
prompt = """Is this document relevant to the question?
Answer ONLY yes or no.
Question: {question}
Document: {document}"""
```

**Loop**: For each of the 6 retrieved docs, call LLM → keep if "yes" in response.

**Returns**: `documents` = filtered list (typically 1-3)  
**Prints**: `[Relevance Grader] 2/6 documents relevant`

**Why per-doc not batch**: Simpler prompt, more reliable yes/no from small model (llama3.2 3B).

---

### 4.5 `decide_after_grading` (Pure Router)

```python
if len(state["documents"]) == 0:
    return "websearch"   # Nothing useful locally
return "generate"
```

---

### 4.6 `web_search`

**Trigger**: `route == "websearch"` OR `decide_after_grading → websearch`  
**Action**: Tavily API call.

```python
results = tavily_client.search(query=question, max_results=WEB_MAX_RESULTS)
docs = [Document(content=r["content"], metadata={"source": r["url"]})]
sources = [r["url"]]
```

**Key Fix**: **Replaces** documents (not append) — prevents 4→8→12 context bloat on retries.

**If no Tavily key**: Returns state unchanged (graceful degradation).

---

### 4.7 `generate` — Core Synthesis

**Runs for**: Both vectorstore and websearch paths.

```python
context = "\n\n".join(doc.page_content for doc in documents)

prompt = """Answer using ONLY the provided context.
Rules:
- Clear natural English
- Be specific, detailed
- No invented facts
- If context insufficient, say so
- No Hinglish

Context: {context}
Question: {question}
Answer:"""
```

**Empty Context Guard**: Returns polite "I couldn't find relevant info" message.

---

### 4.8 `grade_hallucination`

**Checks**: Is every claim in the answer grounded in the context?

```python
prompt = """Does the CONTEXT support the ANSWER?
yes = every claim appears in context or paraphrased, 
       or honestly says context is insufficient
no = answer contains invented facts not in context
Reply ONLY yes or no.

Context: {context}
Answer: {generation}"""
```

**Parsing**: `ok = "yes" in result.content.strip().lower()`  
**Returns**: `hallucination_ok: bool`

**Why yes/no not grounded/hallucinated**: Tested extensively — llama3.2 3B is far more reliable with yes/no format.

---

### 4.9 `grade_answer_quality`

**Checks**: Does the answer actually address the user's question?

```python
prompt = """Does the ANSWER address the QUESTION?
yes = directly responds, or honestly explains it cannot answer
no = empty, off-topic, or doesn't respond at all

Examples:
Q: What is 2+2? A: The answer is 4. => yes
Q: Budget report? A: Context lacks info. => yes
Q: Capital of France? A: Bananas are yellow. => no
Q: What is AI? A: (empty) => no

Reply ONLY yes or no.

Question: {question}
Answer: {generation}"""
```

**Empty Guard**: Returns `False` immediately if generation empty.

---

### 4.10 `decide_after_checks` — Self-Correction Brain

```python
def decide_after_checks(state) -> Literal["end", "retry_generate"]:
    # No evidence at all → stop
    if not state.get("documents"):
        return "end"
    
    # Max retries hit → best effort
    if state["retry_count"] >= MAX_RETRIES:
        return "end"
    
    # ANY failure → retry generate (SAME evidence)
    if not state["hallucination_ok"] or not state["answer_quality_ok"]:
        return "retry_generate"
    
    return "end"
```

**Critical Design Decisions**:
1. **Both graders → same retry path** (`retry_generate`) — never jumps to websearch on hallucination fail. Keeps local docs.
2. **No `retry_websearch`** — removed entirely. Websearch only happens once at the start.
3. **Retry increments** → `increment_retry` node just does `retry_count + 1`.

---

### 4.11 `increment_retry`

```python
def increment_retry(state):
    return {**state, "retry_count": state.get("retry_count", 0) + 1}
```

**Edge**: `generate` reads `retry_count` for potential seed variation (not currently implemented).

---

## 5. Graph Wiring (`build_graph`, lines 329-377)

```python
workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("route_query", route_query)
workflow.add_node("direct_answer", direct_answer)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("grade_hallucination", grade_hallucination)
workflow.add_node("grade_answer_quality", grade_answer_quality)
workflow.add_node("increment_retry", increment_retry)

# Entry
workflow.set_entry_point("route_query")

# Router → 3 branches
workflow.add_conditional_edges(
    "route_query",
    lambda s: s["route"],
    {"direct": "direct_answer", "vectorstore": "retrieve", "websearch": "web_search"}
)

# Direct → END
workflow.add_edge("direct_answer", END)

# Retrieve → Grade
workflow.add_edge("retrieve", "grade_documents")

# Grade → Generate OR Websearch
workflow.add_conditional_edges(
    "grade_documents", decide_after_grading,
    {"websearch": "web_search", "generate": "generate"}
)

# Websearch → Generate
workflow.add_edge("web_search", "generate")

# Generate → Hallucination → Quality
workflow.add_edge("generate", "grade_hallucination")
workflow.add_edge("grade_hallucination", "grade_answer_quality")

# Self-correction loop
workflow.add_conditional_edges(
    "grade_answer_quality", decide_after_checks,
    {"end": END, "retry_generate": "increment_retry"}
)
workflow.add_edge("increment_retry", "generate")  # ← back to generate
```

**Visual Loop**: `generate → hallucination → quality → (fail?) → increment_retry → generate`

---

## 6. Ingestion Pipeline (`ingest.py`)

```
data/*.pdf, *.txt
       │
       ▼
DirectoryLoader (recursive)
       │
       ▼
RecursiveCharacterTextSplitter(chunk_size=800, overlap=100)
       │
       ▼
OllamaEmbeddings(nomic-embed-text) → 384-dim vectors
       │
       ▼
Chroma.from_documents(
    persist_directory="chroma_db",
    collection_name="local_knowledge_base",
    client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
)
```

**Output**: `chroma_db/` folder with SQLite + HNSW index.

---

## 7. CLI Entry (`main.py`)

```python
app = build_graph()  # Compiles graph once

def ask(question):
    initial_state = {
        "question": question,
        "route": "", "documents": [], "sources": [],
        "generation": "", "retry_count": 0,
        "hallucination_ok": False, "answer_quality_ok": False
    }
    final_state = app.invoke(initial_state)  # Blocking, full run
    print(final_state["generation"])
    if final_state["sources"]: print sources
```

**REPL Loop**: `while True: input → ask → print`

---

## 8. Streamlit UI (`app.py`)

**Key Patterns**:
- `@st.cache_resource` on `build_graph()` — compiles once per session
- `st.session_state.messages` — chat history
- `st.chat_input` + `st.chat_message` — modern chat UI
- `rag_graph.stream(initial_state)` — **streaming** for live stage display
- Fallback to `.invoke()` if streaming fails
- `st.status` box shows: `▶ Running: **node_name**` per step
- Error handling: Ollama down → friendly message, `st.stop()`
- Sources displayed in `st.expander("Sources")`

---

## 9. Data Flow Example: "Tell me about this project"

| Step | Node | State Changes | Time |
|---|---|---|---|
| 1 | `route_query` | `route="vectorstore"`, `retry_count=0` | ~0.1s |
| 2 | `retrieve` | `documents=[6 chunks]`, `sources=[pdf, txt]` | ~1s |
| 3 | `grade_documents` | `documents=[1 relevant]` | ~15s (6 LLM calls) |
| 4 | `generate` | `generation="I don't have enough info..."` | ~25s |
| 5 | `grade_hallucination` | `hallucination_ok=True` | ~20s |
| 6 | `grade_answer_quality` | `answer_quality_ok=True` | ~5s |
| 7 | `decide_after_checks` | `"end"` | — |
| **Total** | | | **~66s** |

---

## 10. Key Design Decisions You Can Defend

| Decision | Why |
|---|---|
| **LangGraph not chains** | Explicit state, cycles, conditional edges — needed for retry loop |
| **Two graders** | Hallucination ≠ quality. An answer can be grounded but not answer the question. |
| **Yes/No grader format** | llama3.2 3B is unreliable on "grounded/hallucinated"; yes/no + few-shot works |
| **Retry loop with same evidence** | Keeps local docs grounded; no websearch retry prevents context explosion |
| **Replace not append web docs** | Prevents context explosion (4→8→12) that made websearch slow |
| **Local-first routing** | Heuristics + LLM fallback — fast for obvious cases |
| **Sources in state** | Free provenance for UI/demo — evaluators love "where did this come from" |
| **`.env` + `load_dotenv()`** | 12-factor config — no hardcoded secrets, works in CI/CD |

---

## 11. Defense Answers for Evaluators

| Question | Answer |
|---|---|
| **Why not RAGAS/LangSmith eval?** | College scope — manual verification sufficient; eval harness is next sprint |
| **Why not larger model?** | llama3.2 runs on 8GB RAM; evaluator machines may not have 24GB+ for 70B |
| **Why not async?** | Sync simpler for demo; async adds complexity without visible benefit |
| **Why not Pinecone?** | Zero cost, zero setup, fully local — matches "no cloud" requirement |
| **Why not LCEL chains?** | Need cycles (retry loop) — LCEL is DAG only |
| **Why heuristic + LLM router?** | Pure LLM router = 1 extra call per query; heuristics cover 80% free |

---

## 12. Live Demo Checklist (3 Questions)

| Query | Expected Route | What to Point Out |
|---|---|---|
| `What is 2+2?` | `direct` | Instant, no retrieval, no graders |
| `Summarize the project in my documents` | `vectorstore` | Shows retriever count, relevance grader filter, both graders PASS, Sources expander |
| `Latest AI news` | `websearch` | Tavily call, web sources, graders PASS |

---

## Appendix: File Structure

```
project/
├── graph.py        ← ALL LOGIC (~380 lines)
│   ├── State (TypedDict)
│   ├── Config (env-driven)
│   ├── 10 Nodes (router, retrieve, grade, generate, 2 graders, retry)
│   └── build_graph() → compiled StateGraph
├── ingest.py       ← Build vector DB (run once)
├── main.py         ← Terminal REPL
├── app.py          ← Streamlit chat UI
├── requirements.txt
├── .env.example    ← Config template
├── .gitignore
└── README.md
```

---

## Quick Start for Evaluators

```bash
# 1. Install Ollama & models
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Python environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure (optional: add Tavily key for web search)
copy .env.example .env

# 4. Add documents to data/
# 5. Build index
python ingest.py

# 6. Run
python main.py          # Terminal
streamlit run app.py    # Web UI
```

---

*End of Document*