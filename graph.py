"""
graph.py
--------
Yahan poora Self-Correcting Agentic RAG pipeline LangGraph ke through define hota hai.

Flow:
  route_query -> (direct_answer) OR (retrieve -> grade_documents -> (web_search if needed)
                  -> generate -> grade_hallucination -> grade_answer_quality)
  Agar kahin bhi grader "fail" bole, graph automatically wapas relevant node pe loop karta hai.
"""

import os

# Chroma telemetry warnings band karne ke liye (import se PEHLE set karna zaroori hai)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")

import chromadb

from typing import List, TypedDict, Literal

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

# .env file se config load karo (TAVILY_API_KEY, model names, etc.)
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — sab kuch .env se override kiya ja sakta hai, defaults neeche hain
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")       # jaise "mistral:7b" bhi chalta hai
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
COLLECTION_NAME = "local_knowledge_base"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))      # infinite loop se bachne ke liye
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "4"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()  # .env file me set karo

# Chroma telemetry band + DISK persistence (is_persistent=True ke bina in-memory ho jaata hai!)
_chroma_settings = chromadb.Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=PERSIST_DIR,
)

llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)  # seed handled per-call for retries
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    client_settings=_chroma_settings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


# ---------------------------------------------------------------------------
# STATE — ye dictionary poore graph mein ek node se dusre node tak carry hoti hai
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    route: str # "direct" | "vectorstore" | "websearch"
    documents: List[Document]
    sources: List[str] # demo ke liye — kis file/URL se answer aaya
    generation: str
    retry_count: int
    hallucination_ok: bool
    answer_quality_ok: bool


# ---------------------------------------------------------------------------
# NODE 1: QUERY ROUTER
# ---------------------------------------------------------------------------
def route_query(state: GraphState) -> GraphState:
    question = state["question"].strip()
    lowered = question.lower()

    # Strong heuristic for document-specific or project-related questions.
    if any(keyword in lowered for keyword in [
        "document", "project", "report", "summary", "paper", "pdf", "txt",
        "according to", "based on", "this project", "this document", "describe this",
        "what is this", "what does this", "explain this", "tell me about this"
    ]):
        route = "vectorstore"
    elif any(keyword in lowered for keyword in [
        "latest", "today", "current", "news", "recent", "nowadays", "2026", "2025"
    ]):
        route = "websearch"
    else:
        prompt = ChatPromptTemplate.from_template(
            """You are a query router. Classify the question into exactly one of these categories:
- "direct": general knowledge question that does not require local document retrieval
- "vectorstore": question about a specific local document, project, report, PDF, or uploaded file
- "websearch": question requiring current or recent information not available in local documents

Return only one word: direct, vectorstore, or websearch.

Question: {question}"""
        )
        chain = prompt | llm
        result = chain.invoke({"question": question})
        route = result.content.strip().lower()

        if "direct" in route:
            route = "direct"
        elif "websearch" in route or "web_search" in route or "web search" in route:
            route = "websearch"
        else:
            route = "vectorstore"

    print(f"[Router] -> {route}")
    return {**state, "route": route, "retry_count": 0}


# ---------------------------------------------------------------------------
# NODE 2: DIRECT ANSWER (no retrieval needed)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """Answer the question clearly in English.
Provide a direct, useful response with enough detail to be helpful.
Do not answer in Hinglish or mixed language.

Question: {question}
Answer:"""
    )
    chain = prompt | llm
    result = chain.invoke({"question": state["question"]})
    return {**state, "generation": result.content, "documents": [], "sources": []}


# ---------------------------------------------------------------------------
# NODE 3: RETRIEVER (local Chroma vector store)
# ---------------------------------------------------------------------------
def retrieve(state: GraphState) -> GraphState:
    docs = retriever.invoke(state["question"])
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)
    print(f"[Retriever] {len(docs)} documents mile")
    return {**state, "documents": docs, "sources": sources}


# ---------------------------------------------------------------------------
# NODE 4: DOCUMENT RELEVANCE GRADER
# ---------------------------------------------------------------------------
def grade_documents(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """Is this retrieved document relevant or potentially useful for answering the user's question?
Answer "yes" if the document contains ANY information related to the question topic, even partially.
Answer "no" only if completely unrelated.

Question: {question}
Document: {document}"""
    )
    chain = prompt | llm

    relevant_docs = []
    for doc in state["documents"]:
        result = chain.invoke({"question": state["question"], "document": doc.page_content})
        if "yes" in result.content.strip().lower():
            relevant_docs.append(doc)

    print(f"[Relevance Grader] {len(relevant_docs)}/{len(state['documents'])} documents relevant")
    return {**state, "documents": relevant_docs}


def decide_after_grading(state: GraphState) -> Literal["websearch", "generate"]:
    """Agar koi relevant document nahi bacha, web search pe jao."""
    if len(state["documents"]) == 0:
        return "websearch"
    return "generate"


# ---------------------------------------------------------------------------
# NODE 5: WEB SEARCH FALLBACK (Tavily)
# ---------------------------------------------------------------------------
def web_search(state: GraphState) -> GraphState:
    if tavily_client is None:
        print("[Web Search] TAVILY_API_KEY set nahi hai — skip kar rahe hain")
        return state

    results = tavily_client.search(query=state["question"], max_results=WEB_MAX_RESULTS)
    docs = [
        Document(page_content=r["content"], metadata={"source": r["url"]})
        for r in results.get("results", [])
    ]
    sources = [r["url"] for r in results.get("results", [])]
    print(f"[Web Search] {len(docs)} results mile")

    # Purane documents ko replace karo (append nahi) taaki retry pe
    # context har baar nahi badhta aur generation slow nahi hoti.
    return {**state, "documents": docs, "sources": sources}


# ---------------------------------------------------------------------------
# NODE 6: GENERATOR
# ---------------------------------------------------------------------------
def generate(state: GraphState) -> GraphState:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    if not context.strip():
        return {
            **state,
            "generation": "I could not find relevant information in the uploaded documents to answer this question accurately. Please add the correct project documents or ask a question that matches the available context."
        }

    prompt = ChatPromptTemplate.from_template(
        """Answer the user's question using only the information in the provided context.
Rules:
- Write the answer in clear, natural English.
- Be specific and sufficiently detailed.
- Do not invent facts.
- If the context genuinely does not contain the answer, state that clearly.
- Do not mix Hindi and English in the final answer.

Context:
{context}

Question: {question}

Answer:"""
    )
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "generation": result.content}


# ---------------------------------------------------------------------------
# NODE 7: HALLUCINATION GRADER
# ---------------------------------------------------------------------------
def grade_hallucination(state: GraphState) -> GraphState:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    if not context.strip():
        print("[Hallucination Grader] FAIL (no context available)")
        return {**state, "hallucination_ok": False}

    prompt = ChatPromptTemplate.from_template(
        """Does the CONTEXT support the ANSWER? Answer yes or no.
yes means every claim in the answer appears in the context or is a paraphrase of it, including an honest statement that the context is insufficient.
no means the answer contains invented facts not present in the context.

Context: {context}
Answer: {generation}"""
    )
    chain = prompt | llm
    result = chain.invoke({"context": context, "generation": state["generation"]})
    ok = "yes" in result.content.strip().lower()
    print(f"[Hallucination Grader] {'PASS' if ok else 'FAIL'}")
    return {**state, "hallucination_ok": ok}


# ---------------------------------------------------------------------------
# NODE 8: ANSWER QUALITY GRADER
# ---------------------------------------------------------------------------
def grade_answer_quality(state: GraphState) -> GraphState:
    if not state.get("generation", "").strip():
        print("[Answer Quality Grader] FAIL (empty generation)")
        return {**state, "answer_quality_ok": False}

    prompt = ChatPromptTemplate.from_template(
        """Does the ANSWER address the QUESTION?
"yes" means the answer directly responds to the question, or honestly explains it cannot answer.
"no" means the answer is empty, off-topic, or does not respond to the question at all.

Examples:
Question: What is 2+2?
Answer: The answer is 4.
=> yes

Question: Tell me about the budget report
Answer: The context does not contain enough information about the budget report.
=> yes

Question: What is the capital of France?
Answer: Bananas are yellow fruits.
=> no

Question: What is AI?
Answer: (empty)
=> no

Reply with exactly one word: yes or no.

Question: {question}
Answer: {generation}"""
    )
    chain = prompt | llm
    result = chain.invoke({"question": state["question"], "generation": state["generation"]})
    ok = "yes" in result.content.strip().lower()
    print(f"[Answer Quality Grader] {'PASS' if ok else 'FAIL'}")
    return {**state, "answer_quality_ok": ok}


def decide_after_checks(state: GraphState) -> Literal["end", "retry_generate"]:
    """Self-correction decision: sab pass -> end. Fail hua to retry (max limit ke saath)."""
    if not state.get("documents"):
        print("[Self-Correction] No relevant documents found — stopping instead of generating a generic answer")
        return "end"

    if state["retry_count"] >= MAX_RETRIES:
        print("[Self-Correction] Max retries reached — best-effort answer de rahe hain")
        return "end"

    # Dono failure modes me same evidence se regenerate karo.
    # Isse local documents wala context web search se replace nahi hota
    # aur hallucination check fail hone par bhi grounded context hi use hota hai.
    if not state["hallucination_ok"] or not state["answer_quality_ok"]:
        return "retry_generate"
    return "end"


def increment_retry(state: GraphState) -> GraphState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("route_query", route_query)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_hallucination", grade_hallucination)
    workflow.add_node("grade_answer_quality", grade_answer_quality)
    workflow.add_node("increment_retry", increment_retry)

    workflow.set_entry_point("route_query")

    # Router se 3 raste
    workflow.add_conditional_edges(
        "route_query",
        lambda s: s["route"],
        {
            "direct": "direct_answer",
            "vectorstore": "retrieve",
            "websearch": "web_search",
        },
    )

    workflow.add_edge("direct_answer", END)
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"websearch": "web_search", "generate": "generate"},
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "grade_hallucination")
    workflow.add_edge("grade_hallucination", "grade_answer_quality")

    # Self-correction loop
    workflow.add_conditional_edges(
        "grade_answer_quality",
        decide_after_checks,
        {
            "end": END,
            "retry_generate": "increment_retry",
        },
    )
    workflow.add_edge("increment_retry", "generate")  # retry pe same evidence se dobara generate

    return workflow.compile()
