"""
RAG Engine — handles vector store loading, retrieval, and LLM chain execution.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CHAT_MODEL       = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DB_DIR    = os.getenv("CHROMA_DB_DIR", "./chroma_db")
RETRIEVER_TOP_K  = int(os.getenv("RETRIEVER_TOP_K", "8"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL   = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_N   = int(os.getenv("RERANKER_TOP_N", "8"))


# ---------------------------------------------------------------------------
# Singleton components (loaded once at startup)
# ---------------------------------------------------------------------------
_embeddings: OpenAIEmbeddings | None = None
_vector_store: Chroma | None = None
_rag_chain: Any = None
_reranker: CrossEncoderReranker | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
    return _embeddings


def _get_reranker() -> CrossEncoderReranker:
    """Lazy-load the cross-encoder reranker (downloads model on first call)."""
    global _reranker
    if _reranker is None:
        print(f"[reranker] Loading cross-encoder model: {RERANKER_MODEL}")
        encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        _reranker = CrossEncoderReranker(model=encoder, top_n=RERANKER_TOP_N)
        print(f"[reranker] Ready — will re-score candidates and keep top {RERANKER_TOP_N}.")
    return _reranker


def load_vector_store() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    global _vector_store
    _vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=_get_embeddings(),
    )
    return _vector_store


def is_vector_store_ready() -> bool:
    """Return True if ChromaDB directory exists and contains documents."""
    if not os.path.isdir(CHROMA_DB_DIR):
        return False
    try:
        vs = load_vector_store()
        count = vs._collection.count()
        return count > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# RAG chain construction
# ---------------------------------------------------------------------------

def _build_rag_chain(vector_store: Chroma):
    """
    Build a history-aware retrieval chain with a document QA chain.

    Two-step architecture:
      1. contextualize_q_chain  — rewrites the user question using chat history
                                   so the retriever always receives a standalone query.
      2. question_answer_chain  — stuffs retrieved docs into the prompt and asks
                                   the LLM to answer grounded in the context.
    """
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.1,
        streaming=True,
    )

    # When reranking is enabled MMR fetches 2× more candidates so the
    # cross-encoder has a richer pool to re-score and select the best from.
    mmr_k = RETRIEVER_TOP_K * 2 if RERANKER_ENABLED else RETRIEVER_TOP_K
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": mmr_k, "fetch_k": mmr_k * 5},
    )
    if RERANKER_ENABLED:
        retriever = ContextualCompressionRetriever(
            base_compressor=_get_reranker(),
            base_retriever=retriever,
        )

    # -- Step 1: contextualise + expand the question -------------------------
    # This prompt does two jobs:
    #   a) resolves references from history ("it", "that connector", etc.)
    #   b) expands vague/short queries so the retriever finds more relevant chunks
    #      e.g. "Explain AS2" → "AS2 connector configuration Arc integration protocol"
    contextualize_q_system_prompt = (
        "You are a search query expert. Given the chat history and the user's latest "
        "question, do the following:\n"
        "1. If the question references something from history (e.g. 'it', 'that', "
        "'the connector'), replace those references with explicit terms.\n"
        "2. If the question is short or vague (e.g. 'Explain AS2', 'What is SFTP'), "
        "expand it into a richer search query by adding related technical terms, "
        "synonyms, and context words that would appear in documentation.\n"
        "   Example: 'Explain AS2' → 'AS2 connector configuration setup Arc "
        "integration EDI protocol send receive'\n"
        "3. Return ONLY the rewritten search query. Do NOT answer the question."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # -- Step 2: answer using retrieved docs + LLM's own knowledge -----------
    qa_system_prompt = (
        "You are an expert technical assistant specialising in the Arc integration "
        "platform. You have two sources of knowledge:\n"
        "  1. The Arc DOCUMENTATION provided below (primary source).\n"
        "  2. Your own general technical knowledge (secondary — used only to enrich).\n\n"
        "STRICT RULES — follow these in order:\n\n"
        "RULE 1 — EXAMPLES:\n"
        "  If an example, sample code, or script exists in the DOCUMENTATION CONTEXT,\n"
        "  you MUST copy it EXACTLY, word-for-word, character-for-character.\n"
        "  Do NOT rewrite it, paraphrase it, or 'improve' it.\n"
        "  After showing the exact doc example, you may then explain it and optionally\n"
        "  add a clearly labelled supplementary example.\n"
        "  NEVER invent or generate code for Arc-specific syntax (ArcScript, arc:set,\n"
        "  arc:call, etc.) from your own knowledge — Arc's syntax must come from the docs.\n\n"
        "RULE 2 — PARAMETERS & ATTRIBUTES:\n"
        "  Always list parameters, attributes, and field names exactly as they appear\n"
        "  in the documentation. Do not guess or rename them.\n\n"
        "RULE 3 — EXPLANATIONS:\n"
        "  After presenting the exact doc content, you MAY add:\n"
        "  - Plain-English explanation of what each part does\n"
        "  - General background (e.g. what BASE64 is, what SFTP is)\n"
        "  - Real-world use cases and analogies\n"
        "  - Comparisons with similar standards/protocols\n"
        "  Clearly label any content that comes from your own knowledge vs the docs.\n\n"
        "RULE 4 — MISSING CONTENT:\n"
        "  Only say 'not found in documentation' if the topic is completely absent\n"
        "  from the retrieved context AND unrelated to Arc or integration platforms.\n\n"
        "RULE 5 — STRUCTURE:\n"
        "  Use clear headings, bullet points, numbered steps, and fenced code blocks.\n"
        "  For code, always specify the language tag (xml, json, python, etc.).\n\n"
        "--- ARC DOCUMENTATION CONTEXT ---\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_rag_chain():
    """Return the cached RAG chain, building it if necessary."""
    global _rag_chain, _vector_store
    if _rag_chain is None:
        if _vector_store is None:
            load_vector_store()
        _rag_chain = _build_rag_chain(_vector_store)
    return _rag_chain


# ---------------------------------------------------------------------------
# Public API used by the FastAPI routes
# ---------------------------------------------------------------------------

# Readable labels for common op_ filename prefixes
_OP_PREFIX_MAP = {
    "enc": "encoding",
    "db":  "database",
    "file": "file",
    "http": "HTTP",
    "json": "JSON",
    "xml":  "XML",
    "zip":  "ZIP",
    "sys":  "system",
    "msg":  "message",
    "crypto": "crypto",
    "thread": "thread",
    "flow":   "flow",
    "integrity": "integrity",
}


def _extract_topic_from_path(text: str) -> str | None:
    """
    If the text contains a file path or file:// URL, extract the HTML filename
    and convert it into a human-readable topic string.

    Examples:
      file:///D:/Arc.help.AZN/mft/op_encDecode.html
          → "encDecode encode decode operation Arc"
      D:\\Arc.help.AZN\\mft\\SFTP.html
          → "SFTP"
      op_HTTPGet.html
          → "HTTPGet HTTP GET operation Arc"
    """
    # Match file:// URLs or Windows/Unix absolute paths containing .html
    pattern = r'(?:file:///|file://)?[A-Za-z]:[\\//][^\s"<>]+\.html|[^\s"<>]+\.html'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None

    # Get just the filename without extension
    raw = match.group(0)
    stem = Path(re.sub(r'file:///|file://', '', raw).replace('%20', ' ')).stem
    # e.g. "op_encDecode", "SFTP", "REST"

    # Remove op_ prefix and split on camelCase / underscores / hyphens
    stem = re.sub(r'^op_', '', stem)
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)  # camelCase split
    words = re.sub(r'[-_]', ' ', words)

    return f"{stem} {words} Arc operation connector documentation"


def _normalize_question(question: str) -> str:
    """
    Pre-process the user question before it enters the RAG chain.

    - If it contains a file path or URL, extract the topic and rewrite
      the question so it searches for that topic instead of the raw path.
    - Otherwise return the question unchanged.
    """
    topic = _extract_topic_from_path(question)
    if topic is None:
        return question

    # Remove the raw path/URL from the question text
    cleaned = re.sub(
        r'(?:file:///|file://)?[A-Za-z]:[\\//][^\s"<>]+\.html|[^\s"<>]+\.html',
        '',
        question,
        flags=re.IGNORECASE,
    ).strip()

    # Build a clean question: keep whatever the user typed plus the extracted topic
    if cleaned:
        # e.g. "in this path there is a description" + topic
        rewritten = f"{cleaned} — topic: {topic}"
    else:
        rewritten = f"Explain {topic} in detail with examples"

    return rewritten


def convert_chat_history(raw_history: List[Dict[str, str]]):
    """Convert [{role, content}] list to LangChain message objects."""
    messages = []
    for msg in raw_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def ask(question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Run a RAG query.

    Args:
        question:     The user's question.
        chat_history: Previous turns as [{"role": "user"|"assistant", "content": "..."}].

    Returns:
        {"answer": str, "sources": [{"source": str, "content": str}]}
    """
    if chat_history is None:
        chat_history = []

    # Normalize: convert any pasted file paths/URLs into searchable topic queries
    question = _normalize_question(question)

    chain = get_rag_chain()
    lc_history = convert_chat_history(chat_history)

    result = chain.invoke({"input": question, "chat_history": lc_history})

    # Deduplicate source documents
    seen_sources = set()
    sources = []
    for doc in result.get("context", []):
        src = doc.metadata.get("source", "Unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            sources.append({
                "source": src,
                "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            })

    return {
        "answer": result["answer"],
        "sources": sources,
    }


def reset_chain():
    """Release all cached objects so file handles are freed (important on Windows)."""
    global _rag_chain, _vector_store
    _rag_chain = None
    _vector_store = None
    import gc
    gc.collect()
