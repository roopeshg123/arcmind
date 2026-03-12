"""
Query expander for ArcMind.

Generates semantically related search queries from a single user question
to improve retrieval recall across all topics — not just known connectors.

Strategy: LLM-first expansion using gpt-4o-mini.
The LLM expands acronyms, generates aspect-specific variants (config, errors,
usage, troubleshooting) and handles any topic in the CData Arc domain.
Falls back gracefully to the original query if the LLM call fails.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FAST_MODEL     = "gpt-4o-mini"   # fast/cheap model for expansion

_expansion_llm = None   # lazy singleton


def _get_expansion_llm():
    """Return a cached ChatOpenAI instance for query expansion."""
    global _expansion_llm
    if _expansion_llm is None:
        from langchain_openai import ChatOpenAI
        _expansion_llm = ChatOpenAI(
            model=FAST_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
        )
    return _expansion_llm

_SYSTEM_PROMPT = (
    "You are a search-query expert for the CData Arc enterprise integration "
    "platform (also called ArcESB). Your job is to expand a user question into "
    "5 distinct search queries that together maximise recall against the "
    "Arc documentation and Jira issue tracker.\n\n"
    "Rules:\n"
    "1. Always expand acronyms on first use "
    "   (MDN → Message Disposition Notification, "
    "   AS2, SFTP, OFTP, X12, EDIFACT, FTP, SMTP, REST, HTTP keep their names "
    "   but add the words 'connector' and 'Arc').\n"
    "2. Cover different aspects across the 5 queries:\n"
    "   – one query about configuration / setup\n"
    "   – one about common errors / troubleshooting\n"
    "   – one about a specific sub-feature or concept mentioned\n"
    "   – one broad topic query\n"
    "   – one with alternative phrasing / synonyms\n"
    "3. Use CData Arc / ArcESB terminology where possible.\n"
    "4. Return ONLY the 5 queries, one per line, no numbering, no extra text."
)


def expand_with_llm(query: str) -> list[str]:
    """
    Ask gpt-4o-mini to generate 5 related search queries.
    Falls back to [query] on any error so the pipeline is never blocked.
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate

        llm    = _get_expansion_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "Question: {query}"),
        ])
        result   = (prompt | llm).invoke({"query": query})
        expanded = [q.strip() for q in result.content.strip().splitlines() if q.strip()]
        log.debug("LLM expanded '%s' → %d queries.", query, len(expanded))
        return [query] + expanded[:5]

    except Exception as exc:
        log.warning("LLM query expansion failed (%s) — using original query only.", exc)
        return [query]


def expand_query(
    query:     str,
    connector: str | None = None,  # kept for API compatibility, not used
    use_llm:   bool = True,
) -> list[str]:
    """
    Build the expanded query list for retrieval.

    Args:
        query:     Original user question.
        connector: Unused — kept for backward compatibility.
        use_llm:   Run LLM expansion (default True).

    Returns:
        Deduplicated list of query strings, original query first.
    """
    if use_llm:
        return expand_with_llm(query)
    return [query]

