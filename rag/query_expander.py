"""
Query expander for ArcMind.

Generates multiple semantically related search queries from a single user
question to improve retrieval recall.

Two expansion strategies
------------------------
1. Static  — connector-specific expansion table (fast, deterministic)
2. LLM     — optional GPT-based expansion for general queries (adds ~1 s)

The default pipeline uses static-only expansion.  Set use_llm=True to
additionally invoke the LLM.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FAST_MODEL     = "gpt-4o-mini"   # fast/cheap model for expansion

# ---------------------------------------------------------------------------
# Static expansion tables — one entry per Arc connector
# ---------------------------------------------------------------------------

_CONNECTOR_EXPANSIONS: dict[str, list[str]] = {
    "AS2": [
        "AS2 connector configuration setup Arc",
        "AS2 MDN acknowledgement receipt",
        "AS2 certificate encryption digital signing",
        "AS2 EDI protocol trading partner",
        "AS2 send receive message",
    ],
    "SFTP": [
        "SFTP connector configuration",
        "SFTP file transfer authentication key",
        "SFTP host port connection",
        "SFTP upload download directory",
    ],
    "OFTP": [
        "OFTP Odette connector configuration",
        "OFTP2 file transfer certificate",
        "Odette FTP trading partner",
    ],
    "X12": [
        "X12 EDI connector Arc",
        "X12 transaction set 850 856 810",
        "X12 interchange envelope segment",
        "X12 mapping transformation",
    ],
    "EDIFACT": [
        "EDIFACT connector configuration Arc",
        "EDIFACT ORDERS INVOIC DESADV messages",
        "EDIFACT trading partner mapping",
        "EDIFACT envelope structure UNB UNG UNH",
    ],
    "Peppol": [
        "Peppol BIS connector Arc",
        "Peppol access point configuration",
        "Peppol e-invoicing UBL",
        "Peppol SMP SML lookup participant",
    ],
    "HTTP": [
        "HTTP connector request response Arc",
        "HTTP authentication headers settings",
        "HTTP REST integration endpoint",
    ],
    "FTP": [
        "FTP connector configuration Arc",
        "FTP file transfer active passive mode",
        "FTP authentication TLS",
    ],
    "SMTP": [
        "SMTP email connector Arc",
        "SMTP mail server TLS authentication",
        "SMTP send receive email configuration",
    ],
    "REST": [
        "REST API connector Arc",
        "REST OAuth2 authentication",
        "REST HTTP methods GET POST PUT",
    ],
    "ArcScript": [
        "ArcScript scripting language Arc",
        "ArcScript arc:set arc:call functions",
        "ArcScript conditional logic loop",
        "ArcScript variable operation",
    ],
    "FlatFile": [
        "flat file CSV connector Arc",
        "delimited file format parsing",
        "flat file schema mapping",
    ],
    "XML": [
        "XML connector Arc",
        "XML mapper XSLT transformation",
        "XML schema validation",
    ],
    "JSON": [
        "JSON connector Arc",
        "JSON mapper transformation",
        "JSON schema validation",
    ],
    "Database": [
        "database connector Arc JDBC",
        "database query insert update",
        "database connection pool",
    ],
    "Flows": [
        "Arc flow configuration",
        "Arc workflow automation",
        "Arc flow connector routing",
    ],
    "Profiles": [
        "Arc profile configuration",
        "trading partner profile",
        "Arc profile settings",
    ],
}


# ---------------------------------------------------------------------------
# Static expansion
# ---------------------------------------------------------------------------

def expand_with_connector(query: str, connector: str | None) -> list[str]:
    """
    Return expanded queries using the static connector table.

    Always includes the original *query* as the first element.
    """
    queries = [query]
    if connector and connector in _CONNECTOR_EXPANSIONS:
        for exp in _CONNECTOR_EXPANSIONS[connector]:
            # Build a combined query for richer embedding
            queries.append(f"{query} {exp}")
        # Also add the raw connector phrases for exact-term coverage
        queries.extend(_CONNECTOR_EXPANSIONS[connector][:3])
    return queries


# ---------------------------------------------------------------------------
# LLM expansion (optional)
# ---------------------------------------------------------------------------

def expand_with_llm(query: str) -> list[str]:
    """
    Ask the LLM to generate 4 related search queries.

    Falls back to [query] on any error so the pipeline is never blocked.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(
            model=FAST_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a search-query expert for the CData Arc enterprise "
                "integration platform documentation. Given the user's question, "
                "generate exactly 4 related search queries that target different "
                "aspects of the topic (configuration, usage, errors, API). "
                "Return ONLY the queries, one per line, no numbering."
            )),
            ("human", "Question: {query}"),
        ])
        result = (prompt | llm).invoke({"query": query})
        expanded = [q.strip() for q in result.content.strip().splitlines() if q.strip()]
        return [query] + expanded[:4]

    except Exception as exc:
        log.warning("LLM query expansion failed: %s", exc)
        return [query]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_query(
    query: str,
    connector: str | None = None,
    use_llm: bool = False,
) -> list[str]:
    """
    Build the final expanded query list for retrieval.

    Args:
        query:     Original user question.
        connector: Detected Arc connector (from connector_detector).
        use_llm:   If True AND no static connector expansion was found,
                   also run LLM-based expansion.

    Returns:
        Deduplicated list of query strings (original query first).
    """
    queries = expand_with_connector(query, connector)

    # Use LLM only when static expansion didn't add much
    if use_llm and len(queries) < 4:
        llm_queries = expand_with_llm(query)
        seen = set(queries)
        for q in llm_queries:
            if q not in seen:
                queries.append(q)
                seen.add(q)

    return queries
