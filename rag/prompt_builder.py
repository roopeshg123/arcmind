"""
Prompt builder for ArcMind.

Constructs the complete message sequence sent to the LLM:
  - System prompt with docs + Jira context injected
  - Conversation history (prior turns)
  - Current user question

The prompt instructs the LLM to:
  1. Explain the feature / topic using documentation first
  2. Then reference relevant Jira issues, grouped by theme
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag.jira_clusterer import cluster_jira_docs, format_jira_clusters

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are **ArcMind**, an expert AI assistant for CData Arc — the enterprise \
integration and B2B messaging platform.

You have access to two knowledge sources injected below:
  1. **Arc Documentation** (primary, authoritative)
  2. **Jira Issue History** (for known bugs, regressions, and resolutions)

## Response Structure

Always organise your answer as follows:

### [Feature / Topic Name]

**From Documentation:**
<Explain the feature using the documentation context. Copy code examples and \
ArcScript snippets EXACTLY — do not paraphrase or rewrite them.>

**Relevant Jira Issues:**
<Reference ticket IDs from the Jira context, grouped by theme.  \
If no relevant tickets exist, omit this section.>

## Strict Rules

1. **Exact verbatim quotes** — Any code, configuration snippet, or ArcScript \
from the documentation must be reproduced character-for-character inside a \
fenced code block with the correct language tag (xml, json, python, shell, …).

2. **Parameter names** — Always use parameter and attribute names exactly as \
they appear in the documentation.  Never guess or rename them.

3. **Jira citation format** — Cite tickets as `ARCESB-XXXXX` with a brief \
description of the issue.  Example:
   - ARCESB-12011 — certificate validation failure in AS2 connector

4. **Uncertainty** — If you are unsure, state it explicitly.  Do not fabricate \
information or invent API details.

5. **Structure** — Use Markdown: headings, bullet lists, numbered steps, and \
fenced code blocks.

---

### Arc Documentation Context

{docs_context}

---

### Jira Issues Context

{jira_context}
"""


# ---------------------------------------------------------------------------
# Context formatters
# ---------------------------------------------------------------------------

def build_docs_context(docs: list[Document]) -> str:
    """Format documentation chunks as a readable context block."""
    if not docs:
        return "_No documentation context retrieved._"

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata
        section = meta.get("section", "")
        title   = meta.get("title",   "")
        src     = meta.get("file_path", "") or meta.get("url", "")

        header_parts: list[str] = []
        if section:
            header_parts.append(f"Section: {section}")
        if title:
            header_parts.append(f"Page: {title}")
        if src:
            header_parts.append(f"Source: {src}")

        header = " | ".join(header_parts) if header_parts else f"Document {i}"
        parts.append(f"--- [{header}] ---\n{doc.page_content}")

    return "\n\n".join(parts)


def build_jira_context(jira_docs: list[Document]) -> str:
    """Format Jira chunks as clustered Markdown context."""
    if not jira_docs:
        return "_No Jira issues found for this topic._"
    clusters = cluster_jira_docs(jira_docs)
    return format_jira_clusters(clusters)


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_messages(
    question:     str,
    docs:         list[Document],
    jira_docs:    list[Document],
    chat_history: list[dict],
) -> list:
    """
    Build the complete LangChain message list for the LLM.

    Args:
        question:     The current user question.
        docs:         Retrieved documentation chunks.
        jira_docs:    Retrieved Jira chunks.
        chat_history: Prior conversation turns as
                      [{"role": "user"|"assistant", "content": "…"}].

    Returns:
        [SystemMessage, *history_messages, HumanMessage(question)]
    """
    docs_context = build_docs_context(docs)
    jira_context = build_jira_context(jira_docs)

    system_content = _SYSTEM_TEMPLATE.format(
        docs_context=docs_context,
        jira_context=jira_context,
    )

    messages: list = [SystemMessage(content=system_content)]

    for turn in chat_history:
        role    = turn.get("role",    "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=question))
    return messages
