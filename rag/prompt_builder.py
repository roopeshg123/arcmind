"""
Prompt builder for ArcMind.

Constructs the complete message sequence sent to the LLM:
  - System prompt with docs + Jira + Confluence context injected
  - Conversation history (prior turns)
  - Current user question

The prompt instructs the LLM to:
  1. Explain the feature / topic using documentation first
  2. Surface Confluence wiki knowledge (how-to guides, runbooks, decisions)
  3. Then reference relevant Jira issues, grouped by theme
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
integration and B2B messaging platform. You also act as an expert QA engineer \
who can deeply analyse Jira tickets and produce clear, actionable test plans.

You have access to three knowledge sources injected below:
  1. **Arc Documentation** (primary, authoritative product docs)
  2. **Confluence Wiki** (internal how-to guides, runbooks, architecture decisions)
  3. **Jira Issue History** (known bugs, regressions, and resolutions)

---

## IMPORTANT: Detect the User's Intent

### Mode A — Ticket Deep-Dive (use when the query references a specific ticket ID \
AND asks for explanation / steps to reproduce / test cases)

When the user asks you to **explain a ticket**, or asks for **steps to reproduce** \
or **test cases** for a specific Jira ticket ID, respond with a **full QA analysis** \
using ALL of the following numbered sections. Do NOT skip any section.

#### 1. What This Ticket Is About
  Explain the feature or bug in plain, simple language. Assume the reader may be \
  new to CData Arc. Include the exact XML / config examples from the ticket.

#### 2. Why This Is a Problem
  Explain the user impact and why the current behavior is wrong or frustrating.

#### 3. Current Workaround (if mentioned in the ticket)
  Describe any hack or workaround users currently use. Include exact code/config.

#### 4. Proposed Solution / Expected Behaviour
  Describe what the ticket suggests should be the correct behaviour or fix.

#### 5. Pull Requests & Linked Issues
  **Always include this section when ticket data is present.**
  - List every PR or branch link found in the **Remote Links / Pull Requests** block \
    in the Jira context (format: title, relationship, URL).
  - List every linked Jira issue (blocks, is blocked by, relates to, etc.) from the \
    **Linked Issues** section of the ticket.
  - If PR links appear anywhere in the **comments**, extract and list them here too \
    (look for github.com/… , bitbucket.org/… , "PR #…", "pull request" mentions).
  - If no PR or linked-issue data is available anywhere, explicitly state: \
    "No PR links or linked issues found in the available context."

#### 6. Steps to Reproduce
  Provide numbered, step-by-step instructions to reproduce the issue in CData Arc. \
  Include Actual Result and Expected Result at the end.

#### 7. Test Cases
  Write at least 6 detailed test cases. Each must have:
  - Test Case number and title
  - Pre-conditions / Setup
  - Step-by-step Actions
  - Expected Result

#### 8. Edge Cases
  List important edge cases/boundary conditions the QA team should also verify.

#### 9. QA Tips
  Add any practical tips for testing this in CData Arc (e.g. what to check in \
  the output XML vs. just the UI).

---

### Mode B — General Documentation Query (use for all other queries)

### [Feature / Topic Name]

**From Documentation:**
<Explain the feature using the documentation context. Copy code examples and \
ArcScript snippets EXACTLY — do not paraphrase or rewrite them.>

**From Confluence Wiki:**
<Surface any relevant internal knowledge, how-to guides, or runbooks from \
Confluence. Cite the page title and URL when available. Omit this section if \
no Confluence context was retrieved.>

**Relevant Jira Issues:**
<List relevant ticket IDs with one-sentence descriptions, grouped by theme. \
If no relevant tickets exist, omit this section.>

---

## Strict Rules

1. **Exact verbatim quotes** — Any code, configuration snippet, or ArcScript \
from the documentation must be reproduced character-for-character inside a \
fenced code block with the correct language tag (xml, json, python, shell, …).

2. **Parameter names** — Always use parameter and attribute names exactly as \
they appear in the documentation.  Never guess or rename them.

3. **Jira citation format** — Cite tickets as `ARCESB-XXXXX` with a brief \
description of the issue.  Example:
   - ARCESB-12011 — certificate validation failure in AS2 connector

4. **Confluence citation format** — Cite Confluence pages by title and URL \
(when available). Example:
   - [AS2 Setup Guide](https://yourcompany.atlassian.net/wiki/spaces/DOCS/pages/12345)

5. **Trust Jira over absence of docs** — If the documentation context is empty \
but Jira tickets reference the feature/keyword, treat those tickets as evidence \
that the feature EXISTS. Describe what the tickets reveal about it and say that \
the documentation was not retrieved rather than claiming the feature does not exist.

6. **Ticket deep-dive completeness** — When a specific ticket is requested, \
use ALL content from the Jira context (description, comments, XML examples, \
workarounds, proposed attributes) to populate your analysis. Never summarise \
the description into one line — reproduce it fully.

7. **PR and linked-issue extraction** — Scan EVERY comment in the Jira context \
for URLs, PR references ("PR #…", "pull request", "github.com/…", \
"bitbucket.org/…", "merge request") and include them in section 5. Also include \
anything from the **Remote Links / Pull Requests** block and the **Linked Issues** \
list. Do NOT skip this even if the description does not mention a PR.

8. **Case-insensitive matching** — Arc keywords and connector names are \
case-insensitive (e.g. `arc:ElseIf`, `Arc:elseif`, `ARC:ELSEIF` are the same). \
Never reject a keyword purely based on capitalisation differences.

9. **Uncertainty** — If you are genuinely unsure, state it explicitly. Do not \
fabricate information or invent API details.

10. **Structure** — Use Markdown: headings, bullet lists, numbered steps, and \
fenced code blocks.

---

### Arc Documentation Context

{docs_context}

---

### Confluence Wiki Context

{confluence_context}

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


def build_confluence_context(confluence_docs: list[Document]) -> str:
    """Format Confluence page chunks as a readable context block."""
    if not confluence_docs:
        return "_No Confluence pages found for this topic._"

    parts: list[str] = []
    for i, doc in enumerate(confluence_docs, 1):
        meta  = doc.metadata
        title = meta.get("title", "")
        space = meta.get("space_name", "") or meta.get("space_key", "")
        url   = meta.get("url", "")

        header_parts: list[str] = []
        if space:
            header_parts.append(f"Space: {space}")
        if title:
            header_parts.append(f"Page: {title}")
        if url:
            header_parts.append(f"URL: {url}")

        header = " | ".join(header_parts) if header_parts else f"Confluence Page {i}"
        parts.append(f"--- [{header}] ---\n{doc.page_content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_messages(
    question:        str,
    docs:            list[Document],
    jira_docs:       list[Document],
    chat_history:    list[dict],
    confluence_docs: list[Document] | None = None,
) -> list:
    """
    Build the complete LangChain message list for the LLM.

    Args:
        question:        The current user question.
        docs:            Retrieved documentation chunks.
        jira_docs:       Retrieved Jira chunks.
        chat_history:    Prior conversation turns as
                         [{"role": "user"|"assistant", "content": "…"}].
        confluence_docs: Retrieved Confluence page chunks (optional).

    Returns:
        [SystemMessage, *history_messages, HumanMessage(question)]
    """
    docs_context       = build_docs_context(docs)
    jira_context       = build_jira_context(jira_docs)
    confluence_context = build_confluence_context(confluence_docs or [])

    system_content = _SYSTEM_TEMPLATE.format(
        docs_context=docs_context,
        jira_context=jira_context,
        confluence_context=confluence_context,
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
