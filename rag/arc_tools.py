"""
ArcMind Arc-Specific Intelligence Tools

Six specialized features that differentiate ArcMind from generic RAG assistants:

  decode_error(error_text)        — paste an Arc error log → root cause + fix steps
  explain_edi(edi_text)           — paste a raw X12/EDIFACT message → plain-English walkthrough
  find_similar(ticket_id)         — find Jira tickets related to a given ticket
  generate_ticket_draft(desc)     — plain English description → ready-to-file Jira ticket JSON
  connector_changelog(connector)  — structured changelog for a connector across fix versions
  generate_script(requirement)    — plain English → working ArcScript or Python script
  fix_script(script, error_desc)  — broken script + error description → corrected script

All functions are synchronous (wrap in run_in_executor for async endpoints).
Each returns a dict: { answer, sources, jira_issues, tool, ...optional extras }
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None  # type: ignore

from rag.connector_detector import detect_connector
from rag.prompt_builder import (
    build_confluence_context,
    build_docs_context,
    build_jira_context,
)
from rag.query_expander import expand_query
from rag.reranker import rerank
from rag.retriever import retrieve_all
from vector_db.chroma_store import get_store

load_dotenv()

log = logging.getLogger(__name__)

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CHAT_PROVIDER     = os.getenv("CHAT_PROVIDER",     "openai").lower()
CHAT_MODEL        = os.getenv("CHAT_MODEL",        "gpt-4.1")

_llm: Any | None = None


def _get_llm() -> Any:
    global _llm
    if _llm is None:
        if CHAT_PROVIDER == "claude":
            if ChatAnthropic is None:
                raise RuntimeError(
                    "langchain-anthropic is not installed. "
                    "Run: pip install langchain-anthropic"
                )
            _llm = ChatAnthropic(
                model=CHAT_MODEL,
                anthropic_api_key=ANTHROPIC_API_KEY,
                temperature=0.1,
            )
        else:
            _llm = ChatOpenAI(
                model=CHAT_MODEL,
                openai_api_key=OPENAI_API_KEY,
                temperature=0.1,
            )
    return _llm


def _build_sources(docs: list, jira_docs: list, confluence_docs: list | None = None) -> list[dict]:
    seen: set[str] = set()
    sources: list[dict] = []
    for doc in docs + jira_docs + (confluence_docs or []):
        key = (
            doc.metadata.get("file_path")
            or doc.metadata.get("url")
            or doc.metadata.get("ticket")
            or doc.metadata.get("page_id")
            or "unknown"
        )
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":  key,
                "type":    doc.metadata.get("source",  "unknown"),
                "section": doc.metadata.get("section", ""),
                "ticket":  doc.metadata.get("ticket",  ""),
                "title":   doc.metadata.get("title",   ""),
                "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            })
    return sources


def _jira_issues_from_docs(jira_docs: list) -> list[dict]:
    seen: set[str] = set()
    result = []
    for d in jira_docs:
        t = d.metadata.get("ticket", "")
        if t and t not in seen:
            seen.add(t)
            result.append({
                "ticket":    t,
                "status":    d.metadata.get("status",    ""),
                "component": d.metadata.get("component", ""),
            })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Error Decoder
# ─────────────────────────────────────────────────────────────────────────────

_ERROR_SYSTEM = """\
You are ArcMind's Error Decoder for CData Arc — the enterprise EDI/B2B integration platform.

A user has pasted an Arc error message or log snippet. \
Give them the definitive diagnosis using the knowledge sources below.

Respond with exactly these sections:

## Error Type
Identify the category (e.g., Certificate Error, MDN Failure, Connection Timeout, \
ArcScript Parse Error, SFTP Auth Failure, X12 Validation Error, Encryption Error, etc.)

## Root Cause
Explain precisely why this error occurs in CData Arc. \
Reference Arc-specific internals (connector configs, ArcScript engine, MDN handshake, etc.).

## Known Tickets
List every Jira ticket from the context that matches or closely relates to this error:
- **ARCESB-XXXXX** — one-line description and its resolution
If no exact match exists, say "No matching tickets found — this may be a new issue."

## Step-by-Step Fix
Numbered steps to resolve the error inside CData Arc. \
Include exact XML config parameters where relevant.

## Verify the Fix
How to confirm the issue is resolved (what to check in Arc's output logs/activity log).

---
Strict rules:
- Never guess. State uncertainty explicitly.
- Copy XML/config examples verbatim — do not paraphrase.
- If the error appears in multiple tickets, mention ALL of them.

### Arc Documentation Context
{docs_context}

### Jira Issue History
{jira_context}

### Confluence Wiki
{confluence_context}
"""


def decode_error(error_text: str) -> dict[str, Any]:
    """
    Diagnose a CData Arc error log or exception message.

    Searches docs, Jira history, and Confluence for matching issues,
    then returns a structured root-cause analysis with fix steps.
    """
    connector = detect_connector(error_text)
    expanded  = expand_query(error_text, connector=connector, use_llm=True)

    docs, jira_docs, confluence_docs = retrieve_all(
        expanded, docs_k=5, jira_k=8, confluence_k=3,
        connector_filter=connector,
    )

    combined = docs + jira_docs + confluence_docs
    if combined:
        reranked        = rerank(error_text, combined, top_n=12)
        docs            = [d for d in reranked if d.metadata.get("source") not in ("jira", "confluence")][:5]
        jira_docs       = [d for d in reranked if d.metadata.get("source") == "jira"][:8]
        confluence_docs = [d for d in reranked if d.metadata.get("source") == "confluence"][:3]

    system = _ERROR_SYSTEM.format(
        docs_context       = build_docs_context(docs),
        jira_context       = build_jira_context(jira_docs),
        confluence_context = build_confluence_context(confluence_docs),
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Diagnose this Arc error:\n\n```\n{error_text}\n```"),
    ]

    response = _get_llm().invoke(messages)
    return {
        "answer":      response.content,
        "sources":     _build_sources(docs, jira_docs, confluence_docs),
        "jira_issues": _jira_issues_from_docs(jira_docs),
        "tool":        "error_decoder",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. EDI Message Explainer
# ─────────────────────────────────────────────────────────────────────────────

_EDI_SYSTEM = """\
You are ArcMind's EDI analyst and CData Arc integration expert.

The user has pasted a raw EDI message. Walk through it fully and clearly.

## Format Detection
State the EDI standard (X12, EDIFACT, or other) and the specific transaction \
type / message type (e.g., X12 850 Purchase Order, EDIFACT ORDERS D.96A).

## Segment-by-Segment Breakdown
For each segment, in order:

**SEG_ID** — Full Segment Name
| Element | Value | Meaning |
|---------|-------|---------|
| E001    | ...   | ...     |

Flag ⚠️ any unexpected, invalid, or missing mandatory values.

## Business Summary
What does this entire message represent in plain English? \
(e.g., "This is a Purchase Order from ACME Corp to Supplier XYZ for 500 units of \
product ABC, expected by 2026-04-15.")

## CData Arc Processing Tips
How does Arc handle this message type? Include:
- Which Arc connector and port to use
- Any delimiter or validation settings to configure
- Expected acknowledgement (997/CONTRL) requirements
- Common mistakes when mapping this message type in Arc

---
### Arc Documentation Context (X12 / EDIFACT / Mapping / Connectors)
{docs_context}
"""


def explain_edi(edi_text: str) -> dict[str, Any]:
    """
    Explain a raw X12 or EDIFACT message segment-by-segment in plain English.

    Searches Arc docs for segment definitions and connector guidance,
    then produces a structured walkthrough.
    """
    stripped = edi_text.strip()
    if stripped.startswith("ISA"):
        query = "X12 EDI ISA segment elements transaction set interchange control"
    elif stripped.startswith("UNB") or "UNH" in stripped[:50]:
        query = "EDIFACT UNB UNH segment elements interchange message type"
    else:
        query = f"EDI message format parsing segments connectors {stripped[:120]}"

    docs, _, _ = retrieve_all(
        [query, "EDI connector mapping ArcScript transformation"],
        docs_k=8, jira_k=0, confluence_k=0,
        connector_filter=None,
    )

    system = _EDI_SYSTEM.format(docs_context=build_docs_context(docs))
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Explain this EDI message:\n\n```\n{edi_text}\n```"),
    ]

    response = _get_llm().invoke(messages)
    return {
        "answer":      response.content,
        "sources":     _build_sources(docs, []),
        "jira_issues": [],
        "tool":        "edi_explainer",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Find Similar Tickets
# ─────────────────────────────────────────────────────────────────────────────

_SIMILAR_SYSTEM = """\
You are ArcMind's Jira ticket analyst for CData Arc.

You have a reference ticket and a list of related tickets found in the Jira index.
Analyse the relationships and provide:

## Common Theme
What root cause, feature area, or component connects these tickets?

## Closest Matches
The top 3 most similar tickets and a clear explanation of WHY each is related \
(same error, same connector config, same workaround, same regression area, etc.).

## Pattern Alert
Is there a recurring pattern here the QA or support team should escalate? \
Does this suggest a systemic issue?

## Recommended Action
Should these tickets be linked? Is a parent epic needed? \
Is there an unresolved root cause that spans multiple tickets?

Be concise. Use bullet points. Cite ticket IDs as ARCESB-XXXXX."""


def find_similar(ticket_id: str) -> dict[str, Any]:
    """
    Find Jira tickets related to the given ticket ID.

    Strategy:
      1. Fetch ALL chunks of the target ticket (description + comments).
      2. Build multiple search queries from the ticket title, body, and keywords.
      3. Run two parallel searches — one with connector filter, one without —
         so cross-component similar tickets (e.g. same UI bug in another connector)
         are never missed.
      4. Merge, deduplicate, rerank, and pass top-10 to the LLM for analysis.
    """
    store       = get_store()
    target_docs = store.get_jira_by_tickets([ticket_id.strip().upper()])

    if not target_docs:
        return {
            "answer": (
                f"Ticket **{ticket_id}** was not found in the index.\n\n"
                "Make sure Jira has been indexed and this ticket key is correct."
            ),
            "sources":     [],
            "jira_issues": [],
            "tool":        "similar_tickets",
        }

    # Use ALL chunks so comments and full description are included in the
    # search signal — not just the first 600 chars of the main doc.
    ref_id = ticket_id.strip().upper()
    all_ticket_texts = [d.page_content for d in target_docs]
    # Primary query: first chunk (description head)
    primary_text   = target_docs[0].page_content[:600]
    # Extended query: full concatenated text for keyword richness (capped)
    extended_text  = " ".join(all_ticket_texts)[:900]

    connector      = detect_connector(primary_text)
    target_summary = target_docs[0].metadata.get("summary", ref_id)

    # Build varied search queries to improve recall
    queries = [primary_text, extended_text]
    # Add the summary as a concise standalone query if available
    if target_summary and target_summary != ref_id:
        queries.append(target_summary)

    # Search 1: with connector filter (precise — same component)
    _, filtered_docs, _ = retrieve_all(
        queries, docs_k=0, jira_k=20, confluence_k=0,
        connector_filter=connector,
    )

    # Search 2: without connector filter (broad — catches cross-component matches
    # e.g. the same UI bug in a different connector, same error in another area)
    _, broad_docs, _ = retrieve_all(
        queries, docs_k=0, jira_k=20, confluence_k=0,
        connector_filter=None,
    )

    # Merge both result sets, remove the reference ticket itself
    all_docs = filtered_docs + broad_docs
    all_docs = [d for d in all_docs if d.metadata.get("ticket") != ref_id]

    # Rerank against primary query to surface the most relevant matches
    if all_docs:
        from rag.reranker import rerank as _rerank
        all_docs = _rerank(primary_text, all_docs, top_n=min(30, len(all_docs)))

    # Deduplicate by ticket key, preserving rerank order
    seen: dict[str, dict] = {}
    for d in all_docs:
        t = d.metadata.get("ticket", "")
        if t and t not in seen:
            seen[t] = {
                "ticket":    t,
                "summary":   d.metadata.get("summary",   ""),
                "status":    d.metadata.get("status",    ""),
                "component": d.metadata.get("component", ""),
            }

    top_issues = list(seen.values())[:10]

    if top_issues:
        similar_text = "\n".join(
            f"- **{v['ticket']}** [{v['status']}] {v['component']} — {v['summary']}"
            for v in top_issues
        )
        user_msg = (
            f"Reference ticket: **{ref_id}** — {target_summary}\n\n"
            f"Similar tickets found in the index:\n{similar_text}\n\n"
            "Analyse the relationships and provide your assessment."
        )
    else:
        user_msg = (
            f"Reference ticket: **{ref_id}** — {target_summary}\n\n"
            "No similar tickets were found in the index. "
            "Based on the ticket summary alone, analyse what kind of issues "
            "WOULD be related, what to look for, and what the QA team should check."
        )

    messages = [
        SystemMessage(content=_SIMILAR_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    response = _get_llm().invoke(messages)
    return {
        "answer":           response.content,
        "sources":          [],
        "jira_issues":      top_issues,
        "tool":             "similar_tickets",
        "reference_ticket": ref_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ticket Auto-Generator
# ─────────────────────────────────────────────────────────────────────────────

_TICKET_SYSTEM = """\
You are ArcMind's Jira ticket author for CData Arc support.

Generate a complete, professional Jira ticket based on the user's description.
Output ONLY valid JSON — no markdown fences, no extra text, just the JSON object.

Use this exact schema:
{{
  "summary":            "ComponentName — concise one-line description under 80 chars",
  "type":               "Bug",
  "priority":           "Major",
  "component":          "AS2",
  "description":        "## Problem\\nDetailed description...\\n\\n## Environment\\n- Arc Version: \\n- OS: \\n- Connector: \\n\\n## Impact\\n...",
  "steps_to_reproduce": [
    "Open CData Arc and navigate to Connectors > ...",
    "Configure the ... connector with these settings: ...",
    "Trigger the operation by ...",
    "Observe the error in the Arc activity log"
  ],
  "expected_result": "What should happen according to the Arc documentation",
  "actual_result":   "What actually happens (error message / wrong behaviour)",
  "related_tickets": ["ARCESB-XXXXX"]
}}

Field rules:
- summary: start with the component name; ≤80 chars
- type: "Bug" | "Task" | "Improvement" | "New Feature"
- priority: "Blocker" | "Critical" | "Major" | "Minor"
- component: exact Arc connector (AS2, SFTP, X12, EDIFACT, HTTP, FTP, SMTP, REST, \
  ArcScript, Peppol, OFTP, FlatFile, XML, JSON, Database, Flows, Profiles)
- steps_to_reproduce: 4-6 steps, Arc-specific actions
- related_tickets: ONLY tickets that appear in the Jira context below; use [] if none

### Similar Past Tickets (for context and related_tickets field)
{jira_context}
"""


def generate_ticket_draft(description: str) -> dict[str, Any]:
    """
    Draft a Jira ticket from a plain-English problem description.

    Auto-detects the Arc component, searches for similar past tickets
    to populate related_tickets, and returns both a formatted answer
    and the raw parsed ticket_data dict.
    """
    connector = detect_connector(description)
    expanded  = expand_query(description, connector=connector, use_llm=True)

    _, jira_docs, _ = retrieve_all(
        expanded, docs_k=0, jira_k=8, confluence_k=0,
        connector_filter=connector,
    )

    system   = _TICKET_SYSTEM.format(jira_context=build_jira_context(jira_docs))
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Generate a Jira ticket for this issue:\n\n{description}"),
    ]

    response = _get_llm().invoke(messages)
    raw      = response.content.strip()

    # Extract JSON even if the LLM accidentally wrapped it in markdown fences
    json_match  = re.search(r'\{[\s\S]+\}', raw)
    ticket_data: dict = {}
    if json_match:
        try:
            ticket_data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            ticket_data = {}

    if ticket_data:
        related = ", ".join(ticket_data.get("related_tickets", [])) or "_None found_"
        steps   = "\n".join(
            f"{i + 1}. {s}"
            for i, s in enumerate(ticket_data.get("steps_to_reproduce", []))
        )
        answer = (
            f"## Drafted Jira Ticket\n\n"
            f"**Summary:** {ticket_data.get('summary', '')}\n"
            f"**Type:** {ticket_data.get('type', '')} &nbsp;|&nbsp; "
            f"**Priority:** {ticket_data.get('priority', '')} &nbsp;|&nbsp; "
            f"**Component:** {ticket_data.get('component', '')}\n\n"
            f"### Description\n{ticket_data.get('description', '')}\n\n"
            f"### Steps to Reproduce\n{steps}\n\n"
            f"**Expected:** {ticket_data.get('expected_result', '')}\n\n"
            f"**Actual:** {ticket_data.get('actual_result', '')}\n\n"
            f"**Related Tickets:** {related}"
        )
    else:
        answer = raw  # fallback: return raw LLM text if JSON parse failed

    return {
        "answer":      answer,
        "sources":     _build_sources([], jira_docs),
        "jira_issues": _jira_issues_from_docs(jira_docs),
        "tool":        "ticket_generator",
        "ticket_data": ticket_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Connector Changelog
# ─────────────────────────────────────────────────────────────────────────────

_CHANGELOG_SYSTEM = """\
You are ArcMind's release analyst for CData Arc.

Create a clear, structured changelog for the "{connector}" connector/component \
using the Jira tickets provided below.

## What's New — {connector}
Write 2-3 sentences summarising the most significant recent improvements, \
bug fixes, or new capabilities added to this connector.

## Changelog by Version
Group tickets by Fix Version, newest version first. For each version:

### Version X.Y.Z
- **[BUG FIX]** ARCESB-XXXXX — one-line description of what was fixed
- **[IMPROVEMENT]** ARCESB-XXXXX — one-line description of the improvement
- **[NEW FEATURE]** ARCESB-XXXXX — one-line description of the new capability

Rules:
- Only include tickets with Resolution = Fixed, Done, Closed, or Implemented.
- Sort versions newest-first (higher version numbers = newer).
- If no Fix Version is set, group under **Upcoming / Unversioned**.
- Keep each bullet to one line — ticket ID + one-line description.
- If fewer than 3 tickets exist, note that the changelog is limited.

---
### Jira Tickets for {connector}
{jira_context}
"""


def connector_changelog(connector_name: str) -> dict[str, Any]:
    """
    Return a structured changelog for the given Arc connector/component.

    Searches Jira for resolved tickets tagged with the connector,
    then groups them by Fix Version to form a readable release history.
    """
    query = f"{connector_name} connector fix resolved bug improvement new feature version"

    _, jira_docs, _ = retrieve_all(
        [query], docs_k=0, jira_k=20, confluence_k=0,
        connector_filter=connector_name,
    )

    # Broaden if the filtered search returned nothing
    if not jira_docs:
        _, jira_docs, _ = retrieve_all(
            [query], docs_k=0, jira_k=20, confluence_k=0,
            connector_filter=None,
        )

    system = _CHANGELOG_SYSTEM.format(
        connector    = connector_name,
        jira_context = build_jira_context(jira_docs),
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Generate a changelog for the {connector_name} connector."),
    ]

    response = _get_llm().invoke(messages)
    return {
        "answer":      response.content,
        "sources":     _build_sources([], jira_docs),
        "jira_issues": _jira_issues_from_docs(jira_docs),
        "tool":        "connector_changelog",
        "connector":   connector_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Script Generator
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_SYSTEM = """\
You are ArcMind's Script Engineer for CData Arc.

You generate complete, production-ready scripts that run inside CData Arc using
either ArcScript (Arc's built-in XML-based scripting language) or Python
(via Arc's Python Scripting connector), based on what the user asks for.

Always do the following:

## 1. Detect Script Type
Determine whether the user needs:
- **ArcScript** — Arc's native XML macro language (arc:set, arc:if, arc:for,
  arc:script, arc:call, arc:throw, arc:try, arc:catch, arc:log, arc:map, etc.)
- **Python** — used inside an Arc Script connector (language="python")
- **Both** — if the task benefits from combining them

Statewhich type you are generating and why.

## 2. Generated Script
Output the complete, ready-to-paste script inside a fenced code block with the
correct language tag (xml for ArcScript, python for Python).

Rules for ArcScript:
- Use exact attribute and element names as shown in the documentation.
- All arc: namespace tags must be lowercase (arc:set, arc:if, arc:elseif,
  arc:else, arc:for, arc:while, arc:try, arc:catch, arc:throw, arc:log,
  arc:call, arc:script, arc:map, arc:date, arc:math).
- String values go in the value attribute; XPath/expressions use xpath="...".
- Output variables with arc:set attr="output:FieldName" value="...".
- Wrap the full script in <arc:script xmlns:arc="http://www.cdata.com/arc">...</arc:script>.

Rules for Python:
- Use the `arcInput` dict for incoming Arc message fields.
- Use the `arcOutput` dict to pass values back to Arc.
- Do not use external libraries unless they are standard-library or specified.
- Always include proper error handling (try/except).

## 3. How It Works
A short (3-6 bullet) plain-English explanation of what the script does, step by step.

## 4. Where to Place / Configure It in Arc
Exact instructions:
- Which Arc connector or port to drop this script into
- Which field or script window to paste it
- Any pre-conditions (input fields expected, connector settings needed)

## 5. Real Examples & Patterns from Docs
Copy any directly relevant code examples from the documentation context
VERBATIM into a fenced code block. Label each one with its source.

## 6. Script Patterns Learned from Jira
If any of the Jira tickets below contain ArcScript or Python code snippets,
extract the relevant patterns and explain how you applied or adapted them
to produce the generated script above. Format as:
- **ARCESB-XXXXX** — what pattern was reused and how

If no script patterns were found in the Jira context, omit this section.

## 7. Related Jira Tickets (Scripting Issues)
List any other tickets from the Jira context relevant to this scripting area:
- **ARCESB-XXXXX** — one-line description
If none, omit this section.

---
Strict rules:
- NEVER invent arc: tags — only use tags that appear in the documentation OR
  in the Jira script examples below. Jira examples are authoritative evidence
  of working syntax even when the docs are incomplete.
- NEVER guess Python field names — use only what the requirement or an example confirms.
- If neither docs nor Jira cover part of the requirement, say so explicitly.
- Reproduce all code examples character-for-character — do not paraphrase.
- When a Jira ticket contains a working script, treat it as a REAL example and
  build upon it — adapt it to match the user's requirement.

### Arc Scripting Documentation
{docs_context}

### Jira Tickets (including real script examples from past tickets)
{jira_context}
"""


def _extract_arc_connector_config(text: str) -> dict | None:
    """
    Return a dict of Arc connector config fields if *text* contains a JSON
    blob with at least a ConnectorType key (e.g. from the Arc config panel).

    Returns None if no such structure is found.
    """
    m = re.search(r'\{[^{}]*"ConnectorType"\s*:\s*"[^"]+"[^{}]*\}', text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None


def generate_script(requirement: str) -> dict[str, Any]:
    """
    Generate a working ArcScript or Python script from a plain-English requirement.

    Retrieves scripting docs and relevant Jira tickets (including tickets that
    contain real script examples), then produces a complete, annotated script.

    Also handles Arc connector config JSON blobs (e.g. {"ConnectorId": "Roopesh",
    "WorkspaceId": "Default", "ConnectorType": "JSON"}) by extracting the
    ConnectorType and enriching the LLM message with full connector context so
    the generated script actually sets output fields instead of producing empty files.
    """
    # Pre-process: if the requirement is (or contains) an Arc connector config
    # JSON blob, extract the structured fields and build a richer human message.
    arc_config = _extract_arc_connector_config(requirement)
    if arc_config:
        connector_type  = arc_config.get("ConnectorType", "Unknown")
        connector_id    = arc_config.get("ConnectorId",   "Unknown")
        workspace_id    = arc_config.get("WorkspaceId",   "Default")
        human_requirement = (
            f"Generate a complete Arc script for the following connector configuration:\n\n"
            f"- **ConnectorType**: {connector_type}\n"
            f"- **ConnectorId**:   {connector_id}\n"
            f"- **WorkspaceId**:   {workspace_id}\n\n"
            f"The script must:\n"
            f"1. Read all available input fields from the incoming Arc message "
            f"(use arcInput for Python or XPath in ArcScript).\n"
            f"2. Process or pass through each field to the output "
            f"(use arcOutput for Python or arc:set attr=\"output:FieldName\" in ArcScript).\n"
            f"3. Include error handling so that a missing field does not leave "
            f"the output empty.\n"
            f"4. Log key values using arc:log (ArcScript) or print() (Python) "
            f"for traceability in the Arc activity log.\n\n"
            f"Do NOT generate a no-op or empty script. Every output field must "
            f"be explicitly set."
        )
    else:
        human_requirement = requirement

    # Build search queries focused on scripting docs and relevant behaviour
    scripting_queries = [
        human_requirement,
        f"ArcScript {human_requirement}",
        "ArcScript arc:set arc:if arc:for arc:script syntax examples",
        "Python scripting Arc connector arcInput arcOutput",
        "ArcScript functions methods string date math operations",
    ]

    connector = detect_connector(requirement)  # uses ConnectorType from JSON if present
    expanded  = expand_query(human_requirement, connector=connector, use_llm=True)
    all_queries = list(dict.fromkeys(scripting_queries + expanded))

    docs, jira_docs, _ = retrieve_all(
        all_queries,
        docs_k=10,
        jira_k=6,
        confluence_k=0,
        connector_filter=connector,
    )

    # Second pass: specifically fetch Jira tickets that contain real script
    # examples (tagged has_script=true during ingestion).  These are the most
    # valuable learning signal — actual working scripts from real tickets.
    store = get_store()
    try:
        script_example_docs = store.similarity_search_jira(
            query=human_requirement,
            k=8,
            filter_metadata={"has_script": {"$eq": "true"}},
        )
    except Exception:
        script_example_docs = []

    # Merge and deduplicate by page_content hash
    seen_content: set[str] = {id(d) for d in jira_docs}
    for d in script_example_docs:
        content_key = d.page_content[:120]
        if content_key not in seen_content:
            seen_content.add(content_key)  # type: ignore[arg-type]
            jira_docs.append(d)

    # Rerank the combined pool keeping scripting docs at the top
    combined = docs + jira_docs
    if combined:
        reranked  = rerank(human_requirement, combined, top_n=16)
        docs      = [d for d in reranked if d.metadata.get("source") != "jira"][:10]
        jira_docs = [d for d in reranked if d.metadata.get("source") == "jira"][:8]

    system = _SCRIPT_SYSTEM.format(
        docs_context = build_docs_context(docs),
        jira_context = build_jira_context(jira_docs),
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Generate an Arc script for this requirement:\n\n{human_requirement}"),
    ]

    response = _get_llm().invoke(messages)
    return {
        "answer":      response.content,
        "sources":     _build_sources(docs, jira_docs),
        "jira_issues": _jira_issues_from_docs(jira_docs),
        "tool":        "script_generator",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Script Debugger / Fixer
# ─────────────────────────────────────────────────────────────────────────────

_FIX_SCRIPT_SYSTEM = """\
You are ArcMind's Script Debugger for CData Arc.

The user has provided a broken ArcScript or Python script and a description of
what is wrong or what error they are seeing. Your job is to diagnose and fix it.

Respond with ALL of the following sections:

## 1. What Went Wrong
Identify the exact problem — wrong tag name, wrong attribute, logic error,
missing namespace, wrong variable name, incorrect syntax, etc.
Be specific about which line or element is incorrect.

## 2. Root Cause
Explain WHY it fails. Reference the correct behaviour from the Arc documentation
context below. Quote the relevant doc rule or syntax requirement.

## 3. Fixed Script
Output the complete corrected script inside a fenced code block (`xml` for
ArcScript, `python` for Python).
Mark every changed line with a `← FIXED` inline comment so the user can
immediately see what changed.

## 4. Changes Made
A bullet list of every change and why it was necessary.

## 5. How to Verify
Tell the user what to check in Arc's activity log, output fields, or connector
result to confirm the fix worked.

---
Strict rules:
- NEVER invent arc: tags — only use tags confirmed by the documentation.
- If the documentation does not clarify the correct syntax, say so explicitly.
- Reproduce the fixed script in full — do not abbreviate or use placeholders.

### Arc Scripting Documentation
{docs_context}

### Relevant Jira Issues (known scripting bugs / workarounds)
{jira_context}
"""


def _extract_json_test_data(text: str) -> str | None:
    """
    Return the JSON content from *text* if it looks like test input data
    (a JSON object or array the user pasted to demonstrate the script failed),
    or None if it reads like a natural-language error description.

    Heuristic: text is (or contains) a JSON object/array with ≥2 key-value
    pairs and does NOT start with common error-message phrases.
    """
    stripped = text.strip()
    # Quick rejection: natural-language error messages never start with { or [
    error_phrases = ("error", "exception", "the script", "when i", "it does", "failed")
    if not (stripped.startswith("{") or stripped.startswith("[")):
        # Look for an embedded JSON block
        m = re.search(r'(\{[\s\S]{10,}\}|\[[\s\S]{10,}\])', stripped)
        if not m:
            return None
        candidate = m.group(1)
    else:
        candidate = stripped

    # Must contain at least 2 "key": value pairs to qualify as data (not a tiny cfg blob)
    if len(re.findall(r'"[^"]+"\s*:', candidate)) < 2:
        return None

    try:
        parsed = json.loads(candidate)
        return json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, ValueError):
        # Not perfectly valid JSON but still looks enough like data
        return candidate[:4000]


def fix_script(script: str, error_description: str) -> dict[str, Any]:
    """
    Diagnose and fix a broken ArcScript or Python script.

    Takes the broken script and a description of what is wrong,
    retrieves relevant scripting docs, and returns a fully corrected
    script with a detailed explanation of every change.

    Special case: when the user pastes a JSON test input file as the
    "error description" (the script ran on that data but produced no output),
    the function detects this automatically and reframes the LLM prompt to
    explain the real task — fix the script so it correctly transforms the
    provided JSON data into XML output.
    """
    # ── Detect "JSON test data" vs "natural-language error message" ──────────
    json_test_data = _extract_json_test_data(error_description)

    if json_test_data:
        # The user pasted a JSON input file to show the script produced no output.
        # Build an explicit error context so the LLM understands the actual task.
        effective_error = (
            "The script is supposed to convert the JSON input data shown below "
            "into XML output. When run with this input the script produces no XML "
            "output (empty result). Fix the script so it correctly transforms every "
            "field in the JSON into valid XML elements."
        )
        human_content = (
            f"Here is the broken script:\n\n```\n{script}\n```\n\n"
            f"**JSON test input** (the data the script must convert to XML):\n\n"
            f"```json\n{json_test_data}\n```\n\n"
            f"**Problem:** the script produces **no XML output** when run with "
            f"this input. Diagnose why and provide a fully corrected script that "
            f"transforms every field in the JSON above into valid XML."
        )
        # Focused queries for JSON→XML conversion docs and scripting patterns
        scripting_queries = [
            "JSON to XML conversion ArcScript",
            "ArcScript JSON input parse transform output XML",
            "ArcScript arc:set arc:map JSON XML mapping",
            "Python arcInput arcOutput JSON XML conversion",
            "JSON connector XML output Arc script",
        ]
    else:
        effective_error = error_description
        human_content = (
            f"Here is the broken script:\n\n```\n{script}\n```\n\n"
            f"Problem / error description:\n{error_description}\n\n"
            "Please diagnose and fix it."
        )
        scripting_queries = [
            error_description,
            f"ArcScript {error_description}",
            "ArcScript arc:set arc:if arc:for arc:script syntax correct usage",
            "Python scripting Arc connector arcInput arcOutput error",
            "ArcScript common mistakes debugging",
        ]

    combined_text = f"{effective_error}\n\n{script}"
    connector = detect_connector(combined_text)
    expanded  = expand_query(effective_error, connector=connector, use_llm=True)
    all_queries = list(dict.fromkeys(scripting_queries + expanded))

    docs, jira_docs, _ = retrieve_all(
        all_queries,
        docs_k=10,
        jira_k=6,
        confluence_k=0,
        connector_filter=connector,
    )

    combined = docs + jira_docs
    if combined:
        reranked  = rerank(effective_error, combined, top_n=14)
        docs      = [d for d in reranked if d.metadata.get("source") != "jira"][:10]
        jira_docs = [d for d in reranked if d.metadata.get("source") == "jira"][:6]

    system = _FIX_SCRIPT_SYSTEM.format(
        docs_context = build_docs_context(docs),
        jira_context = build_jira_context(jira_docs),
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=human_content),
    ]

    response = _get_llm().invoke(messages)
    return {
        "answer":      response.content,
        "sources":     _build_sources(docs, jira_docs),
        "jira_issues": _jira_issues_from_docs(jira_docs),
        "tool":        "script_debugger",
    }
