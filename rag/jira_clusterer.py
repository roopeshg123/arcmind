"""
Jira issue clusterer for ArcMind.

Groups a flat list of Jira Documents into thematic clusters so the LLM
receives organised, structured context rather than an unordered pile.

Example output passed to the LLM:

    ### Certificate Issues
    - **ARCESB-12011**: certificate validation failure
    - **ARCESB-12044**: SSL handshake error

    ### MDN / Acknowledgement
    - **ARCESB-13051**: MDN response not received
"""

from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Topic classification patterns
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS: dict[str, list[str]] = {
    "Certificate Issues": [
        r"\bcertificate\b", r"\bcert\b", r"\bSSL\b", r"\bTLS\b",
        r"\bprivate\s+key\b", r"\bpublic\s+key\b",
        r"\bkeystore\b",    r"\btruststore\b",
    ],
    "MDN / Acknowledgement": [
        r"\bMDN\b",  r"\backnowledgement\b", r"\bACK\b",
        r"\breceipt\b", r"\bdelivery\s+notification\b",
    ],
    "Authentication / Login": [
        r"\bauth\w*\b", r"\blogin\b", r"\bpassword\b",
        r"\bcredential\b", r"\bOAuth\b", r"\btoken\b", r"\bAPI\s+key\b",
    ],
    "Connection / Timeout": [
        r"\btimeout\b", r"\bconnection\s*refused\b",
        r"\bfirewall\b", r"\bport\b", r"\bnetwork\s+error\b",
    ],
    "Encryption / Signing": [
        r"\bencrypt\w*\b", r"\bdecrypt\w*\b",
        r"\bsigning\b", r"\bsignature\b", r"\bAES\b", r"\bRSA\b",
    ],
    "Configuration": [
        r"\bconfigur\w*\b", r"\bsetting\b",
        r"\bsetup\b", r"\binstall\w*\b",
    ],
    "Performance": [
        r"\bperformance\b", r"\bslow\b", r"\blatency\b",
        r"\bthroughput\b", r"\bmemory\s+leak\b",
    ],
    "UI / Interface": [
        r"\bUI\b", r"\binterface\b", r"\bdashboard\b",
        r"\bdisplay\b", r"\brender\b",
    ],
    "File Transfer": [
        r"\bfile\s*transfer\b", r"\bupload\b", r"\bdownload\b",
        r"\btransmission\b",
    ],
    "Mapping / Transformation": [
        r"\bmapping\b", r"\btransform\w*\b",
        r"\bXSLT\b", r"\bconvert\w*\b", r"\bfield\s+map\b",
    ],
    "Error / Exception": [
        r"\berror\b", r"\bexception\b", r"\bfailure\b",
        r"\bcrash\b", r"\bstack\s+trace\b",
    ],
}


def _classify(text: str) -> str:
    """Assign a topic label to *text* using the first matching pattern group."""
    for topic, patterns in _TOPIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return topic
    return "Other Issues"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_jira_docs(
    jira_docs: list[Document],
) -> dict[str, list[Document]]:
    """
    Group *jira_docs* into thematic clusters.

    Returns:
        OrderedDict  { cluster_label → [Document, …] }
    """
    clusters: dict[str, list[Document]] = defaultdict(list)
    for doc in jira_docs:
        search_text = (
            doc.metadata.get("ticket", "") + " " + doc.page_content[:500]
        )
        label = _classify(search_text)
        clusters[label].append(doc)
    return dict(clusters)


def format_jira_clusters(clusters: dict[str, list[Document]]) -> str:
    """
    Render *clusters* as a Markdown-formatted string for LLM context.

    Each cluster becomes a heading. Main ticket documents are rendered as a
    bullet summary line. Comment documents are rendered with their full body
    so the LLM can read discussion details (e.g. API key mentions, decisions).
    """
    if not clusters:
        return "No relevant Jira issues found."

    lines: list[str] = []
    for topic, docs in sorted(clusters.items()):
        lines.append(f"### {topic}")
        for doc in docs:
            ticket     = doc.metadata.get("ticket", "Unknown")
            status     = doc.metadata.get("status", "")
            doc_type   = doc.metadata.get("type", "")
            status_tag = f" [{status}]" if status else ""

            if doc_type == "comment":
                # Include full comment content so the LLM can read it
                lines.append(f"- **{ticket}**{status_tag} (comment):\n{doc.page_content.strip()}")
            else:
                # Include FULL ticket content (description + metadata) so the LLM
                # can read XML examples, workarounds, and proposed solutions.
                # The description is already capped at 3000 chars during ingest.
                lines.append(f"#### {ticket}{status_tag}\n{doc.page_content.strip()}")
        lines.append("")   # blank line between clusters

    return "\n".join(lines).strip()
