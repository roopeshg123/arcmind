"""
Query router for ArcMind.

Examines the user query and decides which retrieval strategy to use:

  ticket_direct  — query contains one or more Jira ticket IDs (e.g. ARCESB-12011)
                   → fetch those tickets directly by ID

  hybrid_search  — no ticket IDs detected
                   → run the full hybrid retrieval pipeline
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Matches standard Jira-style ticket IDs: PROJECT-1234 (uppercase letters + digits)
_TICKET_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")


class RouteDecision(NamedTuple):
    strategy:   str          # "ticket_direct" | "hybrid_search"
    ticket_ids: list[str]    # non-empty when strategy == "ticket_direct"


def route_query(query: str) -> RouteDecision:
    """
    Decide the retrieval strategy for *query*.

    Returns a RouteDecision with the chosen strategy and any extracted IDs.
    """
    ids = list(dict.fromkeys(_TICKET_RE.findall(query)))   # deduplicated, order-preserving
    if ids:
        return RouteDecision(strategy="ticket_direct", ticket_ids=ids)
    return RouteDecision(strategy="hybrid_search", ticket_ids=[])


def extract_ticket_ids(query: str) -> list[str]:
    """Return all unique Jira ticket IDs found in *query*."""
    return list(dict.fromkeys(_TICKET_RE.findall(query)))
