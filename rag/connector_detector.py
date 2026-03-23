"""
Arc connector detector for ArcMind.

Scans a user query for mentions of known CData Arc connectors and returns
the primary connector name so downstream components can:
  - boost connector-specific retrieval
  - select static query expansions
  - filter metadata

Connectors covered: AS2, SFTP, OFTP, X12, EDIFACT, Peppol, HTTP, FTP,
                    SMTP, REST, ArcScript, FlatFile, XML, JSON, Database,
                    Flows, Profiles
"""

from __future__ import annotations

import re

# connector name → list of regex patterns that identify it
# (ordered most-specific → least-specific within each entry)
_CONNECTOR_PATTERNS: dict[str, list[str]] = {
    "ArcScript": [r"\bArcScript\b",   r"\barc:set\b",     r"\barc:call\b"],
    "EDIFACT":   [r"\bEDIFACT\b",     r"\bUN/EDIFACT\b"],
    "Peppol":    [r"\bPEPPOL\b",      r"\bPeppol\b",      r"\bPEPP\b"],
    "OFTP":      [r"\bOFTP\b",        r"\bOdette\b"],
    "AS2":       [r"\bAS2\b",         r"\bAS-2\b"],
    "SFTP":      [r"\bSFTP\b"],
    "X12":       [r"\bX\.?12\b",      r"\bANSI\s*X12\b",  r"\bEDI\s*X12\b"],
    "FTP":       [r"\bFTPS?\s+connector\b", r"\bFTP\s+connector\b"],
    "SMTP":      [r"\bSMTP\b",        r"\bPOP3\b",         r"\bIMAP\b"],
    "REST":      [r"\bREST\s*API\s+connector\b", r"\bRESTful\s+connector\b"],
    "HTTP":      [r"\bHTTPS?\s+connector\b"],
    "FlatFile":  [r"\bflat.?file\b",  r"\bCSV\s+connector\b"],
    "XML":       [r"\bXML\s+connector\b", r"\bXML\s+mapper\b"],
    "JSON":      [r"\bJSON\s+connector\b", r"\bJSON\s+mapper\b"],
    "Database":  [r"\bJDBC\b",        r"\bdatabase\s+connector\b"],
    "Flows":     [r"\bflow\b",        r"\bworkflow\b"],
    "Profiles":  [r"\bprofile\b"],
}


def _extract_connector_type_from_json(text: str) -> str | None:
    """
    Extract ConnectorType from an Arc connector config JSON blob.

    Handles inputs like {"ConnectorType": "JSON", ...} that come from
    the Arc connector configuration panel or generate-script requirements.
    """
    m = re.search(r'"ConnectorType"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if not m:
        return None
    connector_type = m.group(1).strip()
    for connector in _CONNECTOR_PATTERNS:
        if connector.lower() == connector_type.lower():
            return connector
    return None


def detect_connector(query: str) -> str | None:
    """
    Return the first Arc connector name matched in *query*, or None.

    Checks for a ConnectorType JSON field first (e.g. Arc connector config
    blobs), then falls back to regex pattern matching.
    Matching is case-insensitive.  Patterns are tested in the order they
    appear in _CONNECTOR_PATTERNS (most-specific first).
    """
    cfg_connector = _extract_connector_type_from_json(query)
    if cfg_connector:
        return cfg_connector
    for connector, patterns in _CONNECTOR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return connector
    return None


def detect_all_connectors(query: str) -> list[str]:
    """Return all connector names mentioned in *query* (no duplicates)."""
    found: list[str] = []
    for connector, patterns in _CONNECTOR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                found.append(connector)
                break   # move to next connector once matched
    return found
