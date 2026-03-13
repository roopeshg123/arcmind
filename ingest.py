"""
ArcMind Ingestion Entry Point

Orchestrates the documentation and Jira ingestion pipelines.

Backward-compatible API
-----------------------
    run_ingestion(docs_dir, reset)   — used by main.py's /api/ingest endpoint

New public helpers
------------------
    run_jira_ingestion(jql, reset)         — full Jira ingest
    run_incremental_jira_sync(hours)       — incremental Jira sync (last N hours)

CLI usage
---------
    python ingest.py                              # ingest docs from DOCS_DIR
    python ingest.py --docs-dir ./my_docs         # custom docs directory
    python ingest.py --reset                      # drop existing data first
    python ingest.py --jira                       # also ingest Jira issues
    python ingest.py --jira-jql "project=ARCESB"  # custom JQL
    python ingest.py --jira-sync                  # incremental sync (last 1 h)
"""

import argparse
import logging
import os

from dotenv import load_dotenv

from ingest.ingest_docs import ingest_docs

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")


# ---------------------------------------------------------------------------
# Public API (consumed by main.py and tests)
# ---------------------------------------------------------------------------

def run_ingestion(docs_dir: str | None = None, reset: bool = True) -> dict:
    """
    Run the documentation ingestion pipeline.

    Backward-compatible entry point used by the FastAPI /api/ingest endpoint.
    Delegates to ingest.ingest_docs.ingest_docs().
    """
    return ingest_docs(docs_dir=docs_dir, reset=reset)


def run_jira_ingestion(jql: str | None = None, reset: bool = False) -> dict:
    """Run the full Jira ingestion pipeline."""
    from ingest.ingest_jira import ingest_jira
    return ingest_jira(jql=jql, reset=reset)


def run_incremental_jira_sync(hours: int = 1) -> dict:
    """Sync Jira issues updated in the last *hours* hours (append mode)."""
    from ingest.ingest_jira import incremental_jira_sync
    return incremental_jira_sync(hours=hours)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ArcMind — documentation & Jira ingestion pipeline"
    )
    parser.add_argument(
        "--docs-dir", default=None,
        help="Path to the HTML documentation directory (overrides DOCS_DIR env).",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop and recreate the affected collection(s) before ingesting.",
    )
    parser.add_argument(
        "--jira", action="store_true",
        help="Also ingest Jira issues after the documentation ingest.",
    )
    parser.add_argument(
        "--jira-jql", default=None,
        help="Custom JQL query for Jira ingestion.",
    )
    parser.add_argument(
        "--jira-sync", action="store_true",
        help="Run an incremental Jira sync (issues updated in the last 1 hour).",
    )
    args = parser.parse_args()

    if args.jira_sync:
        result = run_incremental_jira_sync(hours=1)
        log.info("Incremental Jira sync: %s", result)
    else:
        doc_result = run_ingestion(docs_dir=args.docs_dir, reset=args.reset)
        log.info("Docs ingest: %s", doc_result)

        if args.jira:
            jira_result = run_jira_ingestion(jql=args.jira_jql, reset=args.reset)
            log.info("Jira ingest: %s", jira_result)

