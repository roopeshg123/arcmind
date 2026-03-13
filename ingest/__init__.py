# Re-export top-level functions so `from ingest import run_ingestion` works
# even though ingest/ is a package (which shadows the old ingest.py module).
from ingest.ingest_docs import ingest_docs as _ingest_docs
from ingest.ingest_jira import ingest_jira as _ingest_jira, incremental_jira_sync
import os


def run_ingestion(docs_dir=None, reset=True):
    return _ingest_docs(docs_dir=docs_dir, reset=reset)


def run_jira_ingestion(jql=None, reset=False):
    return _ingest_jira(jql=jql, reset=reset)


def run_incremental_jira_sync(hours=1):
    return incremental_jira_sync(hours=hours)


__all__ = ["run_ingestion", "run_jira_ingestion", "run_incremental_jira_sync"]
