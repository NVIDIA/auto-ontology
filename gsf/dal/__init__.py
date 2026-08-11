"""GSF Neo4j data access layer, organized by topic.

Each module is the single source of truth for a domain:
  datasources     — Database / Schema / Table / Column catalog
  terms           — Term CRUD and synonym reads (semantic compilation)
  attributes      — ColumnAttribute, SemanticFK, join path traversal
  custom_analyses — CustomAnalysis / Sql subgraph
  sql_attributes  — SqlAttribute / Sql subgraph
  connections     — UI-managed database connection metadata on DB nodes
  reset           — Deleting a database's catalog/semantic nodes and embeddings
  candidates      — Vector-hit graph enrichment at retrieval time
  users           — Zone-scope helpers for catalog queries. Users and
                    user-to-zone relationships are not stored in Neo4j.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def close_store() -> None:
    """Release every connection the DAL holds. Idempotent.

    The public shutdown hook for the data store, so callers don't reach into
    driver internals. There are two connections to release, and the shutdown
    path used to close neither cleanly:

    * the auto-commit singleton owned by ``nemo_retriever``, which callers
      previously closed by poking its private module global, and
    * the driver :mod:`gsf.dal.neo4j_tx` opens for explicit write transactions,
      which nothing closed at all.

    Imports are deferred so importing :mod:`gsf.dal` stays cheap and this
    module keeps no import-time dependency on the driver.

    When the Postgres backend lands this also disposes the SQLAlchemy engine —
    the point of the indirection is that ``__main__`` never has to know which.
    """
    from nemo_retriever.tabular_data.neo4j import neo4j_connection

    from gsf.dal import neo4j_tx

    if neo4j_connection._conn is not None:
        try:
            neo4j_connection._conn.close()
        except Exception:
            logger.warning(
                "close_store: shared connection failed to close", exc_info=True
            )
        finally:
            neo4j_connection._conn = None

    if neo4j_tx._driver is not None:
        try:
            neo4j_tx._driver.close()
        except Exception:
            logger.warning(
                "close_store: transaction driver failed to close", exc_info=True
            )
        finally:
            neo4j_tx._driver = None
