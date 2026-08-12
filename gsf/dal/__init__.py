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

    * the auto-commit singleton in :mod:`gsf.catalog.store.neo4j.connection` (owned by
      ``nemo_retriever`` until Phase 1 forked it), which callers previously
      closed by poking its private module global, and
    * the driver :mod:`gsf.dal.neo4j_tx` opens for explicit write transactions,
      which nothing closed at all.

    …and, since Phase 3, the pooled SQLAlchemy engine.

    **All three are closed unconditionally, whatever ``GSF_STORE`` says.** The
    setting decides which store is *used*; it does not tell you which
    connections a process has actually opened. A run that ingested through one
    backend and was reconfigured, or a test that touched both, would leak the
    other's sockets if shutdown trusted the flag. Closing what is open is
    cheaper than reasoning about what should be.

    Imports are deferred so importing :mod:`gsf.dal` stays cheap and this
    module keeps no import-time dependency on either driver.
    """
    from gsf.catalog.store.neo4j import connection

    from gsf.dal import neo4j_tx
    from gsf.dal.pg.session import dispose_engine

    if connection._conn is not None:
        try:
            connection._conn.close()
        except Exception:
            logger.warning(
                "close_store: shared connection failed to close", exc_info=True
            )
        finally:
            connection._conn = None

    if neo4j_tx._driver is not None:
        try:
            neo4j_tx._driver.close()
        except Exception:
            logger.warning(
                "close_store: transaction driver failed to close", exc_info=True
            )
        finally:
            neo4j_tx._driver = None

    # Already idempotent and already swallows its own failures.
    dispose_engine()
