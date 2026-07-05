"""GSF Neo4j data access layer, organized by topic.

Each module is the single source of truth for a domain:
  datasources     — Database / Schema / Table / Column catalog
  terms           — Term CRUD and synonym reads (semantic compilation)
  attributes      — ColumnAttribute, SemanticFK, join path traversal
  custom_analyses — CustomAnalysis / Sql subgraph
  sql_attributes  — SqlAttribute / Sql subgraph
  foreign_keys    — FK and join edge traversal
  connections     — UI-managed database connection metadata on DB nodes
  candidates      — Vector-hit graph enrichment at retrieval time
  users           — User node mirror of the PostgreSQL Better Auth records;
                    each User is linked to every Zone via :PARTICIPANT_OF
"""
