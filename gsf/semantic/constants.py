"""Semantic graph labels and relationship types."""

from __future__ import annotations

SEMANTIC_SOURCE = "semantic"

# Source values stored on SqlAttribute nodes.
SQL_ATTR_SOURCE_MANUAL = "manual"
SQL_ATTR_SOURCE_SQL = "sql"
SQL_ATTR_SOURCE_TABLE = "table"
SQL_ATTR_SOURCE_BRIDGE = "bridgeTable"

# Semantic node labels
LABEL_TERM = "Term"
LABEL_COLUMN_ATTRIBUTE = "ColumnAttribute"
LABEL_SQL_ATTRIBUTE = "SqlAttribute"
LABEL_TEXT_ATTRIBUTE = "TextAttribute"
LABEL_ANALYSIS = "Analysis"

# PQL (predictive) custom analyses — the KumoRFM-prediction twin of CustomAnalysis.
# Stored under their own label so they never mix into the SQL text-to-SQL retrieval;
# retrieved only as few-shot examples for PQL generation.
LABEL_PQL_ANALYSIS = "PqlAnalysis"

# Semantic relationship types
REL_HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
REL_PART_OF = "PART_OF"
REL_PROPERTY_OF = "PROPERTY_OF"
REL_IS_A = "IS_A"
REL_ROLE = "ROLE"
REL_REPRESENTS = "REPRESENTS"
REL_SEMANTIC_FK = "SEMANTIC_FK"

# Additional source value for SqlAttribute nodes generated from pure-FK
# bridge (junction) tables. Sits alongside SQL_ATTR_SOURCE_MANUAL /
# SQL_ATTR_SOURCE_SQL, which remain defined in gsf.dal.sql_attributes.
SQL_ATTR_SOURCE_BRIDGE = "bridgeTable"
