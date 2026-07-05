"""Semantic graph labels and relationship types."""

from __future__ import annotations

SEMANTIC_SOURCE = "semantic"

# Semantic node labels
LABEL_TERM = "Term"
LABEL_COLUMN_ATTRIBUTE = "ColumnAttribute"
LABEL_SQL_ATTRIBUTE = "SqlAttribute"
LABEL_TEXT_ATTRIBUTE = "TextAttribute"
LABEL_ANALYSIS = "Analysis"

# Semantic relationship types
REL_HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
REL_PROPERTY_OF = "PROPERTY_OF"
REL_IS_A = "IS_A"
REL_PART_OF = "PART_OF"
REL_ROLE = "ROLE"
REL_REPRESENTS = "REPRESENTS"
REL_SEMANTIC_FK = "SEMANTIC_FK"
