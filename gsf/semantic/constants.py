"""Semantic graph labels and relationship types."""

from __future__ import annotations

SEMANTIC_SOURCE = "semantic"

# Semantic node labels
LABEL_TERM = "Term"
LABEL_COLUMN_ATTRIBUTE = "ColumnAttribute"
LABEL_SQL_ATTRIBUTE = "SqlAttribute"
LABEL_TEXT_ATTRIBUTE = "TextAttribute"
LABEL_ANALYSIS = "Analysis"

# Semantic node labels
LABEL_ZONE = "zone"

# Semantic relationship types
REL_HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
REL_PARTICIPANT_OF = "participant_of"
REL_PART_OF = "PART_OF"
REL_PROPERTY_OF = "PROPERTY_OF"
REL_IS_A = "IS_A"
REL_ROLE = "ROLE"
REL_REPRESENTS = "REPRESENTS"
REL_SEMANTIC_FK = "SEMANTIC_FK"
REL_ZONE_OF = "zone_of"
