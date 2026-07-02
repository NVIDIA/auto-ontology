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
# Relationship from an Admin node directly to a catalog item that is not
# connected to any zone.  Keeps the Neo4j graph navigable for data that exists
# in the application but has not yet been assigned to any zone.
REL_HAS_DIRECT_ACCESS = "HAS_DIRECT_ACCESS"
