# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j foreign-key and join edge reads.

Only the Cypher query function lives here. The non-graph helpers
(_apply_foreign_key_hints, get_relevant_fks_from_candidates_tables,
get_relevant_tables_with_fks) remain in
gsf/retrieval/data_access/foreign_keys.py.
"""

from __future__ import annotations

import json
import logging

from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

logger = logging.getLogger(__name__)


def get_relevant_fks(tables_ids: list) -> list[dict]:
    """Expand up to 3 FK/join hops and return all FK pairs among connected tables."""
    query = """
    // Start with target tables and expand outward to find connected tables
    WITH $tables_ids as current_ids

    // Level 1: Find tables connected via FK
    OPTIONAL MATCH (t0:table WHERE t0.id IN current_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t1:table)
    WITH current_ids, collect(DISTINCT t1.id) as new_ids_1
    WITH current_ids + new_ids_1 as level_1_ids

    // Level 2
    OPTIONAL MATCH (t1:table WHERE t1.id IN level_1_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t2:table)
    WITH level_1_ids, collect(DISTINCT t2.id) as new_ids_2
    WITH level_1_ids + new_ids_2 as level_2_ids

    // Level 3
    OPTIONAL MATCH (t2:table WHERE t2.id IN level_2_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t3:table)
    WITH level_2_ids, collect(DISTINCT t3.id) as new_ids_3
    WITH level_2_ids + new_ids_3 as all_table_ids

    // Get all FK relationships between these tables
    MATCH (t1:table)-[:schema]->(col1:column)-[:fk]-(col2:column)<-[:schema]-(t2:table)
    WHERE t1.id IN all_table_ids AND t2.id IN all_table_ids
      AND t1.id < t2.id

    RETURN collect(DISTINCT {
        table1: t1.schema_name + '.' + t1.name,
        column1: col1.name,
        column1_datatype: coalesce(col1.data_type, 'None'),
        table2: t2.schema_name + '.' + t2.name,
        column2: col2.name,
        column2_datatype: coalesce(col2.data_type, 'None')
    }) as list_of_foreign_keys
    """
    results = get_neo4j_conn().query_read(query, {"tables_ids": tables_ids})
    result_fks = results[0]["list_of_foreign_keys"] if results else []

    query_joins = """
    OPTIONAL MATCH (t0:table WHERE t0.id IN $tables_ids)-[:join]-(t1:table)
    WITH collect(DISTINCT t1.id) as new_ids_1
    WITH $tables_ids + new_ids_1 as level_1_ids

    OPTIONAL MATCH (t1:table WHERE t1.id IN level_1_ids)-[:join]-(t2:table)
    WITH level_1_ids, collect(DISTINCT t2.id) as new_ids_2
    WITH level_1_ids + new_ids_2 as level_2_ids

    OPTIONAL MATCH (t2:table WHERE t2.id IN level_2_ids)-[:join]-(t3:table)
    WITH level_2_ids, collect(DISTINCT t3.id) as new_ids_3
    WITH level_2_ids + new_ids_3 as all_table_ids

    MATCH (t1:table)-[rel:join]-(t2:table)
    WHERE t1.id IN all_table_ids AND t2.id IN all_table_ids
      AND t1.id < t2.id
      AND rel.join IS NOT NULL

    WITH t1, t2, rel,
         trim(apoc.text.split(rel.join, '<=|>=|=|<|>')[0]) as left_side,
         trim(apoc.text.split(rel.join, '<=|>=|=|<|>')[1]) as right_side

    WITH t1, t2, rel, left_side, right_side,
         trim(split(left_side, '.')[0]) as left_schema,
         trim(split(left_side, '.')[1]) as left_table,
         trim(split(left_side, '.')[2]) as left_column,
         trim(split(right_side, '.')[0]) as right_schema,
         trim(split(right_side, '.')[1]) as right_table,
         trim(split(right_side, '.')[2]) as right_column
    WHERE left_schema IS NOT NULL AND left_table IS NOT NULL AND left_column IS NOT NULL
      AND right_schema IS NOT NULL AND right_table IS NOT NULL AND right_column IS NOT NULL

    OPTIONAL MATCH (s1:schema{name: left_schema})-[:schema]->
        (tbl1:table{name: left_table})-[:schema]->(col1:column{name: left_column})

    OPTIONAL MATCH (s2:schema{name: right_schema})-[:schema]->
        (tbl2:table{name: right_table})-[:schema]->(col2:column{name: right_column})

    RETURN collect(DISTINCT {
        table1: t1.schema_name + '.' + t1.name,
        column1: coalesce(col1.name, left_column),
        column1_datatype: coalesce(col1.data_type, 'None'),
        table2: t2.schema_name + '.' + t2.name,
        column2: coalesce(col2.name, right_column),
        column2_datatype: coalesce(col2.data_type, 'None')
    }) as list_of_foreign_keys
    """
    results = get_neo4j_conn().query_read(query_joins, {"tables_ids": tables_ids})
    result_joins = results[0]["list_of_foreign_keys"] if results else []

    combined = result_fks + result_joins
    unique_strings = set(json.dumps(d, sort_keys=True) for d in combined)
    unique_results = [json.loads(s) for s in unique_strings]

    key_order = [
        "table1",
        "column1",
        "column1_datatype",
        "table2",
        "column2",
        "column2_datatype",
    ]
    return [{key: d[key] for key in key_order} for d in unique_results]
