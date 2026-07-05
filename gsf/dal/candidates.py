# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate retrieval: vector-hit graph enrichment via Neo4j.

Only ``expand_info`` lives here (it calls ``get_neo4j_conn()`` directly).
The orchestration helpers that do not touch Neo4j directly
(``_get_candidates_information``, ``_dedupe_best_score_sort_cap``,
``extract_candidates``) remain in
``gsf/retrieval/data_access/candidates.py``.
"""

from __future__ import annotations

import logging
from itertools import groupby

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn


logger = logging.getLogger(__name__)


def expand_info(ids_and_labels: list | None) -> dict:
    """Fetch Neo4j properties per (label, id).

    Column nodes merge their parent table into ``relevant_tables``.
    """
    items: list[dict] = []
    for x in ids_and_labels or []:
        if not isinstance(x, dict):
            continue
        if x.get("id") is None:
            continue
        if str(x.get("label") or "").strip() == "":
            continue
        items.append({"id": x["id"], "label": x["label"]})

    results: dict = {}

    allowed_labels = set(Labels.LIST_OF_ALL)
    for label, ids in groupby(
        sorted(items, key=lambda d: str(d.get("label") or "").strip()),
        key=lambda d: str(d.get("label") or "").strip(),
    ):
        label_id_pairs_for_current_label = list(ids)
        if not label:
            continue
        if label not in allowed_labels:
            logger.warning("Skipping unknown label %r in expand_info", label)
            continue
        query = f"""UNWIND $label_id_pairs as label_id
                    MATCH (n:{label} {{id: label_id.id}})
                    CALL apoc.case([
                        n:{Labels.CUSTOM_ANALYSIS},
                            'OPTIONAL MATCH(n)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
                            WITH n, head(collect(sql.sql_full_query)) as sql_code, head(collect(sql)) as sql_node
                            OPTIONAL MATCH (sql_node)-[:{Edges.SQL}]->(t:{Labels.TABLE})
                                <-[:{Edges.CONTAINS}]-(schema:{Labels.SCHEMA})
                                <-[:{Edges.CONTAINS}]-(db:{Labels.DB})
                            WITH n, sql_code,
                                 [x IN collect(
                                     CASE WHEN t IS NOT NULL THEN
                                         apoc.map.merge(
                                             properties(t),
                                             {{label: "{Labels.TABLE}",
                                              schema_name: schema.name,
                                              database_name: db.name,
                                              columns: [(t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN}) |
                                                  {{name: c.name,
                                                    data_type: toString(coalesce(c.data_type, "")),
                                                    description: CASE
                                                        WHEN c.description IS NOT NULL AND trim(c.description) <> ""
                                                        THEN c.description ELSE null END,
                                                    sample_values: CASE
                                                        WHEN c.sample_values IS NOT NULL AND size(c.sample_values) > 0
                                                        THEN c.sample_values ELSE null END
                                                  }}]
                                             }}
                                         )
                                     ELSE null END
                                 ) WHERE x IS NOT NULL] AS tables
                            RETURN apoc.map.merge(
                                apoc.map.setKey(properties(n), "sql", coalesce(sql_code, "")),
                                {{relevant_tables: tables}}
                            ) as item',
                        n:{Labels.SQL_ATTRIBUTE},
                            'MATCH(n)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
                            WITH n, head(collect(sql.sql_full_query)) as sql_code, head(collect(sql)) as sql_node
                            MATCH (n)-[:{Edges.PROPERTY_OF}]->(term:{Labels.TERM})
                            MATCH (sql_node)-[:{Edges.SQL}]->(t:{Labels.TABLE})
                                <-[:{Edges.CONTAINS}]-(schema:{Labels.SCHEMA})
                                <-[:{Edges.CONTAINS}]-(db:{Labels.DB})
                            WITH n, sql_code, term,
                                 collect(apoc.map.merge(
                                     properties(t),
                                     {{label: "{Labels.TABLE}",
                                      schema_name: schema.name,
                                      database_name: db.name,
                                      columns: [(t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN}) |
                                          {{name: c.name,
                                            data_type: toString(coalesce(c.data_type, "")),
                                            description: CASE
                                                WHEN c.description IS NOT NULL AND trim(c.description) <> ""
                                                THEN c.description ELSE null END,
                                            sample_values: CASE
                                                WHEN c.sample_values IS NOT NULL AND size(c.sample_values) > 0
                                                THEN c.sample_values ELSE null END
                                          }}]
                                     }}
                                 )) AS tables
                            RETURN apoc.map.merge(
                                apoc.map.setKey(properties(n), "sql", coalesce(sql_code, "")),
                                {{relevant_tables: tables,
                                  term_name: term.name,
                                  term_id: term.id}}
                            ) as item',
                        n:{Labels.COLUMN},
                            'MATCH(n)<-[:{Edges.CONTAINS}]-(parent)<-[:{Edges.CONTAINS}]-(schema:{Labels.SCHEMA})
                            <-[:{Edges.CONTAINS}]-(db:{Labels.DB})
                            WITH n, parent, schema, db,
                                 [(parent)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN}) |
                                  {{name: c.name,
                                    data_type: toString(coalesce(c.data_type, "")),
                                    description: CASE WHEN c.description IS NOT NULL AND trim(c.description) <> ""
                                                      THEN c.description ELSE null END,
                                    sample_values: CASE WHEN c.sample_values IS NOT NULL AND size(c.sample_values) > 0
                                                        THEN c.sample_values ELSE null END
                                  }}] AS column_list
                            WITH n, parent, schema, db, column_list,
                                 apoc.map.merge(
                                     properties(parent),
                                     {{label: coalesce(parent.label,
                                      toLower(head(labels(parent))), "{Labels.TABLE}"),
                                      columns: column_list,
                                      schema_name: schema.name,
                                      database_name: db.name}}
                                 ) AS t0
                            RETURN apoc.map.merge(
                                     apoc.map.setPairs(properties(n),[
                                         ["table_name", parent.name],
                                         ["table_type", parent.table_type],
                                         ["parent_id", parent.id]
                                     ]),
                                     {{relevant_tables: [t0]}}
                                 ) as item'
                        ],
                        'with n RETURN n{{ .*}} as item ',
                        {{n:n, sql_type: $sql_type }}
                        )
                    YIELD value as response
                    WITH collect(response.item) as all_items
                    RETURN apoc.map.groupBy(all_items,'id') as ids_to_props
                    """
        params = {
            "sql_type": Labels.SQL,
            "label_id_pairs": label_id_pairs_for_current_label,
        }
        result = get_neo4j_conn().query_read(
            query=query,
            parameters=params,
        )
        if len(result) > 0:
            results = results | result[0]["ids_to_props"]

    return results
