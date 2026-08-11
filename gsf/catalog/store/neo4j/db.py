# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database-level store access.

Forked verbatim in Phase 1 of the drop-Neo4j refactor
(``docs/refactor/drop-neo4j/PLAN.md``) from::

    nemo_retriever.tabular_data.ingestion.dal.db_dal

(NeMo-Retriever, Apache-2.0). Behaviour is unchanged; only imports and the
``Neo4jNode`` -> ``CatalogNode`` rename differ.
"""

import logging

from gsf.catalog.store.neo4j.connection import get_neo4j_conn
from gsf.catalog.constants import Edges, Labels

# The diff is storage-agnostic and shared with the Postgres backend; only the
# primitives below are Neo4j-specific. See gsf/catalog/diff.py.
from gsf.catalog.diff import (  # noqa: F401
    accumulate_added_column_props,
    accumulate_updated_column,
    update_diff_from_existing_schema,
)

logger = logging.getLogger(__name__)


def db_exists(db_node):
    database_name = db_node.get_name()
    query = f"""
    MATCH (n:{Labels.DB}{{name: $database_name}})
    OPTIONAL MATCH (n)-[r]-(v)
    RETURN n.id AS id, count(r) AS nbrs
    """
    result_data = get_neo4j_conn().query_read(
        query=query, parameters={"database_name": database_name}
    )
    if not result_data or len(result_data) == 0:
        return None, None

    nbrs = result_data[0]["nbrs"]
    return result_data[0]["id"], False if nbrs == 0 else True


def update_node_property(label, node_id, update_properties):
    query = f"""
            match(n:{label}{{id:$node_id}})
            set n += $update_properties
            """
    get_neo4j_conn().query_write(
        query=query,
        parameters={
            "node_id": node_id,
            "update_properties": update_properties,
        },
    )


def delete_schema(schema_node_id):
    query = f"""MATCH (schema:{Labels.SCHEMA} {{id: $schema_node_id}})-[:{Edges.CONTAINS}]->
               (table:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
               DETACH DELETE schema, table, col
             """
    get_neo4j_conn().query_write(
        query=query,
        parameters={"schema_node_id": schema_node_id},
    )


def add_schemas_edge_batch(edges, created):
    """
    If the nodes do not exist in the Neo4j graph, the function adds them.
    Add to the Neo4j graph the given edge.
    :param edge: edge is a tuple of the form (from_node, to_node, edge_properties)
    :return:
    """
    try:
        # in case of match override the existing ID in the graph,
        # in order to correlate with the ID of the parsed Node object
        query = f"""
            UNWIND $edges as e
            CALL apoc.merge.node.eager([e.from_label], e.from_identProps, e.v_props, {{id:e.v_props.id}})
            yield node as v1
            set v1.created = coalesce(v1.created, $created)
            with v1, e
            call apoc.merge.node.eager([e.to_label], e.to_identProps, e.u_props, {{id:e.u_props.id}})
            yield node as v2
            set v2.created = coalesce(v2.created, $created)
            MERGE (v1)-[r:{Edges.CONTAINS}]->(v2)
            SET r = e.optional_edge_props
            """

        get_neo4j_conn().query_write(
            query=query,
            parameters={
                "created": created,
                "edges": edges,
            },
        )
    except Exception as err:
        raise Exception(f'Error in "add_schemas_edge_batch": {err}')


def delete_table(table_id):
    query = f"""MATCH (table:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
               DETACH DELETE table, col
            """
    get_neo4j_conn().query_write(
        query=query,
        parameters={"table_id": table_id},
    )


def update_properties_in_graph_batch(items):
    # Bulk upsert nodes by id+label, updating all props while preserving any existing description.
    query = """
            UNWIND $items as item
            WITH item, item.props.description as new_description,
            apoc.map.removeKeys(item.props, ["description"]) as item_props_no_description
            CALL apoc.merge.node.eager([item.label], {id: item.id}, {}, item_props_no_description)
            YIELD node
            // keep existing description unless it is null
            SET node.description = coalesce(node.description, new_description)
            """
    get_neo4j_conn().query_write(
        query=query,
        parameters={"items": items},
    )


def delete_columns_batch(column_ids):
    query = f"""UNWIND $column_ids as column_id
               MATCH (col:{Labels.COLUMN} {{id: column_id}})
               DETACH DELETE col
            """
    get_neo4j_conn().query_write(
        query=query,
        parameters={"column_ids": column_ids},
    )


def delete_column(column_id):
    query = f"""MATCH (col:{Labels.COLUMN} {{id: $column_id}})
               DETACH DELETE col
            """
    get_neo4j_conn().query_write(
        query=query,
        parameters={"column_id": column_id},
    )
