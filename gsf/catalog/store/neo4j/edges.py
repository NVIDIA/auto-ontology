# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic node/edge merge helpers.

Forked verbatim in Phase 1 of the drop-Neo4j refactor
(``docs/refactor/drop-neo4j/PLAN.md``) from::

    nemo_retriever.tabular_data.ingestion.dal.utils_dal

(NeMo-Retriever, Apache-2.0). Behaviour is unchanged; only imports and the
``Neo4jNode`` -> ``CatalogNode`` rename differ.
"""

import logging

from gsf.catalog.model.node import CatalogNode
from gsf.catalog.constants import (
    Edges,
    Props,
    Labels,
)
from gsf.catalog.store.neo4j.connection import get_neo4j_conn

logger = logging.getLogger(__name__)


def is_flat_dict(properties: dict):
    for key, value in properties.items():
        if isinstance(value, list):
            if value:
                if any(
                    isinstance(item, list) or isinstance(item, dict) for item in value
                ):
                    raise ValueError(
                        f"Invalid property name: {key}\nThe property value: {value}"
                    )
        if isinstance(value, dict):
            raise ValueError(
                f"Invalid property name: {key}\nThe property value: {value}"
            )


def check_properties_compatibility_with_neo4j(
    node_from: CatalogNode, node_to: CatalogNode, edge_props: dict
):
    is_flat_dict(node_from.get_properties())
    if node_from.get_override_existing_props():
        is_flat_dict(node_from.get_override_existing_props())
    is_flat_dict(node_to.get_properties())
    if node_to.get_override_existing_props():
        is_flat_dict(node_to.get_override_existing_props())
    is_flat_dict(edge_props)


def prepare_edge(edge):
    node_from = edge[0]
    node_to = edge[1]

    e_label = _get_edge_label(edge)
    v1_label, v1_identity_props, v1_on_create_props, v1_on_match_props = prepare_node(
        node_from
    )
    v2_label, v2_identity_props, v2_on_create_props, v2_on_match_props = prepare_node(
        node_to
    )
    edge_props = edge[2].copy()

    check_properties_compatibility_with_neo4j(node_from, node_to, edge[2])

    if e_label in (Edges.JOIN, Edges.UNION):
        edge_identity_props = {}
    elif "child_idx" in edge_props and edge_props["child_idx"] is not None:
        edge_identity_props = {"child_idx": edge_props["child_idx"]}
    else:
        edge_identity_props = {}

    return {
        "v1_label": v1_label,
        "v1_identity_props": v1_identity_props,
        "v1_on_create_props": v1_on_create_props,
        "v1_on_match_props": v1_on_match_props,
        "v2_label": v2_label,
        "v2_identity_props": v2_identity_props,
        "v2_on_create_props": v2_on_create_props,
        "v2_on_match_props": v2_on_match_props,
        "edge_props": edge_props,
        "edge_label": e_label,
        "edge_identity_props": edge_identity_props,
    }


def _get_edge_label(edge):
    if Props.JOIN in edge[2]:
        return Edges.JOIN
    elif Props.UNION in edge[2]:
        return Edges.UNION
    elif Props.SQL_ID in edge[2]:
        return Edges.SQL
    elif Props.ANALYSIS_ID in edge[2]:
        return Edges.HAS_SQL
    else:
        return next(iter(edge[2]))


def prepare_node(node: CatalogNode):
    label = node.get_label()
    props = node.get_properties()

    identity_props = node.get_match_props()
    on_create_props = props.copy()
    override_props = (
        node.get_override_existing_props() if node.get_override_existing_props() else {}
    )
    return [label], identity_props, on_create_props, override_props


def add_edges(edges_data):
    """
    If the nodes do not exist in the Neo4j graph, the function adds them.
    Add to the Neo4j graph the given edge.
    :param edge: edge is a tuple of the form (from_node, to_node, edge_properties)
    :return:
    """
    query = """
            unwind $edges_data as data
            call apoc.merge.node.eager(
                data.v1_label,
                data.v1_identity_props,
                data.v1_on_create_props,
                data.v1_on_match_props
            )
            yield node as v1
            call apoc.merge.node.eager(
                data.v2_label,
                data.v2_identity_props,
                data.v2_on_create_props,
                data.v2_on_match_props
            )
            yield node as v2
            with v1, v2, data, data.edge_identity_props as e_identity_props
            call apoc.merge.relationship.eager(v1, data.edge_label, e_identity_props, {}, v2)
            YIELD rel
            with rel, data,
            case
              when data.edge_props.join_refs is not null
                then {join_refs: coalesce(rel.join_refs, []) + data.edge_props.join_refs}
              when data.edge_props.union_refs is not null
                then {union_refs: coalesce(rel.union_refs, []) + data.edge_props.union_refs}
              else data.edge_props
            end as props
            SET rel = props
            RETURN DISTINCT 'true'
            """
    get_neo4j_conn().query_write(query=query, parameters={"edges_data": edges_data})


def get_node_properties_by_id(id, label: str | list[str]):
    if isinstance(label, list):
        label_filter = "|".join(label)
    else:
        label_filter = label
    query = f"""
        MATCH(n:{label_filter}{{id:$id}})
        RETURN apoc.map.setKey(properties(n),"label", labels(n)[0]) as props
    """

    props = get_neo4j_conn().query_read(query, parameters={"id": id})
    if len(props) == 0:
        return None
    else:
        return props[0]["props"]


def delete_bulk_of_nodes(ids, labels):
    for label in labels:
        query = f"""match(n:{label})
                    where n.id in $ids
                    detach delete n
                """
        get_neo4j_conn().query_write(query, parameters={"ids": ids})


def detach_bulk_of_nodes(ids):
    query = """ unwind $ids as id
                match(n:field{qs_id:id})-[r:depends_on]->()
                delete r
            """
    get_neo4j_conn().query_write(query, parameters={"ids": ids})


def get_node_id_by_name_and_label(name: str, label: Labels):
    query = f"""MATCH (n:{label}{{name:$name}})
               RETURN n.id as id"""
    result = get_neo4j_conn().query_read(query=query, parameters={"name": name})
    if len(result) > 0:
        return result[0]["id"]
    return None
