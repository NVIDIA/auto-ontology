// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pageQuery, requests } from './requests';
import { TableType } from '@/enums/datasources';
import { ExplorationLayer } from '@/enums/exploration';
import type {
	ColumnAttributeExplorationDetails,
	ColumnExplorationDetails,
	DataExplorationGraph,
	DataGraphEdgeDto,
	ExplorationDataNode,
	ExplorationLinkPathDto,
	ExplorationNode,
	ExplorationRelationshipsPageDto,
	ExplorationTermNode,
	ExplorationZonesMap,
	SemanticExplorationGraph,
	SqlAttributeExplorationDetails,
	SqlExplorationDetails,
	TableExplorationDetails,
	TermExplorationDetails,
} from '@/types/exploration';
import type { PageParams, ResponseWithCount, ResponseWithError } from './types';

export type ExplorationGraphResponse<Graph> = {
	data: Graph;
};

// Server caps the response at the same number regardless of what's sent (see
// MAX_EXPLORATION_GRAPH_NODES in auto_ontology/dal/datasources.py) — sent explicitly so
// intent is visible at the call site rather than relying on the server default.
const EXPLORATION_GRAPH_NODE_LIMIT = 500;

export const explorationApi = {
	getEdges: () => requests.get<ResponseWithCount<DataGraphEdgeDto[]>>('exploration/edges'),

	/** Full data-layer Exploration graph ({nodes, links}) in a single request. */
	getDataExplorationGraph: () =>
		requests.get<ExplorationGraphResponse<DataExplorationGraph>>('exploration/graph', {
			limit: EXPLORATION_GRAPH_NODE_LIMIT,
		}),

	getTableExplorationDetails: (tableId: string, params?: PageParams) =>
		requests.get<{ data: TableExplorationDetails }>(
			`exploration/tables/${tableId}/details`,
			pageQuery(params),
		),

	getTermExplorationDetails: (termId: string, params?: PageParams) =>
		requests.get<{ data: TermExplorationDetails }>(
			`exploration/terms/${termId}/details`,
			pageQuery(params),
		),

	getColumnExplorationDetails: (columnId: string) =>
		requests.get<{ data: ColumnExplorationDetails }>(`exploration/columns/${columnId}/details`),

	getColumnAttributeExplorationDetails: (attrId: string, params?: PageParams) =>
		requests.get<{ data: ColumnAttributeExplorationDetails }>(
			`exploration/column-attributes/${attrId}/details`,
			pageQuery(params),
		),

	getSqlAttributeExplorationDetails: (attrId: string) =>
		requests.get<{ data: SqlAttributeExplorationDetails }>(
			`exploration/sql-attributes/${attrId}/details`,
		),

	getSqlExplorationDetails: (sqlId: string) =>
		requests.get<{ data: SqlExplorationDetails }>(`exploration/sql/${sqlId}/details`),

	/** The real hop chain connecting two Terms, for highlighting a clicked term↔term edge. */
	getSemanticLinkPath: (termId: string, otherTermId: string) =>
		requests.get<{ data: ExplorationLinkPathDto }>(
			`exploration/terms/${termId}/path/${otherTermId}`,
		),

	getRelatedNodes: async (
		nodeId: string,
		layer: ExplorationLayer,
		params: PageParams,
	): Promise<ResponseWithError<{ data: ExplorationNode[]; total: number }>> => {
		const response = await requests.get<{ data: ExplorationRelationshipsPageDto }>(
			`exploration/nodes/${nodeId}/relationships`,
			{ layer, ...pageQuery(params) },
		);
		if (response.error) {
			return {
				data: [],
				total: 0,
				error: true,
				message: response.message,
			};
		}

		const page = response.data;
		const nodes: ExplorationNode[] = (page?.nodes ?? []).map((node) => {
			if (layer === ExplorationLayer.Semantic) {
				return {
					id: node.id,
					name: node.name,
					description: null,
					synonyms: [],
					zones: [],
					layer: ExplorationLayer.Semantic,
					nodeType: 'term',
					relationshipCount: node.relationship_count,
					columnAttributesCount: 0,
					sqlAttributesCount: 0,
				} satisfies ExplorationTermNode;
			}
			return {
				id: node.id,
				name: node.name,
				description: null,
				layer: ExplorationLayer.Data,
				nodeType: (node.table_type || TableType.BASE_TABLE) as TableType,
				relationshipCount: node.relationship_count,
				databaseId: node.database_id ?? '',
				databaseName: '',
				schemaId: node.schema_id ?? '',
				schemaName: '',
				columnsCount: 0,
				sqlCount: 0,
				termsCount: 0,
				zones: [],
			} satisfies ExplorationDataNode;
		});
		return { data: nodes, total: page?.total ?? 0 };
	},

	getTableZonesMap: () => requests.get<{ data: ExplorationZonesMap }>('exploration/tables/zones'),

	/** Full semantic-layer Exploration graph ({nodes, links}) in a single request. */
	getSemanticExplorationGraph: (): Promise<
		ResponseWithError<ExplorationGraphResponse<SemanticExplorationGraph>>
	> =>
		requests.get<ExplorationGraphResponse<SemanticExplorationGraph>>(
			'exploration/semantic-graph',
			{
				limit: EXPLORATION_GRAPH_NODE_LIMIT,
			},
		),
};
