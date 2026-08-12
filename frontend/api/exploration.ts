// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pageQuery, requests } from './requests';
import { TableType } from '@/enums/datasources';
import { ExplorationLayer } from '@/enums/exploration';
import type {
	DataExplorationGraph,
	DataGraphEdgeDto,
	ExplorationDataNode,
	ExplorationNode,
	ExplorationRelationshipsPageDto,
	ExplorationTermNode,
	ExplorationZonesMap,
	SemanticExplorationGraph,
	TableExplorationDetails,
} from '@/types/exploration';
import type { PageParams, ResponseWithCount, ResponseWithError } from './types';

export type ExplorationGraphResponse<Graph> = {
	data: Graph;
};

// Server caps the response at the same number regardless of what's sent (see
// MAX_EXPLORATION_GRAPH_NODES in gsf/dal/datasources.py) — sent explicitly so
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
