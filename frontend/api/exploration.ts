// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type {
	DataExplorationGraph,
	DataGraphEdgeDto,
	ExplorationZonesMap,
	SemanticExplorationGraph,
	TableExplorationDetails,
} from '@/types/exploration';
import type { ResponseWithCount, ResponseWithError } from './types';

export type ExplorationGraphResponse<Graph> = {
	data: Graph;
	meta?: {
		noZoneAccess?: boolean;
	};
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

	getTableExplorationDetails: (tableId: string) =>
		requests.get<{ data: TableExplorationDetails }>(`exploration/tables/${tableId}/details`),

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
