// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';

import { Icon } from '@/common/icons';
import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import { getTableType } from '@/components/dataPage/get-table-type';
import { ExplorationLayer } from '@/enums/exploration';
import { TableType } from '@/enums/datasources';
import { DetailLinkButton } from '@/common/DetailLinkButton';
import type {
	DataExplorationGraph,
	ExplorationDataNode,
	ExplorationGraph,
	ExplorationLink,
} from '@/types/exploration';
import { ZonesRow } from '@/common/SinglePageComposer';

/** Build the data-layer graph (Tables/Views + their relationships) from the server DTO. */
export const buildDataGraph = (graph: DataExplorationGraph): ExplorationGraph => {
	const nodes: ExplorationDataNode[] = graph.nodes.map((table) => ({
		id: table.id,
		name: table.name,
		description: table.description ?? null,
		layer: ExplorationLayer.Data,
		nodeType: getTableType(table.table_type) as TableType,
		relationshipCount: 0,
		databaseId: table.database_id,
		databaseName: table.database_name,
		schemaId: table.schema_id,
		schemaName: table.schema_name,
		columnsCount: table.columns_count,
		sqlCount: table.sql_count ?? 0,
		termsCount: table.terms_count ?? 0,
		zones: table.zones ?? [],
	}));

	const tableIds = new Set(nodes.map((node) => node.id));
	const links: ExplorationLink[] = graph.links
		.filter(
			(link) =>
				tableIds.has(link.source) &&
				tableIds.has(link.target) &&
				(link.queries.length > 0 || link.via_foreign_key),
		)
		.map((link) => ({
			source: link.source,
			target: link.target,
			queries: link.queries,
			viaForeignKey: link.via_foreign_key,
			foreignKeys: (link.foreign_keys ?? []).map((fk) => ({
				sourceColumn: fk.source_column,
				targetColumn: fk.target_column,
				sourceSampleValues: fk.source_sample_values,
				targetSampleValues: fk.target_sample_values,
			})),
		}));

	const relationshipCountById = new Map<string, number>();
	links.forEach(({ source, target }) => {
		relationshipCountById.set(source, (relationshipCountById.get(source) ?? 0) + 1);
		relationshipCountById.set(target, (relationshipCountById.get(target) ?? 0) + 1);
	});

	return {
		nodes: nodes.map((node) => ({
			...node,
			relationshipCount: relationshipCountById.get(node.id) ?? 0,
		})),
		links,
	};
};

type ActiveDataCardProps = {
	node: ExplorationDataNode;
	onClose: () => void;
	onView: () => void;
	onShowRelationships: () => void;
	onShowColumns: () => void;
	onShowQueries: () => void;
	onShowTerms: () => void;
};

/** Detail card shown when a Table/View node is selected on the data graph. */
export const ActiveDataCard = ({
	node,
	onClose,
	onView,
	onShowRelationships,
	onShowColumns,
	onShowQueries,
	onShowTerms,
}: ActiveDataCardProps) => {
	const [minimized, setMinimized] = useState(false);

	return (
		<section className="absolute bottom-4 left-4 z-20 w-[min(26rem,calc(100%-2rem))] rounded-lg border border-zinc-200 bg-white shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
			<header className="flex items-center justify-between border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
				<p className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">
					Showing info on this Data Object
				</p>
				<div className="flex items-center gap-1">
					<button
						type="button"
						onClick={() => setMinimized((value) => !value)}
						className="cursor-pointer rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
						aria-label={
							minimized
								? 'Expand data object details'
								: 'Minimize data object details'
						}
					>
						{minimized ? '+' : '−'}
					</button>
					<button
						type="button"
						onClick={onClose}
						className="cursor-pointer rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
						aria-label="Close data object details"
					>
						×
					</button>
				</div>
			</header>
			{!minimized && (
				<div className="space-y-3 p-4">
					<div className="flex items-start gap-3">
						<Icon
							name={catalogNodeInfo[node.nodeType].icon}
							className="mt-0.5 h-5 w-5 shrink-0 text-[#3b82b6]"
						/>
						<div className="min-w-0 flex-1">
							<h2 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
								{node.name}
							</h2>
							<p className="mt-0.5 truncate text-xs text-zinc-400 dark:text-zinc-500">
								{node.databaseName} • {node.schemaName}
							</p>
							<p className="mt-1 line-clamp-2 text-xs leading-5 text-zinc-500 dark:text-zinc-400">
								{node.description || 'No Description'}
							</p>
						</div>
						<div className="flex shrink-0 items-center gap-1">
							<button
								type="button"
								onClick={onView}
								className="cursor-pointer rounded-lg bg-[#76b900] px-3 py-1.5 text-xs font-medium text-white hover:bg-[#5e9400]"
							>
								View in Data
							</button>
						</div>
					</div>
					<div className="grid grid-cols-2 gap-2">
						<span className="flex items-center justify-between gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
							Columns: {node.columnsCount}
							<DetailLinkButton
								count={node.columnsCount}
								onClick={onShowColumns}
								label="View columns"
							/>
						</span>
						<span className="flex items-center justify-between gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
							SQL Queries: {node.sqlCount}
							<DetailLinkButton
								count={node.sqlCount}
								onClick={onShowQueries}
								label="View SQL queries"
							/>
						</span>
						<span className="flex items-center justify-between gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
							Terms: {node.termsCount}
							<DetailLinkButton
								count={node.termsCount}
								onClick={onShowTerms}
								label="View Terms"
							/>
						</span>
						<span className="flex items-center justify-between gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
							Related Tables: {node.relationshipCount}
							<DetailLinkButton
								count={node.relationshipCount}
								onClick={onShowRelationships}
								label="View related tables"
							/>
						</span>
					</div>
					<ZonesRow zones={node.zones} />
				</div>
			)}
		</section>
	);
};
