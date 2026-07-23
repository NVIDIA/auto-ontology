// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';

import { Icon, IconName } from '@/common/icons';
import { DetailLinkButton } from '@/common/DetailLinkButton';
import { ExplorationLayer } from '@/enums/exploration';
import { Label } from '@/common/Label';
import type {
	ExplorationGraph,
	ExplorationTermNode,
	SemanticExplorationGraph,
} from '@/types/exploration';

/** Build the semantic-layer graph (Terms + their relationships) from the server DTO. */
export const buildSemanticGraph = (graph: SemanticExplorationGraph): ExplorationGraph => ({
	nodes: graph.nodes.map((node) => ({
		id: node.id,
		name: node.name,
		description: node.description,
		synonyms: node.synonyms,
		zones: node.zones,
		layer: ExplorationLayer.Semantic as const,
		nodeType: 'term' as const,
		relationshipCount: node.relationship_count,
		columnAttributesCount: node.column_attributes_count,
		sqlAttributesCount: node.sql_attributes_count,
	})),
	links: graph.links.map((link) => ({
		source: link.source,
		target: link.target,
		queries: [],
	})),
});

type ActiveTermCardProps = {
	node: ExplorationTermNode;
	onClose: () => void;
	onView: () => void;
	onShowRelationships: () => void;
	onShowColumnAttributes: () => void;
	onShowSqlAttributes: () => void;
};

/** Detail card shown when a Term node is selected on the semantic graph. */
export const ActiveTermCard = ({
	node,
	onClose,
	onView,
	onShowRelationships,
	onShowColumnAttributes,
	onShowSqlAttributes,
}: ActiveTermCardProps) => {
	const [minimized, setMinimized] = useState(false);

	return (
		<section className="absolute bottom-4 left-4 z-20 w-[min(26rem,calc(100%-2rem))] rounded-lg border border-zinc-200 bg-white shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
			<header className="flex items-center justify-between border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
				<p className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">
					Showing info on this Semantic Object
				</p>
				<div className="flex items-center gap-1">
					<button
						type="button"
						onClick={() => setMinimized((value) => !value)}
						className="cursor-pointer rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
						aria-label={minimized ? 'Expand term details' : 'Minimize term details'}
					>
						{minimized ? '+' : '−'}
					</button>
					<button
						type="button"
						onClick={onClose}
						className="cursor-pointer rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
						aria-label="Close term details"
					>
						×
					</button>
				</div>
			</header>
			{!minimized && (
				<div className="space-y-3 p-4">
					<div className="flex items-start gap-3">
						<Icon
							name={IconName.Terms}
							className="mt-0.5 h-5 w-5 shrink-0 text-[#76b900]"
						/>
						<div className="min-w-0 flex-1">
							<h2 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
								{node.name}
							</h2>
							{node.synonyms.length > 0 && (
								<p className="mt-0.5 truncate text-xs text-zinc-400 dark:text-zinc-500">
									Synonyms: {node.synonyms.join(', ')}
								</p>
							)}
							<p className="mt-1 line-clamp-2 text-xs leading-5 text-zinc-500 dark:text-zinc-400">
								{node.description || 'No Description'}
							</p>
						</div>
						<button
							type="button"
							onClick={onView}
							className="shrink-0 cursor-pointer rounded-lg bg-[#76b900] px-3 py-1.5 text-xs font-medium text-white hover:bg-[#5e9400]"
						>
							View Term
						</button>
					</div>
					<div className="flex flex-wrap items-center gap-2">
						<span className="flex items-center gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
							Related Terms: {node.relationshipCount}
							<DetailLinkButton
								count={node.relationshipCount}
								onClick={onShowRelationships}
								label="View related Terms"
							/>
						</span>
						<span className="flex items-center gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
							Attribute Columns: {node.columnAttributesCount}
							<DetailLinkButton
								count={node.columnAttributesCount}
								onClick={onShowColumnAttributes}
								label="View attribute columns"
							/>
						</span>
						<span className="flex items-center gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
							SQL Attributes: {node.sqlAttributesCount}
							<DetailLinkButton
								count={node.sqlAttributesCount}
								onClick={onShowSqlAttributes}
								label="View SQL attributes"
							/>
						</span>
						<span className="flex items-center gap-1.5 text-xs text-zinc-400">
							Zones:
							{node.zones.length > 0 ? (
								node.zones.map((zone) => (
									<Label
										key={zone.id}
										label={zone.name}
										color={zone.color}
										muted={!zone.enabled}
									/>
								))
							) : (
								<span className="text-zinc-500 dark:text-zinc-400">-</span>
							)}
						</span>
					</div>
				</div>
			)}
		</section>
	);
};
