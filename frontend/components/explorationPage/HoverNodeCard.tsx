// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/common/icons';
import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import { ExplorationLayer } from '@/enums/exploration';
import type { ExplorationNode } from '@/types/exploration';
import { ZonesRow } from '@/common/SinglePageComposer';

type HoverNodeCardProps = {
	node: ExplorationNode;
	x: number;
	y: number;
};

/** Read-only preview card that follows the cursor while hovering a graph node. */
export const HoverNodeCard = ({ node, x, y }: HoverNodeCardProps) => {
	const isSemantic = node.layer === ExplorationLayer.Semantic;

	return (
		<aside
			// Cytoscape positions are runtime canvas coordinates and cannot be static Tailwind classes.
			// Read-only preview: pointer-events-none so it never intercepts hover/click
			// on the graph beneath it. Actionable links only live in the click card.
			style={{ left: x, top: y }}
			className="pointer-events-none absolute z-30 w-[min(25rem,calc(100%-2rem))] rounded-lg border border-zinc-200 bg-white p-3 shadow-xl dark:border-zinc-700 dark:bg-zinc-900"
		>
			<div className="flex items-start gap-2">
				<Icon
					name={
						node.layer === ExplorationLayer.Semantic
							? IconName.Terms
							: catalogNodeInfo[node.nodeType].icon
					}
					className={`mt-0.5 h-5 w-5 shrink-0 ${
						isSemantic ? 'text-[#47bac5]' : 'text-[#31b9c5]'
					}`}
				/>
				<div className="min-w-0 flex-1">
					<div className="flex items-baseline gap-2">
						<h2 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
							{node.name}
						</h2>
						<span className="shrink-0 text-xs capitalize text-zinc-400">
							{node.layer === ExplorationLayer.Semantic
								? 'Term'
								: catalogNodeInfo[node.nodeType].title}
						</span>
					</div>
					{node.layer === ExplorationLayer.Data && (
						<p className="mt-0.5 truncate text-xs text-zinc-400 dark:text-zinc-500">
							{node.databaseName} • {node.schemaName}
						</p>
					)}
					{node.layer === ExplorationLayer.Semantic && node.synonyms.length > 0 && (
						<p className="mt-0.5 truncate text-xs text-zinc-400 dark:text-zinc-500">
							Synonyms: {node.synonyms.join(', ')}
						</p>
					)}
					<p className="mt-1 line-clamp-2 text-xs leading-5 text-zinc-500 dark:text-zinc-400">
						{node.description || 'No Description'}
					</p>
				</div>
			</div>

			{node.layer === ExplorationLayer.Data ? (
				<dl className="mt-3 grid grid-cols-2 overflow-hidden rounded-md border border-zinc-200 text-xs dark:border-zinc-700">
					{[
						{ label: 'Columns', value: node.columnsCount },
						{ label: 'SQL Queries', value: node.sqlCount },
						{ label: 'Terms', value: node.termsCount },
						{ label: 'Related Tables', value: node.relationshipCount },
					].map((item, index) => (
						<div
							key={item.label}
							className={`px-3 py-2 ${
								index % 2 === 0
									? 'border-r border-zinc-200 dark:border-zinc-700'
									: ''
							} ${index < 2 ? 'border-b border-zinc-200 dark:border-zinc-700' : ''}`}
						>
							<dt className="text-zinc-400">{item.label}</dt>
							<dd className="mt-0.5 font-medium text-zinc-700 dark:text-zinc-200">
								{item.value}
							</dd>
						</div>
					))}
				</dl>
			) : (
				<dl className="mt-3 grid grid-cols-3 overflow-hidden rounded-md border border-zinc-200 text-xs dark:border-zinc-700">
					{[
						{ label: 'Related Terms', value: node.relationshipCount },
						{ label: 'Attribute Columns', value: node.columnAttributesCount },
						{ label: 'SQL Attributes', value: node.sqlAttributesCount },
					].map((item, index) => (
						<div
							key={item.label}
							className={`px-3 py-2 ${
								index < 2 ? 'border-r border-zinc-200 dark:border-zinc-700' : ''
							}`}
						>
							<dt className="text-zinc-400">{item.label}</dt>
							<dd className="mt-0.5 font-medium text-zinc-700 dark:text-zinc-200">
								{item.value}
							</dd>
						</div>
					))}
				</dl>
			)}

			<ZonesRow zones={node.zones} />
		</aside>
	);
};
