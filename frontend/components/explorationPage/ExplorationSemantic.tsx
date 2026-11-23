// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { Icon, IconName } from '@/common/icons';
import { DetailLinkButton } from '@/common/DetailLinkButton';
import { PropertyRow } from '@/common/PropertyRow';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import { ExplorationLayer } from '@/enums/exploration';
import { ZonesRow } from '@/common/SinglePageComposer';
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
		relationshipTypes: link.relationship_types ?? [],
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
}: ActiveTermCardProps) => (
	<section className="absolute right-4 top-20 bottom-28 z-20 flex w-[380px] flex-col overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
		<header className="flex shrink-0 items-center justify-between border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
			<p className="text-xs font-semibold text-body dark:text-zinc-300">
				Showing info on this Semantic Object
			</p>
			<Button
				theme={ButtonTheme.IconNeutral}
				size={Size.SMALL}
				iconOnly
				type="button"
				onClick={onClose}
				aria-label="Close term details"
			>
				<Icon name={IconName.Close} className="h-4 w-4" />
			</Button>
		</header>
		<div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-4">
			<div className="flex items-start gap-2">
				<Icon
					name={IconName.Terms}
					className="mt-0.5 h-5 w-5 shrink-0 text-body dark:text-zinc-300"
				/>
				<div className="min-w-0 flex-1 space-y-1">
					<Text as="h2" text={node.name} variant={TextVariant.Heading} />
					{node.synonyms.length > 0 && (
						<Text as="p" variant={TextVariant.Caption}>
							Synonyms: {node.synonyms.join(', ')}
						</Text>
					)}
				</div>
			</div>
			<div className="self-start">
				<Button
					theme={ButtonTheme.Primary}
					size={Size.SMALL}
					type="button"
					onClick={onView}
				>
					View Term
				</Button>
			</div>
			<div className="flex flex-col gap-2">
				<PropertyRow label="Name" value={node.name} />
				<PropertyRow label="Description" value={node.description ?? ''} />
				<PropertyRow label="ID" value={node.id} monospace />
			</div>
			<div className="grid grid-cols-2 gap-2">
				<span className="flex items-center justify-between gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-body dark:border-zinc-700 dark:text-zinc-300">
					Related Terms: {node.relationshipCount}
					<DetailLinkButton
						count={node.relationshipCount}
						onClick={onShowRelationships}
						label="View related Terms"
					/>
				</span>
				<span className="flex items-center justify-between gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-body dark:border-zinc-700 dark:text-zinc-300">
					Attribute Columns: {node.columnAttributesCount}
					<DetailLinkButton
						count={node.columnAttributesCount}
						onClick={onShowColumnAttributes}
						label="View Attribute Columns"
					/>
				</span>
				<span className="flex items-center justify-between gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs text-body dark:border-zinc-700 dark:text-zinc-300">
					SQL Attributes: {node.sqlAttributesCount}
					<DetailLinkButton
						count={node.sqlAttributesCount}
						onClick={onShowSqlAttributes}
						label="View SQL attributes"
					/>
				</span>
			</div>
			<ZonesRow zones={node.zones} />
		</div>
	</section>
);
