// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { EmptyState } from '@/common/EmptyState';
import { EmptyStateVariant } from '@/enums/emptyState';
import { Icon, IconName } from '@/common/icons';
import { PropertyRow } from '@/common/PropertyRow';
import { SqlBlock } from '@/common/SqlBlock';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import type {
	ExpansionEntity,
	ExpansionEntityKind,
	ExplorationLinkPathHopDto,
} from '@/types/exploration';

const TYPE_ICON: Record<ExpansionEntityKind, IconName> = {
	schema: IconName.Schema,
	column: IconName.Column,
	term: IconName.Terms,
	table: IconName.Table,
	columnAttribute: IconName.Key,
	sqlAttribute: IconName.ChartLine,
	sql: IconName.CodeBracket,
	customAnalysis: IconName.ChartBar,
	connection: IconName.Connection,
};

const TYPE_LABEL: Record<ExpansionEntityKind, string> = {
	schema: 'Schema',
	column: 'Column',
	term: 'Term',
	table: 'Table',
	columnAttribute: 'Column Attribute',
	sqlAttribute: 'SQL Attribute',
	sql: 'SQL Query',
	customAnalysis: 'Custom Analysis',
	connection: 'Connection',
};

const TYPE_ICON_COLOR: Record<ExpansionEntityKind, string> = {
	schema: 'text-[#8b7fd6]',
	column: 'text-[#8a94a6]',
	term: 'text-[#76b900]',
	table: 'text-[#3b82b6]',
	columnAttribute: 'text-[#c99a2e]',
	sqlAttribute: 'text-[#c2577a]',
	sql: 'text-[#4f83cc]',
	customAnalysis: 'text-[#d17a3f]',
	connection: 'text-[#76b900]',
};

// The graph id `ExplorationView.tsx` gives most expansion node kinds is
// prefixed (`column:${id}`, `attribute:${id}` for both `columnAttribute`
// and `sqlAttribute`, `sql:${id}`, `customAnalysis:${id}`, `schema:${id}`)
// so it can't collide with a Table/Term's own raw catalog id sharing the
// graph — meaningless to a user reading the "ID" row below, which wants
// the raw catalog/Term id alone. `table`/`term` ids are never prefixed to
// begin with, so they're simply absent here.
const TYPE_ID_PREFIX: Partial<Record<ExpansionEntityKind, string>> = {
	schema: 'schema:',
	column: 'column:',
	columnAttribute: 'attribute:',
	sqlAttribute: 'attribute:',
	sql: 'sql:',
	customAnalysis: 'customAnalysis:',
};

const displayId = (node: ExpansionEntity): string => {
	const prefix = TYPE_ID_PREFIX[node.kind];
	return prefix != null && node.id.startsWith(prefix) ? node.id.slice(prefix.length) : node.id;
};

// No single per-kind "View" destination makes sense for a `connection`
// entity (it names *two* terms, not one) — see the button's own guard
// below, which skips rendering it entirely for that kind instead of
// reading this map.
const TYPE_VIEW_LABEL: Record<Exclude<ExpansionEntityKind, 'connection'>, string> = {
	schema: 'View in Data',
	column: 'View in Data',
	term: 'View Term',
	table: 'View in Data',
	columnAttribute: 'View in Data',
	sqlAttribute: 'View Term',
	sql: 'View Term',
	customAnalysis: 'View in Analysis',
};

// Matches `ExplorationLinkPathNodeDto.type`/this file's own `TYPE_ICON`/
// `TYPE_ICON_COLOR` — a path hop's node is always one of these four kinds
// (see `find_term_link_path` in `auto_ontology/dal/attributes.py`).
const PATH_NODE_ICON: Record<string, IconName> = {
	term: IconName.Terms,
	table: IconName.Table,
	column: IconName.Column,
	columnAttribute: IconName.Key,
};

const PATH_NODE_ICON_COLOR: Record<string, string> = {
	term: 'text-[#76b900]',
	table: 'text-[#3b82b6]',
	column: 'text-[#8a94a6]',
	columnAttribute: 'text-[#c99a2e]',
};

/** One Term/Table/Column/ColumnAttribute stop, connected to the next by its own relationship type. */
const PathChain = ({ hops }: { hops: ExplorationLinkPathHopDto[] }) => {
	// Every hop's `target` chains into the next hop's `source` (see
	// `find_term_link_path`'s ordered chain) — so the full node sequence is
	// just the first hop's source followed by every hop's target.
	const nodes = [hops[0].source, ...hops.map((hop) => hop.target)];

	return (
		<div className="flex flex-col">
			{nodes.map((node, index) => (
				<div key={`${node.type}:${node.id}:${index}`} className="flex flex-col">
					<div className="flex items-center gap-2">
						<Icon
							name={PATH_NODE_ICON[node.type] ?? IconName.Connection}
							className={`h-4 w-4 shrink-0 ${PATH_NODE_ICON_COLOR[node.type] ?? 'text-zinc-400'}`}
						/>
						<Text as="p" text={node.name ?? node.id} variant={TextVariant.Label} />
					</div>
					{index < hops.length && (
						<div className="ml-2 flex items-center gap-1.5 py-1 pl-[7px]">
							<div className="h-4 w-px bg-zinc-300 dark:bg-zinc-600" />
							<span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] font-medium text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
								{hops[index].relationship}
							</span>
						</div>
					)}
				</div>
			))}
		</div>
	);
};

/**
 * The real path connecting a `connection` entity's two terms — `null`
 * while `ExplorationView`'s `getSemanticLinkPath` request is still in
 * flight, `[]` on the (now rare — see `find_term_link_path`'s own doc
 * comment) case where they genuinely share no traceable path.
 */
const ConnectionDetails = ({ hops }: { hops: ExplorationLinkPathHopDto[] | null | undefined }) => {
	if (hops == null) {
		return (
			<Text as="p" variant={TextVariant.Caption}>
				Loading connection…
			</Text>
		);
	}
	if (hops.length === 0) {
		return (
			<EmptyState
				variant={EmptyStateVariant.Inline}
				icon={IconName.Link}
				title="No connection found"
				description="These terms no longer share a common table."
			/>
		);
	}
	return (
		<div className="flex flex-col gap-1.5">
			<Text as="p" variant={TextVariant.Caption}>
				Details
			</Text>
			<PathChain hops={hops} />
		</div>
	);
};

type ActiveExpansionCardProps = {
	node: ExpansionEntity;
	onClose: () => void;
	onView: () => void;
};

/**
 * Trimmed sibling of `ActiveTermCard`/`ActiveDataCard` for the Schema/Column/Term
 * nodes grafted onto the graph by expanding a Table node, and the Table/
 * ColumnAttribute/SqlAttribute nodes grafted on in turn by expanding one of
 * those Terms — see `expandTableNode`/`expandTermNode` in
 * `ExplorationView.tsx`. These aren't part of the base graph payload, so
 * all that's known about them is their name/description/id, not the full
 * stats the other two cards show. A `sql` node — grafted by expanding a
 * SqlAttribute in turn (see `expandSqlAttributeNode`) — shows its own raw
 * query text via `SqlBlock` instead of the plain-text Description row
 * every other kind gets. A `connection` entity is the odd one out — not a
 * graph node, but a clicked Semantic-layer term↔term *edge* itself (see
 * `activeSemanticConnectionEntity` in `ExplorationView.tsx`); it shows the
 * real hop chain connecting the two terms in place of a Description, ends
 * with one "ID" row per Term endpoint (`connectionSource`/`connectionTarget`)
 * instead of a single row for the raw composite `id`, and skips the "View"
 * button entirely since it doesn't name a single entity to view.
 */
export const ActiveExpansionCard = ({ node, onClose, onView }: ActiveExpansionCardProps) => (
	<section className="absolute right-4 top-20 bottom-28 z-20 flex w-[380px] flex-col overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
		<header className="flex shrink-0 items-center justify-between border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
			<p className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">
				Showing info on this {TYPE_LABEL[node.kind]}
			</p>
			<Button
				theme={ButtonTheme.IconNeutral}
				size={Size.SMALL}
				iconOnly
				type="button"
				onClick={onClose}
				aria-label={`Close ${TYPE_LABEL[node.kind].toLowerCase()} details`}
			>
				<Icon name={IconName.Close} className="h-4 w-4" />
			</Button>
		</header>
		<div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-4">
			<div className="flex items-start gap-3">
				<Icon
					name={TYPE_ICON[node.kind]}
					className={`mt-0.5 h-5 w-5 shrink-0 ${TYPE_ICON_COLOR[node.kind]}`}
				/>
				<div className="min-w-0 flex-1 space-y-1">
					<Text as="h2" text={node.name} variant={TextVariant.Heading} />
					<Text as="p" variant={TextVariant.Caption}>
						{TYPE_LABEL[node.kind]}
					</Text>
				</div>
			</div>
			{node.kind !== 'connection' && (
				<div className="self-start">
					<Button
						theme={ButtonTheme.Primary}
						size={Size.SMALL}
						type="button"
						onClick={onView}
					>
						{TYPE_VIEW_LABEL[node.kind]}
					</Button>
				</div>
			)}
			<div className="flex flex-col gap-2">
				<PropertyRow label="Name" value={node.name} />
				{node.kind === 'column' && node.dataType != null && (
					<PropertyRow label="Data Type" value={node.dataType} monospace />
				)}
				{node.kind === 'columnAttribute' && node.relationshipType != null && (
					<PropertyRow label="Relationship" value={node.relationshipType} monospace />
				)}
				{node.kind === 'sql' ? (
					<SqlBlock sql={node.sqlText ?? ''} />
				) : node.kind === 'connection' ? (
					<ConnectionDetails hops={node.connectionHops} />
				) : (
					<PropertyRow
						label="Description"
						value={
							node.description ??
							(node.kind === 'column'
								? 'No description on record for this column.'
								: '')
						}
					/>
				)}
				{node.kind === 'connection' ? (
					<>
						<PropertyRow
							label={node.connectionSource?.name ?? 'Term 1'}
							value={node.connectionSource?.id ?? node.id}
							monospace
						/>
						<PropertyRow
							label={node.connectionTarget?.name ?? 'Term 2'}
							value={node.connectionTarget?.id ?? node.id}
							monospace
						/>
					</>
				) : (
					<PropertyRow label="ID" value={displayId(node)} monospace />
				)}
			</div>
		</div>
	</section>
);
