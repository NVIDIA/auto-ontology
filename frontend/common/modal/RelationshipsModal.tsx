// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import NextLink from 'next/link';

import { explorationApi } from '@/api/exploration';
import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import { ExplorationLayer } from '@/enums/exploration';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import type { ExplorationNode } from '@/types/exploration';
import type { TableColumn } from '@/types/table';
import { Icon, IconName } from '@/common/icons';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { SkeletonTable } from '@/common/Skeleton';
import { Table } from '@/common/Table';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import { usePagination } from '@/hooks/usePagination';
import { Modal } from './Modal';

type RelationshipsModalProps = {
	node: ExplorationNode | null;
	onClose: () => void;
	onFocus: (nodeId: string) => void;
	focusableNodeIds: ReadonlySet<string>;
};

/**
 * Destination of a row's external link, or `null` when there is nowhere to go.
 *
 * A related table is listed even when no Database/Schema sits above it (see
 * `_data_related_tables_match` in auto_ontology/dal/exploration.py — requiring the
 * catalog path there would make the list shorter than the node's degree on
 * the graph). The `/data` page addresses a table by its full
 * `dbId|schemaId|tableId` path, so such a row has no address to link to.
 */
const nodeHref = (row: ExplorationNode): string | null => {
	if (row.layer === ExplorationLayer.Semantic) {
		return `/terms?focus=${encodeURIComponent(row.id)}`;
	}
	if (!row.databaseId || !row.schemaId) return null;
	return catalogPathFromFocusId(`${row.databaseId}|${row.schemaId}|${row.id}`);
};

/** Related-entities modal shared by both Exploration layers (Terms and Tables/Views). */
export const RelationshipsModal = ({
	node,
	onClose,
	onFocus,
	focusableNodeIds,
}: RelationshipsModalProps) => {
	const nodeId = node?.id ?? null;
	const [rows, setRows] = useState<ExplorationNode[]>([]);
	const [total, setTotal] = useState(0);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const { skip, pageSize, pageRowCount, pagination } = usePagination({
		totalItems: total,
		resetKey: nodeId,
	});

	useEffect(() => {
		if (node == null) return undefined;
		let cancelled = false;

		const load = async () => {
			setLoading(true);
			setError(null);
			const response = await explorationApi.getRelatedNodes(node.id, node.layer, {
				skip,
				limit: pageSize,
			});
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load related entities');
			} else {
				setRows(response.data ?? []);
				setTotal(response.total ?? 0);
			}
			setLoading(false);
		};

		void load();
		return () => {
			cancelled = true;
		};
	}, [node, pageSize, skip]);

	const columns: TableColumn<ExplorationNode>[] = [
		{
			key: 'type',
			header: 'Type',
			width: 'w-24',
			cell: (row) => (
				<div className="flex items-center gap-1.5 capitalize">
					<Icon
						name={
							row.layer === ExplorationLayer.Semantic
								? IconName.Terms
								: catalogNodeInfo[row.nodeType].icon
						}
						className="h-4 w-4 text-[#76b900]"
					/>
					{row.layer === ExplorationLayer.Semantic
						? 'Term'
						: catalogNodeInfo[row.nodeType].title}
				</div>
			),
		},
		{
			key: 'name',
			header: 'Name',
			cell: (row) => row.name,
			title: (row) => row.name,
			truncate: true,
		},
		{
			key: 'relationships',
			header: 'Relationships',
			width: 'w-32',
			cell: (row) => row.relationshipCount,
		},
		{
			key: 'link',
			header: '',
			width: 'w-10',
			cell: (row) => {
				const href = nodeHref(row);
				if (href == null) {
					return (
						<span
							className="flex h-6 w-6 items-center justify-center rounded text-zinc-300 dark:text-zinc-600"
							title={`${row.name} has no catalog location to open`}
						>
							<Icon name={IconName.ExternalLink} className="h-3.5 w-3.5" />
						</span>
					);
				}
				return (
					<NextLink
						href={href}
						className="flex h-6 w-6 items-center justify-center rounded text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:hover:bg-zinc-700"
						aria-label={`Open ${row.name}`}
						title={`Open ${row.name}`}
					>
						<Icon name={IconName.ExternalLink} className="h-3.5 w-3.5" />
					</NextLink>
				);
			},
		},
		{
			key: 'focus',
			header: 'Focus',
			width: 'w-20',
			cell: (row) => {
				// The graph is capped at a fixed number of nodes, so a related
				// entity can exist without being drawn — there is nothing on the
				// canvas to focus in that case.
				const focusable = focusableNodeIds.has(row.id);
				return (
					<Button
						theme={ButtonTheme.Secondary}
						size={Size.SMALL}
						type="button"
						onClick={() => onFocus(row.id)}
						disabled={!focusable}
						title={
							focusable
								? `Focus ${row.name} on the graph`
								: `${row.name} is not shown on the current graph`
						}
					>
						Focus
					</Button>
				);
			},
		},
	];

	return (
		<Modal open={node != null} onClose={onClose} className="w-full max-w-3xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon name={IconName.Link} className="h-5 w-5 shrink-0 text-[#76b900]" />
					<Text as="h2" variant={TextVariant.Heading}>
						{node?.name} — Related Entities ({total})
					</Text>
				</div>
				<Button
					theme={ButtonTheme.IconNeutral}
					size={Size.SMALL}
					iconOnly
					type="button"
					onClick={onClose}
					aria-label="Close related entities"
				>
					<Icon name={IconName.Close} className="h-4 w-4" />
				</Button>
			</header>
			<div className="max-h-[70dvh] overflow-y-auto p-5">
				{loading ? (
					<div role="status" aria-label="Loading related entities">
						<SkeletonTable columns={columns.length} rows={pageRowCount} />
					</div>
				) : error != null ? (
					<p className="text-sm text-red-600 dark:text-red-300">{error}</p>
				) : (
					<Table
						columns={columns}
						rows={rows}
						rowKey={(row) => row.id}
						pagination={pagination}
						containerClassName="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
						emptyMessage="No related entities"
					/>
				)}
			</div>
		</Modal>
	);
};
