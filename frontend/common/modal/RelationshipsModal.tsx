// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import NextLink from 'next/link';

import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import { ExplorationLayer } from '@/enums/exploration';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import type { ExplorationNode } from '@/types/exploration';
import type { TableColumn } from '@/types/table';
import { Icon, IconName } from '@/common/icons';
import { Table } from '@/common/Table';
import { Modal } from './Modal';

type RelationshipsModalProps = {
	node: ExplorationNode | null;
	rows: ExplorationNode[];
	onClose: () => void;
	onFocus: (nodeId: string) => void;
};

/** Related-entities modal shared by both Exploration layers (Terms and Tables/Views). */
export const RelationshipsModal = ({ node, rows, onClose, onFocus }: RelationshipsModalProps) => {
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
			cell: (row) => (
				<NextLink
					href={
						row.layer === ExplorationLayer.Semantic
							? `/terms?focus=${encodeURIComponent(row.id)}`
							: catalogPathFromFocusId(`${row.databaseId}|${row.schemaId}|${row.id}`)
					}
					className="flex h-6 w-6 items-center justify-center rounded text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:hover:bg-zinc-700"
					aria-label={`Open ${row.name}`}
					title={`Open ${row.name}`}
				>
					<Icon name={IconName.ExternalLink} className="h-3.5 w-3.5" />
				</NextLink>
			),
		},
		{
			key: 'focus',
			header: 'Focus',
			width: 'w-20',
			cell: (row) => (
				<button
					type="button"
					onClick={() => onFocus(row.id)}
					className="cursor-pointer rounded-md border border-zinc-200 px-2 py-1 text-xs font-medium text-zinc-600 transition-colors hover:border-[#76b900] hover:text-[#76b900] dark:border-zinc-700 dark:text-zinc-300"
				>
					Focus
				</button>
			),
		},
	];

	return (
		<Modal open={node != null} onClose={onClose} className="w-full max-w-3xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon name={IconName.Link} className="h-5 w-5 shrink-0 text-[#76b900]" />
					<h2 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{node?.name} — Related Entities ({rows.length})
					</h2>
				</div>
				<button
					type="button"
					onClick={onClose}
					className="cursor-pointer rounded p-1 text-lg text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
					aria-label="Close related entities"
				>
					×
				</button>
			</header>
			<div className="max-h-[70dvh] overflow-y-auto p-5">
				<Table
					columns={columns}
					rows={rows}
					rowKey={(row) => row.id}
					containerClassName="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
					scrollClassName="max-h-[28rem] overflow-auto"
					emptyMessage="No related entities"
				/>
			</div>
		</Modal>
	);
};
