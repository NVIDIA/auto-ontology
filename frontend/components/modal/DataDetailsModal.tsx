// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import NextLink from 'next/link';

import { datasources } from '@/api/datasources';
import { explorationApi } from '@/api/exploration';
import { Icon, IconName } from '@/components/icons';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import type { Column } from '@/types/datasources';
import type { TableExplorationDetails } from '@/types/exploration';
import type { TableColumn } from '@/types/table';
import { SqlBlock } from '@/components/SqlBlock';
import { Table } from '@/components/Table';
import { Modal } from './Modal';

export type DataDetailsKind = 'columns' | 'queries' | 'terms';

/** Minimal data-object reference — decoupled from any specific page's node shape. */
export type DataDetailsModalTarget = {
	id: string;
	name: string;
	databaseId: string;
	schemaId: string;
};

type DataDetailsModalProps = {
	target: DataDetailsModalTarget | null;
	kind: DataDetailsKind | null;
	onClose: () => void;
};

/** Generic modal for a Table/View's columns, SQL queries, or related Terms. */
export const DataDetailsModal = ({ target, kind, onClose }: DataDetailsModalProps) => {
	const [columns, setColumns] = useState<Column[]>([]);
	const [details, setDetails] = useState<TableExplorationDetails>({
		queries: [],
		terms: [],
	});
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		if (target == null || kind == null) return undefined;
		let cancelled = false;

		const load = async () => {
			setLoading(true);
			setError(null);
			if (kind === 'columns') {
				const response = await datasources.getColumnsForTable(target.id);
				if (cancelled) return;
				if (response.error) {
					setError(response.message ?? 'Failed to load columns');
				} else {
					setColumns(response.data ?? []);
				}
			} else {
				const response = await explorationApi.getTableExplorationDetails(target.id);
				if (cancelled) return;
				if (response.error) {
					setError(response.message ?? 'Failed to load details');
				} else {
					setDetails(response.data);
				}
			}
			setLoading(false);
		};

		void load();
		return () => {
			cancelled = true;
		};
	}, [kind, target]);

	const title = kind === 'columns' ? 'Columns' : kind === 'queries' ? 'SQL Queries' : 'Terms';

	const renderTable = () => {
		if (kind === 'columns') {
			const tableColumns: TableColumn<Column>[] = [
				{
					key: 'column',
					header: 'Name',
					cell: (row) => row.column_name,
					title: (row) => row.column_name,
					truncate: true,
				},
				{
					key: 'link',
					header: '',
					width: 'w-10',
					cell: (row) =>
						target != null ? (
							<NextLink
								href={catalogPathFromFocusId(
									`${target.databaseId}|${target.schemaId}|${target.id}|${row.id}`,
								)}
								className="flex h-6 w-6 items-center justify-center rounded text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:hover:bg-zinc-700"
								aria-label={`Open ${row.column_name} in Data`}
								title={`Open ${row.column_name} in Data`}
							>
								<Icon name={IconName.ExternalLink} className="h-3.5 w-3.5" />
							</NextLink>
						) : null,
				},
			];
			return (
				<Table
					columns={tableColumns}
					rows={columns}
					rowKey={(row) => row.id}
					containerClassName="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
					scrollClassName="max-h-[28rem] overflow-auto"
					emptyMessage="No columns"
				/>
			);
		}

		if (kind === 'queries') {
			if (details.queries.length === 0) {
				return (
					<p className="text-sm italic text-zinc-500 dark:text-zinc-400">
						No SQL queries
					</p>
				);
			}
			return (
				<ul className="space-y-4">
					{details.queries.map((query, index) => (
						<li key={query.id || `${index}`}>
							<SqlBlock sql={query.sql} label={`Query ${index + 1}`} />
						</li>
					))}
				</ul>
			);
		}

		const termColumns: TableColumn<TableExplorationDetails['terms'][number]>[] = [
			{ key: 'name', header: 'Name', width: 'w-48', cell: (row) => row.name },
			{
				key: 'description',
				header: 'Description',
				cell: (row) => row.description || 'No Description',
				title: (row) => row.description ?? '',
				truncate: true,
			},
			{
				key: 'link',
				header: '',
				width: 'w-10',
				cell: (row) => (
					<NextLink
						href={`/terms?focus=${encodeURIComponent(row.id)}`}
						className="flex h-6 w-6 items-center justify-center rounded text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:hover:bg-zinc-700"
						aria-label={`Open ${row.name} term`}
						title={`Open ${row.name} term`}
					>
						<Icon name={IconName.ExternalLink} className="h-3.5 w-3.5" />
					</NextLink>
				),
			},
		];
		return (
			<Table
				columns={termColumns}
				rows={details.terms}
				rowKey={(row) => row.id}
				containerClassName="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
				scrollClassName="max-h-[28rem] overflow-auto"
				emptyMessage="No Terms"
			/>
		);
	};

	return (
		<Modal open={target != null && kind != null} onClose={onClose} className="w-full max-w-4xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon
						name={
							kind === 'columns'
								? IconName.Column
								: kind === 'terms'
									? IconName.Terms
									: IconName.Link
						}
						className="h-5 w-5 shrink-0 text-[#76b900]"
					/>
					<h2 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{target?.name} ({title})
					</h2>
				</div>
				<button
					type="button"
					onClick={onClose}
					className="cursor-pointer rounded p-1 text-lg text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
					aria-label={`Close ${title}`}
				>
					×
				</button>
			</header>
			<div className="max-h-[70dvh] overflow-y-auto p-5">
				{loading ? (
					<div className="flex h-32 items-center justify-center">
						<div
							className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900]"
							role="status"
							aria-label={`Loading ${title}`}
						/>
					</div>
				) : error != null ? (
					<p className="text-sm text-red-600 dark:text-red-300">{error}</p>
				) : (
					renderTable()
				)}
			</div>
		</Modal>
	);
};
