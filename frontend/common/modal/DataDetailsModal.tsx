// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import NextLink from 'next/link';

import { datasources } from '@/api/datasources';
import { explorationApi } from '@/api/exploration';
import { Icon, IconName } from '@/common/icons';
import { Button } from '@/common/Button';
import { EmptyState } from '@/common/EmptyState';
import { Size, ButtonTheme } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import type { Column } from '@/types/datasources';
import type { TableExplorationDetails } from '@/types/exploration';
import type { TableColumn } from '@/types/table';
import { SkeletonSqlBlocks, SkeletonTable } from '@/common/Skeleton';
import { SqlBlock } from '@/common/SqlBlock';
import { Table } from '@/common/Table';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import { usePagination } from '@/hooks/usePagination';
import { Modal } from './Modal';

export type DataDetailsType = 'columns' | 'queries' | 'terms';

/** Minimal data-object reference — decoupled from any specific page's node shape. */
export type DataDetailsModalTarget = {
	id: string;
	name: string;
	databaseId: string;
	schemaId: string;
};

type DataDetailsModalProps = {
	target: DataDetailsModalTarget | null;
	type: DataDetailsType | null;
	onClose: () => void;
};

/** Generic modal for a Table/View's columns, SQL queries, or related Terms. */
export const DataDetailsModal = ({ target, type, onClose }: DataDetailsModalProps) => {
	const [columns, setColumns] = useState<Column[]>([]);
	const [details, setDetails] = useState<TableExplorationDetails>({
		queries: [],
		terms: [],
		terms_total: 0,
	});
	const [columnsTotal, setColumnsTotal] = useState(0);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	// Columns are read a page at a time, so `columns` already holds one page.
	const columnsPage = usePagination({
		totalItems: columnsTotal,
		resetKey: target?.id ?? null,
	});
	const { skip: columnsSkip, pageSize: columnsPageSize } = columnsPage;
	// Terms are read from the server one page at a time.
	const termsPage = usePagination({
		totalItems: details.terms_total,
		resetKey: target?.id ?? null,
	});

	useEffect(() => {
		if (target == null || type !== 'columns') return undefined;
		let cancelled = false;

		const load = async () => {
			setLoading(true);
			setError(null);
			const response = await datasources.getColumnsForTable(target.id, {
				skip: columnsSkip,
				limit: columnsPageSize,
			});
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load columns');
			} else {
				setColumns(response.data ?? []);
				setColumnsTotal(response.total ?? 0);
			}
			setLoading(false);
		};

		void load();
		return () => {
			cancelled = true;
		};
	}, [target, type, columnsSkip, columnsPageSize]);

	useEffect(() => {
		if (target == null || type == null || type === 'columns') return undefined;
		let cancelled = false;

		const load = async () => {
			setLoading(true);
			setError(null);
			const response = await explorationApi.getTableExplorationDetails(target.id, {
				skip: termsPage.skip,
				limit: termsPage.pageSize,
			});
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load details');
			} else {
				setDetails(response.data);
			}
			setLoading(false);
		};

		void load();
		return () => {
			cancelled = true;
		};
	}, [type, target, termsPage.skip, termsPage.pageSize]);

	const title = type === 'columns' ? 'Columns' : type === 'queries' ? 'SQL Queries' : 'Terms';

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

	const termTableColumns: TableColumn<TableExplorationDetails['terms'][number]>[] = [
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

	const renderTable = () => {
		if (type === 'columns') {
			return (
				<Table
					columns={tableColumns}
					rows={columns}
					rowKey={(row) => row.id}
					pagination={columnsPage.pagination}
					containerClassName="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
					emptyMessage="No columns"
				/>
			);
		}

		if (type === 'queries') {
			if (details.queries.length === 0) {
				return <EmptyState variant={EmptyStateVariant.Inline} title="No SQL queries" />;
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

		return (
			<Table
				columns={termTableColumns}
				rows={details.terms}
				rowKey={(row) => row.id}
				pagination={termsPage.pagination}
				containerClassName="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
				emptyMessage="No Terms"
			/>
		);
	};

	// Queries arrive as a list of SQL cards, the other two tabs as a table, so
	// the placeholder follows whichever one is about to appear.
	const renderSkeleton = () => {
		if (type === 'queries') return <SkeletonSqlBlocks />;
		return type === 'columns' ? (
			<SkeletonTable columns={tableColumns.length} rows={columnsPage.pageRowCount} />
		) : (
			<SkeletonTable columns={termTableColumns.length} rows={termsPage.pageRowCount} />
		);
	};

	return (
		<Modal open={target != null && type != null} onClose={onClose} className="w-full max-w-4xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon
						name={
							type === 'columns'
								? IconName.Column
								: type === 'terms'
									? IconName.Terms
									: IconName.Link
						}
						className="h-5 w-5 shrink-0 text-[#76b900]"
					/>
					<Text as="h2" variant={TextVariant.Heading}>
						{target?.name} ({title})
					</Text>
				</div>
				<Button
					theme={ButtonTheme.IconNeutral}
					size={Size.SMALL}
					iconOnly
					type="button"
					onClick={onClose}
					aria-label={`Close ${title}`}
				>
					<Icon name={IconName.Close} className="h-4 w-4" />
				</Button>
			</header>
			<div className="max-h-[70dvh] overflow-y-auto p-5">
				{loading ? (
					<div role="status" aria-label={`Loading ${title}`}>
						{renderSkeleton()}
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
