// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';

import { termsApi } from '@/api/terms';
import { Icon, IconName } from '@/common/icons';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import type { ColumnAttribute } from '@/types/terms';
import type { TableColumn } from '@/types/table';
import { LabelList } from '@/common/SinglePageComposer';
import { SkeletonTable } from '@/common/Skeleton';
import { Table } from '@/common/Table';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import { usePagination } from '@/hooks/usePagination';
import { Modal } from './Modal';

/** Minimal Term reference — decoupled from any specific page's node/row shape. */
export type ColumnAttributesModalTerm = {
	id: string;
	name: string;
};

type ColumnAttributesModalProps = {
	term: ColumnAttributesModalTerm | null;
	onClose: () => void;
};

/** Generic modal listing a Term's attribute Columns. Reusable from any page that has a term id. */
export const ColumnAttributesModal = ({ term, onClose }: ColumnAttributesModalProps) => {
	const termId = term?.id ?? null;
	const [attributes, setAttributes] = useState<ColumnAttribute[]>([]);
	const [total, setTotal] = useState(0);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const { skip, pageSize, pageRowCount, pagination } = usePagination({
		totalItems: total,
		resetKey: termId,
	});

	useEffect(() => {
		if (termId == null) return undefined;
		let cancelled = false;

		const load = async () => {
			setLoading(true);
			setError(null);
			const response = await termsApi.getColumnAttributes(termId, {
				skip,
				limit: pageSize,
			});
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load attribute columns');
			} else {
				setAttributes(response.data ?? []);
				setTotal(response.total ?? 0);
			}
			setLoading(false);
		};

		void load();
		return () => {
			cancelled = true;
		};
	}, [termId, skip, pageSize]);

	const columns: TableColumn<ColumnAttribute>[] = [
		{
			key: 'name',
			header: 'Attribute Name',
			cell: (row) => row.name,
			title: (row) => row.name,
			truncate: true,
			width: 'w-40',
		},
		{
			key: 'description',
			header: 'Description',
			cell: (row) => row.description || '—',
			title: (row) => row.description ?? '',
			truncate: true,
		},
		{
			key: 'sample_values',
			header: 'Sample Values',
			width: 'w-56',
			cell: (row) => <LabelList values={row.sample_values ?? []} />,
		},
	];

	return (
		<Modal open={term != null} onClose={onClose} className="w-full max-w-2xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon
						name={IconName.Column}
						className="h-5 w-5 shrink-0 text-body dark:text-zinc-300"
					/>
					<Text as="h2" variant={TextVariant.Heading}>
						{term?.name} — Attribute Columns ({total})
					</Text>
				</div>
				<Button
					theme={ButtonTheme.IconNeutral}
					size={Size.SMALL}
					iconOnly
					type="button"
					onClick={onClose}
					aria-label="Close attribute columns"
				>
					<Icon name={IconName.Close} className="h-4 w-4" />
				</Button>
			</header>
			<div className="max-h-[70dvh] overflow-y-auto p-5">
				{loading ? (
					<div role="status" aria-label="Loading attribute columns">
						<SkeletonTable columns={columns.length} rows={pageRowCount} />
					</div>
				) : error != null ? (
					<p className="text-sm text-red-600 dark:text-red-300">{error}</p>
				) : (
					<Table
						columns={columns}
						rows={attributes}
						rowKey={(row) => row.id}
						pagination={pagination}
						containerClassName="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
						emptyMessage="No Attribute Columns"
					/>
				)}
			</div>
		</Modal>
	);
};
