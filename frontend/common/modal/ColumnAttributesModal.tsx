// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';

import { termsApi } from '@/api/terms';
import { Icon, IconName } from '@/common/icons';
import type { ColumnAttribute } from '@/types/terms';
import type { TableColumn } from '@/types/table';
import { LabelList } from '@/common/SinglePageComposer';
import { Table } from '@/common/Table';
import { TruncatedText } from '@/common/TruncatedText';
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
	const [attributes, setAttributes] = useState<ColumnAttribute[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		if (term == null) return undefined;
		let cancelled = false;

		const load = async () => {
			setLoading(true);
			setError(null);
			const response = await termsApi.getColumnAttributes(term.id);
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load attribute columns');
			} else {
				setAttributes(response.data ?? []);
			}
			setLoading(false);
		};

		void load();
		return () => {
			cancelled = true;
		};
	}, [term]);

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
			cell: (row) =>
				row.description ? (
					<TruncatedText text={row.description} maxWidthClass="max-w-none" />
				) : (
					'—'
				),
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
					<Icon name={IconName.Column} className="h-5 w-5 shrink-0 text-[#76b900]" />
					<h2 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{term?.name} — Attribute Columns ({attributes.length})
					</h2>
				</div>
				<button
					type="button"
					onClick={onClose}
					className="cursor-pointer rounded p-1 text-lg text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
					aria-label="Close attribute columns"
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
							aria-label="Loading attribute columns"
						/>
					</div>
				) : error != null ? (
					<p className="text-sm text-red-600 dark:text-red-300">{error}</p>
				) : (
					<Table
						columns={columns}
						rows={attributes}
						rowKey={(row) => row.id}
						containerClassName="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
						scrollClassName="max-h-[28rem] overflow-auto"
						emptyMessage="No attribute columns"
					/>
				)}
			</div>
		</Modal>
	);
};
