// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useMemo } from 'react';
import type { ParsedTable, TableRow } from '@/lib/parseSqlResponse';
import { EmptyState } from '@/common/EmptyState';
import { Table } from '@/common/Table';
import { EmptyStateVariant } from '@/enums/emptyState';
import { usePagination } from '@/hooks/usePagination';
import type { TableColumn } from '@/types/table';

type DynamicTableProps = {
	table: ParsedTable;
};

export const DynamicTable = ({ table }: DynamicTableProps) => {
	const { columns, rows } = table;
	const { skip, pageSize, pagination } = usePagination({ totalItems: rows.length });
	// A fresh array every render would re-render `Table` on every parent render,
	// which chat result sets are large enough to feel.
	const pageRows = useMemo(() => rows.slice(skip, skip + pageSize), [rows, skip, pageSize]);

	const tableColumns = useMemo<TableColumn<TableRow>[]>(
		() =>
			columns.map((col) => ({
				key: col,
				header: col,
				truncate: true,
				maxWidthClass: 'max-w-[240px]',
				className: 'text-zinc-700 dark:text-zinc-300',
				cell: (row) => row[col],
				title: (row) => row[col],
			})),
		[columns],
	);

	if (columns.length === 0 || rows.length === 0) {
		return <EmptyState variant={EmptyStateVariant.Inline} title="No data available" />;
	}

	return (
		<Table
			columns={tableColumns}
			rows={pageRows}
			rowKey={(_, index) => String(index)}
			layout="auto"
			textClassName="text-xs"
			cellClassName="px-3 py-2"
			containerClassName="overflow-hidden rounded-lg border border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-900"
			theadClassName="sticky top-0 border-b border-zinc-200 bg-zinc-50 text-left font-semibold text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
			bodyClassName="text-zinc-700 dark:text-zinc-300"
			rowClassName="border-b border-zinc-100 last:border-b-0 dark:border-zinc-800"
			scrollClassName="max-h-[478px] overflow-auto"
			pagination={pagination}
		/>
	);
};
