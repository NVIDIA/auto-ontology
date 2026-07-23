// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useMemo, useState } from 'react';
import type { ParsedTable, TableRow } from '@/lib/parseSqlResponse';
import { Table } from '@/common/Table';
import type { TableColumn } from '@/types/table';

const PAGE_SIZE = 10;

type DynamicTableProps = {
	table: ParsedTable;
};

export const DynamicTable = ({ table }: DynamicTableProps) => {
	const { columns, rows } = table;
	const [page, setPage] = useState(1);

	const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
	const currentPage = Math.min(page, pageCount);

	const paginatedRows = useMemo(() => {
		const start = (currentPage - 1) * PAGE_SIZE;
		return rows.slice(start, start + PAGE_SIZE);
	}, [rows, currentPage]);

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
		return <p className="text-xs text-zinc-400">No data available</p>;
	}

	return (
		<Table
			columns={tableColumns}
			rows={paginatedRows}
			rowKey={(_, index) => String(index)}
			layout="auto"
			textClassName="text-xs"
			cellClassName="px-3 py-2"
			containerClassName="overflow-hidden rounded-lg border border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-900"
			theadClassName="sticky top-0 border-b border-zinc-200 bg-zinc-50 text-left font-semibold text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
			bodyClassName="text-zinc-700 dark:text-zinc-300"
			rowClassName="border-b border-zinc-100 last:border-b-0 dark:border-zinc-800"
			scrollClassName="max-h-[478px] overflow-auto"
			pagination={{
				page: currentPage,
				pageSize: PAGE_SIZE,
				totalItems: rows.length,
				onPageChange: setPage,
			}}
		/>
	);
};
