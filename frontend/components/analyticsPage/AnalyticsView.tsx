// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';

import { Icon, IconName } from '@/components/icons';
import { Table } from '@/components/Table';
import { Toast } from '@/components/Toast';
import { analyticsApi } from '@/api/analytics';
import { formatDate } from '@/common/date';
import type { ConversationAnalytics } from '@/types/analytics';
import type { TableColumn } from '@/types/table';

const PAGE_SIZE = 10;

const CSV_HEADERS = ['Timestamp', 'User', 'Source', 'Question', 'Reasoning', 'SQL'];
const escapeCsv = (value: string) => `"${value.replace(/"/g, '""')}"`;

const COLUMNS: TableColumn<ConversationAnalytics>[] = [
	{
		key: 'questionTimestamp',
		header: 'Timestamp',
		width: 'w-44',
		nowrap: true,
		className: 'text-zinc-600 dark:text-zinc-300',
		cell: (row) => formatDate(row.questionTimestamp),
	},
	{
		key: 'user',
		header: 'User',
		width: 'w-40',
		nowrap: true,
		truncate: true,
		className: 'text-zinc-700 dark:text-zinc-300',
		cell: (row) => row.user.name || row.user.email,
		title: (row) => row.user.name || row.user.email,
	},
	{
		key: 'source',
		header: 'Source',
		width: 'w-24',
		nowrap: true,
		className: 'text-zinc-600 dark:text-zinc-300 uppercase',
		cell: (row) => row.source ?? '—',
		title: (row) => row.source ?? '',
	},
	{
		key: 'question',
		header: 'Question',
		truncate: true,
		className: 'text-zinc-700 dark:text-zinc-300',
		cell: (row) => row.question,
		title: (row) => row.question ?? '',
	},
	{
		key: 'response',
		header: 'Reasoning',
		truncate: true,
		className: 'text-zinc-700 dark:text-zinc-300',
		cell: (row) => row.response,
		title: (row) => row.response ?? '',
	},
	{
		key: 'sql',
		header: 'SQL',
		truncate: true,
		cell: (row) => row.sql,
		title: (row) => row.sql ?? '',
	},
];

export const AnalyticsView = () => {
	const [rows, setRows] = useState<ConversationAnalytics[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [page, setPage] = useState(1);

	useEffect(() => {
		let cancelled = false;

		(async () => {
			setLoading(true);
			const res = await analyticsApi.list();
			if (cancelled) return;
			if (res.error === true) {
				setError(res.message ?? 'Failed to load analytics');
				setRows([]);
			} else {
				setError(null);
				setRows(res.data ?? []);
			}
			setLoading(false);
		})();

		return () => {
			cancelled = true;
		};
	}, []);

	const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
	const currentPage = Math.min(page, pageCount);
	const pageRows = rows.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

	const handleDownload = () => {
		const lines = [
			CSV_HEADERS.join(','),
			...rows.map((row) =>
				[
					formatDate(row.questionTimestamp),
					row.user.name || row.user.email,
					row.source ?? '',
					row.question ?? '',
					row.response ?? '',
					row.sql ?? '',
				]
					.map(escapeCsv)
					.join(','),
			),
		];
		const csv = lines.join('\r\n');

		const blob = new Blob([`\uFEFF${csv}`], { type: 'text/csv;charset=utf-8;' });
		const url = URL.createObjectURL(blob);
		const filename = `gsf-analytics-${formatDate(Date.now(), 'YYYY-MM')}.csv`;
		const anchor = document.createElement('a');
		anchor.href = url;
		anchor.download = filename;
		document.body.appendChild(anchor);
		anchor.click();
		anchor.remove();
		URL.revokeObjectURL(url);
	};

	return (
		<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
			<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
				<Icon name={IconName.ChartLine} className="h-5 w-5 text-[#76b900]" />
				<h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					Analytics
				</h1>
				<button
					type="button"
					onClick={handleDownload}
					disabled={rows.length === 0}
					className="ml-auto flex cursor-pointer items-center gap-2 rounded-lg bg-[#76b900] px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[#5e9400] disabled:cursor-not-allowed disabled:opacity-40"
				>
					Download
				</button>
			</header>

			<div className="flex-1 overflow-y-auto px-6 py-6">
				{loading && (
					<div className="flex h-full items-center justify-center">
						<div
							className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
							role="status"
							aria-label="Loading analytics"
						/>
					</div>
				)}

				{!loading && error == null && rows.length === 0 && (
					<div className="flex h-full min-h-[40dvh] flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-zinc-300/80 bg-white/60 p-12 text-center dark:border-zinc-600 dark:bg-zinc-950/40">
						<Icon
							name={IconName.ChartLine}
							className="h-8 w-8 text-zinc-300 dark:text-zinc-600"
						/>
						<p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
							No analytics recorded yet
						</p>
						<p className="text-xs text-zinc-500 dark:text-zinc-500">
							Analytics are captured automatically when you send messages in a
							conversation.
						</p>
					</div>
				)}

				{!loading && error == null && rows.length > 0 && (
					<Table
						columns={COLUMNS}
						rows={pageRows}
						rowKey={(row) => row.id}
						pagination={{
							page: currentPage,
							pageSize: PAGE_SIZE,
							totalItems: rows.length,
							onPageChange: setPage,
						}}
					/>
				)}
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				title="Couldn't load analytics"
				variant="error"
				onClose={() => setError(null)}
			/>
		</div>
	);
};
