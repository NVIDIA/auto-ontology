// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';

import { Button } from '@/common/Button';
import { EmptyState } from '@/common/EmptyState';
import { Size, ButtonTheme } from '@/enums/button';
import { Icon, IconName } from '@/common/icons';
import { SkeletonTable } from '@/common/Skeleton';
import { Table } from '@/common/Table';
import { Toast } from '@/common/Toast';
import { analyticsApi } from '@/api/analytics';
import { formatDate } from '@/common/date';
import { usePagination } from '@/hooks/usePagination';
import type { ConversationAnalytics } from '@/types/analytics';
import type { TableColumn } from '@/types/table';

const CSV_HEADERS = ['Timestamp', 'User', 'Source', 'Question', 'Reasoning', 'SQL'];
const escapeCsv = (value: string) => `"${value.replace(/"/g, '""')}"`;

const COLUMNS: TableColumn<ConversationAnalytics>[] = [
	{
		key: 'question_timestamp',
		header: 'Timestamp',
		width: 'w-44',
		nowrap: true,
		className: 'text-zinc-600 dark:text-zinc-300',
		cell: (row) => formatDate(row.question_timestamp),
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
	const [total, setTotal] = useState(0);
	const [loading, setLoading] = useState(true);
	const [downloading, setDownloading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	// Kept apart from `error`: the table below only renders while `error` is
	// null, so a failed export must not take the rows already on screen with it.
	const [downloadError, setDownloadError] = useState<string | null>(null);
	const { skip, pageSize, pagination } = usePagination({ totalItems: total });

	useEffect(() => {
		let cancelled = false;

		(async () => {
			setLoading(true);
			const res = await analyticsApi.list({ skip, limit: pageSize });
			if (cancelled) return;
			if (res.error) {
				// The page already on screen is left alone: dropping the total
				// would collapse the pager to page one and send the request for
				// it, so a single failed request would cost two.
				setError(res.message ?? 'Failed to load analytics');
			} else {
				setError(null);
				setRows(res.data ?? []);
				setTotal(res.total);
			}
			setLoading(false);
		})();

		return () => {
			cancelled = true;
		};
	}, [skip, pageSize]);

	const downloadRows = (analytics: ConversationAnalytics[]) => {
		const lines = [
			CSV_HEADERS.join(','),
			...analytics.map((row) =>
				[
					formatDate(row.question_timestamp),
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

	const handleDownload = async () => {
		setDownloading(true);
		setDownloadError(null);
		const res = await analyticsApi.list();
		setDownloading(false);
		if (res.error) {
			setDownloadError(res.message ?? 'Failed to download analytics');
			return;
		}
		downloadRows(res.data ?? []);
	};

	return (
		<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
			<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
				<Icon name={IconName.ChartLine} className="h-5 w-5 text-[#76b900]" />
				<h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					Analytics
				</h1>
				<div className="ml-auto">
					<Button
						theme={ButtonTheme.Primary}
						size={Size.REGULAR}
						type="button"
						onClick={handleDownload}
						disabled={total === 0 || downloading}
						shadow
					>
						{downloading ? 'Preparing download…' : 'Download'}
					</Button>
				</div>
			</header>

			<div className="flex-1 overflow-y-auto px-6 py-6">
				{loading && (
					<div role="status" aria-label="Loading analytics">
						<SkeletonTable columns={6} rows={10} />
					</div>
				)}

				{!loading && error == null && total === 0 && (
					<EmptyState
						icon={IconName.ChartLine}
						title="No analytics recorded yet"
						description="Analytics are captured automatically when you send messages in a conversation."
					/>
				)}

				{!loading && error == null && rows.length > 0 && (
					<Table
						columns={COLUMNS}
						rows={rows}
						rowKey={(row) => row.id}
						pagination={pagination}
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

			<Toast
				open={downloadError !== null}
				message={downloadError ?? ''}
				title="Couldn't download analytics"
				variant="error"
				onClose={() => setDownloadError(null)}
			/>
		</div>
	);
};
