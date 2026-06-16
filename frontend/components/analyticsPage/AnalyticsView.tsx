// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import {
	PaginationArrowButton,
	PaginationItemRangeText,
	PaginationNavigationGroup,
	PaginationPageList,
	PaginationRoot,
} from '@nvidia/foundations-react-core';

import { Icon, IconName } from '@/components/icons';
import { analyticsApi } from '@/api/analytics';
import type { ConversationAnalytics } from '@/types/analytics';

const formatDate = (iso: string) =>
	new Intl.DateTimeFormat(undefined, {
		dateStyle: 'medium',
		timeStyle: 'short',
	}).format(new Date(iso));

const PAGE_SIZE = 10;

const CSV_HEADERS = ['Timestamp', 'Question', 'Reasoning', 'Response SQL'];
const escapeCsv = (value: string) => `"${value.replace(/"/g, '""')}"`;

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
		const now = new Date();
		const month = String(now.getMonth() + 1).padStart(2, '0');
		const filename = `gsf-analytics-${now.getFullYear()}-${month}.csv`;
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

				{!loading && error != null && (
					<div className="mx-auto max-w-lg rounded-2xl border border-red-200/80 bg-white/90 px-8 py-10 text-center shadow-xl shadow-red-100/50 dark:border-red-900/50 dark:bg-zinc-950/80 dark:shadow-none">
						<h2 className="text-lg font-semibold tracking-tight text-red-800 dark:text-red-300">
							Couldn&apos;t load analytics
						</h2>
						<pre className="mt-4 max-w-full overflow-x-auto rounded-lg border border-red-100 bg-red-50/80 p-3 text-left text-xs text-red-900/80 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
							{error}
						</pre>
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
					<div className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
						<table className="w-full table-fixed text-sm">
							<thead>
								<tr className="border-b border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-800/60">
									<th className="w-44 px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
										Timestamp
									</th>
									<th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
										Question
									</th>
									<th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
										Response
									</th>
									<th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
										SQL
									</th>
								</tr>
							</thead>
							<tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
								{pageRows.map((row) => (
									<tr
										key={row.id}
										className="transition-colors hover:bg-zinc-50 dark:hover:bg-zinc-800/40"
									>
										<td className="whitespace-nowrap px-4 py-3 text-zinc-600 dark:text-zinc-300">
											{formatDate(row.questionTimestamp)}
										</td>
										<td className="max-w-0 px-4 py-3 text-zinc-700 dark:text-zinc-300">
											<span
												className="block truncate"
												title={row.question ?? ''}
											>
												{row.question}
											</span>
										</td>
										<td className="max-w-0 px-4 py-3 text-zinc-700 dark:text-zinc-300">
											<span
												className="block truncate"
												title={row.response ?? ''}
											>
												{row.response}
											</span>
										</td>
										<td className="max-w-0 px-4 py-3">
											<span className="block truncate" title={row.sql ?? ''}>
												{row.sql}
											</span>
										</td>
									</tr>
								))}
							</tbody>
						</table>
						<PaginationRoot
							totalItems={rows.length}
							pageSize={PAGE_SIZE}
							page={currentPage}
							onPageChange={setPage}
							className="border-t border-zinc-200 px-4 py-3 dark:border-zinc-800"
						>
							<PaginationNavigationGroup withTabs>
								<PaginationArrowButton direction="previous" />
								<PaginationPageList />
								<PaginationArrowButton direction="next" />
							</PaginationNavigationGroup>
							<PaginationItemRangeText
								rangeTextFormatFn={({
									firstItemIndex,
									lastItemIndex,
									totalItems,
								}) => `${firstItemIndex}-${lastItemIndex} out of ${totalItems}`}
							/>
						</PaginationRoot>
					</div>
				)}
			</div>
		</div>
	);
};
