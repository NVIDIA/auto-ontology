// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import { om } from '@/api/openmetadata';
import type { OmTable } from '@/types/openmetadata';
import { TagPill } from '@/components/catalog/TagPill';

type Filter = 'all' | 'pii' | 'undocumented' | 'tagged';

const filterOf = (params: URLSearchParams): Filter => {
	if (params.get('pii') === '1') return 'pii';
	if (params.get('filter') === 'undocumented') return 'undocumented';
	if (params.get('filter') === 'tagged') return 'tagged';
	return 'all';
};

const piiColumnCount = (t: OmTable): number =>
	(t.columns ?? []).reduce(
		(acc, c) => acc + ((c.tags ?? []).some((tag) => tag.tagFQN.startsWith('PII.')) ? 1 : 0),
		0,
	);

export default function TablesListPage() {
	const params = useSearchParams();
	const initialFilter = filterOf(new URLSearchParams(params.toString()));
	const initialQuery = params.get('q') ?? '';

	const [tables, setTables] = useState<OmTable[] | null>(null);
	const [err, setErr] = useState<string | null>(null);
	const [q, setQ] = useState(initialQuery);
	const [filter, setFilter] = useState<Filter>(initialFilter);

	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				const r = await om.tables.list({ fields: 'columns,tags,description', limit: 200 });
				if (!cancelled) setTables(r.data);
			} catch (e) {
				if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
			}
		})();
		return () => {
			cancelled = true;
		};
	}, []);

	const filtered = useMemo(() => {
		if (!tables) return null;
		const needle = q.trim().toLowerCase();
		return tables.filter((t) => {
			if (needle && !t.fullyQualifiedName.toLowerCase().includes(needle)) return false;
			if (filter === 'pii' && piiColumnCount(t) === 0) return false;
			if (filter === 'undocumented' && t.description) return false;
			if (filter === 'tagged' && !((t.tags ?? []).length > 0)) return false;
			return true;
		});
	}, [tables, q, filter]);

	return (
		<div className="mx-auto w-full max-w-7xl px-6 py-8">
			<header className="mb-6 flex items-end justify-between gap-4">
				<div>
					<h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
						Data Dictionary
					</h1>
					<p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
						{tables == null
							? 'Loading tables…'
							: `${filtered?.length ?? 0} of ${tables.length} tables`}
					</p>
				</div>
				<div className="flex w-72 items-center gap-2 rounded-lg border border-zinc-300 bg-white px-3 py-1.5 shadow-sm dark:border-zinc-700 dark:bg-zinc-900">
					<svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4 text-zinc-400">
						<path
							fillRule="evenodd"
							d="M9 3a6 6 0 1 0 3.84 10.61l3.27 3.28 1.42-1.42-3.28-3.27A6 6 0 0 0 9 3zm0 2a4 4 0 1 1 0 8 4 4 0 0 1 0-8z"
							clipRule="evenodd"
						/>
					</svg>
					<input
						value={q}
						onChange={(e) => setQ(e.target.value)}
						placeholder="Search table FQN…"
						className="w-full bg-transparent text-sm outline-none placeholder:text-zinc-400 dark:text-zinc-100"
					/>
				</div>
			</header>

			<div className="mb-4 flex flex-wrap gap-2">
				{(
					[
						['all', 'All'],
						['pii', 'PII'],
						['tagged', 'Has tags'],
						['undocumented', 'Undocumented'],
					] as const
				).map(([id, label]) => (
					<button
						key={id}
						type="button"
						onClick={() => setFilter(id)}
						className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
							filter === id
								? 'border-[#76b900] bg-[#76b900]/10 text-[#76b900]'
								: 'border-zinc-300 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800'
						}`}
					>
						{label}
					</button>
				))}
			</div>

			{err ? (
				<div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
					{err}
				</div>
			) : tables == null ? (
				<TableSkeleton />
			) : filtered && filtered.length === 0 ? (
				<p className="rounded-lg border border-dashed border-zinc-300 p-12 text-center text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
					No tables match these filters.
				</p>
			) : (
				<div className="overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
					<table className="w-full text-sm">
						<thead className="border-b border-zinc-200 bg-zinc-50 text-left text-xs font-medium uppercase tracking-wider text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950/40 dark:text-zinc-400">
							<tr>
								<th className="px-4 py-2.5">Table</th>
								<th className="px-4 py-2.5">Schema</th>
								<th className="px-4 py-2.5 text-right">Cols</th>
								<th className="px-4 py-2.5 text-right">PII</th>
								<th className="px-4 py-2.5">Tags</th>
								<th className="px-4 py-2.5">Description</th>
							</tr>
						</thead>
						<tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
							{(filtered ?? []).map((t) => {
								const piiN = piiColumnCount(t);
								return (
									<tr
										key={t.id}
										className="group transition-colors hover:bg-[#76b900]/5"
									>
										<td className="px-4 py-3">
											<Link
												href={`/catalog/tables/${encodeURIComponent(t.fullyQualifiedName)}`}
												className="font-medium text-zinc-900 group-hover:text-[#76b900] dark:text-zinc-100"
											>
												{t.name}
											</Link>
											{t.displayName && t.displayName !== t.name ? (
												<span className="ml-2 text-xs text-zinc-500">
													{t.displayName}
												</span>
											) : null}
										</td>
										<td className="px-4 py-3 font-mono text-xs text-zinc-600 dark:text-zinc-400">
											{t.database.name} · {t.databaseSchema.name}
										</td>
										<td className="px-4 py-3 text-right tabular-nums text-zinc-700 dark:text-zinc-300">
											{t.columns?.length ?? 0}
										</td>
										<td className="px-4 py-3 text-right">
											{piiN > 0 ? (
												<span className="inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-900 dark:bg-amber-950/60 dark:text-amber-200">
													{piiN}
												</span>
											) : (
												<span className="text-xs text-zinc-400">—</span>
											)}
										</td>
										<td className="px-4 py-3">
											{(t.tags ?? []).length === 0 ? (
												<span className="text-xs text-zinc-400">—</span>
											) : (
												<div className="flex flex-wrap gap-1">
													{(t.tags ?? []).slice(0, 3).map((tag) => (
														<TagPill
															key={tag.tagFQN}
															label={tag}
															dense
														/>
													))}
												</div>
											)}
										</td>
										<td className="px-4 py-3 text-xs text-zinc-600 dark:text-zinc-400">
											<span className="line-clamp-2 max-w-md">
												{t.description ?? (
													<em className="text-zinc-400">
														— not documented —
													</em>
												)}
											</span>
										</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</div>
			)}
		</div>
	);
}

const TableSkeleton = () => (
	<div className="space-y-2">
		{Array.from({ length: 6 }).map((_, i) => (
			<div key={i} className="h-12 animate-pulse rounded-lg bg-zinc-100 dark:bg-zinc-900" />
		))}
	</div>
);
