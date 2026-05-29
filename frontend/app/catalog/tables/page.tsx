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
import {
	CategoryTabs,
	CertificationBadge,
	OwnerChip,
	UsageMeter,
	readinessOf,
	usageLevelOf,
} from '@/components/catalog/ui';

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

const matchesFilter = (t: OmTable, filter: Filter): boolean => {
	if (filter === 'pii') return piiColumnCount(t) > 0;
	if (filter === 'undocumented') return !t.description;
	if (filter === 'tagged') return (t.tags ?? []).length > 0;
	return true;
};

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
				const r = await om.tables.list({
					fields: 'columns,tags,description,owners,usageSummary',
					limit: 200,
				});
				if (!cancelled) setTables(r.data);
			} catch (e) {
				if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
			}
		})();
		return () => {
			cancelled = true;
		};
	}, []);

	const counts = useMemo(() => {
		const base = { all: 0, pii: 0, tagged: 0, undocumented: 0 };
		if (!tables) return base;
		for (const t of tables) {
			base.all += 1;
			if (piiColumnCount(t) > 0) base.pii += 1;
			if ((t.tags ?? []).length > 0) base.tagged += 1;
			if (!t.description) base.undocumented += 1;
		}
		return base;
	}, [tables]);

	const filtered = useMemo(() => {
		if (!tables) return null;
		const needle = q.trim().toLowerCase();
		return tables.filter((t) => {
			if (
				needle &&
				!t.fullyQualifiedName.toLowerCase().includes(needle) &&
				!(t.tags ?? []).some((tag) => tag.tagFQN.toLowerCase().includes(needle))
			)
				return false;
			return matchesFilter(t, filter);
		});
	}, [tables, q, filter]);

	return (
		<div className="mx-auto w-full max-w-7xl px-6 py-8">
			<header className="mb-5">
				<h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
					Data Dictionary
				</h1>
				<p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
					Every table OpenMetadata harvested, with ownership, certification status, usage
					and tags.
				</p>
			</header>

			<div className="mb-4 flex flex-wrap items-center justify-between gap-3">
				<CategoryTabs
					tabs={[
						{ id: 'all', label: 'All tables', count: counts.all },
						{ id: 'pii', label: 'PII', count: counts.pii },
						{ id: 'tagged', label: 'Tagged', count: counts.tagged },
						{ id: 'undocumented', label: 'Undocumented', count: counts.undocumented },
					]}
					active={filter}
					onSelect={setFilter}
				/>
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
						placeholder="Search tables or tags…"
						className="w-full bg-transparent text-sm outline-none placeholder:text-zinc-400 dark:text-zinc-100"
					/>
				</div>
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
								<th className="px-4 py-2.5">Owner</th>
								<th className="px-4 py-2.5">Status</th>
								<th className="px-4 py-2.5">Usage</th>
								<th className="px-4 py-2.5 text-right">Cols</th>
								<th className="px-4 py-2.5 text-right">PII</th>
								<th className="px-4 py-2.5">Tags</th>
							</tr>
						</thead>
						<tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
							{(filtered ?? []).map((t) => {
								const piiN = piiColumnCount(t);
								const ownerName =
									t.owners && t.owners.length > 0
										? (t.owners[0].displayName ?? t.owners[0].name)
										: null;
								return (
									<tr
										key={t.id}
										className="group transition-colors hover:bg-[#76b900]/5"
									>
										<td className="px-4 py-3">
											<Link
												href={`/catalog/tables/${encodeURIComponent(t.fullyQualifiedName)}`}
												className="block"
											>
												<span className="font-medium text-zinc-900 group-hover:text-[#76b900] dark:text-zinc-100">
													{t.name}
												</span>
												<span className="block font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
													{t.database.name} · {t.databaseSchema.name}
												</span>
											</Link>
										</td>
										<td className="px-4 py-3">
											<OwnerChip name={ownerName} />
										</td>
										<td className="px-4 py-3">
											<CertificationBadge status={readinessOf(t)} dense />
										</td>
										<td className="px-4 py-3">
											<UsageMeter level={usageLevelOf(t.usageSummary)} />
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
													{(t.tags ?? []).length > 3 ? (
														<span className="text-[11px] text-zinc-400">
															+{(t.tags ?? []).length - 3}
														</span>
													) : null}
												</div>
											)}
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
