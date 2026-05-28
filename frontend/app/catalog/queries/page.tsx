// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { om } from '@/api/openmetadata';
import type { OmQuery } from '@/types/openmetadata';

type Sort = 'recent' | 'most-used';

const wordCount = (s: string) => s.split(/\s+/).filter(Boolean).length;

const summarise = (sql: string): string => {
	const cleaned = sql.replace(/\s+/g, ' ').trim();
	const m = cleaned.match(/^(SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|WITH)\b/i);
	const verb = m ? m[1].toUpperCase() : 'QUERY';
	const fromMatch = cleaned.match(/\bFROM\s+([^\s,;()]+)/i);
	const target = fromMatch ? fromMatch[1].replace(/"/g, '') : null;
	const cols = cleaned.match(/SELECT\s+(.+?)\s+FROM/i)?.[1] ?? '';
	const colN = cols ? cols.split(',').length : 0;
	const tail = target ? ` from ${target}` : '';
	const colDesc = colN > 0 ? ` (${colN} column${colN === 1 ? '' : 's'})` : '';
	return `${verb}${tail}${colDesc}`;
};

export default function QueriesPage() {
	const [queries, setQueries] = useState<OmQuery[] | null>(null);
	const [err, setErr] = useState<string | null>(null);
	const [sort, setSort] = useState<Sort>('recent');
	const [expanded, setExpanded] = useState<Record<string, boolean>>({});
	const [translate, setTranslate] = useState(false);

	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				const r = await om.queries.list({ limit: 100 });
				if (!cancelled) setQueries(r.data);
			} catch (e) {
				if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
			}
		})();
		return () => {
			cancelled = true;
		};
	}, []);

	const sorted = useMemo(() => {
		if (!queries) return null;
		const out = [...queries];
		if (sort === 'recent') {
			out.sort((a, b) => (b.queryDate ?? 0) - (a.queryDate ?? 0));
		} else {
			out.sort((a, b) => (b.queryUsedIn?.length ?? 0) - (a.queryUsedIn?.length ?? 0));
		}
		return out;
	}, [queries, sort]);

	return (
		<div className="mx-auto w-full max-w-7xl px-6 py-8">
			<header className="mb-6 flex flex-wrap items-end justify-between gap-4">
				<div>
					<h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
						Query Tab
					</h1>
					<p className="mt-1 max-w-2xl text-sm text-zinc-600 dark:text-zinc-400">
						Every query OpenMetadata observed in your warehouse, with the tables it
						touched and a plain-English summary.
					</p>
				</div>
				<div className="flex items-center gap-2">
					<label className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-300">
						<input
							type="checkbox"
							checked={translate}
							onChange={(e) => setTranslate(e.target.checked)}
							className="accent-[#76b900]"
						/>
						Translate with AI
					</label>
					<div className="flex items-center gap-1 rounded-lg border border-zinc-300 bg-white p-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900">
						{(
							[
								['recent', 'Most recent'],
								['most-used', 'Most used'],
							] as const
						).map(([id, label]) => (
							<button
								key={id}
								type="button"
								onClick={() => setSort(id)}
								className={`rounded px-2 py-1 font-medium ${
									sort === id
										? 'bg-[#76b900]/10 text-[#76b900]'
										: 'text-zinc-600 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800'
								}`}
							>
								{label}
							</button>
						))}
					</div>
				</div>
			</header>

			{err ? (
				<div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
					{err}
				</div>
			) : sorted == null ? (
				<div className="space-y-2">
					{Array.from({ length: 4 }).map((_, i) => (
						<div
							key={i}
							className="h-24 animate-pulse rounded-xl bg-zinc-100 dark:bg-zinc-900"
						/>
					))}
				</div>
			) : sorted.length === 0 ? (
				<p className="rounded-lg border border-dashed border-zinc-300 p-12 text-center text-sm text-zinc-500 dark:border-zinc-700">
					No queries tracked yet. Run the OpenMetadata usage ingestion against your
					Snowflake service.
				</p>
			) : (
				<ul className="space-y-3">
					{sorted.map((q) => {
						const isExpanded = expanded[q.id] === true;
						const tables = q.queryUsedIn ?? [];
						const wc = wordCount(q.query);
						return (
							<li
								key={q.id}
								className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
							>
								<div className="flex flex-wrap items-start justify-between gap-3">
									<div className="min-w-0 flex-1">
										{translate ? (
											<p className="mb-2 text-sm text-zinc-800 dark:text-zinc-200">
												<span className="mr-1.5 inline-block rounded bg-[#76b900]/10 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-[#76b900]">
													AI
												</span>
												{summarise(q.query)}
											</p>
										) : null}
										<pre
											className={`overflow-x-auto whitespace-pre-wrap break-all font-mono text-xs leading-relaxed text-zinc-800 dark:text-zinc-200 ${
												isExpanded ? '' : 'line-clamp-3'
											}`}
										>
											{q.query}
										</pre>
										{wc > 30 ? (
											<button
												type="button"
												onClick={() =>
													setExpanded((p) => ({
														...p,
														[q.id]: !isExpanded,
													}))
												}
												className="mt-1 text-[11px] font-medium text-[#76b900] hover:underline"
											>
												{isExpanded ? 'Collapse' : 'Show full SQL'}
											</button>
										) : null}
									</div>

									<aside className="flex flex-col items-end gap-1 text-[11px] text-zinc-500 dark:text-zinc-500">
										<span className="font-mono">
											{q.queryDate
												? new Date(q.queryDate).toLocaleString()
												: 'unknown'}
										</span>
										{q.users && q.users.length > 0 ? (
											<span>by {q.users.map((u) => u.name).join(', ')}</span>
										) : null}
									</aside>
								</div>

								{tables.length > 0 ? (
									<div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-zinc-100 pt-3 text-[11px] dark:border-zinc-800">
										<span className="text-zinc-500">Tables:</span>
										{tables.map((ref) => (
											<Link
												key={ref.id}
												href={`/catalog/tables/${encodeURIComponent(ref.fullyQualifiedName ?? '')}`}
												className="rounded-md bg-zinc-100 px-1.5 py-0.5 font-mono text-zinc-700 hover:bg-[#76b900]/10 hover:text-[#76b900] dark:bg-zinc-800 dark:text-zinc-300"
											>
												{ref.name}
											</Link>
										))}
									</div>
								) : null}
							</li>
						);
					})}
				</ul>
			)}
		</div>
	);
}
