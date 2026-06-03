// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { om } from '@/api/openmetadata';
import { explainQuery } from '@/api/queries';
import { Icon, IconName } from '@/components/icons/Icon';
import type { OmQuery } from '@/types/openmetadata';

type Sort = 'recent' | 'slowest';

type Translation = { status: 'loading' | 'done' | 'error'; text?: string };

const wordCount = (s: string) => s.split(/\s+/).filter(Boolean).length;

export default function QueriesPage() {
	const [queries, setQueries] = useState<OmQuery[] | null>(null);
	const [err, setErr] = useState<string | null>(null);
	const [sort, setSort] = useState<Sort>('recent');
	const [expanded, setExpanded] = useState<Record<string, boolean>>({});
	const [translations, setTranslations] = useState<Record<string, Translation>>({});
	const [copiedId, setCopiedId] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				const r = await om.queries.list({ limit: 100 });
				if (cancelled) return;
				setQueries(r.data);
				// Seed translations already persisted as the query's description so
				// they always show and are never re-processed.
				const seed: Record<string, Translation> = {};
				for (const q of r.data) {
					const saved = (q.description ?? '').trim();
					if (saved) seed[q.id] = { status: 'done', text: saved };
				}
				setTranslations(seed);
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
			out.sort((a, b) => (b.duration ?? 0) - (a.duration ?? 0));
		}
		return out;
	}, [queries, sort]);

	const copySql = async (q: OmQuery) => {
		try {
			await navigator.clipboard.writeText(q.query);
			setCopiedId(q.id);
			window.setTimeout(() => {
				setCopiedId((current) => (current === q.id ? null : current));
			}, 2000);
		} catch {
			/* clipboard unavailable (e.g. insecure context) */
		}
	};

	// Translate a single query on demand and persist the result on the
	// OpenMetadata query's description so it survives reloads.
	const translateOne = async (q: OmQuery) => {
		setTranslations((prev) => ({ ...prev, [q.id]: { status: 'loading' } }));
		const res = await explainQuery(q.query);
		if (res.error || !res.explanation) {
			setTranslations((prev) => ({ ...prev, [q.id]: { status: 'error' } }));
			return;
		}
		const explanation = res.explanation;
		setTranslations((prev) => ({ ...prev, [q.id]: { status: 'done', text: explanation } }));
		// Best-effort: a failed save still shows the translation this session.
		void om.queries
			.patch(q.id, [{ op: 'add', path: '/description', value: explanation }])
			.catch(() => {});
	};

	return (
		<div className="mx-auto w-full max-w-7xl px-6 py-8">
			<header className="mb-6 flex flex-wrap items-end justify-between gap-4">
				<div>
					<h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
						Query Tab
					</h1>
					<p className="mt-1 max-w-2xl text-sm text-zinc-600 dark:text-zinc-400">
						Every query observed in your warehouse, with the tables it touched and a
						plain-English summary.
					</p>
				</div>
				<div className="flex items-center gap-1 rounded-lg border border-zinc-300 bg-white p-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900">
					{(
						[
							['recent', 'Most recent'],
							['slowest', 'Slowest'],
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
					No queries tracked yet. Run usage ingestion against your warehouse.
				</p>
			) : (
				<ul className="space-y-3">
					{sorted.map((q) => {
						const isExpanded = expanded[q.id] === true;
						const tables = q.queryUsedIn ?? [];
						const wc = wordCount(q.query);
						const t = translations[q.id];
						return (
							<li
								key={q.id}
								className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
							>
								<div className="flex flex-wrap items-start justify-between gap-3">
									<div className="min-w-0 flex-1">
										{t?.status === 'done' || t?.status === 'loading' ? (
											<div className="mb-3 rounded-lg border-l-2 border-[#76b900] bg-[#76b900]/5 px-3 py-2">
												<div className="mb-1 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#76b900]">
													<span className="rounded bg-[#76b900]/10 px-1 py-0.5">
														AI
													</span>
													Summary
												</div>
												{t.status === 'done' ? (
													<p className="text-sm leading-relaxed text-zinc-700 dark:text-zinc-200">
														{t.text}
													</p>
												) : (
													<p className="animate-pulse text-sm text-zinc-400 dark:text-zinc-500">
														Translating…
													</p>
												)}
											</div>
										) : null}
										<div className="overflow-hidden rounded-lg ring-1 ring-zinc-200 dark:ring-zinc-800">
											<div className="flex items-center justify-between border-b border-zinc-200 bg-zinc-50 px-3 py-1 dark:border-zinc-800 dark:bg-zinc-950/50">
												<span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
													SQL
												</span>
												<button
													type="button"
													onClick={() => void copySql(q)}
													aria-label="Copy SQL to clipboard"
													className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium text-zinc-500 transition-colors hover:bg-zinc-200/60 hover:text-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
												>
													<Icon
														name={
															copiedId === q.id
																? IconName.Check
																: IconName.Copy
														}
														className={`h-3.5 w-3.5 ${copiedId === q.id ? 'text-[#76b900]' : ''}`}
													/>
													{copiedId === q.id
														? 'Copied to clipboard'
														: 'Copy'}
												</button>
											</div>
											<pre
												className={`overflow-x-auto whitespace-pre-wrap break-all bg-zinc-50/50 px-3 py-2 font-mono text-xs leading-relaxed text-zinc-800 dark:bg-zinc-950/30 dark:text-zinc-200 ${
													isExpanded ? '' : 'line-clamp-3'
												}`}
											>
												{q.query}
											</pre>
										</div>
										<div className="mt-1.5 flex flex-wrap items-center gap-3">
											{wc > 30 ? (
												<button
													type="button"
													onClick={() =>
														setExpanded((p) => ({
															...p,
															[q.id]: !isExpanded,
														}))
													}
													className="text-[11px] font-medium text-[#76b900] hover:underline"
												>
													{isExpanded ? 'Collapse' : 'Show full SQL'}
												</button>
											) : null}
											{t?.status !== 'loading' ? (
												<button
													type="button"
													onClick={() => void translateOne(q)}
													className="inline-flex items-center gap-1 rounded-md border border-zinc-300 px-2 py-1 text-[11px] font-medium text-zinc-600 transition-colors hover:border-[#76b900] hover:text-[#76b900] dark:border-zinc-700 dark:text-zinc-300"
												>
													<span className="rounded bg-[#76b900]/10 px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-[#76b900]">
														AI
													</span>
													{t?.status === 'done'
														? 'Retranslate'
														: t?.status === 'error'
															? 'Retry translation'
															: 'Translate with AI'}
												</button>
											) : null}
											{t?.status === 'error' ? (
												<span className="text-[11px] text-zinc-400 dark:text-zinc-500">
													Translation unavailable.
												</span>
											) : null}
										</div>
									</div>

									<aside className="flex flex-col items-end gap-1 text-[11px] text-zinc-500 dark:text-zinc-500">
										<span className="font-mono">
											{q.queryDate
												? new Date(q.queryDate).toLocaleString()
												: 'unknown'}
										</span>
										{q.duration != null ? (
											<span className="font-mono tabular-nums">
												{Math.round(q.duration)} ms
											</span>
										) : null}
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
