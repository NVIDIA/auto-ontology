// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import {
	Badge,
	Button,
	CodeSnippet,
	SegmentedControl,
	Skeleton,
	Text,
} from '@kui/foundations-react';
import { om } from '@/api/openmetadata';
import { explainQuery } from '@/api/queries';
import type { OmQuery } from '@/types/openmetadata';

type Sort = 'recent' | 'slowest';

type Translation = { status: 'loading' | 'done' | 'error'; text?: string };

export default function QueriesPage() {
	const [queries, setQueries] = useState<OmQuery[] | null>(null);
	const [err, setErr] = useState<string | null>(null);
	const [sort, setSort] = useState<Sort>('recent');
	const [translations, setTranslations] = useState<Record<string, Translation>>({});

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
					<Text asChild kind="title/md">
						<h1 className="text-[var(--text-color-primary)]">Query Tab</h1>
					</Text>
					<p className="mt-1 max-w-2xl text-sm text-[var(--text-color-base)]">
						Every query observed in your warehouse, with the tables it touched and a
						plain-English summary.
					</p>
				</div>
				<SegmentedControl
					name="query-sort"
					size="small"
					value={sort}
					onValueChange={(v) => setSort(v as Sort)}
					items={[
						{ value: 'recent', children: 'Most recent' },
						{ value: 'slowest', children: 'Slowest' },
					]}
				/>
			</header>

			{err ? (
				<div className="rounded-[var(--radius-lg)] border border-[var(--border-color-feedback-danger)] bg-[var(--background-color-feedback-danger-subtle-hover)] p-4 text-sm text-[var(--text-color-feedback-danger-strong)]">
					{err}
				</div>
			) : sorted == null ? (
				<div className="space-y-3">
					{Array.from({ length: 4 }).map((_, i) => (
						<div
							key={i}
							className="space-y-3 rounded-[var(--radius-lg)] border border-[var(--border-color-base)] bg-[var(--background-color-surface-raised)] p-4"
						>
							<Skeleton kind="line" />
							<Skeleton kind="line" />
						</div>
					))}
				</div>
			) : sorted.length === 0 ? (
				<p className="rounded-[var(--radius-lg)] border border-dashed border-[var(--border-color-base)] p-12 text-center text-sm text-[var(--text-color-subtle)]">
					No queries tracked yet. Run usage ingestion against your warehouse.
				</p>
			) : (
				<ul className="space-y-3">
					{sorted.map((q) => {
						const tables = q.queryUsedIn ?? [];
						const t = translations[q.id];
						const translateLabel =
							t?.status === 'done'
								? 'Retranslate'
								: t?.status === 'error'
									? 'Retry translation'
									: 'Translate with AI';
						return (
							<li
								key={q.id}
								className="rounded-[var(--radius-lg)] border border-[var(--border-color-base)] bg-[var(--background-color-surface-raised)] p-4"
							>
								<div className="flex flex-wrap items-start justify-between gap-3">
									<div className="min-w-0 flex-1">
										{t?.status === 'done' || t?.status === 'loading' ? (
											<div className="mb-3 rounded-[var(--radius-md)] border-l-2 border-[var(--color-brand)] bg-[var(--background-color-accent-green-subtle)] px-3 py-2">
												<div className="mb-1 flex items-center gap-1.5">
													<Badge color="green" kind="solid">
														AI
													</Badge>
													<span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-color-accent-green)]">
														Summary
													</span>
												</div>
												{t.status === 'done' ? (
													<p className="text-sm leading-relaxed text-[var(--text-color-secondary)]">
														{t.text}
													</p>
												) : (
													<p className="animate-pulse text-sm text-[var(--text-color-subtle)]">
														Translating…
													</p>
												)}
											</div>
										) : null}
										<CodeSnippet
											kind="block"
											language="text"
											value={q.query}
											collapsible
											rows={3}
										/>
										<div className="mt-2 flex flex-wrap items-center gap-3">
											{t?.status !== 'loading' ? (
												<Button
													kind="secondary"
													size="tiny"
													onClick={() => void translateOne(q)}
												>
													<Badge color="green" kind="solid">
														AI
													</Badge>
													{translateLabel}
												</Button>
											) : null}
											{t?.status === 'error' ? (
												<span className="text-[11px] text-[var(--text-color-subtle)]">
													Translation unavailable.
												</span>
											) : null}
										</div>
									</div>

									<aside className="flex flex-col items-end gap-1 text-[11px] text-[var(--text-color-subtle)]">
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
									<div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-[var(--border-color-base)] pt-3 text-[11px]">
										<span className="text-[var(--text-color-subtle)]">
											Tables:
										</span>
										{tables.map((ref) => (
											<Link
												key={ref.id}
												href={`/catalog/tables/${encodeURIComponent(ref.fullyQualifiedName ?? '')}`}
												className="rounded-[var(--radius-md)] bg-[var(--background-color-accent-gray-subtle)] px-1.5 py-0.5 font-mono text-[var(--text-color-secondary)] hover:text-[var(--text-color-brand)]"
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
