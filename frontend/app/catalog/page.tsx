// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { om } from '@/api/openmetadata';
import type { OmQuery, OmTable } from '@/types/openmetadata';
import { TagPill } from '@/components/catalog/TagPill';

type Stats = {
	tables: number;
	columns: number;
	piiColumns: number;
	piiTables: number;
	queries: number;
};

const fqnSchemaPath = (t: OmTable): string => `${t.database.name} · ${t.databaseSchema.name}`;

export default function CatalogOverviewPage() {
	const [stats, setStats] = useState<Stats | null>(null);
	const [piiTables, setPiiTables] = useState<OmTable[]>([]);
	const [recentQueries, setRecentQueries] = useState<OmQuery[]>([]);
	const [err, setErr] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				const [tablesRes, queriesRes] = await Promise.all([
					om.tables.list({ fields: 'columns,tags', limit: 200 }),
					om.queries.list({ limit: 50 }),
				]);
				if (cancelled) return;

				let columns = 0;
				let piiColumns = 0;
				const piiByTable: OmTable[] = [];
				for (const t of tablesRes.data) {
					columns += t.columns?.length ?? 0;
					const tablePii = (t.columns ?? []).filter((c) =>
						(c.tags ?? []).some((tag) => tag.tagFQN.startsWith('PII.')),
					);
					piiColumns += tablePii.length;
					if (tablePii.length > 0) piiByTable.push(t);
				}

				setStats({
					tables: tablesRes.data.length,
					columns,
					piiColumns,
					piiTables: piiByTable.length,
					queries: queriesRes.data.length,
				});
				setPiiTables(piiByTable.slice(0, 8));
				setRecentQueries(queriesRes.data.slice(0, 5));
			} catch (e) {
				if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
			}
		})();
		return () => {
			cancelled = true;
		};
	}, []);

	return (
		<div className="mx-auto w-full max-w-7xl px-6 py-8">
			<header className="mb-8">
				<p className="text-xs font-medium uppercase tracking-wider text-[#76b900]">
					Generative Semantic Fabric · Catalog
				</p>
				<h1 className="mt-1 text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
					Your data, organised and AI-ready.
				</h1>
				<p className="mt-2 max-w-2xl text-sm text-zinc-600 dark:text-zinc-400">
					Tables, columns, descriptions, tags, PII, lineage, and query history — all
					harvested from your warehouses by OpenMetadata and surfaced here in one place.
				</p>
			</header>

			{err ? (
				<div className="mb-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
					Couldn&apos;t reach OpenMetadata: {err}
				</div>
			) : null}

			<section className="grid grid-cols-2 gap-3 md:grid-cols-5">
				<StatCard label="Tables" value={stats?.tables} href="/catalog/tables" />
				<StatCard label="Columns" value={stats?.columns} />
				<StatCard
					label="PII columns"
					value={stats?.piiColumns}
					accent="amber"
					href="/catalog/tables?pii=1"
				/>
				<StatCard label="Tables with PII" value={stats?.piiTables} accent="red" />
				<StatCard label="Tracked queries" value={stats?.queries} href="/catalog/queries" />
			</section>

			<section className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-2">
				<div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
					<div className="mb-3 flex items-center justify-between">
						<h2 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
							Tables with PII
						</h2>
						<Link
							href="/catalog/tables?pii=1"
							className="text-xs text-[#76b900] hover:underline"
						>
							View all →
						</Link>
					</div>
					{piiTables.length === 0 ? (
						<EmptyState label="No tables with PII tags yet. Run the Auto Classification workflow." />
					) : (
						<ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
							{piiTables.map((t) => {
								const piiCols = (t.columns ?? []).filter((c) =>
									(c.tags ?? []).some((tg) => tg.tagFQN.startsWith('PII.')),
								);
								return (
									<li key={t.id} className="py-2.5">
										<Link
											href={`/catalog/tables/${encodeURIComponent(t.fullyQualifiedName)}`}
											className="group block"
										>
											<div className="flex items-center justify-between">
												<div className="min-w-0">
													<p className="truncate text-sm font-medium text-zinc-900 group-hover:text-[#76b900] dark:text-zinc-100">
														{t.name}
													</p>
													<p className="truncate text-xs text-zinc-500 dark:text-zinc-400">
														{fqnSchemaPath(t)}
													</p>
												</div>
												<span className="ml-3 shrink-0 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-900 dark:bg-amber-950/60 dark:text-amber-200">
													{piiCols.length} PII
												</span>
											</div>
											<div className="mt-2 flex flex-wrap gap-1">
												{piiCols.slice(0, 6).map((c) => (
													<span
														key={c.name}
														className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[11px] text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
													>
														{c.name}
													</span>
												))}
												{piiCols.length > 6 ? (
													<span className="text-[11px] text-zinc-400">
														+{piiCols.length - 6}
													</span>
												) : null}
											</div>
										</Link>
									</li>
								);
							})}
						</ul>
					)}
				</div>

				<div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
					<div className="mb-3 flex items-center justify-between">
						<h2 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
							Recent queries
						</h2>
						<Link
							href="/catalog/queries"
							className="text-xs text-[#76b900] hover:underline"
						>
							View all →
						</Link>
					</div>
					{recentQueries.length === 0 ? (
						<EmptyState label="No queries ingested yet. Run the usage workflow." />
					) : (
						<ul className="space-y-2">
							{recentQueries.map((q) => (
								<li
									key={q.id}
									className="rounded-lg border border-zinc-100 bg-zinc-50/60 p-3 dark:border-zinc-800 dark:bg-zinc-950/40"
								>
									<pre className="line-clamp-2 whitespace-pre-wrap break-all font-mono text-[11px] leading-relaxed text-zinc-700 dark:text-zinc-300">
										{q.query}
									</pre>
									{q.queryUsedIn && q.queryUsedIn.length > 0 ? (
										<div className="mt-1.5 flex flex-wrap gap-1">
											{q.queryUsedIn.slice(0, 3).map((ref) => (
												<Link
													key={ref.id}
													href={`/catalog/tables/${encodeURIComponent(ref.fullyQualifiedName ?? '')}`}
													className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-700 hover:text-[#76b900] dark:bg-zinc-800 dark:text-zinc-300"
												>
													{ref.name}
												</Link>
											))}
										</div>
									) : null}
								</li>
							))}
						</ul>
					)}
				</div>
			</section>

			<section className="mt-10">
				<h2 className="mb-3 text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					What you can do here
				</h2>
				<div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
					<FeatureCard
						label="Data Dictionary"
						desc="Browse every table and column with descriptions, tags, and types."
						href="/catalog/tables"
					/>
					<FeatureCard
						label="Tags"
						desc="Create custom classifications and apply PII labels to columns."
						href="/catalog/tags"
					/>
					<FeatureCard
						label="Auto PII"
						desc="Surface columns the engine flagged as Personally Identifiable Information."
						href="/catalog/tables?pii=1"
						pill={
							<TagPill
								label={{
									tagFQN: 'PII.Sensitive',
									source: 'Classification',
									state: 'Confirmed',
									labelType: 'Generated',
								}}
								dense
							/>
						}
					/>
					<FeatureCard
						label="Query Tab"
						desc="See what queries hit each table, who ran them, and how often."
						href="/catalog/queries"
					/>
				</div>
			</section>
		</div>
	);
}

const StatCard = ({
	label,
	value,
	href,
	accent,
}: {
	label: string;
	value: number | undefined;
	href?: string;
	accent?: 'amber' | 'red';
}) => {
	const accentRing =
		accent === 'amber'
			? 'ring-amber-200 dark:ring-amber-900'
			: accent === 'red'
				? 'ring-red-200 dark:ring-red-900'
				: 'ring-zinc-200 dark:ring-zinc-800';
	const inner = (
		<div
			className={`rounded-xl bg-white p-4 ring-1 ${accentRing} dark:bg-zinc-900 ${
				href ? 'transition-shadow hover:shadow-md' : ''
			}`}
		>
			<p className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
				{label}
			</p>
			<p className="mt-1 text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
				{value == null ? '—' : value.toLocaleString()}
			</p>
		</div>
	);
	return href ? <Link href={href}>{inner}</Link> : inner;
};

const EmptyState = ({ label }: { label: string }) => (
	<p className="rounded-lg border border-dashed border-zinc-200 p-6 text-center text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-500">
		{label}
	</p>
);

const FeatureCard = ({
	label,
	desc,
	href,
	pill,
}: {
	label: string;
	desc: string;
	href: string;
	pill?: React.ReactNode;
}) => (
	<Link
		href={href}
		className="group rounded-xl border border-zinc-200 bg-white p-4 transition-colors hover:border-[#76b900]/60 hover:bg-[#76b900]/5 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-[#76b900]/60 dark:hover:bg-[#76b900]/5"
	>
		<div className="flex items-start justify-between">
			<p className="text-sm font-semibold text-zinc-900 group-hover:text-[#76b900] dark:text-zinc-100">
				{label}
			</p>
			{pill}
		</div>
		<p className="mt-1.5 text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">{desc}</p>
	</Link>
);
