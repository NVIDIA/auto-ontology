// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { use, useCallback, useEffect, useMemo, useState } from 'react';
import { om } from '@/api/openmetadata';
import type { OmQuery, OmTable } from '@/types/openmetadata';
import { TagPill, TagsRow } from '@/components/catalog/TagPill';
import {
	CertificationBadge,
	DocIcon,
	GridIcon,
	LinkIcon,
	MetaField,
	OwnerChip,
	Panel,
	StatsIcon,
	UsageMeter,
	readinessOf,
	usageLevelOf,
} from '@/components/catalog/ui';

type PageProps = { params: Promise<{ fqn: string }> };

export default function TableDetailPage(props: PageProps) {
	const { fqn: rawFqn } = use(props.params);
	const fqn = decodeURIComponent(rawFqn);

	const [table, setTable] = useState<OmTable | null>(null);
	const [queries, setQueries] = useState<OmQuery[] | null>(null);
	const [err, setErr] = useState<string | null>(null);
	const [busy, setBusy] = useState(false);

	const reload = useCallback(async () => {
		try {
			const t = await om.tables.getByFqn(fqn);
			setTable(t);
			try {
				const qs = await om.tables.queriesForTable(t.id, 20);
				setQueries(qs.data);
			} catch {
				setQueries([]);
			}
		} catch (e) {
			setErr(e instanceof Error ? e.message : String(e));
		}
	}, [fqn]);

	useEffect(() => {
		void reload();
	}, [reload]);

	const confirmTag = useCallback(
		async (colIdx: number, tagIdx: number) => {
			if (!table) return;
			setBusy(true);
			try {
				await om.tables.patch(table.id, [
					{
						op: 'replace',
						path: `/columns/${colIdx}/tags/${tagIdx}/state`,
						value: 'Confirmed',
					},
				]);
				await reload();
			} catch (e) {
				setErr(e instanceof Error ? e.message : String(e));
			} finally {
				setBusy(false);
			}
		},
		[table, reload],
	);

	const rejectTag = useCallback(
		async (colIdx: number, tagIdx: number) => {
			if (!table) return;
			setBusy(true);
			try {
				await om.tables.patch(table.id, [
					{ op: 'remove', path: `/columns/${colIdx}/tags/${tagIdx}` },
				]);
				await reload();
			} catch (e) {
				setErr(e instanceof Error ? e.message : String(e));
			} finally {
				setBusy(false);
			}
		},
		[table, reload],
	);

	const related = useMemo(() => {
		if (!queries) return [];
		const counts = new Map<string, { name: string; fqn: string; n: number }>();
		for (const q of queries) {
			for (const ref of q.queryUsedIn ?? []) {
				if (!ref.fullyQualifiedName || ref.fullyQualifiedName === fqn) continue;
				const prev = counts.get(ref.fullyQualifiedName);
				counts.set(ref.fullyQualifiedName, {
					name: ref.name,
					fqn: ref.fullyQualifiedName,
					n: (prev?.n ?? 0) + 1,
				});
			}
		}
		return [...counts.values()].sort((a, b) => b.n - a.n).slice(0, 8);
	}, [queries, fqn]);

	if (err) {
		return (
			<div className="mx-auto max-w-3xl px-6 py-12">
				<div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
					{err}
				</div>
			</div>
		);
	}
	if (!table) {
		return (
			<div className="mx-auto max-w-6xl px-6 py-12">
				<div className="h-8 w-72 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
				<div className="mt-2 h-4 w-96 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
				<div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr_240px]">
					<div className="h-72 animate-pulse rounded-xl bg-zinc-100 dark:bg-zinc-900" />
					<div className="h-72 animate-pulse rounded-xl bg-zinc-100 dark:bg-zinc-900" />
					<div className="h-72 animate-pulse rounded-xl bg-zinc-100 dark:bg-zinc-900" />
				</div>
			</div>
		);
	}

	const columns = table.columns ?? [];
	const piiCols = columns.filter((c) => (c.tags ?? []).some((t) => t.tagFQN.startsWith('PII.')));
	const documentedCols = columns.filter((c) => c.description && c.description.trim().length > 0);
	const status = readinessOf(table);
	const usage = usageLevelOf(table.usageSummary);
	const ownerName =
		table.owners && table.owners.length > 0
			? (table.owners[0].displayName ?? table.owners[0].name)
			: null;
	const monthlyHits = table.usageSummary?.monthlyStats?.count;
	const weeklyHits = table.usageSummary?.weeklyStats?.count;

	return (
		<div className="mx-auto w-full max-w-6xl px-6 py-8">
			<nav className="mb-2 flex flex-wrap items-center gap-1 text-xs text-zinc-500 dark:text-zinc-400">
				<Link href="/catalog/tables" className="hover:text-[#76b900]">
					Data Dictionary
				</Link>
				<Chevron />
				<span className="font-mono">{table.database.name}</span>
				<Chevron />
				<span className="font-mono">{table.databaseSchema.name}</span>
				<Chevron />
				<span className="font-mono font-semibold text-zinc-700 dark:text-zinc-200">
					{table.name}
				</span>
			</nav>

			<header className="mb-6 flex flex-wrap items-start justify-between gap-4">
				<div className="min-w-0">
					<div className="flex flex-wrap items-center gap-3">
						<h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
							<GridIcon className="h-5 w-5 text-zinc-400" />
							{table.displayName ?? table.name}
						</h1>
						<CertificationBadge status={status} />
						{piiCols.length > 0 ? (
							<span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-900 ring-1 ring-amber-300 dark:bg-amber-950/50 dark:text-amber-200 dark:ring-amber-800">
								<ShieldIcon />
								Contains PII
							</span>
						) : null}
					</div>
				</div>
				<div className="flex items-center gap-2">
					<button
						type="button"
						className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
					>
						<DocIcon className="h-4 w-4" />
						Export
					</button>
				</div>
			</header>

			<div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_minmax(0,1fr)_240px]">
				{/* ---------------------------------------------------------- left rail */}
				<aside className="flex flex-col gap-4">
					<Panel title="Table owner" icon={<UserIcon />}>
						<div className="px-4 py-3">
							<OwnerChip name={ownerName} />
						</div>
					</Panel>

					<Panel title="View in Exploration" icon={<LinkIcon className="h-3.5 w-3.5" />}>
						<Link
							href={`/catalog/lineage?fqn=${encodeURIComponent(table.fullyQualifiedName)}`}
							className="group block px-4 py-3"
						>
							<ExplorationThumb />
							<span className="mt-2 block text-xs font-medium text-[#76b900] group-hover:underline">
								Open lineage graph →
							</span>
						</Link>
					</Panel>

					<Panel title="Details">
						<div className="divide-y divide-zinc-100 dark:divide-zinc-800">
							<MetaField label="Type" mono>
								{table.tableType ?? '—'}
							</MetaField>
							<MetaField label="Database" mono>
								{table.database.name}
							</MetaField>
							<MetaField label="Schema" mono>
								{table.databaseSchema.name}
							</MetaField>
							<MetaField label="Updated">
								{table.updatedAt
									? new Date(table.updatedAt).toLocaleDateString()
									: '—'}
							</MetaField>
						</div>
					</Panel>

					<Panel title="Tags" action={<span className="text-zinc-300">＋</span>}>
						<div className="px-4 py-3">
							<TagsRow tags={table.tags ?? []} />
						</div>
					</Panel>
				</aside>

				{/* -------------------------------------------------------- center pane */}
				<div className="flex min-w-0 flex-col gap-6">
					<Panel title="Description" icon={<DocIcon className="h-3.5 w-3.5" />}>
						<p className="px-4 py-4 text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
							{table.description || (
								<em className="text-zinc-400">No description provided yet.</em>
							)}
						</p>
					</Panel>

					<Panel title="Certification" icon={<CheckBadgeIcon />}>
						<div className="grid grid-cols-3 divide-x divide-zinc-100 dark:divide-zinc-800">
							<CertCell
								label="Description"
								icon={<DocIcon className="h-3.5 w-3.5" />}
								ok={Boolean(table.description)}
							/>
							<CertCell label="Owner" icon={<UserIcon />} ok={Boolean(ownerName)} />
							<CertCell
								label="Tags"
								icon={<TagIcon />}
								ok={(table.tags ?? []).length > 0}
							/>
						</div>
					</Panel>

					<div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
						<Panel title="Usage" icon={<StatsIcon className="h-3.5 w-3.5" />}>
							<div className="grid grid-cols-2 divide-x divide-zinc-100 dark:divide-zinc-800">
								<div className="px-4 py-3">
									<p className="text-xs text-zinc-500">Popularity</p>
									<div className="mt-1">
										<UsageMeter level={usage} />
									</div>
								</div>
								<div className="px-4 py-3">
									<p className="text-xs text-zinc-500">Queries</p>
									<p className="mt-0.5 text-lg font-semibold tabular-nums text-zinc-900 dark:text-zinc-100">
										{queries == null ? '—' : queries.length}
									</p>
								</div>
							</div>
							{monthlyHits != null || weeklyHits != null ? (
								<div className="grid grid-cols-2 divide-x divide-zinc-100 border-t border-zinc-100 dark:divide-zinc-800 dark:border-zinc-800">
									<div className="px-4 py-3">
										<p className="text-xs text-zinc-500">Last 7 days</p>
										<p className="mt-0.5 text-lg font-semibold tabular-nums text-zinc-900 dark:text-zinc-100">
											{weeklyHits?.toLocaleString() ?? '—'}
										</p>
									</div>
									<div className="px-4 py-3">
										<p className="text-xs text-zinc-500">Last 30 days</p>
										<p className="mt-0.5 text-lg font-semibold tabular-nums text-zinc-900 dark:text-zinc-100">
											{monthlyHits?.toLocaleString() ?? '—'}
										</p>
									</div>
								</div>
							) : null}
						</Panel>

						<Panel title="Entities" icon={<GridIcon className="h-3.5 w-3.5" />}>
							<div className="grid grid-cols-2 divide-x divide-y divide-zinc-100 dark:divide-zinc-800">
								<EntityCell label="Columns" value={columns.length} />
								<EntityCell
									label="Documented"
									value={`${documentedCols.length}/${columns.length}`}
								/>
								<EntityCell label="PII columns" value={piiCols.length} />
								<EntityCell label="Tags" value={(table.tags ?? []).length} />
							</div>
						</Panel>
					</div>

					<Panel
						title="Columns"
						icon={<GridIcon className="h-3.5 w-3.5" />}
						action={
							<span className="text-xs font-normal normal-case tracking-normal text-zinc-400">
								{columns.length} total
							</span>
						}
					>
						<div className="overflow-x-auto">
							<table className="w-full text-sm">
								<thead className="border-b border-zinc-100 bg-zinc-50 text-left text-xs font-medium uppercase tracking-wider text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950/40 dark:text-zinc-400">
									<tr>
										<th className="w-12 px-3 py-2.5 text-right">#</th>
										<th className="px-3 py-2.5">Column</th>
										<th className="px-3 py-2.5">Type</th>
										<th className="px-3 py-2.5">Tags</th>
										<th className="px-3 py-2.5">Description</th>
									</tr>
								</thead>
								<tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
									{columns.map((c, colIdx) => {
										const suggestedPii = (c.tags ?? [])
											.map((t, tagIdx) => ({ t, tagIdx }))
											.filter((x) => x.t.state === 'Suggested');
										return (
											<tr key={c.name}>
												<td className="px-3 py-3 text-right font-mono text-xs text-zinc-400">
													{c.ordinalPosition ?? colIdx + 1}
												</td>
												<td className="px-3 py-3 font-mono text-sm font-medium text-zinc-900 dark:text-zinc-100">
													{c.name}
												</td>
												<td className="px-3 py-3 font-mono text-xs text-zinc-600 dark:text-zinc-400">
													{c.dataType}
												</td>
												<td className="px-3 py-3">
													<div className="flex flex-col gap-1.5">
														<TagsRow tags={c.tags ?? []} />
														{suggestedPii.length > 0 ? (
															<div className="flex flex-wrap items-center gap-1.5 text-[11px]">
																<span className="text-amber-700 dark:text-amber-300">
																	Auto-detected — review:
																</span>
																{suggestedPii.map(
																	({ t, tagIdx }) => (
																		<span
																			key={tagIdx}
																			className="inline-flex items-center gap-1"
																		>
																			<button
																				type="button"
																				disabled={busy}
																				onClick={() =>
																					void confirmTag(
																						colIdx,
																						tagIdx,
																					)
																				}
																				className="rounded bg-emerald-600 px-1.5 py-0.5 text-[10px] font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
																				title={`Accept ${t.tagFQN}`}
																			>
																				Accept
																			</button>
																			<button
																				type="button"
																				disabled={busy}
																				onClick={() =>
																					void rejectTag(
																						colIdx,
																						tagIdx,
																					)
																				}
																				className="rounded bg-zinc-200 px-1.5 py-0.5 text-[10px] font-medium text-zinc-700 hover:bg-zinc-300 disabled:opacity-50 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
																				title={`Reject ${t.tagFQN}`}
																			>
																				Reject
																			</button>
																		</span>
																	),
																)}
															</div>
														) : null}
													</div>
												</td>
												<td className="px-3 py-3 text-xs text-zinc-600 dark:text-zinc-400">
													<span className="line-clamp-2 max-w-md">
														{c.description ?? (
															<em className="text-zinc-400">—</em>
														)}
													</span>
												</td>
											</tr>
										);
									})}
								</tbody>
							</table>
						</div>
						<p className="border-t border-zinc-100 px-4 py-2.5 text-[11px] text-zinc-500 dark:border-zinc-800 dark:text-zinc-500">
							Tags with the <span aria-hidden>✦</span> icon were auto-generated by
							Auto-Classification. &quot;Suggested&quot; tags await review — accept or
							reject to confirm.
						</p>
					</Panel>

					<Panel
						title="Recent queries"
						icon={<StatsIcon className="h-3.5 w-3.5" />}
						action={
							<Link
								href="/catalog/queries"
								className="text-xs font-normal normal-case tracking-normal text-[#76b900] hover:underline"
							>
								View all →
							</Link>
						}
					>
						{queries == null ? (
							<div className="m-4 h-20 animate-pulse rounded-lg bg-zinc-100 dark:bg-zinc-800" />
						) : queries.length === 0 ? (
							<p className="px-4 py-6 text-center text-sm text-zinc-500">
								No tracked queries yet — run the usage ingestion against this
								source.
							</p>
						) : (
							<ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
								{queries.slice(0, 6).map((q) => (
									<li key={q.id} className="px-4 py-3">
										<div className="mb-1 flex flex-wrap items-center gap-3 text-[11px] text-zinc-500 dark:text-zinc-500">
											<span className="font-mono">
												{q.queryDate
													? new Date(q.queryDate).toLocaleDateString()
													: 'unknown date'}
											</span>
											{q.users && q.users.length > 0 ? (
												<span>
													by {q.users.map((u) => u.name).join(', ')}
												</span>
											) : null}
										</div>
										<pre className="overflow-x-auto whitespace-pre-wrap break-all font-mono text-xs leading-relaxed text-zinc-800 dark:text-zinc-200">
											{q.query}
										</pre>
									</li>
								))}
							</ul>
						)}
					</Panel>
				</div>

				{/* ---------------------------------------------------------- right rail */}
				<aside className="flex flex-col gap-4">
					<Panel title="Related tables" icon={<LinkIcon className="h-3.5 w-3.5" />}>
						{related.length === 0 ? (
							<p className="px-4 py-3 text-xs text-zinc-500">
								No co-queried tables found.
							</p>
						) : (
							<ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
								{related.map((r) => (
									<li key={r.fqn}>
										<Link
											href={`/catalog/tables/${encodeURIComponent(r.fqn)}`}
											className="flex items-center justify-between gap-2 px-4 py-2.5 hover:bg-[#76b900]/5"
										>
											<span className="truncate font-mono text-xs text-zinc-700 dark:text-zinc-300">
												{r.name}
											</span>
											<span className="shrink-0 rounded-full bg-zinc-100 px-1.5 py-0.5 text-[10px] tabular-nums text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
												{r.n}×
											</span>
										</Link>
									</li>
								))}
							</ul>
						)}
					</Panel>

					{piiCols.length > 0 ? (
						<Panel title="PII columns" icon={<ShieldIcon />}>
							<ul className="flex flex-wrap gap-1.5 px-4 py-3">
								{piiCols.map((c) => (
									<li key={c.name}>
										<span className="rounded bg-amber-50 px-1.5 py-0.5 font-mono text-[11px] text-amber-900 ring-1 ring-amber-200 dark:bg-amber-950/40 dark:text-amber-200 dark:ring-amber-900">
											{c.name}
										</span>
									</li>
								))}
							</ul>
						</Panel>
					) : null}

					<Panel title="Common filters" icon={<TagIcon />}>
						<div className="flex flex-wrap gap-1.5 px-4 py-3">
							{(table.tags ?? []).length === 0 ? (
								<span className="text-xs text-zinc-400">No tags to filter by.</span>
							) : (
								(table.tags ?? []).map((t) => (
									<Link
										key={t.tagFQN}
										href={`/catalog/tables?q=${encodeURIComponent(t.tagFQN.split('.').slice(-1)[0])}`}
									>
										<TagPill label={t} dense />
									</Link>
								))
							)}
						</div>
					</Panel>
				</aside>
			</div>
		</div>
	);
}

const ExplorationThumb = () => (
	<div className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 bg-[radial-gradient(circle,_var(--color-zinc-200)_1px,_transparent_1px)] bg-[length:14px_14px] dark:border-zinc-800 dark:bg-zinc-950/40 dark:bg-[radial-gradient(circle,_var(--color-zinc-800)_1px,_transparent_1px)]">
		<svg viewBox="0 0 220 96" className="h-24 w-full" aria-hidden>
			<line
				x1="40"
				y1="48"
				x2="110"
				y2="28"
				className="stroke-zinc-300 dark:stroke-zinc-700"
			/>
			<line
				x1="40"
				y1="48"
				x2="110"
				y2="68"
				className="stroke-zinc-300 dark:stroke-zinc-700"
			/>
			<line
				x1="110"
				y1="48"
				x2="180"
				y2="30"
				className="stroke-zinc-300 dark:stroke-zinc-700"
			/>
			<line
				x1="110"
				y1="48"
				x2="180"
				y2="66"
				className="stroke-zinc-300 dark:stroke-zinc-700"
			/>
			<circle cx="40" cy="48" r="7" className="fill-violet-400" />
			<circle cx="110" cy="28" r="6" className="fill-amber-400" />
			<circle cx="110" cy="68" r="6" className="fill-zinc-400" />
			<circle cx="110" cy="48" r="9" className="fill-[#76b900]" />
			<circle cx="180" cy="30" r="5" className="fill-zinc-300 dark:fill-zinc-600" />
			<circle cx="180" cy="66" r="5" className="fill-zinc-300 dark:fill-zinc-600" />
			<circle cx="180" cy="48" r="5" className="fill-zinc-300 dark:fill-zinc-600" />
		</svg>
	</div>
);

const CertCell = ({ label, icon, ok }: { label: string; icon: React.ReactNode; ok: boolean }) => (
	<div className="flex items-center justify-between gap-2 px-4 py-3">
		<span className="flex items-center gap-1.5 text-sm text-zinc-700 dark:text-zinc-300">
			<span className="text-zinc-400">{icon}</span>
			{label}
		</span>
		{ok ? (
			<span
				className="flex h-4 w-4 items-center justify-center rounded-full bg-emerald-500 text-[10px] text-white"
				title="Present"
			>
				✓
			</span>
		) : (
			<span
				className="h-4 w-4 rounded-full border border-dashed border-zinc-300 dark:border-zinc-600"
				title="Missing"
				aria-hidden
			/>
		)}
	</div>
);

const EntityCell = ({ label, value }: { label: string; value: number | string }) => (
	<div className="px-4 py-3">
		<p className="text-xs text-zinc-500">{label}</p>
		<p className="mt-0.5 text-lg font-semibold tabular-nums text-zinc-900 dark:text-zinc-100">
			{value}
		</p>
	</div>
);

const Chevron = () => (
	<svg viewBox="0 0 20 20" fill="currentColor" className="h-3 w-3 text-zinc-400" aria-hidden>
		<path
			fillRule="evenodd"
			d="M7.21 14.77a.75.75 0 0 1 .02-1.06L11.168 10 7.23 6.29a.75.75 0 1 1 1.04-1.08l4.5 4.25a.75.75 0 0 1 0 1.08l-4.5 4.25a.75.75 0 0 1-1.06-.02z"
			clipRule="evenodd"
		/>
	</svg>
);

const ShieldIcon = () => (
	<svg viewBox="0 0 20 20" fill="currentColor" className="h-3 w-3" aria-hidden>
		<path
			fillRule="evenodd"
			d="M10 1.944A11.954 11.954 0 0 1 2.166 5C2.056 5.649 2 6.319 2 7c0 5.225 3.34 9.67 8 11.317C14.66 16.67 18 12.225 18 7c0-.682-.057-1.35-.166-2.001A11.954 11.954 0 0 1 10 1.944zM11 14a1 1 0 1 1-2 0 1 1 0 0 1 2 0zm0-7a1 1 0 1 0-2 0v3a1 1 0 1 0 2 0V7z"
			clipRule="evenodd"
		/>
	</svg>
);

const UserIcon = () => (
	<svg viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5" aria-hidden>
		<path
			fillRule="evenodd"
			d="M10 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm-7 9a7 7 0 1 1 14 0H3z"
			clipRule="evenodd"
		/>
	</svg>
);

const CheckBadgeIcon = () => (
	<svg viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5" aria-hidden>
		<path
			fillRule="evenodd"
			d="M16.403 12.652a3 3 0 0 0 0-5.304 3 3 0 0 0-3.751-3.751 3 3 0 0 0-5.304 0 3 3 0 0 0-3.751 3.751 3 3 0 0 0 0 5.304 3 3 0 0 0 3.751 3.751 3 3 0 0 0 5.304 0 3 3 0 0 0 3.751-3.751zm-2.546-4.46a.75.75 0 0 0-1.214-.883l-3.236 4.53-1.55-1.55a.75.75 0 0 0-1.06 1.061l2.171 2.172a.75.75 0 0 0 1.137-.089l3.752-5.252z"
			clipRule="evenodd"
		/>
	</svg>
);

const TagIcon = () => (
	<svg viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5" aria-hidden>
		<path
			fillRule="evenodd"
			d="M5.5 3A2.5 2.5 0 0 0 3 5.5v2.879a2.5 2.5 0 0 0 .732 1.767l6.5 6.5a2.5 2.5 0 0 0 3.536 0l2.878-2.878a2.5 2.5 0 0 0 0-3.536l-6.5-6.5A2.5 2.5 0 0 0 8.38 3H5.5zM6 7a1 1 0 1 0 0-2 1 1 0 0 0 0 2z"
			clipRule="evenodd"
		/>
	</svg>
);
