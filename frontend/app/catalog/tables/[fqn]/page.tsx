// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { use, useCallback, useEffect, useState } from 'react';
import { om } from '@/api/openmetadata';
import type { OmQuery, OmTable } from '@/types/openmetadata';
import { TagsRow } from '@/components/catalog/TagPill';

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
			<div className="mx-auto max-w-5xl px-6 py-12">
				<div className="h-8 w-72 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
				<div className="mt-2 h-4 w-96 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
				<div className="mt-8 h-72 animate-pulse rounded-xl bg-zinc-100 dark:bg-zinc-900" />
			</div>
		);
	}

	const piiCols = (table.columns ?? []).filter((c) =>
		(c.tags ?? []).some((t) => t.tagFQN.startsWith('PII.')),
	);

	return (
		<div className="mx-auto w-full max-w-7xl px-6 py-8">
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

			<header className="mb-6 flex flex-wrap items-start justify-between gap-4 border-b border-zinc-200 pb-6 dark:border-zinc-800">
				<div className="min-w-0">
					<div className="flex flex-wrap items-center gap-3">
						<h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
							{table.displayName ?? table.name}
						</h1>
						{piiCols.length > 0 ? (
							<span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-900 ring-1 ring-amber-300 dark:bg-amber-950/50 dark:text-amber-200 dark:ring-amber-800">
								<svg
									viewBox="0 0 20 20"
									fill="currentColor"
									className="h-3 w-3"
									aria-hidden
								>
									<path
										fillRule="evenodd"
										d="M10 1.944A11.954 11.954 0 0 1 2.166 5C2.056 5.649 2 6.319 2 7c0 5.225 3.34 9.67 8 11.317C14.66 16.67 18 12.225 18 7c0-.682-.057-1.35-.166-2.001A11.954 11.954 0 0 1 10 1.944zM11 14a1 1 0 1 1-2 0 1 1 0 0 1 2 0zm0-7a1 1 0 1 0-2 0v3a1 1 0 1 0 2 0V7z"
										clipRule="evenodd"
									/>
								</svg>
								Contains PII
							</span>
						) : null}
					</div>
					<p className="mt-1 max-w-3xl text-sm text-zinc-600 dark:text-zinc-400">
						{table.description || <em className="text-zinc-400">No description.</em>}
					</p>
					{(table.tags ?? []).length > 0 ? (
						<div className="mt-3">
							<TagsRow tags={table.tags ?? []} />
						</div>
					) : null}
				</div>

				<aside className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
					<MetaRow label="Type" value={table.tableType ?? '—'} mono />
					<MetaRow label="Columns" value={String(table.columns?.length ?? 0)} mono />
					<MetaRow
						label="Owners"
						value={
							table.owners && table.owners.length > 0
								? table.owners.map((o) => o.displayName ?? o.name).join(', ')
								: 'unassigned'
						}
					/>
					<MetaRow label="Schema" value={table.databaseSchema.name} mono />
					<MetaRow label="Database" value={table.database.name} mono />
					<MetaRow
						label="Updated"
						value={
							table.updatedAt ? new Date(table.updatedAt).toLocaleDateString() : '—'
						}
					/>
				</aside>
			</header>

			<section>
				<h2 className="mb-3 text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					Columns
				</h2>
				<div className="overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
					<table className="w-full text-sm">
						<thead className="border-b border-zinc-200 bg-zinc-50 text-left text-xs font-medium uppercase tracking-wider text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950/40 dark:text-zinc-400">
							<tr>
								<th className="w-12 px-3 py-2.5 text-right">#</th>
								<th className="px-3 py-2.5">Column</th>
								<th className="px-3 py-2.5">Type</th>
								<th className="px-3 py-2.5">Tags</th>
								<th className="px-3 py-2.5">Description</th>
							</tr>
						</thead>
						<tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
							{(table.columns ?? []).map((c, colIdx) => {
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
														{suggestedPii.map(({ t, tagIdx }) => (
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
														))}
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
				<p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-500">
					<span className="font-medium">Tip:</span> tags with the{' '}
					<span aria-hidden>✦</span> icon were auto-generated by the OpenMetadata
					Auto-Classification workflow. &quot;Suggested&quot; tags are awaiting human
					review; accept or reject them to confirm.
				</p>
			</section>

			<section className="mt-10">
				<h2 className="mb-3 text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					Recent queries on this table
				</h2>
				{queries == null ? (
					<div className="h-24 animate-pulse rounded-xl bg-zinc-100 dark:bg-zinc-900" />
				) : queries.length === 0 ? (
					<p className="rounded-lg border border-dashed border-zinc-300 p-8 text-center text-sm text-zinc-500 dark:border-zinc-700">
						No tracked queries yet — run the usage ingestion against this source.
					</p>
				) : (
					<ul className="space-y-2">
						{queries.map((q) => (
							<li
								key={q.id}
								className="rounded-xl border border-zinc-200 bg-white p-3 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
							>
								<div className="mb-1 flex flex-wrap items-center gap-3 text-[11px] text-zinc-500 dark:text-zinc-500">
									<span className="font-mono">
										{q.queryDate
											? new Date(q.queryDate).toLocaleDateString()
											: 'unknown date'}
									</span>
									{q.queryUsedIn && q.queryUsedIn.length > 1 ? (
										<span>used in {q.queryUsedIn.length} tables</span>
									) : null}
									{q.users && q.users.length > 0 ? (
										<span>by {q.users.map((u) => u.name).join(', ')}</span>
									) : null}
								</div>
								<pre className="overflow-x-auto whitespace-pre-wrap break-all font-mono text-xs leading-relaxed text-zinc-800 dark:text-zinc-200">
									{q.query}
								</pre>
							</li>
						))}
					</ul>
				)}
			</section>
		</div>
	);
}

const Chevron = () => (
	<svg viewBox="0 0 20 20" fill="currentColor" className="h-3 w-3 text-zinc-400" aria-hidden>
		<path
			fillRule="evenodd"
			d="M7.21 14.77a.75.75 0 0 1 .02-1.06L11.168 10 7.23 6.29a.75.75 0 1 1 1.04-1.08l4.5 4.25a.75.75 0 0 1 0 1.08l-4.5 4.25a.75.75 0 0 1-1.06-.02z"
			clipRule="evenodd"
		/>
	</svg>
);

const MetaRow = ({ label, value, mono }: { label: string; value: string; mono?: boolean }) => (
	<div className="grid grid-cols-[auto_1fr] gap-x-3">
		<span className="text-zinc-500 dark:text-zinc-500">{label}</span>
		<span className={`text-zinc-800 dark:text-zinc-200 ${mono ? 'font-mono text-xs' : ''}`}>
			{value}
		</span>
	</div>
);
