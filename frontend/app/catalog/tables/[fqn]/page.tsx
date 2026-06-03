// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { use, useCallback, useEffect, useMemo, useState } from 'react';
import {
	Badge,
	Button,
	CodeSnippet,
	Skeleton,
	TableBody,
	TableDataCell,
	TableHead,
	TableHeaderCell,
	TableRoot,
	TableRow,
	Text,
} from '@kui/foundations-react';
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
				<div className="rounded-[var(--radius-lg)] border border-[var(--border-color-feedback-danger)] bg-[var(--background-color-feedback-danger-subtle-hover)] p-4 text-sm text-[var(--text-color-feedback-danger-strong)]">
					{err}
				</div>
			</div>
		);
	}
	if (!table) {
		return (
			<div className="mx-auto max-w-6xl space-y-3 px-6 py-12">
				<Skeleton kind="line" />
				<Skeleton kind="line" />
				<div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr_240px]">
					{Array.from({ length: 3 }).map((_, i) => (
						<div
							key={i}
							className="h-72 animate-pulse rounded-[var(--radius-lg)] bg-[var(--background-color-component-skeleton)]"
						/>
					))}
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
			<nav className="mb-2 flex flex-wrap items-center gap-1 text-xs text-[var(--text-color-base)]">
				<Link href="/catalog/tables" className="hover:text-[var(--text-color-brand)]">
					Data Dictionary
				</Link>
				<Chevron />
				<span className="font-mono">{table.database.name}</span>
				<Chevron />
				<span className="font-mono">{table.databaseSchema.name}</span>
				<Chevron />
				<span className="font-mono font-semibold text-[var(--text-color-secondary)]">
					{table.name}
				</span>
			</nav>

			<header className="mb-6 flex flex-wrap items-start justify-between gap-4">
				<div className="min-w-0">
					<div className="flex flex-wrap items-center gap-3">
						<Text asChild kind="title/md">
							<h1 className="flex items-center gap-2 text-[var(--text-color-primary)]">
								<GridIcon className="h-5 w-5 text-[var(--text-color-subtle)]" />
								{table.displayName ?? table.name}
							</h1>
						</Text>
						<CertificationBadge status={status} />
						{piiCols.length > 0 ? (
							<Badge color="yellow" kind="solid">
								<ShieldIcon />
								Contains PII
							</Badge>
						) : null}
					</div>
				</div>
				<Button kind="secondary">
					<DocIcon className="h-4 w-4" />
					Export
				</Button>
			</header>

			<div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_minmax(0,1fr)_240px]">
				{/* ---------------------------------------------------------- left rail */}
				<aside className="flex flex-col gap-4">
					<Panel title="Table owner" icon={<UserIcon />}>
						<OwnerChip name={ownerName} />
					</Panel>

					<Panel title="View in Exploration" icon={<LinkIcon className="h-3.5 w-3.5" />}>
						<Link
							href={`/catalog/lineage?fqn=${encodeURIComponent(table.fullyQualifiedName)}`}
							className="group block"
						>
							<ExplorationThumb />
							<span className="mt-2 block text-xs font-medium text-[var(--text-color-brand)] group-hover:underline">
								Open lineage graph →
							</span>
						</Link>
					</Panel>

					<Panel title="Details">
						<div className="divide-y divide-[var(--border-color-base)]">
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

					<Panel title="Tags">
						<TagsRow tags={table.tags ?? []} />
					</Panel>
				</aside>

				{/* -------------------------------------------------------- center pane */}
				<div className="flex min-w-0 flex-col gap-6">
					<Panel title="Description" icon={<DocIcon className="h-3.5 w-3.5" />}>
						<p className="text-sm leading-relaxed text-[var(--text-color-secondary)]">
							{table.description || (
								<em className="text-[var(--text-color-subtle)]">
									No description provided yet.
								</em>
							)}
						</p>
					</Panel>

					<Panel title="Certification" icon={<CheckBadgeIcon />}>
						<div className="grid grid-cols-3 gap-3">
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
							<div className="grid grid-cols-2 gap-4">
								<div>
									<p className="text-xs text-[var(--text-color-subtle)]">
										Popularity
									</p>
									<div className="mt-1">
										<UsageMeter level={usage} />
									</div>
								</div>
								<Stat
									label="Queries"
									value={queries == null ? '—' : queries.length}
								/>
								{weeklyHits != null ? (
									<Stat label="Last 7 days" value={weeklyHits.toLocaleString()} />
								) : null}
								{monthlyHits != null ? (
									<Stat
										label="Last 30 days"
										value={monthlyHits.toLocaleString()}
									/>
								) : null}
							</div>
						</Panel>

						<Panel title="Entities" icon={<GridIcon className="h-3.5 w-3.5" />}>
							<div className="grid grid-cols-2 gap-4">
								<Stat label="Columns" value={columns.length} />
								<Stat
									label="Documented"
									value={`${documentedCols.length}/${columns.length}`}
								/>
								<Stat label="PII columns" value={piiCols.length} />
								<Stat label="Tags" value={(table.tags ?? []).length} />
							</div>
						</Panel>
					</div>

					<Panel
						title="Columns"
						icon={<GridIcon className="h-3.5 w-3.5" />}
						action={
							<Badge color="gray" kind="outline">
								{columns.length} total
							</Badge>
						}
					>
						<TableRoot>
							<TableHead>
								<TableRow>
									<TableHeaderCell>#</TableHeaderCell>
									<TableHeaderCell>Column</TableHeaderCell>
									<TableHeaderCell>Type</TableHeaderCell>
									<TableHeaderCell>Tags</TableHeaderCell>
									<TableHeaderCell>Description</TableHeaderCell>
								</TableRow>
							</TableHead>
							<TableBody>
								{columns.map((c, colIdx) => {
									const suggestedPii = (c.tags ?? [])
										.map((t, tagIdx) => ({ t, tagIdx }))
										.filter((x) => x.t.state === 'Suggested');
									return (
										<TableRow key={c.name}>
											<TableDataCell>
												<span className="block text-right font-mono text-xs text-[var(--text-color-subtle)]">
													{c.ordinalPosition ?? colIdx + 1}
												</span>
											</TableDataCell>
											<TableDataCell>
												<span className="font-mono text-sm font-medium text-[var(--text-color-primary)]">
													{c.name}
												</span>
											</TableDataCell>
											<TableDataCell>
												<span className="font-mono text-xs text-[var(--text-color-base)]">
													{c.dataType}
												</span>
											</TableDataCell>
											<TableDataCell>
												<div className="flex flex-col gap-1.5">
													<TagsRow tags={c.tags ?? []} />
													{suggestedPii.length > 0 ? (
														<div className="flex flex-wrap items-center gap-1.5 text-[11px]">
															<span className="text-[var(--text-color-accent-yellow)]">
																Auto-detected — review:
															</span>
															{suggestedPii.map(({ t, tagIdx }) => (
																<span
																	key={tagIdx}
																	className="inline-flex items-center gap-1"
																>
																	<Button
																		size="tiny"
																		color="brand"
																		disabled={busy}
																		onClick={() =>
																			void confirmTag(
																				colIdx,
																				tagIdx,
																			)
																		}
																		title={`Accept ${t.tagFQN}`}
																	>
																		Accept
																	</Button>
																	<Button
																		size="tiny"
																		kind="secondary"
																		disabled={busy}
																		onClick={() =>
																			void rejectTag(
																				colIdx,
																				tagIdx,
																			)
																		}
																		title={`Reject ${t.tagFQN}`}
																	>
																		Reject
																	</Button>
																</span>
															))}
														</div>
													) : null}
												</div>
											</TableDataCell>
											<TableDataCell>
												<span className="line-clamp-2 block max-w-md text-xs text-[var(--text-color-base)]">
													{c.description ?? (
														<em className="text-[var(--text-color-subtle)]">
															—
														</em>
													)}
												</span>
											</TableDataCell>
										</TableRow>
									);
								})}
							</TableBody>
						</TableRoot>
						<p className="mt-3 text-[11px] text-[var(--text-color-subtle)]">
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
								className="text-xs font-medium text-[var(--text-color-brand)] hover:underline"
							>
								View all →
							</Link>
						}
					>
						{queries == null ? (
							<Skeleton kind="line" />
						) : queries.length === 0 ? (
							<p className="py-6 text-center text-sm text-[var(--text-color-subtle)]">
								No tracked queries yet — run the usage ingestion against this
								source.
							</p>
						) : (
							<ul className="flex flex-col gap-4">
								{queries.slice(0, 6).map((q) => (
									<li key={q.id}>
										<div className="mb-1 flex flex-wrap items-center gap-3 text-[11px] text-[var(--text-color-subtle)]">
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
										<CodeSnippet
											kind="block"
											language="text"
											value={q.query}
											collapsible
											rows={3}
										/>
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
							<p className="text-xs text-[var(--text-color-subtle)]">
								No co-queried tables found.
							</p>
						) : (
							<ul className="divide-y divide-[var(--border-color-base)]">
								{related.map((r) => (
									<li key={r.fqn}>
										<Link
											href={`/catalog/tables/${encodeURIComponent(r.fqn)}`}
											className="flex items-center justify-between gap-2 py-2.5"
										>
											<span className="truncate font-mono text-xs text-[var(--text-color-secondary)]">
												{r.name}
											</span>
											<Badge color="gray" kind="outline">
												{r.n}×
											</Badge>
										</Link>
									</li>
								))}
							</ul>
						)}
					</Panel>

					{piiCols.length > 0 ? (
						<Panel title="PII columns" icon={<ShieldIcon />}>
							<ul className="flex flex-wrap gap-1.5">
								{piiCols.map((c) => (
									<li key={c.name}>
										<Badge color="yellow" kind="outline">
											<span className="font-mono">{c.name}</span>
										</Badge>
									</li>
								))}
							</ul>
						</Panel>
					) : null}

					<Panel title="Common filters" icon={<TagIcon />}>
						<div className="flex flex-wrap gap-1.5">
							{(table.tags ?? []).length === 0 ? (
								<span className="text-xs text-[var(--text-color-subtle)]">
									No tags to filter by.
								</span>
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
	<div className="overflow-hidden rounded-[var(--radius-lg)] border border-[var(--border-color-base)] bg-[var(--background-color-surface-sunken)]">
		<svg viewBox="0 0 220 96" className="h-24 w-full" aria-hidden>
			<line x1="40" y1="48" x2="110" y2="28" stroke="var(--border-color-accent-gray)" />
			<line x1="40" y1="48" x2="110" y2="68" stroke="var(--border-color-accent-gray)" />
			<line x1="110" y1="48" x2="180" y2="30" stroke="var(--border-color-accent-gray)" />
			<line x1="110" y1="48" x2="180" y2="66" stroke="var(--border-color-accent-gray)" />
			<circle cx="40" cy="48" r="7" className="fill-violet-400" />
			<circle cx="110" cy="28" r="6" className="fill-amber-400" />
			<circle cx="110" cy="68" r="6" fill="var(--text-color-subtle)" />
			<circle cx="110" cy="48" r="9" fill="var(--color-brand)" />
			<circle cx="180" cy="30" r="5" fill="var(--border-color-accent-gray)" />
			<circle cx="180" cy="66" r="5" fill="var(--border-color-accent-gray)" />
			<circle cx="180" cy="48" r="5" fill="var(--border-color-accent-gray)" />
		</svg>
	</div>
);

const CertCell = ({ label, icon, ok }: { label: string; icon: React.ReactNode; ok: boolean }) => (
	<div className="flex items-center justify-between gap-2">
		<span className="flex items-center gap-1.5 text-sm text-[var(--text-color-secondary)]">
			<span className="text-[var(--text-color-subtle)]">{icon}</span>
			{label}
		</span>
		{ok ? (
			<Badge color="green" kind="solid" title="Present">
				✓
			</Badge>
		) : (
			<span
				className="h-4 w-4 rounded-full border border-dashed border-[var(--border-color-base)]"
				title="Missing"
				aria-hidden
			/>
		)}
	</div>
);

const Stat = ({ label, value }: { label: string; value: number | string }) => (
	<div>
		<p className="text-xs text-[var(--text-color-subtle)]">{label}</p>
		<p className="mt-0.5 text-lg font-semibold tabular-nums text-[var(--text-color-primary)]">
			{value}
		</p>
	</div>
);

const Chevron = () => (
	<svg
		viewBox="0 0 20 20"
		fill="currentColor"
		className="h-3 w-3 text-[var(--text-color-subtle)]"
		aria-hidden
	>
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
