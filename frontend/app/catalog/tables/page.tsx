// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Suspense, useEffect, useMemo, useState } from 'react';
import {
	Badge,
	Skeleton,
	TableBody,
	TableDataCell,
	TableHead,
	TableHeaderCell,
	TableRoot,
	TableRow,
	Text,
	TextInput,
} from '@kui/foundations-react';
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

function TablesListContent() {
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
				<Text asChild kind="title/md">
					<h1 className="text-[var(--text-color-primary)]">Data Dictionary</h1>
				</Text>
				<p className="mt-1 text-sm text-[var(--text-color-base)]">
					Every table in the catalog, with ownership, certification status, usage and
					tags.
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
				<TextInput
					className="w-72"
					type="search"
					dismissible
					value={q}
					onValueChange={(v) => setQ(v)}
					onDismiss={() => setQ('')}
					slotStart={
						<svg
							viewBox="0 0 20 20"
							fill="currentColor"
							className="h-4 w-4"
							aria-hidden
						>
							<path
								fillRule="evenodd"
								d="M9 3a6 6 0 1 0 3.84 10.61l3.27 3.28 1.42-1.42-3.28-3.27A6 6 0 0 0 9 3zm0 2a4 4 0 1 1 0 8 4 4 0 0 1 0-8z"
								clipRule="evenodd"
							/>
						</svg>
					}
					attributes={{ Input: { placeholder: 'Search tables or tags…' } }}
				/>
			</div>

			{err ? (
				<div className="rounded-[var(--radius-lg)] border border-[var(--border-color-feedback-danger)] bg-[var(--background-color-feedback-danger-subtle-hover)] p-4 text-sm text-[var(--text-color-feedback-danger-strong)]">
					{err}
				</div>
			) : tables == null ? (
				<TableSkeleton />
			) : filtered && filtered.length === 0 ? (
				<p className="rounded-[var(--radius-lg)] border border-dashed border-[var(--border-color-base)] p-12 text-center text-sm text-[var(--text-color-subtle)]">
					No tables match these filters.
				</p>
			) : (
				<TableRoot hoverableRows>
					<TableHead>
						<TableRow>
							<TableHeaderCell>Table</TableHeaderCell>
							<TableHeaderCell>Owner</TableHeaderCell>
							<TableHeaderCell>Status</TableHeaderCell>
							<TableHeaderCell>Usage</TableHeaderCell>
							<TableHeaderCell>Cols</TableHeaderCell>
							<TableHeaderCell>PII</TableHeaderCell>
							<TableHeaderCell>Tags</TableHeaderCell>
						</TableRow>
					</TableHead>
					<TableBody>
						{(filtered ?? []).map((t) => {
							const piiN = piiColumnCount(t);
							const ownerName =
								t.owners && t.owners.length > 0
									? (t.owners[0].displayName ?? t.owners[0].name)
									: null;
							return (
								<TableRow key={t.id}>
									<TableDataCell>
										<Link
											href={`/catalog/tables/${encodeURIComponent(t.fullyQualifiedName)}`}
											className="block"
										>
											<span className="font-medium text-[var(--text-color-primary)] hover:text-[var(--text-color-brand)]">
												{t.name}
											</span>
											<span className="block font-mono text-[11px] text-[var(--text-color-subtle)]">
												{t.database.name} · {t.databaseSchema.name}
											</span>
										</Link>
									</TableDataCell>
									<TableDataCell>
										<OwnerChip name={ownerName} />
									</TableDataCell>
									<TableDataCell>
										<CertificationBadge status={readinessOf(t)} dense />
									</TableDataCell>
									<TableDataCell>
										<UsageMeter level={usageLevelOf(t.usageSummary)} />
									</TableDataCell>
									<TableDataCell>
										<span className="block text-right tabular-nums">
											{t.columns?.length ?? 0}
										</span>
									</TableDataCell>
									<TableDataCell>
										<span className="block text-right">
											{piiN > 0 ? (
												<Badge color="yellow" kind="outline">
													{piiN}
												</Badge>
											) : (
												<span className="text-xs text-[var(--text-color-subtle)]">
													—
												</span>
											)}
										</span>
									</TableDataCell>
									<TableDataCell>
										{(t.tags ?? []).length === 0 ? (
											<span className="text-xs text-[var(--text-color-subtle)]">
												—
											</span>
										) : (
											<div className="flex flex-wrap gap-1">
												{(t.tags ?? []).slice(0, 3).map((tag) => (
													<TagPill key={tag.tagFQN} label={tag} dense />
												))}
												{(t.tags ?? []).length > 3 ? (
													<span className="text-[11px] text-[var(--text-color-subtle)]">
														+{(t.tags ?? []).length - 3}
													</span>
												) : null}
											</div>
										)}
									</TableDataCell>
								</TableRow>
							);
						})}
					</TableBody>
				</TableRoot>
			)}
		</div>
	);
}

const TableSkeleton = () => (
	<div className="space-y-3 rounded-[var(--radius-lg)] border border-[var(--border-color-base)] p-4">
		{Array.from({ length: 6 }).map((_, i) => (
			<Skeleton key={i} kind="line" />
		))}
	</div>
);

export default function TablesListPage() {
	return (
		<Suspense fallback={null}>
			<TablesListContent />
		</Suspense>
	);
}
