// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { Suspense, useEffect, useMemo, useState } from 'react';
import { Badge, Select, Skeleton, Text } from '@kui/foundations-react';
import { om } from '@/api/openmetadata';
import type { OmLineageResponse, OmQuery, OmTable } from '@/types/openmetadata';
import { LineageGraph, type GraphEdge, type GraphNode } from '@/components/catalog/LineageGraph';

const MAX_SIBLINGS = 6;
const MAX_RELATED = 8;

/**
 * Extract referenced tables from raw SQL by scanning FROM / JOIN clauses.
 * Returns the parsed `{schema?, name}` pairs (last two dotted segments), which
 * lets us reconstruct table↔table relationships OpenMetadata's `queryUsedIn`
 * doesn't resolve (it only ever links a query to a single table).
 */
const parseTableRefs = (sql: string): { schema?: string; name: string }[] => {
	const s = sql.replace(/\s+/g, ' ');
	const re = /(?:FROM|JOIN)\s+([A-Za-z0-9_."]+(?:\.[A-Za-z0-9_."]+)*)/gi;
	const out: { schema?: string; name: string }[] = [];
	for (const m of s.matchAll(re)) {
		const parts = m[1].replace(/"/g, '').split('.').filter(Boolean);
		if (parts.length === 0) continue;
		const name = parts[parts.length - 1].toUpperCase();
		if (!name || name.startsWith('(')) continue;
		const schema = parts.length >= 2 ? parts[parts.length - 2].toUpperCase() : undefined;
		out.push({ schema, name });
	}
	return out;
};

const refMatchesTable = (ref: { schema?: string; name: string }, t: OmTable): boolean => {
	if (t.name.toUpperCase() !== ref.name) return false;
	if (ref.schema && t.databaseSchema.name.toUpperCase() !== ref.schema) return false;
	return true;
};

/** Resolve a parsed ref to a concrete table, preferring the focused table's schema. */
const resolveRef = (
	ref: { schema?: string; name: string },
	all: OmTable[],
	preferSchemaId: string,
): OmTable | null => {
	const candidates = all.filter((t) => refMatchesTable(ref, t));
	if (candidates.length === 0) return null;
	return candidates.find((t) => t.databaseSchema.id === preferSchemaId) ?? candidates[0];
};

/**
 * Build a table-centric relationship graph:
 *   siblings / upstream (left) → focused table (centre) → related / downstream (right)
 * Relationships come from real signals only — OpenMetadata lineage edges when
 * present, and table joins parsed from observed query SQL.
 */
const buildGraph = (
	table: OmTable,
	allTables: OmTable[],
	queries: OmQuery[],
	lineage: OmLineageResponse | null,
): { nodes: GraphNode[]; edges: GraphEdge[]; relatedCount: number } => {
	const nodes: GraphNode[] = [];
	const edges: GraphEdge[] = [];
	const seen = new Set<string>();
	const tblId = (id: string) => `tbl:${id}`;
	const add = (n: GraphNode) => {
		if (seen.has(n.id)) return;
		seen.add(n.id);
		nodes.push(n);
	};
	const tableHref = (t: OmTable) => `/catalog/tables/${encodeURIComponent(t.fullyQualifiedName)}`;

	// focused table (centre)
	add({
		id: tblId(table.id),
		kind: 'table',
		label: table.name,
		sublabel: `${table.database.name} · ${table.databaseSchema.name}`,
		layer: 1,
		focused: true,
	});

	// query-join relationships parsed from real SQL
	const relWeight = new Map<string, { t: OmTable; n: number }>();
	for (const q of queries) {
		const refs = parseTableRefs(q.query ?? '');
		if (!refs.some((r) => refMatchesTable(r, table))) continue;
		const resolved = new Map<string, OmTable>();
		for (const r of refs) {
			const t = resolveRef(r, allTables, table.databaseSchema.id);
			if (t && t.id !== table.id) resolved.set(t.id, t);
		}
		for (const t of resolved.values()) {
			const prev = relWeight.get(t.id);
			relWeight.set(t.id, { t, n: (prev?.n ?? 0) + 1 });
		}
	}
	const related = [...relWeight.values()].sort((a, b) => b.n - a.n).slice(0, MAX_RELATED);
	for (const { t, n } of related) {
		add({
			id: tblId(t.id),
			kind: 'table',
			label: t.name,
			sublabel: t.databaseSchema.name,
			layer: 2,
			href: tableHref(t),
		});
		edges.push({
			from: tblId(table.id),
			to: tblId(t.id),
			kind: 'related',
			label: n > 1 ? `${n} queries` : '1 query',
		});
	}

	// same-schema siblings (left) for structural context
	const siblings = allTables
		.filter(
			(t) =>
				t.databaseSchema.id === table.databaseSchema.id &&
				t.id !== table.id &&
				!seen.has(tblId(t.id)),
		)
		.slice(0, MAX_SIBLINGS);
	for (const s of siblings) {
		add({
			id: tblId(s.id),
			kind: 'table',
			label: s.name,
			sublabel: 'same schema',
			layer: 0,
			href: tableHref(s),
		});
		edges.push({ from: tblId(s.id), to: tblId(table.id), kind: 'sibling' });
	}

	// directed lineage from OpenMetadata, if any
	if (lineage) {
		const refById = new Map(lineage.nodes.map((n) => [n.id, n]));
		const addTableNode = (id: string, layer: number) => {
			const ref = refById.get(id);
			if (!ref) return;
			add({
				id: tblId(id),
				kind: 'table',
				label: ref.name,
				sublabel: 'lineage',
				layer,
				href: ref.fullyQualifiedName
					? `/catalog/tables/${encodeURIComponent(ref.fullyQualifiedName)}`
					: undefined,
			});
		};
		for (const e of lineage.upstreamEdges) {
			addTableNode(e.fromEntity, 0);
			edges.push({ from: tblId(e.fromEntity), to: tblId(e.toEntity), kind: 'lineage' });
		}
		for (const e of lineage.downstreamEdges) {
			addTableNode(e.toEntity, 2);
			edges.push({ from: tblId(e.fromEntity), to: tblId(e.toEntity), kind: 'lineage' });
		}
	}

	return { nodes, edges, relatedCount: related.length };
};

const LineageExplorer = () => {
	const router = useRouter();
	const params = useSearchParams();
	const fqn = params.get('fqn');

	const [allTables, setAllTables] = useState<OmTable[] | null>(null);
	const [table, setTable] = useState<OmTable | null>(null);
	const [queries, setQueries] = useState<OmQuery[]>([]);
	const [lineage, setLineage] = useState<OmLineageResponse | null>(null);
	const [err, setErr] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	// table list for the picker + query history for relationship parsing
	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				const [tablesRes, queriesRes] = await Promise.all([
					om.tables.list({ fields: 'columns', limit: 200 }),
					om.queries.list({ limit: 200 }).catch(() => ({ data: [] as OmQuery[] })),
				]);
				if (cancelled) return;
				setAllTables(tablesRes.data);
				setQueries(queriesRes.data);
			} catch (e) {
				if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
			}
		})();
		return () => {
			cancelled = true;
		};
	}, []);

	// default focus to the first table when none selected
	useEffect(() => {
		if (!fqn && allTables && allTables.length > 0) {
			router.replace(
				`/catalog/lineage?fqn=${encodeURIComponent(allTables[0].fullyQualifiedName)}`,
			);
		}
	}, [fqn, allTables, router]);

	// load the focused table + its OpenMetadata lineage
	useEffect(() => {
		if (!fqn) return;
		let cancelled = false;
		setLoading(true);
		void (async () => {
			try {
				const t = await om.tables.getByFqn(fqn, 'columns,tags,description');
				if (cancelled) return;
				setTable(t);
				const lin = await om.lineage.forTable(fqn).catch(() => null);
				if (cancelled) return;
				setLineage(lin);
			} catch (e) {
				if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
			} finally {
				if (!cancelled) setLoading(false);
			}
		})();
		return () => {
			cancelled = true;
		};
	}, [fqn]);

	const graph = useMemo(
		() => (table && allTables ? buildGraph(table, allTables, queries, lineage) : null),
		[table, allTables, queries, lineage],
	);

	const omEdgeCount = lineage ? lineage.upstreamEdges.length + lineage.downstreamEdges.length : 0;
	const relatedCount = graph?.relatedCount ?? 0;

	return (
		<div className="mx-auto w-full max-w-7xl px-6 py-8">
			<header className="mb-5 flex flex-wrap items-end justify-between gap-4">
				<div>
					<Text asChild kind="title/md">
						<h1 className="text-[var(--text-color-primary)]">Exploration</h1>
					</Text>
					<p className="mt-1 max-w-2xl text-sm text-[var(--text-color-base)]">
						How a table relates to the rest of your warehouse — directed lineage from
						the catalog, table joins observed in real query history, and same-schema
						neighbours, centred on the table you pick.
					</p>
				</div>
				<label className="flex items-center gap-2 text-sm">
					<span className="text-[var(--text-color-base)]">Focus</span>
					<Select
						className="w-72"
						placeholder={allTables == null ? 'Loading…' : 'Select a table'}
						value={fqn ?? ''}
						onValueChange={(v) =>
							router.push(`/catalog/lineage?fqn=${encodeURIComponent(v)}`)
						}
						items={(allTables ?? []).map((t) => ({
							value: t.fullyQualifiedName,
							children: `${t.database.name} · ${t.databaseSchema.name} · ${t.name}`,
						}))}
					/>
				</label>
			</header>

			{err ? (
				<div className="rounded-[var(--radius-lg)] border border-[var(--border-color-feedback-danger)] bg-[var(--background-color-feedback-danger-subtle-hover)] p-4 text-sm text-[var(--text-color-feedback-danger-strong)]">
					{err}
				</div>
			) : null}

			<div className="mb-3 flex flex-wrap items-center gap-4 text-xs text-[var(--text-color-base)]">
				<LegendDot className="bg-[var(--color-brand)]" label="Focused table" />
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-px w-5 border-t-2 border-[var(--color-brand)]" />
					Related — query join{' '}
					<Badge color="gray" kind="outline">
						{relatedCount}
					</Badge>
				</span>
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-px w-5 border-t border-[var(--border-color-accent-gray)]" />
					Lineage — catalog{' '}
					<Badge color="gray" kind="outline">
						{omEdgeCount}
					</Badge>
				</span>
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-px w-5 border-t border-dashed border-[var(--border-color-accent-gray)]" />
					Same schema
				</span>
			</div>

			{loading && !graph ? (
				<div className="h-[480px] animate-pulse rounded-[var(--radius-lg)] bg-[var(--background-color-component-skeleton)]" />
			) : graph ? (
				<>
					<LineageGraph nodes={graph.nodes} edges={graph.edges} />
					{table ? (
						<div className="mt-4 text-sm">
							<Link
								href={`/catalog/tables/${encodeURIComponent(table.fullyQualifiedName)}`}
								className="text-[var(--text-color-brand)] hover:underline"
							>
								Open {table.name} in the Data Dictionary →
							</Link>
						</div>
					) : null}
				</>
			) : (
				<p className="rounded-[var(--radius-lg)] border border-dashed border-[var(--border-color-base)] p-12 text-center text-sm text-[var(--text-color-subtle)]">
					Pick a table to explore.
				</p>
			)}
		</div>
	);
};

const LegendDot = ({ className, label }: { className: string; label: string }) => (
	<span className="flex items-center gap-1.5">
		<span className={`inline-block h-2.5 w-2.5 rounded-sm ${className}`} />
		{label}
	</span>
);

export default function LineagePage() {
	return (
		<Suspense
			fallback={
				<div className="mx-auto w-full max-w-7xl space-y-6 px-6 py-8">
					<Skeleton kind="line" />
					<div className="h-[480px] animate-pulse rounded-[var(--radius-lg)] bg-[var(--background-color-component-skeleton)]" />
				</div>
			}
		>
			<LineageExplorer />
		</Suspense>
	);
}
