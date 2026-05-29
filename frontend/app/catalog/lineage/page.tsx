// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { Suspense, useEffect, useMemo, useState } from 'react';
import { om } from '@/api/openmetadata';
import type { OmLineageResponse, OmQuery, OmTable } from '@/types/openmetadata';
import { LineageGraph, type GraphEdge, type GraphNode } from '@/components/catalog/LineageGraph';
import { Panel } from '@/components/catalog/ui';

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
					<h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
						Exploration
					</h1>
					<p className="mt-1 max-w-2xl text-sm text-zinc-600 dark:text-zinc-400">
						How a table relates to the rest of your warehouse — directed lineage from
						OpenMetadata, table joins observed in real query history, and same-schema
						neighbours, centred on the table you pick.
					</p>
				</div>
				<label className="flex items-center gap-2 text-sm">
					<span className="text-zinc-500 dark:text-zinc-400">Focus</span>
					<select
						value={fqn ?? ''}
						onChange={(e) =>
							router.push(
								`/catalog/lineage?fqn=${encodeURIComponent(e.target.value)}`,
							)
						}
						className="w-72 rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-sm text-zinc-800 shadow-sm outline-none focus:border-[#76b900] dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
					>
						{allTables == null ? (
							<option>Loading…</option>
						) : (
							allTables.map((t) => (
								<option key={t.id} value={t.fullyQualifiedName}>
									{t.database.name} · {t.databaseSchema.name} · {t.name}
								</option>
							))
						)}
					</select>
				</label>
			</header>

			{err ? (
				<div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
					{err}
				</div>
			) : null}

			<div className="mb-3 flex flex-wrap items-center gap-4 text-xs text-zinc-500 dark:text-zinc-400">
				<LegendDot className="bg-[#76b900]" label="Focused table" />
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-px w-5 border-t-2 border-[#76b900]/60" />
					Related — query join{' '}
					<span className="rounded-full bg-zinc-100 px-1.5 py-0.5 tabular-nums dark:bg-zinc-800">
						{relatedCount}
					</span>
				</span>
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-px w-5 border-t border-zinc-400" />
					Lineage — OpenMetadata{' '}
					<span className="rounded-full bg-zinc-100 px-1.5 py-0.5 tabular-nums dark:bg-zinc-800">
						{omEdgeCount}
					</span>
				</span>
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-px w-5 border-t border-dashed border-zinc-400" />
					Same schema
				</span>
			</div>

			{loading && !graph ? (
				<div className="h-[480px] animate-pulse rounded-xl bg-zinc-100 dark:bg-zinc-900" />
			) : graph ? (
				<>
					<LineageGraph nodes={graph.nodes} edges={graph.edges} />
					{omEdgeCount === 0 ? (
						<Panel className="mt-4">
							<p className="px-4 py-3 text-xs text-zinc-500 dark:text-zinc-400">
								{relatedCount > 0 ? (
									<>
										Showing{' '}
										<span className="font-medium">
											{relatedCount} relationship
											{relatedCount === 1 ? '' : 's'} derived from query joins
										</span>{' '}
										plus same-schema neighbours. OpenMetadata has no ingested
										table-to-table lineage for this entity yet — run the lineage
										workflow with query parsing enabled and directed edges will
										appear here automatically.
									</>
								) : (
									<>
										No query joins reference this table and OpenMetadata has no
										ingested lineage for it yet, so only same-schema neighbours
										are shown. Relationships will appear here as lineage is
										ingested or as joins against this table are observed.
									</>
								)}
							</p>
						</Panel>
					) : null}
					{table ? (
						<div className="mt-4 text-sm">
							<Link
								href={`/catalog/tables/${encodeURIComponent(table.fullyQualifiedName)}`}
								className="text-[#76b900] hover:underline"
							>
								Open {table.name} in the Data Dictionary →
							</Link>
						</div>
					) : null}
				</>
			) : (
				<p className="rounded-lg border border-dashed border-zinc-300 p-12 text-center text-sm text-zinc-500 dark:border-zinc-700">
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
				<div className="mx-auto w-full max-w-7xl px-6 py-8">
					<div className="h-8 w-48 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
					<div className="mt-6 h-[480px] animate-pulse rounded-xl bg-zinc-100 dark:bg-zinc-900" />
				</div>
			}
		>
			<LineageExplorer />
		</Suspense>
	);
}
