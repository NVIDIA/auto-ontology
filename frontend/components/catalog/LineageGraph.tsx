// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * A dependency-free, SVG-backed lineage / exploration graph.
 *
 * Nodes are laid out in vertical layers (left → right); edges are drawn as
 * horizontal cubic-bezier curves behind the node cards. This mirrors illumex's
 * "View in Exploration" canvas while staying in the GSF visual language
 * (zinc neutrals, #76b900 accent, rounded cards, dark mode).
 *
 * Geometry (left/top/width/height) is computed at runtime and therefore applied
 * via inline style — Tailwind handles every other visual concern.
 */

'use client';

import Link from 'next/link';
import { useMemo, useState, type ReactNode } from 'react';

export type GraphNodeKind = 'service' | 'database' | 'schema' | 'table' | 'column' | 'pii';

export type GraphNode = {
	id: string;
	label: string;
	sublabel?: string;
	kind: GraphNodeKind;
	/** Layer index — lower is further upstream (left). */
	layer: number;
	focused?: boolean;
	href?: string;
	badge?: string;
};

export type GraphEdge = {
	from: string;
	to: string;
	kind?: 'contains' | 'lineage' | 'related' | 'sibling';
	label?: string;
};

const NODE_W = 196;
const NODE_H = 56;
const COL_GAP = 248;
const ROW_GAP = 80;
const PAD_X = 24;
const PAD_Y = 28;

const KIND_STYLE: Record<GraphNodeKind, { ring: string; chip: string; icon: ReactNode }> = {
	service: {
		ring: 'ring-sky-300 dark:ring-sky-800',
		chip: 'bg-sky-100 text-sky-700 dark:bg-sky-950/50 dark:text-sky-300',
		icon: (
			<path d="M3 5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5zm0 8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2zm3-7a1 1 0 1 0 0 2 1 1 0 0 0 0-2zm0 8a1 1 0 1 0 0 2 1 1 0 0 0 0-2z" />
		),
	},
	database: {
		ring: 'ring-violet-300 dark:ring-violet-800',
		chip: 'bg-violet-100 text-violet-700 dark:bg-violet-950/50 dark:text-violet-300',
		icon: (
			<path d="M10 2c3.866 0 7 1.343 7 3s-3.134 3-7 3-7-1.343-7-3 3.134-3 7-3zm7 5.5C17 9.157 13.866 10.5 10 10.5S3 9.157 3 7.5V10c0 1.657 3.134 3 7 3s7-1.343 7-3V7.5zm0 5C17 14.157 13.866 15.5 10 15.5S3 14.157 3 12.5V15c0 1.657 3.134 3 7 3s7-1.343 7-3v-2.5z" />
		),
	},
	schema: {
		ring: 'ring-amber-300 dark:ring-amber-800',
		chip: 'bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300',
		icon: (
			<path d="M3 5a2 2 0 0 1 2-2h3.172a2 2 0 0 1 1.414.586L10.828 5H15a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5z" />
		),
	},
	table: {
		ring: 'ring-zinc-300 dark:ring-zinc-700',
		chip: 'bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300',
		icon: (
			<path d="M4 4h12a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zm1 3v2h4V7H5zm6 0v2h4V7h-4zm-6 4v2h4v-2H5zm6 0v2h4v-2h-4z" />
		),
	},
	column: {
		ring: 'ring-zinc-200 dark:ring-zinc-800',
		chip: 'bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400',
		icon: (
			<path d="M7.2 3a1 1 0 0 1 .98 1.2l-.4 2H10.78l.42-2.2a1 1 0 1 1 1.96.4l-.36 1.8H14.5a1 1 0 1 1 0 2h-2.1l-.4 2H13.5a1 1 0 1 1 0 2h-1.9l-.42 2.2a1 1 0 1 1-1.96-.4l.36-1.8H6.22l-.42 2.2a1 1 0 1 1-1.96-.4l.36-1.8H2.5a1 1 0 1 1 0-2h2.1l.4-2H3.5a1 1 0 0 1 0-2h1.9l.42-2.2A1 1 0 0 1 7.2 3zm-.6 5-.4 2h3.58l.4-2H6.6z" />
		),
	},
	pii: {
		ring: 'ring-amber-400 dark:ring-amber-700',
		chip: 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-200',
		icon: (
			<path
				fillRule="evenodd"
				d="M10 1.944A11.954 11.954 0 0 1 2.166 5C2.056 5.649 2 6.319 2 7c0 5.225 3.34 9.67 8 11.317C14.66 16.67 18 12.225 18 7c0-.682-.057-1.35-.166-2.001A11.954 11.954 0 0 1 10 1.944zM11 14a1 1 0 1 1-2 0 1 1 0 0 1 2 0zm0-7a1 1 0 1 0-2 0v3a1 1 0 1 0 2 0V7z"
				clipRule="evenodd"
			/>
		),
	},
};

type Positioned = GraphNode & { x: number; y: number };

const layout = (nodes: GraphNode[]): { placed: Positioned[]; width: number; height: number } => {
	const byLayer = new Map<number, GraphNode[]>();
	for (const n of nodes) {
		const arr = byLayer.get(n.layer) ?? [];
		arr.push(n);
		byLayer.set(n.layer, arr);
	}
	const layers = [...byLayer.keys()].sort((a, b) => a - b);
	const maxCount = Math.max(1, ...[...byLayer.values()].map((a) => a.length));
	const contentH = maxCount * ROW_GAP;
	const placed: Positioned[] = [];
	layers.forEach((layer, col) => {
		const arr = byLayer.get(layer) ?? [];
		const start = (contentH - arr.length * ROW_GAP) / 2;
		arr.forEach((n, i) => {
			placed.push({
				...n,
				x: PAD_X + col * COL_GAP,
				y: PAD_Y + start + i * ROW_GAP,
			});
		});
	});
	const width = PAD_X * 2 + (layers.length - 1) * COL_GAP + NODE_W;
	const height = PAD_Y * 2 + contentH;
	return { placed, width, height };
};

const edgePath = (a: Positioned, b: Positioned): string => {
	// connect from the right edge of the upstream node to the left of downstream
	const [src, dst] = a.x <= b.x ? [a, b] : [b, a];
	const x1 = src.x + NODE_W;
	const y1 = src.y + NODE_H / 2;
	const x2 = dst.x;
	const y2 = dst.y + NODE_H / 2;
	const dx = Math.max(40, (x2 - x1) / 2);
	return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
};

const midpoint = (a: Positioned, b: Positioned): { x: number; y: number } => {
	const [src, dst] = a.x <= b.x ? [a, b] : [b, a];
	return {
		x: (src.x + NODE_W + dst.x) / 2,
		y: (src.y + dst.y) / 2 + NODE_H / 2,
	};
};

const NodeIcon = ({ kind }: { kind: GraphNodeKind }) => (
	<svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4" aria-hidden>
		{KIND_STYLE[kind].icon}
	</svg>
);

export const LineageGraph = ({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
	const [hover, setHover] = useState<string | null>(null);
	const { placed, width, height } = useMemo(() => layout(nodes), [nodes]);
	const pos = useMemo(() => new Map(placed.map((p) => [p.id, p])), [placed]);

	const isEdgeActive = (e: GraphEdge): boolean =>
		hover != null && (e.from === hover || e.to === hover);

	return (
		<div className="relative overflow-auto rounded-xl border border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950/40">
			<div
				className="relative bg-[radial-gradient(circle,_var(--color-zinc-200)_1px,_transparent_1px)] bg-[length:22px_22px] dark:bg-[radial-gradient(circle,_var(--color-zinc-800)_1px,_transparent_1px)]"
				style={{ width, height }}
			>
				<svg
					className="absolute inset-0 h-full w-full"
					width={width}
					height={height}
					aria-hidden
				>
					{edges.map((e, i) => {
						const a = pos.get(e.from);
						const b = pos.get(e.to);
						if (!a || !b) return null;
						const active = isEdgeActive(e);
						const solid = e.kind === 'lineage' || e.kind === 'related';
						const accent = e.kind === 'related';
						const stroke = active
							? 'stroke-[#76b900]'
							: accent
								? 'stroke-[#76b900]/60'
								: e.kind === 'lineage'
									? 'stroke-zinc-400 dark:stroke-zinc-600'
									: 'stroke-zinc-300 dark:stroke-zinc-700';
						const d = edgePath(a, b);
						const mid = midpoint(a, b);
						return (
							<g key={`${e.from}-${e.to}-${i}`}>
								<path
									d={d}
									fill="none"
									strokeWidth={active ? 2.5 : accent ? 2 : 1.5}
									strokeDasharray={solid ? undefined : '4 4'}
									className={stroke}
								/>
								{e.label && (active || accent) ? (
									<text
										x={mid.x}
										y={mid.y - 4}
										textAnchor="middle"
										className="fill-zinc-500 text-[10px] dark:fill-zinc-400"
									>
										{e.label}
									</text>
								) : null}
							</g>
						);
					})}
				</svg>

				{placed.map((n) => {
					const style = KIND_STYLE[n.kind];
					const active = hover === n.id;
					const card = (
						<div
							onMouseEnter={() => setHover(n.id)}
							onMouseLeave={() => setHover((h) => (h === n.id ? null : h))}
							className={`absolute flex items-center gap-2 rounded-xl bg-white px-3 shadow-sm ring-1 transition-shadow dark:bg-zinc-900 ${
								n.focused
									? 'ring-2 ring-[#76b900]'
									: active
										? 'ring-[#76b900]'
										: style.ring
							} ${n.href ? 'cursor-pointer hover:shadow-md' : ''}`}
							style={{
								left: n.x,
								top: n.y,
								width: NODE_W,
								height: NODE_H,
							}}
						>
							<span
								className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${style.chip}`}
							>
								<NodeIcon kind={n.kind} />
							</span>
							<span className="min-w-0">
								<span className="block truncate text-sm font-medium text-zinc-900 dark:text-zinc-100">
									{n.label}
								</span>
								{n.sublabel ? (
									<span className="block truncate text-[11px] text-zinc-500 dark:text-zinc-400">
										{n.sublabel}
									</span>
								) : null}
							</span>
							{n.badge ? (
								<span className="ml-auto shrink-0 rounded-full bg-amber-100 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-amber-800 dark:bg-amber-950/60 dark:text-amber-200">
									{n.badge}
								</span>
							) : null}
						</div>
					);
					return n.href ? (
						<Link key={n.id} href={n.href} className="contents">
							{card}
						</Link>
					) : (
						<div key={n.id} className="contents">
							{card}
						</div>
					);
				})}
			</div>
		</div>
	);
};
