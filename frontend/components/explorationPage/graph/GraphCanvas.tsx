// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef } from 'react';
import { UndirectedGraph } from 'graphology';
import Sigma from 'sigma';
import type { EdgeDisplayData, MouseCoords, NodeDisplayData } from 'sigma/types';
import { forceSimulation, forceManyBody, forceLink, forceCollide, forceX, forceY } from 'd3-force';
import type { SimulationNodeDatum, SimulationLinkDatum } from 'd3-force';
import { createNodeBorderProgram } from '@sigma/node-border';
import { createNodeCompoundProgram, NodeCircleProgram } from 'sigma/rendering';
import type {
	NodeHoverDrawingFunction,
	NodeLabelDrawingFunction,
	NodeProgramType,
} from 'sigma/rendering';

import { ExplorationLayer } from '@/enums/exploration';
import type { ExplorationGraph } from '@/types/exploration';
import { NODE_TYPE_ACCENT_COLOR } from './nodeTypeColors';
import type { NodeType } from './nodeTypeColors';

export type HoveredNode = {
	id: string;
	x: number;
	y: number;
};

// A node's border reuses its own icon accent color (rather than a single
// neutral outline for every kind) so the ring reads as "this node's own
// color, just more saturated" instead of a generic UI chrome line — the same
// pastel-fill/vivid-accent pairing Neo4j Browser uses for its node styling.
// `NODE_TYPE_ACCENT_COLOR` lives in `nodeTypeColors.ts` (rather than here)
// so `ExplorationView.tsx`'s own "Viewing: ..." legend can color its
// per-kind swatches with these exact same values too, without a
// value-level import of this file — see that module's own doc comment.
const NODE_TYPE_BORDER_COLOR = NODE_TYPE_ACCENT_COLOR;

// A fixed screen-pixel width (via the border program's `mode: 'pixels'`
// below) rather than a fraction of the node's radius, so the ring reads the
// same crisp thickness on a small Column node and a large hub Table alike.
const NODE_BORDER_WIDTH = 2;

// A single, unfilled ring read off each node's own `borderColor` attribute.
// With no `fill: true` entry, `@sigma/node-border`'s shader leaves
// everything inside the ring untouched — so compounding it *after*
// `NodeCircleProgram` below only draws a thin accent stroke around the
// already-drawn disc, rather than replacing it.
const NodeBorderProgram = createNodeBorderProgram({
	borders: [
		{ size: { value: NODE_BORDER_WIDTH, mode: 'pixels' }, color: { attribute: 'borderColor' } },
	],
});

// One compound program so both draws still happen under the single `circle`
// node type every node already carries — no per-node type juggling needed
// elsewhere (reducers, `addExpansion`, hover program overrides, etc). Nodes
// used to also carry a per-kind SVG icon (via `@sigma/node-image`) baked
// into this same disc — dropped so a kind's icon only ever appears once, in
// `ActiveExpansionCard`'s own header, rather than duplicated on every node
// on the canvas too.
const NodeCircleBorderProgram = createNodeCompoundProgram([NodeCircleProgram, NodeBorderProgram]);

// Sigma always redraws whichever node is currently under the mouse a
// second time, on its own WebGL layer stacked *above* the one captions are
// drawn on — purely so a hovered node visually sits in front of any
// overlapping siblings. For Sigma's own default label placement (beside the
// node) that second draw doesn't matter, but `drawNodeLabel` below instead
// draws each caption centered *inside* its own node's circle (à la Neo4j
// Browser) — so redrawing that same opaque disc a second time, on a layer
// above the one the caption was just drawn on, blots it right back out, in
// the exact place users look right when they mouse over a node. Bringing
// hovered/active nodes to the front is already handled by the `zIndex`
// reducer output below (sorted within the *normal* node layer), so this
// second draw is redundant for us — swapping in a program that renders
// nothing keeps the caption visible under the cursor instead of losing it.
class NoopNodeProgram {
	drawLabel: NodeLabelDrawingFunction | undefined;

	drawHover: NodeHoverDrawingFunction | undefined;

	kill(): void {
		// Nothing was ever allocated for this no-op program.
	}

	reallocate(): void {
		// Nothing to allocate — see the class comment above.
	}

	process(): void {
		// Nothing to process — see the class comment above.
	}

	render(): void {
		// Nothing to render — see the class comment above.
	}
}

// The canvas is plain WebGL/Canvas2D, not DOM, so it can't pick up Tailwind's
// `dark:` variants — node/label colors have to be swapped by hand based on
// the OS-level color scheme instead. Light values are pale tints of each
// type's own `NODE_TYPE_ACCENT_COLOR` hue (the same pastel-fill/vivid-accent
// node styling Neo4j Browser uses, just with this file's own more evenly
// spread hues — see that module's own doc comment); dark values are muted
// tints of the same hues (rather than that same near-white pastel) so nodes
// read as colored shapes instead of glowing white blobs against a black
// canvas.
const NODE_TYPE_COLOR_LIGHT: Record<NodeType, string> = {
	term: '#daf5c7',
	table: '#fae3c2',
	schema: '#f7c5e6',
	column: '#e1caf2',
	columnAttribute: '#caf2d4',
	sqlAttribute: '#c7cef5',
	sql: '#f8c3c3',
	customAnalysis: '#c5f3f7',
};
const NODE_TYPE_COLOR_DARK: Record<NodeType, string> = {
	term: '#293a1d',
	table: '#3e2e19',
	schema: '#3b1c31',
	column: '#2d2037',
	columnAttribute: '#203725',
	sqlAttribute: '#1d223a',
	sql: '#3c1a1a',
	customAnalysis: '#1c383b',
};
const LABEL_COLOR_LIGHT = '#3f3f46';
const LABEL_COLOR_DARK = '#e4e4e7';

// Matches the `bg-zinc-50`/`dark:bg-zinc-950` canvas backdrop set by
// `ExplorationData.tsx`, so dimmed nodes below can be pre-mixed toward it.
const CANVAS_BACKGROUND_LIGHT = '#fafafa';
const CANVAS_BACKGROUND_DARK = '#09090b';

// Matches the light/dark surface colors used by HoverNodeCard and other
// panels (`bg-white` / `dark:bg-zinc-900`), so Sigma's own hover label
// background reads as an intentional themed surface instead of the
// library's hardcoded white pill (which swallowed the light label text
// used in dark mode above).
const HOVER_LABEL_BACKGROUND_LIGHT = '#ffffff';
const HOVER_LABEL_BACKGROUND_DARK = '#18181b';

const prefersDarkMode = () =>
	typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches;

const EDGE_BRAND_COLOR = '#76b900';

const hexToRgb = (hex: string) => {
	const value = hex.replace('#', '');
	return [
		parseInt(value.substring(0, 2), 16),
		parseInt(value.substring(2, 4), 16),
		parseInt(value.substring(4, 6), 16),
	] as const;
};

/** Converts a `#rrggbb` color into an `rgba(...)` string at the given alpha. */
const withAlpha = (hex: string, alpha: number) => {
	const [r, g, b] = hexToRgb(hex);
	return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

// Sigma's own node programs read a plain (non-`rgba`) hex `color` attribute,
// so a dimmed node needs its own pre-mixed hex color rather than a
// translucent `rgba(...)` layered over whatever's behind it — the approach
// that works fine for edges below, which really are drawn as translucent
// strokes over one another. Pre-mixing toward the canvas background as a
// *flat, fully-opaque* color instead means dimmed nodes end up looking like
// an honest faded copy of the active ones rather than blending toward white.
const mixTowardColor = (hex: string, target: string, opacity: number) => {
	const [r1, g1, b1] = hexToRgb(hex);
	const [r2, g2, b2] = hexToRgb(target);
	const mix = (a: number, b: number) =>
		Math.round(a * opacity + b * (1 - opacity))
			.toString(16)
			.padStart(2, '0');
	return `#${mix(r1, r2)}${mix(g1, g2)}${mix(b1, b2)}`;
};

// How much of the original color survives in a dimmed node — applied
// uniformly to both the disc and its border (see
// `NODE_TYPE_BORDER_COLOR_DIMMED_LIGHT`/`_DARK` below) so a dimmed node
// reads as the exact same node, just faded. Kept low so active/connected
// nodes (drawn at full color) stand out clearly against the dimmed ones
// rather than reading as a similar, only-slightly-lighter hue.
const NODE_DIMMED_OPACITY = 0.22;

const NODE_TYPE_COLOR_DIMMED_LIGHT: Record<NodeType, string> = {
	term: mixTowardColor(NODE_TYPE_COLOR_LIGHT.term, CANVAS_BACKGROUND_LIGHT, NODE_DIMMED_OPACITY),
	table: mixTowardColor(
		NODE_TYPE_COLOR_LIGHT.table,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	schema: mixTowardColor(
		NODE_TYPE_COLOR_LIGHT.schema,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	column: mixTowardColor(
		NODE_TYPE_COLOR_LIGHT.column,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	columnAttribute: mixTowardColor(
		NODE_TYPE_COLOR_LIGHT.columnAttribute,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	sqlAttribute: mixTowardColor(
		NODE_TYPE_COLOR_LIGHT.sqlAttribute,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	sql: mixTowardColor(NODE_TYPE_COLOR_LIGHT.sql, CANVAS_BACKGROUND_LIGHT, NODE_DIMMED_OPACITY),
	customAnalysis: mixTowardColor(
		NODE_TYPE_COLOR_LIGHT.customAnalysis,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
};
const NODE_TYPE_COLOR_DIMMED_DARK: Record<NodeType, string> = {
	term: mixTowardColor(NODE_TYPE_COLOR_DARK.term, CANVAS_BACKGROUND_DARK, NODE_DIMMED_OPACITY),
	table: mixTowardColor(NODE_TYPE_COLOR_DARK.table, CANVAS_BACKGROUND_DARK, NODE_DIMMED_OPACITY),
	schema: mixTowardColor(
		NODE_TYPE_COLOR_DARK.schema,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	column: mixTowardColor(
		NODE_TYPE_COLOR_DARK.column,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	columnAttribute: mixTowardColor(
		NODE_TYPE_COLOR_DARK.columnAttribute,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	sqlAttribute: mixTowardColor(
		NODE_TYPE_COLOR_DARK.sqlAttribute,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	sql: mixTowardColor(NODE_TYPE_COLOR_DARK.sql, CANVAS_BACKGROUND_DARK, NODE_DIMMED_OPACITY),
	customAnalysis: mixTowardColor(
		NODE_TYPE_COLOR_DARK.customAnalysis,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
};

// Faded the same way as `NODE_TYPE_COLOR_DIMMED_LIGHT`/`_DARK` above so a
// dimmed node's ring fades in step with its disc.
const NODE_TYPE_BORDER_COLOR_DIMMED_LIGHT: Record<NodeType, string> = {
	term: mixTowardColor(NODE_TYPE_BORDER_COLOR.term, CANVAS_BACKGROUND_LIGHT, NODE_DIMMED_OPACITY),
	table: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.table,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	schema: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.schema,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	column: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.column,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	columnAttribute: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.columnAttribute,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	sqlAttribute: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.sqlAttribute,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
	sql: mixTowardColor(NODE_TYPE_BORDER_COLOR.sql, CANVAS_BACKGROUND_LIGHT, NODE_DIMMED_OPACITY),
	customAnalysis: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.customAnalysis,
		CANVAS_BACKGROUND_LIGHT,
		NODE_DIMMED_OPACITY,
	),
};
const NODE_TYPE_BORDER_COLOR_DIMMED_DARK: Record<NodeType, string> = {
	term: mixTowardColor(NODE_TYPE_BORDER_COLOR.term, CANVAS_BACKGROUND_DARK, NODE_DIMMED_OPACITY),
	table: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.table,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	schema: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.schema,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	column: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.column,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	columnAttribute: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.columnAttribute,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	sqlAttribute: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.sqlAttribute,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
	sql: mixTowardColor(NODE_TYPE_BORDER_COLOR.sql, CANVAS_BACKGROUND_DARK, NODE_DIMMED_OPACITY),
	customAnalysis: mixTowardColor(
		NODE_TYPE_BORDER_COLOR.customAnalysis,
		CANVAS_BACKGROUND_DARK,
		NODE_DIMMED_OPACITY,
	),
};

// A translucent edge that reads fine against a near-black dark canvas turns
// almost invisible against a near-white light one at the same alpha (green
// desaturated toward gray blends into `bg-zinc-50`) — so each theme gets its
// own alpha rather than sharing one. Edges are thin strokes rather than
// filled shapes, so (unlike nodes above) letting them fade via real alpha
// reads fine.
//
// Every edge — active/connected/hovered or not, and on both the Terms and
// Tables layers, since they share this same code — uses this *one* alpha,
// with only line thickness (see `edgeReducer` below) signalling emphasis.
// A separate, darker/near-opaque alpha for connected/hovered edges used to
// make the brand green look like two different shades of green depending
// on state; keeping a single alpha avoids that "some edges are darker
// green than others" inconsistency.
const EDGE_ALPHA_LIGHT = 0.85;
const EDGE_ALPHA_DARK = 0.7;

// Dimmed edges (unrelated to the active node) used real alpha too, but at a
// low enough value that the brand green desaturated almost entirely into
// the light-mode canvas background — some edges read green, others read
// plain white depending on what happened to be layered underneath. Mixing
// toward the canvas background as a flat, opaque color (same trick used for
// dimmed nodes above) guarantees every edge keeps a visibly green tint.
const EDGE_DIMMED_OPACITY_LIGHT = 0.6;
const EDGE_DIMMED_OPACITY_DARK = 0.5;

const getEdgeColors = (isDark: boolean) => {
	const alpha = isDark ? EDGE_ALPHA_DARK : EDGE_ALPHA_LIGHT;
	const canvasBackground = isDark ? CANVAS_BACKGROUND_DARK : CANVAS_BACKGROUND_LIGHT;
	const dimmedOpacity = isDark ? EDGE_DIMMED_OPACITY_DARK : EDGE_DIMMED_OPACITY_LIGHT;
	return {
		base: withAlpha(EDGE_BRAND_COLOR, alpha),
		dimmed: mixTowardColor(EDGE_BRAND_COLOR, canvasBackground, dimmedOpacity),
	};
};

// How much closer than the "whole graph" fit the camera starts by default —
// applied once, right after the initial fit/center, so the first thing a
// user sees is already legibly zoomed in rather than the entire (often
// sparse-looking) graph shrunk to fit the container.
const INITIAL_ZOOM_IN_FACTOR = 1.5;

// How close the camera is ever allowed to zoom in — the `minCameraRatio`
// Sigma setting below.
const MIN_CAMERA_RATIO = 0.2;

// Node radii (half of the previous Cytoscape diameters, since Sigma sizes are radii).
const getNodeRadius = (relationshipCount: number) => {
	if (relationshipCount >= 8) return 50;
	if (relationshipCount >= 4) return 40;
	if (relationshipCount >= 1) return 30;
	return 20;
};

// Schema/column/columnAttribute/sqlAttribute/sql/customAnalysis nodes have
// no `relationshipCount` of their own (they're structural/leaf, not
// analytical), so they get one fixed size each instead of scaling like
// tables/terms do — small enough that a table's whole column list (or a
// term's whole attribute list) doesn't dominate the canvas once expanded.
const NODE_TYPE_FIXED_SIZE: Partial<Record<NodeType, number>> = {
	schema: 26,
	column: 14,
	columnAttribute: 14,
	sqlAttribute: 14,
	sql: 14,
	customAnalysis: 14,
};

const getNodeSize = (kind: NodeType, relationshipCount: number) =>
	NODE_TYPE_FIXED_SIZE[kind] ?? getNodeRadius(relationshipCount);

/** Attributes stored on every Sigma/graphology node for the Exploration graph. */
type GraphNodeAttributes = {
	x: number;
	y: number;
	size: number;
	label: string;
	color: string;
	type: 'circle';
	kind: NodeType;
	/** Read by `NodeBorderProgram` (see `NodeCircleBorderProgram` above). */
	borderColor: string;
};

/** Attributes stored on every Sigma/graphology edge for the Exploration graph. */
type GraphEdgeAttributes = {
	color: string;
	size: number;
	/**
	 * `relationship` edges come from the base graph payload (shared SQL/FK
	 * between tables, or shared-table between terms) and are clickable —
	 * selecting one shows the real hop chain behind it (see
	 * `ExplorationView`'s `highlightedPath`) rather than naming the
	 * relationship type(s) on the edge itself. `structural` edges are added
	 * by `addExpansion` (table→schema, table→column, table→term) and are
	 * click-inert.
	 */
	kind: 'relationship' | 'structural';
};

/** A d3-force particle mirroring one graphology node, kept in sync by ID. */
type SimNode = SimulationNodeDatum & { id: string; size: number };
type SimLink = SimulationLinkDatum<SimNode>;

/**
 * One node to graft onto the live graph from `GraphController.addExpansion`.
 * `kind` can be any `NodeType` — a Table expansion only ever grafts on
 * `schema`/`column`/`term` nodes, but a Term expansion (see `expandTermNode`
 * in `ExplorationView.tsx`) grafts on the `table`/`term` nodes it links to
 * (some of which may already be permanent nodes on the base graph — see
 * `GraphController`'s own doc comment for how those are told apart) plus
 * its own `columnAttribute`/`sqlAttribute` nodes.
 */
export type ExpansionNodeInput = {
	id: string;
	kind: NodeType;
	label: string;
};

/** One edge to graft onto the live graph from `GraphController.addExpansion`. */
export type ExpansionEdgeInput = {
	source: string;
	target: string;
};

/**
 * Imperative handle shared with `ZoomControls`/`ExplorationView`. `addExpansion`
 * mutates the live graphology graph and d3-force simulation directly instead of
 * going through the `graph` prop, so growing the graph on a node click doesn't
 * replay the full teardown/rebuild (and its fade-out/re-scatter/fade-in intro)
 * that a `graph` prop change triggers below.
 */
export type GraphController = {
	getCamera: () => ReturnType<Sigma<GraphNodeAttributes, GraphEdgeAttributes>['getCamera']>;
	refresh: () => void;
	/**
	 * Re-derives the frozen graph-to-viewport mapping (see `setCustomBBox`
	 * below) from every node's *current* position — including any node
	 * dragged/pinned outside the last-frozen box — then re-renders. Plain
	 * `refresh()` above only reprocesses/re-renders against whatever box is
	 * already frozen; it never touches that box itself, so it can't bring a
	 * node dragged past it back into frame. This is what `ZoomControls`'
	 * "reset view" actually needs to recover one.
	 */
	resetExtent: () => void;
	/**
	 * Pans/zooms the camera to bring the given node into view when it's
	 * currently off-screen (a no-op when it's already visible) — the same
	 * "only move the camera when it actually needs to" behavior
	 * `addExpansion` gets from `focusExpansionIfOffscreen` below, exposed
	 * standalone for callers that select a node without expanding it (e.g.
	 * picking a result from the search dropdown or a `?semanticId=`/
	 * `?dataId=` deep link).
	 */
	focusNode: (nodeId: string) => void;
	/**
	 * Also pans/zooms the camera to bring the origin and its newly-connected
	 * nodes into view when any of them land off-screen — see
	 * `focusExpansionIfOffscreen`'s own comment below for why that matters
	 * most for a Term's expansion onto Tables elsewhere in the base graph.
	 * A no-op camera-wise when everything's already visible.
	 */
	addExpansion: (
		originNodeId: string,
		nodes: ExpansionNodeInput[],
		edges: ExpansionEdgeInput[],
	) => void;
	/**
	 * Reverses one origin's `addExpansion` call (a double click collapsing a
	 * table or term back down). Nodes/edges shared with another still-live
	 * expansion (e.g. a Schema or Term reachable from two expanded tables, or
	 * a structural edge two different expansions both grafted) are kept —
	 * only removed once every origin referencing them has been collapsed.
	 * That includes the origin node itself: if it has its own still-live
	 * expansion rooted on it (e.g. a Column grafted by a Table's expansion,
	 * then separately double-clicked to graft its own children), it stays
	 * put — collapsing whichever expansion first grafted it on must never
	 * orphan its own children. A node that was already part of the *base*
	 * graph when this origin expanded onto it (e.g. a Term expanding onto
	 * one of its tables that's already on the graph) is never removed at
	 * all, no matter how many origins referenced it — only nodes an
	 * expansion itself created are ever dropped. Returns the node ids that
	 * were actually dropped from the graph (which may include the origin
	 * itself) so callers can prune their own id → entity lookups, or `null`
	 * if this origin has no recorded expansion yet (e.g. its `addExpansion`
	 * fetch is still in flight) — callers use that to know a pending expand
	 * should be cancelled instead of grafted on once it resolves.
	 */
	removeExpansion: (originNodeId: string) => string[] | null;
};

const buildGraphologyGraph = (
	graph: ExplorationGraph,
	nodeTypeColor: Record<NodeType, string>,
	edgeColor: string,
	containerAspectRatio: number,
) => {
	const graphology = new UndirectedGraph<GraphNodeAttributes, GraphEdgeAttributes>();

	graph.nodes.forEach((node) => {
		const kind: NodeType = node.layer === ExplorationLayer.Data ? 'table' : 'term';
		graphology.addNode(node.id, {
			x: 0,
			y: 0,
			size: getNodeSize(kind, node.relationshipCount),
			label: node.layer === ExplorationLayer.Data ? node.name.toUpperCase() : node.name,
			color: nodeTypeColor[kind],
			type: 'circle',
			kind,
			borderColor: NODE_TYPE_BORDER_COLOR[kind],
		});
	});

	graph.links.forEach((link) => {
		if (!graphology.hasNode(link.source) || !graphology.hasNode(link.target)) return;
		if (link.source === link.target || graphology.hasEdge(link.source, link.target)) return;
		// A Data-layer (Table↔Table) edge used to open its own `LinkPathCard`
		// with the SQL query/foreign key behind it — no longer wanted, so
		// these are drawn as plain `structural` edges instead: click-inert,
		// so hover keeps the canvas `grab` cursor rather than `pointer` (see
		// `syncCanvasCursor`). A Semantic-layer (Term↔Term) edge still
		// opens its own `connection` card, so it keeps the `relationship` kind.
		const isDataLayerEdge = graphology.getNodeAttribute(link.source, 'kind') === 'table';
		graphology.addEdgeWithKey(`${link.source}:${link.target}`, link.source, link.target, {
			color: edgeColor,
			size: 1.5,
			kind: isDataLayerEdge ? 'structural' : 'relationship',
		});
	});

	// Nodes just need *some* non-zero, non-overlapping starting position
	// before the force simulation below can take over; a plain random
	// scatter (rather than a perfect circle) is enough, since d3-force
	// doesn't need — and shouldn't get — a pre-arranged shape to relax away
	// from. Sigma auto-fits whatever bounding box the nodes occupy to the
	// container on every frame, but it preserves aspect ratio rather than
	// stretching — so a square-ish scatter inside a wide container would
	// only fill the height, leaving big empty margins on the sides instead
	// of the "spread across the whole page" look from before the Sigma
	// migration. Shaping the scatter after the container's own aspect ratio
	// avoids that from the very first frame, before physics has even had a
	// chance to run.
	const area = Math.max(200 * 200, graphology.order * 6000);
	const scatterHeight = Math.sqrt(area / containerAspectRatio);
	const scatterWidth = scatterHeight * containerAspectRatio;
	graphology.forEachNode((node) => {
		graphology.mergeNodeAttributes(node, {
			x: (Math.random() - 0.5) * scatterWidth,
			y: (Math.random() - 0.5) * scatterHeight,
		});
	});

	return graphology;
};

/**
 * Node/edge ids to keep at full color while everything else on the canvas
 * dims — the same visual treatment `nodeReducer`/`edgeReducer` already give
 * an active node's neighbourhood, generalized so selecting an edge (see
 * `ExplorationView`'s `highlightedPath`) can highlight the real hop chain
 * behind it instead. Takes precedence over `activeNodeId`'s own dimming
 * while set.
 */
export type HighlightedPath = {
	nodeIds: string[];
	/** Edge keys in the same `${source}:${target}` form used everywhere else. */
	edgeKeys: string[];
};

type GraphCanvasProps = {
	graph: ExplorationGraph;
	activeNodeId: string | null;
	highlightedPath: HighlightedPath | null;
	/** Clicking empty canvas — deselects whatever node/edge is active. */
	onSelectNode: (nodeId: string | null) => void;
	onSelectEdge: (edgeId: string) => void;
	/** Single-clicking a node — selects/opens it (its side panel), without
	 * touching its expansion state. Sigma still tells this apart from a
	 * node drag (see `draggedEventsTolerance` in its own mouse captor), so
	 * this doesn't fire while repositioning a node. */
	onClickNode: (nodeId: string) => void;
	/** Double-clicking a node — toggles expanding/collapsing a table, or a
	 * term grafted onto the graph by expanding one, on top of whatever
	 * `onClickNode` already did for the click that started this double
	 * click. */
	onDoubleClickNode: (nodeId: string) => void;
	onHoverNode: (hoveredNode: HoveredNode | null) => void;
	onControllerChange: (controller: GraphController | null) => void;
};

/** Sigma.js-backed graph canvas shared by both the semantic and data Exploration layers. */
export const GraphCanvas = ({
	graph,
	activeNodeId,
	highlightedPath,
	onSelectNode,
	onSelectEdge,
	onClickNode,
	onDoubleClickNode,
	onHoverNode,
	onControllerChange,
}: GraphCanvasProps) => {
	const containerRef = useRef<HTMLDivElement>(null);
	const activeNodeIdRef = useRef<string | null>(activeNodeId);
	const highlightedPathRef = useRef<{ nodeIds: Set<string>; edgeKeys: Set<string> } | null>(null);
	// The renderer itself, mirrored out of the big mount effect below so
	// these two small prop-sync effects can force a repaint right after
	// updating their ref. The node/edge reducers only ever read
	// `activeNodeIdRef`/`highlightedPathRef` — never the `activeNodeId`/
	// `highlightedPath` props directly — so without an explicit `refresh()`
	// here, a selection made while the force simulation has already
	// settled (alpha decayed to 0, no more per-frame repaints) would sync
	// the ref but never actually redraw, leaving whichever node was
	// active/highlighted *before* stuck looking selected until some
	// unrelated repaint happens to come along.
	const rendererRef = useRef<Sigma<GraphNodeAttributes, GraphEdgeAttributes> | null>(null);

	useEffect(() => {
		activeNodeIdRef.current = activeNodeId;
		rendererRef.current?.refresh();
	}, [activeNodeId]);

	useEffect(() => {
		highlightedPathRef.current =
			highlightedPath == null
				? null
				: {
						nodeIds: new Set(highlightedPath.nodeIds),
						edgeKeys: new Set(highlightedPath.edgeKeys),
					};
		rendererRef.current?.refresh();
	}, [highlightedPath]);

	useEffect(() => {
		if (containerRef.current == null) return undefined;

		let isDark = prefersDarkMode();
		let edgeColors = getEdgeColors(isDark);
		const containerRect = containerRef.current.getBoundingClientRect();
		const containerAspectRatio =
			containerRect.height > 0 ? containerRect.width / containerRect.height : 1;
		const graphology = buildGraphologyGraph(
			graph,
			isDark ? NODE_TYPE_COLOR_DARK : NODE_TYPE_COLOR_LIGHT,
			edgeColors.base,
			containerAspectRatio,
		);
		// Permanent nodes from the base graph payload — an expansion (see
		// `addExpansion` below) can graft an edge onto one of these (e.g. a
		// Term expanding onto a Table already drawn from `graph.nodes`), but
		// must never ref-count or remove it: it stays on the graph regardless
		// of any expansion's own collapsed/expanded state.
		const baseNodeIds = new Set(graph.nodes.map((node) => node.id));
		const hoveredNodeIdRef = { current: null as string | null };
		const hoveredEdgeIdRef = { current: null as string | null };

		// Sigma's own `drawDiscNodeHover` (and our previous reimplementation
		// of it) drew the node's name a second time, in a side pill — right
		// next to the exact same caption `drawNodeLabel` below already draws
		// centered inside the node once zoomed in, and redundant with the
		// separate `HoverNodeCard`/active-node panel shown in the DOM for
		// hovered/selected nodes. A soft shadow ring is enough to mark a
		// node as hovered/active without duplicating its name on screen.
		const drawNodeHover: NodeHoverDrawingFunction<GraphNodeAttributes, GraphEdgeAttributes> = (
			context,
			data,
		) => {
			context.shadowOffsetX = 0;
			context.shadowOffsetY = 0;
			context.shadowBlur = 8;
			context.shadowColor = '#000';
			context.strokeStyle = isDark
				? HOVER_LABEL_BACKGROUND_DARK
				: HOVER_LABEL_BACKGROUND_LIGHT;
			context.lineWidth = 2;
			context.beginPath();
			context.arc(data.x, data.y, data.size, 0, Math.PI * 2);
			context.stroke();
			context.shadowOffsetX = 0;
			context.shadowOffsetY = 0;
			context.shadowBlur = 0;
		};

		// The persistent (non-hover) caption: always drawn, centered *inside*
		// the node's own circle (à la Neo4j Browser) rather than Sigma's
		// default placement to the right of it. A stroked halo behind the
		// fill keeps the caption legible over the node's own color and icon
		// artwork underneath, the same trick used for text labels on maps.
		const drawNodeLabel: NodeLabelDrawingFunction<GraphNodeAttributes, GraphEdgeAttributes> = (
			context,
			data,
			settings,
		) => {
			if (!data.label) return;

			const { labelFont: font, labelWeight: weight } = settings;
			// Scaled to the node's own on-screen radius so captions on the
			// smallest nodes don't dwarf their circle, while bigger hub nodes
			// get a caption that's actually easy to read.
			const fontSize = Math.max(9, Math.min(13, Math.round(data.size * 0.4)));
			context.font = `${weight} ${fontSize}px ${font}`;
			context.textAlign = 'center';
			context.textBaseline = 'middle';

			// Truncated with an ellipsis instead of letting long names spill
			// past the node's own circle, mirroring how Neo4j clips captions
			// that don't fit inside their node.
			const maxWidth = data.size * 1.6;
			let { label } = data;
			if (context.measureText(label).width > maxWidth) {
				while (label.length > 1 && context.measureText(`${label}…`).width > maxWidth) {
					label = label.slice(0, -1);
				}
				label = `${label}…`;
			}

			context.lineJoin = 'round';
			context.lineWidth = 3;
			context.strokeStyle = isDark ? '#000000' : '#ffffff';
			context.strokeText(label, data.x, data.y);
			context.fillStyle = isDark ? LABEL_COLOR_DARK : LABEL_COLOR_LIGHT;
			context.fillText(label, data.x, data.y);
		};

		// Re-evaluated every frame by the continuously running physics layout
		// below, so selection/hover state changes surface within one frame
		// without needing to force a manual re-render.
		// Sigma's own `NodeDisplayData` type doesn't know about the
		// `borderColor` attribute `NodeBorderProgram` reads off the merged
		// node data at render time, so the reducer's return type has to be
		// widened past what `Sigma.Settings['nodeReducer']` declares in
		// order to swap it per-node below.
		const nodeReducer = (
			node: string,
			data: GraphNodeAttributes,
		): Partial<NodeDisplayData> & Pick<GraphNodeAttributes, 'borderColor'> => {
			const currentActiveNodeId = activeNodeIdRef.current;
			const isActive = currentActiveNodeId === node;
			const isHovered = hoveredNodeIdRef.current === node;
			const currentHighlightedPath = highlightedPathRef.current;
			// A highlighted path (see `HighlightedPath`) takes over dimming
			// entirely while set — `ExplorationView` clears `activeNodeId`
			// whenever it sets one, so the two never actually compete, but
			// checking it first keeps that an implementation detail of this
			// component rather than something callers have to guarantee.
			// `currentActiveNodeId` is a React-state value mirrored into this
			// ref by an effect one render behind — if the active node is a
			// Schema/Column/Term that `removeExpansion` just dropped from the
			// graph (e.g. collapsing its origin table drops it too, in the
			// same synchronous click handler that's also re-selecting the
			// table), this ref can transiently point at an id that no longer
			// exists. `areNeighbors` throws on an unknown node, so guard with
			// `hasNode` first rather than let one stale frame crash Sigma.
			const isDimmed =
				currentHighlightedPath != null
					? !currentHighlightedPath.nodeIds.has(node)
					: currentActiveNodeId != null &&
						graphology.hasNode(currentActiveNodeId) &&
						!isActive &&
						!graphology.areNeighbors(currentActiveNodeId, node);

			const nodeTypeColorDimmed = isDark
				? NODE_TYPE_COLOR_DIMMED_DARK
				: NODE_TYPE_COLOR_DIMMED_LIGHT;
			const nodeTypeBorderColorDimmed = isDark
				? NODE_TYPE_BORDER_COLOR_DIMMED_DARK
				: NODE_TYPE_BORDER_COLOR_DIMMED_LIGHT;

			return {
				...data,
				color: isDimmed ? nodeTypeColorDimmed[data.kind] : data.color,
				borderColor: isDimmed
					? nodeTypeBorderColorDimmed[data.kind]
					: NODE_TYPE_BORDER_COLOR[data.kind],
				zIndex: isActive || isHovered ? 1 : 0,
				highlighted: isActive || isHovered,
			};
		};

		const edgeReducer = (edge: string, data: GraphEdgeAttributes): Partial<EdgeDisplayData> => {
			const currentActiveNodeId = activeNodeIdRef.current;
			const currentHighlightedPath = highlightedPathRef.current;
			const isHovered = hoveredEdgeIdRef.current === edge;
			const isConnected =
				currentHighlightedPath != null
					? currentHighlightedPath.edgeKeys.has(edge)
					: currentActiveNodeId != null &&
						graphology.extremities(edge).includes(currentActiveNodeId);
			const isDimmed =
				currentHighlightedPath != null
					? !isConnected
					: currentActiveNodeId != null && !isConnected;
			const isStructural = data.kind === 'structural';

			// Emphasis for connected/hovered edges comes only from thickness now —
			// they stay the exact same color as every other edge, just drawn
			// wider, instead of also switching to a darker/more-opaque shade.
			// Structural edges get a subtler bump than relationship edges so
			// they still read as the "quieter" kind of link even when active,
			// even though both kinds now share the same brand-green color.
			let size = data.size;
			if (isHovered) size = isStructural ? 2.5 : 3.5;
			else if (isConnected) size = isStructural ? 1.5 : 2.5;

			const color = isDimmed ? edgeColors.dimmed : edgeColors.base;

			return { ...data, color, size };
		};

		const renderer = new Sigma<GraphNodeAttributes, GraphEdgeAttributes>(
			graphology,
			containerRef.current,
			{
				// "screen" (the default) still makes nodes grow when zooming in
				// (scaled by camera ratio), but keeps their pixel size decoupled
				// from the graph's raw coordinate spread. With "positions" the
				// on-screen size is scaled by how spread out the *whole* graph
				// currently is, which constantly shifts while the physics runs
				// forever — nodes would balloon or shrink to invisible dots as
				// the simulation's bounding box changes, independently of zoom.
				itemSizesReference: 'screen',
				minCameraRatio: MIN_CAMERA_RATIO,
				maxCameraRatio: 5,
				// The container can briefly report a zero size during layer
				// switches/route transitions before the surrounding flex layout
				// settles; falling back to a 1px size for that one frame avoids a
				// hard crash instead of forcing every consumer to guard against it.
				allowInvalidContainer: true,
				enableEdgeEvents: true,
				zIndex: true,
				labelColor: { color: isDark ? LABEL_COLOR_DARK : LABEL_COLOR_LIGHT },
				// Sigma normally shows only as many labels per screen area as fit
				// without overlapping (its density-based label grid), which at the
				// default density hides most node captions until zoomed in close.
				// Cranked up so every node's caption is a candidate regardless of
				// on-screen crowding — `drawNodeLabel` below always draws once
				// selected, so this is what actually makes "all titles visible by
				// default" happen.
				labelDensity: 100,
				defaultDrawNodeHover: drawNodeHover,
				defaultDrawNodeLabel: drawNodeLabel,
				// Sigma's own program classes are typed generically over the
				// default `Attributes` type; our stricter node attributes are a
				// compatible subtype at runtime, so this cast is safe.
				nodeProgramClasses: {
					circle: NodeCircleBorderProgram as unknown as NodeProgramType<
						GraphNodeAttributes,
						GraphEdgeAttributes
					>,
				},
				// See `NoopNodeProgram`'s own comment above for why hovered nodes
				// get a do-nothing program here instead of redrawing the disc.
				nodeHoverProgramClasses: {
					circle: NoopNodeProgram as unknown as NodeProgramType<
						GraphNodeAttributes,
						GraphEdgeAttributes
					>,
				},
				nodeReducer,
				edgeReducer,
			},
		);
		rendererRef.current = renderer;

		// Mirror the graphology graph as a d3-force simulation. This replaces
		// graphology-layout-force, whose repulsion never decays with distance
		// (it's a constant push per node pair, regardless of how far apart
		// they already are). That makes any group of nodes with similar
		// connectivity — e.g. dozens of terms all linked only to one shared
		// "User"/"Zone" node — settle at the *same* radius from their hub no
		// matter how the force constants are tuned, since nothing but the
		// (identical, quadratic-in-distance) attraction differs between them:
		// the whole graph reliably collapses onto one big ring.
		//
		// d3-force's `forceManyBody` charge decays with distance (Barnes-Hut
		// approximated), so only nodes that are *actually* pulled together by
		// real edges end up close, while unrelated nodes/clusters can settle
		// at very different distances — producing the same kind of organic,
		// clustered layout the previous Cytoscape/euler setup had.
		// `forceLink` also weakens its own pull automatically for high-degree
		// hub nodes, so one heavily-connected node doesn't flatten all its
		// neighbours onto a uniform circle either.
		const simNodesById = new Map<string, SimNode>();
		// `simNodes`/`simLinks` are reassigned (not just pushed to) by
		// `removeExpansion` below when collapsing a table drops nodes/edges, so
		// both are `let` rather than `const` — every reference to them here is
		// a closure over the binding, which continues to see later reassignments.
		let simNodes: SimNode[] = graphology.mapNodes((node, attributes) => {
			const simNode: SimNode = {
				id: node,
				size: attributes.size,
				x: attributes.x,
				y: attributes.y,
			};
			simNodesById.set(node, simNode);
			return simNode;
		});
		let simLinks: SimLink[] = graphology.mapEdges((_edge, _attributes, source, target) => ({
			source,
			target,
		}));

		// Kept as its own binding (rather than looked up via `simulation.force('link')`
		// later) so `addExpansion` below can re-feed it a grown `simLinks` array
		// without an `as` cast back to `ForceLink`.
		const linkForce = forceLink<SimNode, SimLink>(simLinks)
			.id((node) => node.id)
			.distance(220);

		// Bookkeeping for `addExpansion`/`removeExpansion`: how many *live*
		// expanded tables currently depend on a given expansion node (a Schema
		// or Term can be shared by more than one expanded table), and exactly
		// which node ids/edge keys each origin table's own `addExpansion` call
		// contributed — so collapsing one table only removes what it uniquely
		// added, leaving anything still shared with another expanded table in
		// place. `edgeRefCount` mirrors `nodeRefCount` for the same reason on
		// the edge side — e.g. a link-path graft and a Column's own expansion
		// can both want the exact same Table↔Column structural edge; whichever
		// grafted it *second* still needs it to survive should the first one
		// collapse.
		const nodeRefCount = new Map<string, number>();
		const edgeRefCount = new Map<string, number>();
		const expansionsByOrigin = new Map<string, { nodeIds: string[]; edgeKeys: string[] }>();
		const structuralEdgeKey = (source: string, target: string) => `${source}:${target}`;
		const simLinkEndpointId = (endpoint: SimLink['source'] | SimLink['target']): string =>
			typeof endpoint === 'string' ? endpoint : (endpoint as SimNode).id;

		// Whether the `'end'` handler below (physics naturally settling) is
		// allowed to re-freeze `customBBox`. Only ever armed right before a
		// `simulation...restart()` call that grows/shrinks the *graph itself*
		// (the initial layout, or `addExpansion`/`removeExpansion` below) —
		// never by a node drag. Dragging pins every node and stops the
		// simulation (see `handleDownNode`), so a drop has nothing new to
		// fit; re-freezing the box there would still hop the canvas if the
		// dragged node had been pulled past the last-frozen extent.
		let bboxFreezeArmed = true;

		// Isolated drag pins every existing node (see `handleDownNode`), so
		// filtering to unpinned nodes would after an expansion fit only the
		// handful of newly grafted children and zoom the canvas onto that
		// cluster. Always use the full live extent; `setCustomBBox` is not
		// applied during drag, so a node pulled off-frame still doesn't
		// rescale the canvas live. `resetExtent` remains the way to bring
		// it back.
		const computeStableBBox = (): { x: [number, number]; y: [number, number] } | null => {
			let minX = Infinity;
			let maxX = -Infinity;
			let minY = Infinity;
			let maxY = -Infinity;
			simNodes.forEach((node) => {
				if (node.x == null || node.y == null) return;
				minX = Math.min(minX, node.x);
				maxX = Math.max(maxX, node.x);
				minY = Math.min(minY, node.y);
				maxY = Math.max(maxY, node.y);
			});
			if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null;
			return { x: [minX, maxX], y: [minY, maxY] };
		};

		// `setCustomBBox` itself is a single synchronous jump — it swaps the
		// graph-to-framed-space mapping that *every* node's rendered position
		// derives from in one frame, so applying it directly on every re-arm
		// (an `addExpansion`/`removeExpansion`, or clicking a semantic edge —
		// which grafts the path via `addExpansion` under the hood) reads as
		// the whole graph abruptly hopping, the same jarring "shake" freezing
		// the box was meant to prevent in the first place. Tweening it over a
		// short animation — mirroring `focusExpansionIfOffscreen`'s own
		// camera animations below — turns that hop into a smooth zoom/pan
		// instead, while still keeping the mapping perfectly static (hence
		// jitter-free) the rest of the time, between animations.
		const BBOX_FREEZE_ANIMATION_DURATION_MS = 350;
		let bboxAnimationFrame: number | null = null;
		const animateCustomBBoxTo = (target: { x: [number, number]; y: [number, number] }) => {
			if (bboxAnimationFrame != null) cancelAnimationFrame(bboxAnimationFrame);
			const start = renderer.getCustomBBox();
			if (start == null) {
				renderer.setCustomBBox(target);
				return;
			}
			const startTime = performance.now();
			const lerp = (from: number, to: number, t: number) => from + (to - from) * t;
			const step = (now: number) => {
				const t = Math.min(1, (now - startTime) / BBOX_FREEZE_ANIMATION_DURATION_MS);
				// Quadratic ease-out, matching the "quadraticOut" easing Sigma's
				// own camera animations (e.g. `focusExpansionIfOffscreen`) use.
				const eased = 1 - (1 - t) * (1 - t);
				renderer.setCustomBBox({
					x: [lerp(start.x[0], target.x[0], eased), lerp(start.x[1], target.x[1], eased)],
					y: [lerp(start.y[0], target.y[0], eased), lerp(start.y[1], target.y[1], eased)],
				});
				bboxAnimationFrame = t < 1 ? requestAnimationFrame(step) : null;
			};
			bboxAnimationFrame = requestAnimationFrame(step);
		};

		// d3-force's own default `alphaDecay` (`1 - alphaMin^(1/300)`, ≈0.0228)
		// gives a reheat ~300 ticks (~4-5s) to fully cool — plenty of time for
		// the charge/collision forces to finish re-relaxing a newly-grown or
		// -shrunk graph after `addExpansion`/`removeExpansion` below.
		const DEFAULT_ALPHA_DECAY = 1 - 0.001 ** (1 / 300);

		const simulation = forceSimulation<SimNode>(simNodes)
			// More damping than d3-force's own default (0.4) so a reheated
			// simulation settles into place rather than visibly overshooting
			// and bouncing back a few times first.
			.velocityDecay(0.55)
			// Stronger repulsion (and a proportionally longer reach) plus a
			// wider collision margin than the link `distance` above alone
			// would give, so unconnected/loosely-connected nodes settle with
			// visible breathing room between them instead of packing in
			// tight, even in dense clusters.
			.force('charge', forceManyBody<SimNode>().strength(-700).distanceMax(1400))
			.force('link', linkForce)
			.force('collide', forceCollide<SimNode>((node) => node.size + 24).iterations(2))
			// A very weak pull toward the origin — just enough to stop the
			// whole graph drifting off-center over time, far too weak to
			// override the clustering forces above (that imbalance, gravity
			// dominating attraction, was what caused the ring artifact).
			//
			// Crucially, the x/y strengths are *not* equal: repulsion and
			// collision are radially symmetric, so an equal pull in both
			// axes always relaxes toward a roughly circular/square blob,
			// regardless of how wide the initial scatter was — leaving big
			// empty margins on the sides of a wide container. Weakening the
			// pull along the container's long axis (and strengthening it
			// along the short one) biases the settled shape itself into an
			// ellipse matching the container, so the layout actually uses
			// the full width instead of shrinking back to a centered blob.
			.force('x', forceX<SimNode>(0).strength(0.02 / containerAspectRatio))
			.force('y', forceY<SimNode>(0).strength(0.02 * containerAspectRatio))
			.on('tick', () => {
				graphology.updateEachNodeAttributes(
					(node, attributes) => {
						const simNode = simNodesById.get(node);
						if (simNode?.x == null || simNode.y == null) return attributes;
						return { ...attributes, x: simNode.x, y: simNode.y };
					},
					{ attributes: ['x', 'y'] },
				);
			})
			// Re-freezes the custom bounding box (see the `bboxFreezeArmed`/
			// `computeStableBBox` comments above) every time the physics
			// naturally comes to rest with a structural change armed — the
			// initial layout, or an `addExpansion`/`removeExpansion` grows or
			// shrinks the graph — but never after a drag. Doing it only here
			// (never mid-tick) is what keeps the canvas's scale/pan rock
			// steady while nodes are actually moving, while still letting it
			// grow to fit legitimately new content.
			.on('end', () => {
				if (!bboxFreezeArmed) return;
				animateCustomBBoxTo(computeStableBBox() ?? renderer.getBBox());
				bboxFreezeArmed = false;
			});

		let draggedNode: string | null = null;
		// Empty-canvas pan (mousedown on the stage, not a node). Distinct
		// from `draggedNode`: that one moves a single term, this one slides
		// the whole camera. Cursor language matches the two gestures —
		// `pointer` on a node ("this is clickable/draggable"), `grab` /
		// `grabbing` on the background ("you can pan the view"), never
		// `pointer` on empty space, which would promise a click target.
		let isPanning = false;

		const setCanvasCursor = (cursor: string) => {
			if (containerRef.current) containerRef.current.style.cursor = cursor;
		};
		const syncCanvasCursor = () => {
			if (draggedNode != null) {
				setCanvasCursor('pointer');
				return;
			}
			if (isPanning) {
				setCanvasCursor('grabbing');
				return;
			}
			if (hoveredNodeIdRef.current != null) {
				setCanvasCursor('pointer');
				return;
			}
			if (
				hoveredEdgeIdRef.current != null &&
				graphology.hasEdge(hoveredEdgeIdRef.current) &&
				graphology.getEdgeAttribute(hoveredEdgeIdRef.current, 'kind') === 'relationship'
			) {
				setCanvasCursor('pointer');
				return;
			}
			setCanvasCursor('grab');
		};

		const handleDownNode = ({ node, event }: { node: string; event: MouseCoords }) => {
			// Stop Sigma treating this as a camera pan — otherwise the whole
			// graph slides with the cursor even when node positions are frozen.
			event.preventSigmaDefault();
			draggedNode = node;
			// Pin *every* node at its current position and stop the simulation
			// for the whole drag. The previous Neo4j-style reheat let link /
			// charge forces pull neighbours (and, transitively, the rest of
			// the graph) along with the dragged node — first-degree children
			// of a Term especially, since they're on a short, strong edge.
			// Isolated drag is what the exploration-page review asked for:
			// only the grabbed node moves, everyone else stays put. Leaving
			// `fx`/`fy` set after drop (see `endDrag`) keeps that freeze so
			// nothing settles into a new equilibrium the instant the mouse
			// is released. Newly grafted expansion nodes stay unpinned until
			// the next drag, so `addExpansion` can still lay them out.
			simNodes.forEach((simNode) => {
				simNode.fx = simNode.x;
				simNode.fy = simNode.y;
				simNode.vx = 0;
				simNode.vy = 0;
			});
			bboxFreezeArmed = false;
			simulation.alphaTarget(0).alpha(0).stop();
			// Camera pan is a second, independent gesture on the same
			// mousedown+move. Even with the per-move `preventSigmaDefault()`
			// that `handleMoveBody` below issues, one un-prevented move (mouse
			// slipping off the node disc, or onto a chrome overlay at the
			// canvas edge) would slide the *whole* graph. Lock panning for the
			// duration of the node drag.
			renderer.getCamera().enabledPanning = false;
			syncCanvasCursor();
		};
		const handleDownStage = () => {
			isPanning = true;
			syncCanvasCursor();
		};
		// Single funnel for "the gesture is over": the real `mouseup` plus the
		// recovery paths below, for releases the page never sees at all.
		// Idempotent, so calling it spuriously costs nothing.
		const endDrag = () => {
			const wasNodeDrag = draggedNode != null;
			isPanning = false;
			// Sigma's captor bails out of its own `handleUp` unless
			// `isMouseDown` is set, and pans the camera on every move while it
			// stays set. On the recovery paths it never got its `mouseup`
			// either, so clear it here or the graph slides along with a cursor
			// that isn't pressing anything. No-op after a real `mouseup`,
			// which clears the flag before emitting.
			renderer.getMouseCaptor().isMouseDown = false;
			if (wasNodeDrag) {
				// Deliberately leaves every node's `fx`/`fy` set (rather than
				// nulling the non-dragged ones back out) so the rest of the graph
				// cannot start sliding toward a new equilibrium after drop.
				draggedNode = null;
				renderer.getCamera().enabledPanning = true;
				renderer.refresh();
			}
			syncCanvasCursor();
		};
		const handleMoveBody = (coords: MouseCoords) => {
			// Also bound on `document`, so this fires for the move back *into*
			// the page after a release the page never saw — the button let go
			// outside the browser window delivers no `mouseup`, and no `blur`
			// either, since focus never left. No button held while a gesture is
			// still armed is the earliest available evidence that it ended.
			const { original } = coords;
			if ('buttons' in original && original.buttons === 0) {
				if (draggedNode != null || isPanning) endDrag();
				return;
			}
			if (draggedNode == null) return;
			const simNode = simNodesById.get(draggedNode);
			if (!simNode) return;
			const position = renderer.viewportToGraph({ x: coords.x, y: coords.y });
			simNode.fx = position.x;
			simNode.fy = position.y;
			simNode.x = position.x;
			simNode.y = position.y;
			// The simulation is stopped for the drag, so the `'tick'` handler
			// above will not copy this over — write the live graphology
			// position ourselves. `partialGraph` + `skipIndexation` updates
			// only this node and its edges: a full `refresh()` would re-run
			// `process()`, recompute `nodeExtent` from the dragged-out
			// position, and (if `customBBox` isn't frozen yet) rescale the
			// entire canvas every frame — the "whole graph follows the node
			// to the border" glitch.
			graphology.mergeNodeAttributes(draggedNode, { x: position.x, y: position.y });
			renderer.refresh({
				skipIndexation: true,
				partialGraph: {
					nodes: [draggedNode],
					edges: graphology.hasNode(draggedNode) ? graphology.edges(draggedNode) : [],
				},
			});
			coords.preventSigmaDefault();
		};

		// A Term's own expansion (see `expandTermNode` in `ExplorationView.tsx`)
		// almost always grafts onto Tables that are *already* permanent nodes
		// somewhere else in the base graph (`baseNodeIds` below) — often far
		// outside the current viewport, or already linked by an edge from the
		// reverse direction — so the only change `addExpansion` makes can be a
		// single edge stretching off past the container's edge, easy to miss
		// entirely. Panning/zooming just enough to bring the whole
		// newly-connected cluster into view (but doing nothing when it's
		// already fully on screen) makes every expansion visibly register,
		// without the camera jumping around for the common table expansion
		// case where the new Schema/Column/Term nodes are already jittered
		// right next to the origin.
		const FIT_VIEWPORT_PADDING_PX = 72;
		const FIT_ANIMATION_DURATION_MS = 400;
		const focusExpansionIfOffscreen = (nodeIds: string[]) => {
			const viewportPoints = nodeIds
				.filter((nodeId) => graphology.hasNode(nodeId))
				.map((nodeId) => {
					const attributes = graphology.getNodeAttributes(nodeId);
					return renderer.graphToViewport({ x: attributes.x, y: attributes.y });
				});
			if (viewportPoints.length === 0) return;

			const { width, height } = renderer.getDimensions();
			const minX = Math.min(...viewportPoints.map((point) => point.x));
			const maxX = Math.max(...viewportPoints.map((point) => point.x));
			const minY = Math.min(...viewportPoints.map((point) => point.y));
			const maxY = Math.max(...viewportPoints.map((point) => point.y));
			const alreadyVisible = minX >= 0 && maxX <= width && minY >= 0 && maxY <= height;
			if (alreadyVisible) return;

			const availableWidth = Math.max(width - FIT_VIEWPORT_PADDING_PX * 2, 40);
			const availableHeight = Math.max(height - FIT_VIEWPORT_PADDING_PX * 2, 40);
			const zoomOutFactor = Math.max(
				1,
				(maxX - minX) / availableWidth,
				(maxY - minY) / availableHeight,
			);
			const centerFramed = renderer.viewportToFramedGraph({
				x: (minX + maxX) / 2,
				y: (minY + maxY) / 2,
			});
			const camera = renderer.getCamera();
			void camera.animate(
				{ x: centerFramed.x, y: centerFramed.y, ratio: camera.ratio * zoomOutFactor },
				{ duration: FIT_ANIMATION_DURATION_MS },
			);
		};

		// Grafts new nodes/edges onto the *live* graphology graph and d3-force
		// arrays, bypassing the `graph` prop entirely — see the `GraphController`
		// type comment above for why. Idempotent: nodes/edges that already exist
		// (a shared schema, or a table expanded twice) aren't recreated, but are
		// still ref-counted (`nodeRefCount`/`edgeRefCount`) and recorded in this
		// origin's own entry, so a later `removeExpansion` for *this* origin
		// behaves correctly either way and every origin that shares a node/edge
		// has to let go before it actually disappears.
		const addExpansion: GraphController['addExpansion'] = (originNodeId, nodes, edges) => {
			const origin = graphology.hasNode(originNodeId)
				? graphology.getNodeAttributes(originNodeId)
				: null;
			const originX = origin?.x ?? 0;
			const originY = origin?.y ?? 0;
			const nodeTypeColor = isDark ? NODE_TYPE_COLOR_DARK : NODE_TYPE_COLOR_LIGHT;
			const addedNodeIds: string[] = [];
			const addedEdgeKeys: string[] = [];

			// Self-protects a non-base origin (e.g. a Column grafted on by a
			// Table's own expansion, then itself double-clicked to graft its
			// *own* Table/ColumnAttribute/FK-column children) against some
			// *other* origin's `removeExpansion` dropping it later — without
			// this, collapsing (or `clearHighlightedPath`-ing) whichever
			// expansion first grafted this node would rip it out from under
			// its own still-live children, orphaning them even though the
			// node they're rooted on is still very much in active use. Never
			// applies to a virtual origin like `LINK_PATH_ORIGIN_ID` (no real
			// node exists for it) or a base-graph origin (already exempt from
			// `nodeRefCount` entirely). Released in `removeExpansion` below.
			if (origin != null && !baseNodeIds.has(originNodeId)) {
				nodeRefCount.set(originNodeId, (nodeRefCount.get(originNodeId) ?? 0) + 1);
			}

			nodes.forEach((node) => {
				addedNodeIds.push(node.id);
				if (baseNodeIds.has(node.id)) {
					// Already a permanent node — record it above so this
					// origin's own edge to it is still created below, but
					// never ref-count it: `removeExpansion` must never drop
					// a node from the base graph payload.
					return;
				}
				if (graphology.hasNode(node.id)) {
					nodeRefCount.set(node.id, (nodeRefCount.get(node.id) ?? 0) + 1);
					return;
				}
				nodeRefCount.set(node.id, 1);
				// Scattered in a small ring around the table that was just
				// expanded, rather than at a fixed offset, so a table with many
				// new nodes blooms outward instead of stacking them all in the
				// same spot for the simulation to untangle from scratch.
				const jitterRadius = 60;
				const angle = Math.random() * Math.PI * 2;
				const x = originX + Math.cos(angle) * jitterRadius;
				const y = originY + Math.sin(angle) * jitterRadius;
				const size = getNodeSize(node.kind, 0);
				graphology.addNode(node.id, {
					x,
					y,
					size,
					label: node.label,
					color: nodeTypeColor[node.kind],
					type: 'circle',
					kind: node.kind,
					borderColor: NODE_TYPE_BORDER_COLOR[node.kind],
				});
				const simNode: SimNode = { id: node.id, size, x, y };
				simNodesById.set(node.id, simNode);
				simNodes.push(simNode);
			});

			edges.forEach((edge) => {
				if (!graphology.hasNode(edge.source) || !graphology.hasNode(edge.target)) return;
				if (edge.source === edge.target) return;
				// The graph is undirected, so a *different* origin describing
				// this same pair of nodes in the opposite order still resolves
				// to the one edge graphology already created for it —
				// `graph.edge` (unlike the `structuralEdgeKey` we'd otherwise
				// compute) looks it up by node pair and returns whichever key
				// actually won that race, so ref-counting/removal later keys
				// off the same string every origin agrees on.
				const existingEdgeKey = graphology.hasEdge(edge.source, edge.target)
					? graphology.edge(edge.source, edge.target)
					: null;
				if (existingEdgeKey != null) {
					edgeRefCount.set(existingEdgeKey, (edgeRefCount.get(existingEdgeKey) ?? 1) + 1);
					addedEdgeKeys.push(existingEdgeKey);
					return;
				}
				const edgeKey = structuralEdgeKey(edge.source, edge.target);
				graphology.addEdgeWithKey(edgeKey, edge.source, edge.target, {
					color: edgeColors.base,
					size: 1,
					kind: 'structural',
				});
				simLinks.push({ source: edge.source, target: edge.target });
				edgeRefCount.set(edgeKey, 1);
				addedEdgeKeys.push(edgeKey);
			});

			expansionsByOrigin.set(originNodeId, {
				nodeIds: addedNodeIds,
				edgeKeys: addedEdgeKeys,
			});

			// d3-force snapshots `nodes`/`links` when assigned rather than
			// watching the arrays live, so the grown arrays have to be re-fed in —
			// existing nodes keep their current position/velocity either way,
			// only the newly-pushed ones get initialized.
			simulation.nodes(simNodes);
			linkForce.links(simLinks);
			bboxFreezeArmed = true;
			simulation.alphaDecay(DEFAULT_ALPHA_DECAY).alpha(0.5).restart();
			renderer.refresh();
			// Also frames every *pre-existing* neighbour of the origin (e.g. another
			// Table already linked to it by a direct SQL/FOREIGN_KEY edge on the base
			// graph) — not just the nodes/edges this specific expansion just grafted
			// on. A Table's own expansion never re-states that already-drawn edge in
			// `nodes`/`edges` above (it's already on the canvas), so without this, a
			// real, already-visible connection sitting off-screen looks exactly like
			// a missing one the moment a user expands the table expecting to see it.
			const neighborIds = graphology.hasNode(originNodeId)
				? graphology.neighbors(originNodeId)
				: [];
			focusExpansionIfOffscreen([originNodeId, ...addedNodeIds, ...neighborIds]);
		};

		// Reverses one origin's `addExpansion` — see the `GraphController` type
		// comment for the shared-node semantics. Drops nodes and edges whose
		// ref count reaches zero (dropping a node also drops its incident
		// edges via graphology's own `dropNode`, so those are simply skipped
		// when the edge loop below gets to them), leaving anything still kept
		// alive by another expansion — including, since `addExpansion` self-
		// protects a non-base origin, this very node's *own* still-live
		// expansion rooted on it — in place.
		const removeExpansion: GraphController['removeExpansion'] = (originNodeId) => {
			const entry = expansionsByOrigin.get(originNodeId);
			if (entry == null) return null;
			expansionsByOrigin.delete(originNodeId);

			const removedNodeIds: string[] = [];
			entry.nodeIds.forEach((nodeId) => {
				if (baseNodeIds.has(nodeId)) return;
				const nextCount = (nodeRefCount.get(nodeId) ?? 1) - 1;
				if (nextCount > 0) {
					nodeRefCount.set(nodeId, nextCount);
					return;
				}
				nodeRefCount.delete(nodeId);
				removedNodeIds.push(nodeId);
			});

			// Releases the self-protection `addExpansion` granted this origin
			// (see the comment there) now that its own expansion is gone. If
			// whichever *other* expansion originally grafted this node is
			// still live, it survives here exactly as it did before self-
			// protection existed; only once every referrer — including this
			// origin's own now-closed expansion — has let go does it finally
			// get swept up alongside its former children above.
			if (!baseNodeIds.has(originNodeId) && graphology.hasNode(originNodeId)) {
				const nextOriginCount = (nodeRefCount.get(originNodeId) ?? 1) - 1;
				if (nextOriginCount > 0) {
					nodeRefCount.set(originNodeId, nextOriginCount);
				} else {
					nodeRefCount.delete(originNodeId);
					removedNodeIds.push(originNodeId);
				}
			}

			const removedNodeIdSet = new Set(removedNodeIds);
			// The React-state `activeNodeId` this ref mirrors updates one render
			// behind (it's set from an effect) — if it currently points at a node
			// this collapse is about to drop (e.g. a Column/Term whose panel was
			// open), null it out immediately rather than leaving it dangling
			// until that effect catches up, since `renderer.refresh()` below
			// re-runs `nodeReducer` synchronously against the *already*-mutated
			// graph.
			if (activeNodeIdRef.current != null && removedNodeIdSet.has(activeNodeIdRef.current)) {
				activeNodeIdRef.current = null;
			}
			removedNodeIds.forEach((nodeId) => {
				if (graphology.hasNode(nodeId)) graphology.dropNode(nodeId);
				simNodesById.delete(nodeId);
			});
			// Mirrors the node ref-counting above: an edge this origin
			// contributed to `edgeRefCount` (whether it created it outright or
			// just found another origin's already there — see `addExpansion`)
			// only actually gets dropped once every origin that wanted it has
			// let go. Edges to a node dropped above are already gone via
			// graphology's own `dropNode` cascade, hence the `hasEdge` guard.
			const removedEdgeKeys: string[] = [];
			entry.edgeKeys.forEach((edgeKey) => {
				const nextCount = (edgeRefCount.get(edgeKey) ?? 1) - 1;
				if (nextCount > 0) {
					edgeRefCount.set(edgeKey, nextCount);
					return;
				}
				edgeRefCount.delete(edgeKey);
				removedEdgeKeys.push(edgeKey);
			});
			removedEdgeKeys.forEach((edgeKey) => {
				if (graphology.hasEdge(edgeKey)) graphology.dropEdge(edgeKey);
			});

			simNodes = simNodes.filter((node) => !removedNodeIdSet.has(node.id));
			const removedEdgeKeySet = new Set(removedEdgeKeys);
			simLinks = simLinks.filter((link) => {
				const sourceId = simLinkEndpointId(link.source);
				const targetId = simLinkEndpointId(link.target);
				if (removedNodeIdSet.has(sourceId) || removedNodeIdSet.has(targetId)) return false;
				return (
					!removedEdgeKeySet.has(structuralEdgeKey(sourceId, targetId)) &&
					!removedEdgeKeySet.has(structuralEdgeKey(targetId, sourceId))
				);
			});

			simulation.nodes(simNodes);
			linkForce.links(simLinks);
			bboxFreezeArmed = true;
			simulation.alphaDecay(DEFAULT_ALPHA_DECAY).alpha(0.4).restart();
			renderer.refresh();

			return removedNodeIds;
		};

		const handleClickStage = () => onSelectNode(null);
		const handleClickNode = ({ node }: { node: string }) => onClickNode(node);
		const handleDoubleClickNode = ({ node, event }: { node: string; event: MouseCoords }) => {
			// Sigma's default double-click behavior is to zoom the camera in on
			// the node — toggling its expansion instead, so suppress that zoom.
			event.preventSigmaDefault();
			onDoubleClickNode(node);
		};
		const handleClickEdge = ({ edge }: { edge: string }) => {
			// Structural edges (see `GraphEdgeAttributes.kind`) have no real
			// hop chain of their own — they're already the raw structural link.
			if (graphology.getEdgeAttribute(edge, 'kind') === 'structural') return;
			onSelectEdge(edge);
		};
		const handleEnterNode = ({ node, event }: { node: string; event: MouseCoords }) => {
			hoveredNodeIdRef.current = node;
			syncCanvasCursor();
			const displayData = renderer.getNodeDisplayData(node);
			const radius = displayData ? renderer.scaleSize(displayData.size) : 0;
			onHoverNode({ id: node, x: event.x + radius + 8, y: event.y + radius + 8 });
		};
		const handleLeaveNode = () => {
			hoveredNodeIdRef.current = null;
			onHoverNode(null);
			syncCanvasCursor();
		};
		const handleEnterEdge = ({ edge }: { edge: string }) => {
			hoveredEdgeIdRef.current = edge;
			syncCanvasCursor();
		};
		const handleLeaveEdge = () => {
			hoveredEdgeIdRef.current = null;
			syncCanvasCursor();
		};
		const handleCameraUpdated = () => {
			onHoverNode(null);
			// A pan/zoom — whether the user's own scroll/drag or a
			// programmatic one like `focusNode`/`focusExpansionIfOffscreen`
			// re-centering on a newly-selected or -expanded node — moves
			// every node relative to a cursor that hasn't itself moved, so
			// Sigma never gets the real `mousemove` it needs to notice the
			// node underneath changed (or disappeared). Left alone,
			// whichever node was hovered right before the camera moved
			// stays stuck "hovered" — and, since the node reducer above
			// draws the exact same ring for `isHovered` as for `isActive`,
			// that stale ring can visually read as the new selection
			// itself sitting on the wrong node once the camera settles.
			// Clearing both refs here (rather than waiting for a `leaveNode`
			// that may never come) and forcing one reducer pass drops it
			// immediately; a real hover resumes on the next actual
			// `mousemove` regardless.
			let staleHoverCleared = false;
			if (hoveredNodeIdRef.current != null) {
				hoveredNodeIdRef.current = null;
				staleHoverCleared = true;
			}
			if (hoveredEdgeIdRef.current != null) {
				hoveredEdgeIdRef.current = null;
				staleHoverCleared = true;
			}
			if (staleHoverCleared) {
				syncCanvasCursor();
				renderer.refresh();
			}
		};

		renderer.on('downNode', handleDownNode);
		renderer.on('downStage', handleDownStage);
		renderer.on('clickStage', handleClickStage);
		renderer.on('clickNode', handleClickNode);
		renderer.on('doubleClickNode', handleDoubleClickNode);
		renderer.on('clickEdge', handleClickEdge);
		renderer.on('enterNode', handleEnterNode);
		renderer.on('leaveNode', handleLeaveNode);
		renderer.on('enterEdge', handleEnterEdge);
		renderer.on('leaveEdge', handleLeaveEdge);
		renderer.getMouseCaptor().on('mousemovebody', handleMoveBody);
		// mouseup is bound on `document` inside Sigma's captor, so a release
		// outside the canvas still drops the node. Deliberately *not* also
		// listening to `mouseleave`: treating "cursor hit the canvas edge /
		// a chrome overlay" as a drop cancelled the drag and let the next
		// move pan the camera, which is the "graph goes weird at the border"
		// bug.
		renderer.getMouseCaptor().on('mouseup', endDrag);
		// Alt-tabbing mid-gesture (or anything else that pulls focus away:
		// devtools, an OS dialog) means the eventual release lands in another
		// window and no `mouseup` ever reaches the captor. Without this, a node
		// drag leaves `enabledPanning` `false` for the rest of the component's
		// life, and a stage pan leaves `isPanning` — hence a "grabbing" cursor
		// — stuck the same way.
		const handleWindowBlur = () => endDrag();
		window.addEventListener('blur', handleWindowBlur);
		renderer.getCamera().on('updated', handleCameraUpdated);

		// The graph starts from a random scatter (see `buildGraphologyGraph`
		// above), so for the first moment every edge is stretched across a
		// huge, chaotic area rather than connecting its two settled
		// endpoints. Those long, criss-crossing, mostly-empty-space edges
		// read as faint/washed-out at the same alpha that looks like a solid
		// brand green once the layout has actually settled — not a wrong
		// color, just a much sparser one. Hiding the canvas for that one
		// unsettled beat (instead of trying to make the color read correctly
		// while mid-scatter) means the very first thing anyone sees is
		// already-settled, correctly-colored edges.
		containerRef.current.style.transition = 'none';
		containerRef.current.style.opacity = '0';
		// Sigma's mouse/touch captors keep listening even while the canvas is
		// invisible — without this, hovering over the still-scattering (but
		// unseen) initial layout fires `enterNode` for whatever happens to be
		// under the cursor, popping a hover card for a node the user can't
		// see and never touched.
		containerRef.current.style.pointerEvents = 'none';
		// Also clear out any hover state left over from a previous graph (e.g.
		// switching Data/Semantic layers while the cursor sits over the
		// canvas) — otherwise its popover would keep floating on screen for
		// this entire hidden window, over a graph it no longer refers to.
		onHoverNode(null);

		// Give the layout a moment to settle from its random scatter, then
		// fit/center the view once, mirroring the previous "layout stop" reset.
		const centerTimeout = window.setTimeout(() => {
			renderer.refresh();
			// Freezes the graph-to-viewport scale to this just-settled bounding
			// box instead of Sigma's default of recomputing it from *live* node
			// positions on every single simulation tick (`autoRescale`). Without
			// this, dragging one node far outside the rest of the graph — or
			// even just the physics still gently resettling after a drag —
			// keeps shifting that box, which rescales/repans the *entire*
			// canvas in lockstep every frame: every other node visibly
			// trembles even though only the one node actually moved. Freezing
			// it means a node dragged past this box simply goes off-frame
			// (pan/zoom to see it) instead of dragging the whole graph's scale
			// along with it.
			renderer.setCustomBBox(renderer.getBBox());
			void renderer
				.getCamera()
				.animatedReset()
				.then(() => {
					// Zoom in a bit further as the actual default starting view,
					// rather than leaving the user at the (often zoomed-far-out)
					// whole-graph fit.
					void renderer
						.getCamera()
						.animatedZoom({ duration: 300, factor: INITIAL_ZOOM_IN_FACTOR });
				});
			if (containerRef.current) {
				containerRef.current.style.transition = 'opacity 300ms ease-out';
				containerRef.current.style.opacity = '1';
				containerRef.current.style.pointerEvents = 'auto';
				syncCanvasCursor();
			}
		}, 1200);

		// The OS color scheme can change without remounting this component
		// (e.g. the system switching themes at sunset); react to it live
		// instead of leaving stale light/dark colors until the next navigation.
		const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: dark)');
		const handleColorSchemeChange = (event: MediaQueryListEvent) => {
			isDark = event.matches;
			edgeColors = getEdgeColors(isDark);
			const nextNodeTypeColor = isDark ? NODE_TYPE_COLOR_DARK : NODE_TYPE_COLOR_LIGHT;
			graphology.forEachNode((node, attributes) => {
				graphology.setNodeAttribute(node, 'color', nextNodeTypeColor[attributes.kind]);
			});
			renderer.setSetting('labelColor', {
				color: isDark ? LABEL_COLOR_DARK : LABEL_COLOR_LIGHT,
			});
			renderer.refresh();
		};
		colorSchemeQuery.addEventListener('change', handleColorSchemeChange);

		const resizeObserver = new ResizeObserver((entries) => {
			// ResizeObserver fires immediately on observe() and again during
			// layer/route transitions where the flex layout can momentarily
			// collapse the container to 0×0; skip those frames instead of
			// resizing into (and rendering at) an invalid size.
			const { width, height } = entries[0]?.contentRect ?? { width: 0, height: 0 };
			if (width === 0 || height === 0) return;
			renderer.resize();
		});
		resizeObserver.observe(containerRef.current);

		onControllerChange({
			getCamera: () => renderer.getCamera(),
			refresh: () => renderer.refresh(),
			resetExtent: () => {
				renderer.setCustomBBox(renderer.getBBox());
				renderer.refresh();
			},
			focusNode: (nodeId) => focusExpansionIfOffscreen([nodeId]),
			addExpansion,
			removeExpansion,
		});

		return () => {
			window.clearTimeout(centerTimeout);
			if (bboxAnimationFrame != null) cancelAnimationFrame(bboxAnimationFrame);
			colorSchemeQuery.removeEventListener('change', handleColorSchemeChange);
			window.removeEventListener('blur', handleWindowBlur);
			resizeObserver.disconnect();
			onControllerChange(null);
			onHoverNode(null);
			simulation.stop();
			rendererRef.current = null;
			renderer.kill();
		};
	}, [
		graph,
		onClickNode,
		onControllerChange,
		onDoubleClickNode,
		onHoverNode,
		onSelectEdge,
		onSelectNode,
	]);

	return <div ref={containerRef} className="h-full w-full" aria-label="Exploration graph" />;
};
