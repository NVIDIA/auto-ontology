// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import cytoscape, { type Core, type ElementDefinition, type EventObject } from 'cytoscape';
import euler from 'cytoscape-euler';

import SnowflakeSvg from '@/common/icons/svg/snowflake.svg';
import TermsSvg from '@/common/icons/svg/terms.svg';
import type { ExplorationGraph } from '@/types/exploration';

cytoscape.use(euler);

export type HoveredNode = {
	id: string;
	x: number;
	y: number;
};

const DATA_OBJECT_CONNECTOR_ICON = encodeURI(
	`data:image/svg+xml;utf-8,${renderToStaticMarkup(
		<SnowflakeSvg width={24} height={24} />,
	).replaceAll('currentColor', 'rgb(49, 185, 197)')}`,
);
const TERM_OBJECT_ICON = encodeURI(
	`data:image/svg+xml;utf-8,${renderToStaticMarkup(
		<TermsSvg width={24} height={24} />,
	).replaceAll('currentColor', 'rgb(71, 186, 197)')}`,
);
const EULER_LAYOUT: cytoscapeEuler.EulerLayoutOptions = {
	name: 'euler',
	springLength: () => 800,
	springCoeff: () => 0.000001,
	mass: () => 4,
	gravity: -10.2,
	pull: 0.001,
	theta: 0.666,
	dragCoeff: 0.02,
	movementThreshold: 1,
	timeStep: 20,
	refresh: 10,
	animate: false,
	maxIterations: 3000,
	maxSimulationTime: 1000,
	ungrabifyWhileSimulating: false,
	fit: false,
	padding: 30,
	randomize: true,
};

const getNodeSize = (relationshipCount: number) => {
	if (relationshipCount >= 8) return 140;
	if (relationshipCount >= 4) return 80;
	if (relationshipCount >= 1) return 60;
	return 40;
};

const createElements = (graph: ExplorationGraph): ElementDefinition[] => [
	...graph.nodes.map((node) => {
		const size = getNodeSize(node.relationshipCount);
		return {
			data: {
				id: node.id,
				label: node.layer === 'data' ? node.name.toUpperCase() : node.name,
				size,
				borderRadius: Math.max(10, Math.round(size * 0.2)),
				layer: node.layer,
				nodeType: node.nodeType,
				icon: node.layer === 'data' ? DATA_OBJECT_CONNECTOR_ICON : TERM_OBJECT_ICON,
			},
		};
	}),
	...graph.links.map((link) => ({
		data: {
			id: `${link.source}:${link.target}`,
			source: link.source,
			target: link.target,
		},
	})),
];

type GraphCanvasProps = {
	graph: ExplorationGraph;
	activeNodeId: string | null;
	onSelectNode: (nodeId: string | null) => void;
	onSelectEdge: (edgeId: string) => void;
	onHoverNode: (hoveredNode: HoveredNode | null) => void;
	onControllerChange: (controller: Core | null) => void;
};

/** Cytoscape-backed graph canvas shared by both the semantic and data Exploration layers. */
export const GraphCanvas = ({
	graph,
	activeNodeId,
	onSelectNode,
	onSelectEdge,
	onHoverNode,
	onControllerChange,
}: GraphCanvasProps) => {
	const containerRef = useRef<HTMLDivElement>(null);
	const controllerRef = useRef<Core | null>(null);

	useEffect(() => {
		if (containerRef.current == null) return undefined;

		const controller = cytoscape({
			container: containerRef.current,
			elements: createElements(graph),
			boxSelectionEnabled: false,
			minZoom: 0.2,
			maxZoom: 3,
			style: [
				{
					selector: 'edge',
					style: {
						width: 1,
						'line-color': '#76b900',
						opacity: 0.24,
						'curve-style': 'bezier',
					},
				},
				{
					selector: 'node',
					style: {
						width: 'data(size)',
						height: 'data(size)',
						label: 'data(label)',
						'background-color': '#eef7df',
						'border-color': '#76b900',
						'border-width': 1,
						color: '#27272a',
						'font-size': 12,
						'font-weight': 500,
						'text-halign': 'center',
						'text-valign': 'bottom',
						'text-margin-y': 8,
						'text-max-width': '130px',
						'text-overflow-wrap': 'whitespace',
						'text-wrap': 'ellipsis',
					},
				},
				{
					selector: 'node[layer = "semantic"]',
					style: {
						shape: 'ellipse',
						'background-color': '#e3f4f5',
						'border-color': '#47bac5',
						'background-image': 'data(icon)',
						'background-width': '24px',
						'background-height': '24px',
						'background-fit': 'none',
					},
				},
				{
					selector: 'node[layer = "data"]',
					style: {
						shape: 'roundrectangle',
						'corner-radius': 'data(borderRadius)',
						'background-color': '#fceee8',
						'border-color': '#e4b9a9',
						'background-image': 'data(icon)',
						'background-width': '24px',
						'background-height': '24px',
						'background-fit': 'none',
					},
				},
				{
					selector: 'node.hovered',
					style: {
						'underlay-color': '#76b900',
						'underlay-opacity': 0.14,
						'underlay-padding': 10,
					},
				},
				{
					selector: 'node.selected',
					style: {
						'border-width': 3,
						'border-color': '#76b900',
						'background-color': '#e4f4ca',
						opacity: 1,
					},
				},
				{
					selector: 'node[layer = "semantic"].selected',
					style: {
						'border-color': '#47bac5',
						'background-color': '#d5eff1',
					},
				},
				{
					selector: 'node.dimmed',
					style: {
						opacity: 0.35,
					},
				},
				{
					selector: 'edge.dimmed',
					style: {
						opacity: 0.06,
					},
				},
				{
					selector: 'edge.connected',
					style: {
						width: 2,
						opacity: 0.9,
					},
				},
				{
					selector: 'edge.hovered',
					style: {
						width: 3,
						opacity: 1,
					},
				},
			],
			layout: { name: 'preset' },
		});

		const handleNodeTap = (event: EventObject) => {
			onSelectNode(event.target.id());
		};
		const handleEdgeTap = (event: EventObject) => {
			onSelectEdge(event.target.id());
		};
		const handleBackgroundTap = (event: EventObject) => {
			if (event.target === controller) onSelectNode(null);
		};

		controller.on('tap', 'node', handleNodeTap);
		controller.on('tap', 'edge', handleEdgeTap);
		controller.on('tap', handleBackgroundTap);
		controller.on('mouseover', 'node', (event) => {
			event.target.addClass('hovered');
			const position = event.target.renderedPosition();
			const offset = event.target.renderedWidth() / 2 + 8;
			onHoverNode({
				id: event.target.id(),
				x: position.x + offset,
				y: position.y + offset,
			});
		});
		controller.on('mouseout', 'node', (event) => {
			event.target.removeClass('hovered');
			onHoverNode(null);
		});
		controller.on('drag', 'node', () => onHoverNode(null));
		controller.on('pan zoom', () => onHoverNode(null));
		controller.on('mouseover', 'edge', (event) => event.target.addClass('hovered'));
		controller.on('mouseout', 'edge', (event) => event.target.removeClass('hovered'));
		controller.one('layoutstop', () => {
			controller.zoom(1);
			controller.center();
		});
		const layout = controller.layout(EULER_LAYOUT);
		layout.run();

		const resizeObserver = new ResizeObserver(() => {
			controller.resize();
		});
		resizeObserver.observe(containerRef.current);

		controllerRef.current = controller;
		onControllerChange(controller);

		return () => {
			resizeObserver.disconnect();
			onControllerChange(null);
			layout.stop();
			controller.destroy();
			controllerRef.current = null;
		};
	}, [graph, onControllerChange, onHoverNode, onSelectEdge, onSelectNode]);

	useEffect(() => {
		const controller = controllerRef.current;
		if (controller == null) return;

		controller.nodes().removeClass('selected dimmed');
		controller.edges().removeClass('connected dimmed');
		if (activeNodeId == null) return;

		const activeNode = controller.getElementById(activeNodeId);
		if (activeNode.empty()) return;

		controller.nodes().addClass('dimmed');
		controller.edges().addClass('dimmed');
		activeNode.removeClass('dimmed').addClass('selected');
		activeNode.neighborhood('node').removeClass('dimmed');
		activeNode.connectedEdges().removeClass('dimmed').addClass('connected');
	}, [activeNodeId]);

	return <div ref={containerRef} className="h-full w-full" aria-label="Exploration graph" />;
};
