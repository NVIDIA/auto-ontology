// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import type { Core } from 'cytoscape';

import { explorationApi } from '@/api/exploration';
import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import { Icon, IconName } from '@/common/icons';
import { SearchInput } from '@/common/SearchInput';
import {
	ColumnAttributesModal,
	DataDetailsModal,
	QueryCarouselModal,
	RelationshipsModal,
	SemanticRelationshipModal,
	SqlAttributesModal,
	type DataDetailsKind,
} from '@/common/modal';
import { TableType } from '@/enums/datasources';
import { ExplorationLayer } from '@/enums/exploration';
import type { ExplorationGraph } from '@/types/exploration';
import { GraphCanvas, type HoveredNode } from './graph/GraphCanvas';
import { ViewToggle } from './graph/ViewToggle';
import { ZoomControls } from './graph/ZoomControls';
import { HoverNodeCard } from './HoverNodeCard';
import { ActiveDataCard, buildDataGraph } from './ExplorationData';
import { ActiveTermCard, buildSemanticGraph } from './ExplorationSemantic';

const EMPTY_GRAPH: ExplorationGraph = { nodes: [], links: [] };

export const ExplorationView = () => {
	const router = useRouter();
	const searchParams = useSearchParams();
	const semanticId = searchParams.get('semanticId');
	const dataId = searchParams.get('dataId');
	const layer: ExplorationLayer =
		dataId != null || searchParams.get('view') === 'data'
			? ExplorationLayer.Data
			: ExplorationLayer.Semantic;
	const activeNodeIdFromUrl = layer === ExplorationLayer.Semantic ? semanticId : dataId;
	const [semanticGraph, setSemanticGraph] = useState<ExplorationGraph>(EMPTY_GRAPH);
	const [dataGraph, setDataGraph] = useState<ExplorationGraph>(EMPTY_GRAPH);
	const [semanticLoading, setSemanticLoading] = useState(true);
	const [dataLoading, setDataLoading] = useState(false);
	const [semanticError, setSemanticError] = useState<string | null>(null);
	const [dataError, setDataError] = useState<string | null>(null);
	const [dataLoaded, setDataLoaded] = useState(false);
	const [semanticNoZoneAccess, setSemanticNoZoneAccess] = useState(false);
	const [dataNoZoneAccess, setDataNoZoneAccess] = useState(false);
	const [search, setSearch] = useState('');
	const [activeNodeId, setActiveNodeId] = useState<string | null>(activeNodeIdFromUrl);
	const [hoveredNodePosition, setHoveredNodePosition] = useState<HoveredNode | null>(null);
	const [selectedLinkId, setSelectedLinkId] = useState<string | null>(null);
	const [selectedSemanticEdgeId, setSelectedSemanticEdgeId] = useState<string | null>(null);
	const [relationshipsNodeId, setRelationshipsNodeId] = useState<string | null>(null);
	const [dataDetailsKind, setDataDetailsKind] = useState<DataDetailsKind | null>(null);
	const [columnAttributesNodeId, setColumnAttributesNodeId] = useState<string | null>(null);
	const [sqlAttributesNodeId, setSqlAttributesNodeId] = useState<string | null>(null);
	const [controller, setController] = useState<Core | null>(null);

	useEffect(() => {
		let cancelled = false;

		const loadGraph = async () => {
			setSemanticLoading(true);
			const response = await explorationApi.getSemanticExplorationGraph();
			if (cancelled) return;

			if (response.error) {
				setSemanticError(response.message ?? 'Failed to load terms');
				setSemanticLoading(false);
				return;
			}

			const nextGraph = buildSemanticGraph(response.data ?? { nodes: [], links: [] });
			if (cancelled) return;
			setSemanticGraph(nextGraph);
			setSemanticNoZoneAccess(response.meta?.noZoneAccess ?? false);
			setSemanticError(null);
			setSemanticLoading(false);
		};

		void loadGraph();
		return () => {
			cancelled = true;
		};
	}, []);

	useEffect(() => {
		if (layer !== ExplorationLayer.Data || dataLoaded) return undefined;
		let cancelled = false;

		const loadGraph = async () => {
			setDataLoading(true);
			try {
				const response = await explorationApi.getDataExplorationGraph();
				if (cancelled) return;
				if (response.error) {
					throw new Error(response.message ?? 'Failed to load data objects');
				}
				const nextGraph = buildDataGraph(response.data ?? { nodes: [], links: [] });
				if (cancelled) return;
				setDataGraph(nextGraph);
				setDataNoZoneAccess(response.meta?.noZoneAccess ?? false);
				setDataError(null);
				setDataLoaded(true);
			} catch (loadError) {
				if (cancelled) return;
				setDataError(
					loadError instanceof Error ? loadError.message : 'Failed to load data objects',
				);
			} finally {
				if (!cancelled) setDataLoading(false);
			}
		};

		void loadGraph();
		return () => {
			cancelled = true;
		};
	}, [dataLoaded, layer]);

	useEffect(() => {
		setActiveNodeId(activeNodeIdFromUrl);
	}, [activeNodeIdFromUrl]);

	const handleSelectNode = useCallback(
		(nodeId: string | null) => {
			setHoveredNodePosition(null);
			setSelectedLinkId(null);
			setSelectedSemanticEdgeId(null);
			setRelationshipsNodeId(null);
			setDataDetailsKind(null);
			setColumnAttributesNodeId(null);
			setSqlAttributesNodeId(null);
			setActiveNodeId(nodeId);
			let nextUrl = '/exploration';
			if (layer === ExplorationLayer.Data) {
				nextUrl = nodeId
					? `/exploration?view=data&dataId=${encodeURIComponent(nodeId)}`
					: '/exploration?view=data';
			} else if (nodeId) {
				nextUrl = `/exploration?semanticId=${encodeURIComponent(nodeId)}`;
			}
			router.replace(nextUrl, { scroll: false });
		},
		[layer, router],
	);

	const handleSelectEdge = useCallback(
		(edgeId: string) => {
			if (layer === ExplorationLayer.Data) setSelectedLinkId(edgeId);
			else setSelectedSemanticEdgeId(edgeId);
		},
		[layer],
	);

	const handleToggleLayer = useCallback(() => {
		setSearch('');
		setActiveNodeId(null);
		setHoveredNodePosition(null);
		setSelectedLinkId(null);
		setSelectedSemanticEdgeId(null);
		setRelationshipsNodeId(null);
		setDataDetailsKind(null);
		setColumnAttributesNodeId(null);
		setSqlAttributesNodeId(null);
		router.replace(
			layer === ExplorationLayer.Semantic ? '/exploration?view=data' : '/exploration',
			{ scroll: false },
		);
	}, [layer, router]);

	const handleControllerChange = useCallback((nextController: Core | null) => {
		setController(nextController);
	}, []);
	const handleHoverNode = useCallback((hoveredNode: HoveredNode | null) => {
		setHoveredNodePosition(hoveredNode);
	}, []);

	const graph = layer === ExplorationLayer.Semantic ? semanticGraph : dataGraph;
	const loading = layer === ExplorationLayer.Semantic ? semanticLoading : dataLoading;
	const error = layer === ExplorationLayer.Semantic ? semanticError : dataError;
	const noZoneAccess =
		layer === ExplorationLayer.Semantic ? semanticNoZoneAccess : dataNoZoneAccess;

	const filteredNodes = useMemo(() => {
		const query = search.trim().toLowerCase();
		if (query === '') return [];
		return graph.nodes.filter((node) => node.name.toLowerCase().includes(query));
	}, [graph.nodes, search]);

	const activeNode = useMemo(
		() => graph.nodes.find((node) => node.id === activeNodeId) ?? null,
		[activeNodeId, graph.nodes],
	);
	const relationshipsNode = useMemo(
		() => graph.nodes.find((node) => node.id === relationshipsNodeId) ?? null,
		[graph.nodes, relationshipsNodeId],
	);
	const relatedNodes = useMemo(() => {
		if (relationshipsNodeId == null) return [];
		const relatedIds = new Set<string>();
		graph.links.forEach((link) => {
			if (link.source === relationshipsNodeId) relatedIds.add(link.target);
			if (link.target === relationshipsNodeId) relatedIds.add(link.source);
		});
		return graph.nodes.filter((node) => relatedIds.has(node.id));
	}, [graph.links, graph.nodes, relationshipsNodeId]);
	const columnAttributesNode = useMemo(() => {
		const found = graph.nodes.find((node) => node.id === columnAttributesNodeId) ?? null;
		return found?.layer === ExplorationLayer.Semantic ? found : null;
	}, [columnAttributesNodeId, graph.nodes]);
	const sqlAttributesNode = useMemo(() => {
		const found = graph.nodes.find((node) => node.id === sqlAttributesNodeId) ?? null;
		return found?.layer === ExplorationLayer.Semantic ? found : null;
	}, [graph.nodes, sqlAttributesNodeId]);
	const hoveredNode = useMemo(
		() => graph.nodes.find((node) => node.id === hoveredNodePosition?.id) ?? null,
		[graph.nodes, hoveredNodePosition?.id],
	);
	const selectedLink = useMemo(
		() =>
			graph.links.find((link) => `${link.source}:${link.target}` === selectedLinkId) ?? null,
		[graph.links, selectedLinkId],
	);
	const selectedLinkSource =
		graph.nodes.find((node) => node.id === selectedLink?.source)?.name ?? '';
	const selectedLinkTarget =
		graph.nodes.find((node) => node.id === selectedLink?.target)?.name ?? '';

	const selectedSemanticEdge = useMemo(
		() =>
			graph.links.find(
				(link) => `${link.source}:${link.target}` === selectedSemanticEdgeId,
			) ?? null,
		[graph.links, selectedSemanticEdgeId],
	);
	const semanticEdgeSourceNode = useMemo(() => {
		const found = graph.nodes.find((node) => node.id === selectedSemanticEdge?.source) ?? null;
		return found?.layer === ExplorationLayer.Semantic ? found : null;
	}, [graph.nodes, selectedSemanticEdge?.source]);
	const semanticEdgeTargetNode = useMemo(() => {
		const found = graph.nodes.find((node) => node.id === selectedSemanticEdge?.target) ?? null;
		return found?.layer === ExplorationLayer.Semantic ? found : null;
	}, [graph.nodes, selectedSemanticEdge?.target]);

	const graphBackground =
		'bg-[radial-gradient(circle,#e4e4e7_1px,transparent_1px)] bg-[size:8px_8px] dark:bg-[radial-gradient(circle,#3f3f46_1px,transparent_1px)]';

	return (
		<main
			className={`relative h-full min-h-0 w-full overflow-hidden bg-zinc-50 dark:bg-zinc-950 ${graphBackground}`}
		>
			<div className="absolute left-4 right-4 top-4 z-20 flex items-start justify-between gap-4">
				<div className="flex items-start gap-2">
					<div className="relative w-72">
						<SearchInput
							value={search}
							onChange={setSearch}
							placeholder={
								layer === ExplorationLayer.Semantic
									? 'Search terms…'
									: 'Search data objects…'
							}
							aria-label={
								layer === ExplorationLayer.Semantic
									? 'Search terms in exploration'
									: 'Search data objects in exploration'
							}
							className="h-10 bg-white shadow-md dark:bg-zinc-900"
						/>
						{filteredNodes.length > 0 && (
							<ul className="absolute top-12 max-h-[calc(100dvh-8.5rem)] w-full overflow-y-auto rounded-lg border border-zinc-200 bg-white py-1 shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
								{filteredNodes.map((node) => (
									<li key={node.id}>
										<button
											type="button"
											onClick={() => {
												setSearch('');
												handleSelectNode(node.id);
											}}
											className="flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left text-sm text-zinc-700 hover:bg-zinc-100 dark:text-zinc-200 dark:hover:bg-zinc-800"
										>
											<Icon
												name={
													node.layer === ExplorationLayer.Semantic
														? IconName.Terms
														: catalogNodeInfo[node.nodeType].icon
												}
												className={`h-4 w-4 shrink-0 ${
													node.layer === ExplorationLayer.Semantic
														? 'text-[#76b900]'
														: 'text-[#3b82b6]'
												}`}
											/>
											<span className="truncate">{node.name}</span>
										</button>
									</li>
								))}
							</ul>
						)}
					</div>
					<ViewToggle layer={layer} onToggle={handleToggleLayer} />
				</div>
				<ZoomControls controller={controller} />
			</div>

			{loading && (
				<div className="flex h-full items-center justify-center">
					<div
						className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
						role="status"
						aria-label="Loading exploration"
					/>
				</div>
			)}

			{!loading && error != null && (
				<div className="flex h-full items-center justify-center px-6 text-center">
					<div className="rounded-xl border border-red-200 bg-white px-8 py-6 text-sm text-red-700 shadow-lg dark:border-red-900/50 dark:bg-zinc-900 dark:text-red-300">
						Couldn&apos;t load exploration: {error}
					</div>
				</div>
			)}

			{!loading && error == null && graph.nodes.length === 0 && (
				<div className="flex h-full items-center justify-center text-sm text-zinc-500">
					{noZoneAccess
						? 'You do not have access to any data. Contact an administrator to request access.'
						: layer === ExplorationLayer.Semantic
							? 'No Terms Created Yet'
							: 'No Data Objects Found'}
				</div>
			)}

			{!loading && error == null && graph.nodes.length > 0 && (
				<GraphCanvas
					graph={graph}
					activeNodeId={activeNodeId}
					onSelectNode={handleSelectNode}
					onSelectEdge={handleSelectEdge}
					onHoverNode={handleHoverNode}
					onControllerChange={handleControllerChange}
				/>
			)}

			{hoveredNode != null && hoveredNodePosition != null && (
				<HoverNodeCard
					node={hoveredNode}
					x={hoveredNodePosition.x}
					y={hoveredNodePosition.y}
				/>
			)}

			{activeNode?.layer === ExplorationLayer.Semantic && (
				<ActiveTermCard
					key={activeNode.id}
					node={activeNode}
					onClose={() => handleSelectNode(null)}
					onView={() => router.push(`/terms?focus=${encodeURIComponent(activeNode.id)}`)}
					onShowRelationships={() => setRelationshipsNodeId(activeNode.id)}
					onShowColumnAttributes={() => setColumnAttributesNodeId(activeNode.id)}
					onShowSqlAttributes={() => setSqlAttributesNodeId(activeNode.id)}
				/>
			)}

			{activeNode?.layer === ExplorationLayer.Data && (
				<ActiveDataCard
					key={activeNode.id}
					node={activeNode}
					onClose={() => handleSelectNode(null)}
					onView={() =>
						router.push(
							`/data?focus=${encodeURIComponent(
								`${activeNode.databaseId}|${activeNode.schemaId}|${activeNode.id}`,
							)}`,
						)
					}
					onShowRelationships={() => setRelationshipsNodeId(activeNode.id)}
					onShowColumns={() => setDataDetailsKind('columns')}
					onShowQueries={() => setDataDetailsKind('queries')}
					onShowTerms={() => setDataDetailsKind('terms')}
				/>
			)}

			<div className="absolute bottom-4 right-4 z-20 rounded-lg border border-zinc-200 bg-white px-3 py-2 shadow-lg dark:border-zinc-700 dark:bg-zinc-900">
				<p className="text-xs text-zinc-500 dark:text-zinc-400">
					Viewing:{' '}
					{layer === ExplorationLayer.Semantic ? 'Semantic Objects' : 'Data Objects'}
				</p>
				{layer === ExplorationLayer.Semantic ? (
					<div className="mt-2 flex items-center gap-2">
						<span className="h-3 w-3 rounded-full border border-[#76b900] bg-[#eef7df]" />
						<Icon name={IconName.Terms} className="h-4 w-4 text-[#76b900]" />
						<span className="text-xs font-medium text-zinc-700 dark:text-zinc-200">
							Terms {graph.nodes.length}
						</span>
					</div>
				) : (
					<div className="mt-2 flex items-center gap-3">
						{[
							{ type: TableType.BASE_TABLE, label: 'Tables' },
							{ type: TableType.VIEW, label: 'Views' },
							{ type: TableType.MATERIALIZED_VIEW, label: 'Materialized' },
						].map((item) => (
							<span
								key={item.type}
								className="flex items-center gap-1 text-xs font-medium text-zinc-700 dark:text-zinc-200"
							>
								<Icon
									name={catalogNodeInfo[item.type].icon}
									className="h-4 w-4 text-[#3b82b6]"
								/>
								{item.label}{' '}
								{
									graph.nodes.filter(
										(node) =>
											node.layer === ExplorationLayer.Data &&
											node.nodeType === item.type,
									).length
								}
							</span>
						))}
					</div>
				)}
			</div>

			<QueryCarouselModal
				key={selectedLinkId ?? 'closed-query'}
				link={selectedLink}
				sourceName={selectedLinkSource}
				targetName={selectedLinkTarget}
				onClose={() => setSelectedLinkId(null)}
			/>
			<RelationshipsModal
				node={relationshipsNode}
				rows={relatedNodes}
				onClose={() => setRelationshipsNodeId(null)}
				onFocus={handleSelectNode}
			/>
			<DataDetailsModal
				target={activeNode?.layer === ExplorationLayer.Data ? activeNode : null}
				kind={dataDetailsKind}
				onClose={() => setDataDetailsKind(null)}
			/>
			<ColumnAttributesModal
				term={columnAttributesNode}
				onClose={() => setColumnAttributesNodeId(null)}
			/>
			<SqlAttributesModal
				term={sqlAttributesNode}
				onClose={() => setSqlAttributesNodeId(null)}
			/>
			<SemanticRelationshipModal
				sourceTerm={semanticEdgeSourceNode}
				targetTerm={semanticEdgeTargetNode}
				onClose={() => setSelectedSemanticEdgeId(null)}
				onView={(termId) => router.push(`/terms?focus=${encodeURIComponent(termId)}`)}
			/>
		</main>
	);
};
