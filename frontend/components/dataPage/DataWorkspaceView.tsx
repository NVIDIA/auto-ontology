// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { BackPanelLayout } from '@/common/BackPanelLayout';
import type { NodePatch } from '@/api/types';
import { EmptyState } from '@/common/EmptyState';
import { IconName } from '@/common/icons';
import { SkeletonBlock, SkeletonCard, SkeletonRows } from '@/common/Skeleton';
import { DataModels } from '@/enums/datasources';
import { EmptyStateVariant } from '@/enums/emptyState';
import { TagItemType } from '@/enums/tags';
import { ToastVariant } from '@/enums/toast';
import { DataTree } from './DataTree';
import { SinglePageView, type SinglePageFormat } from '@/common/SinglePageView';
import type { ComposerEditValue } from '@/common/SinglePageComposer';
import { Toast } from '@/common/Toast';
import type { Column, Database, Schema, Table } from '@/types/datasources';
import { isCatalogBranchLoadedForFocus } from '@/lib/data/catalog-branch-loaded';
import { buildTreeFocusPageFormat, resolveTreeNode } from '@/lib/data/tree-focus-page';
import { TAGS_SECTION_ID, fetchTagOptions, stagedTagIds, syncTags } from '@/lib/tags';
import {
	mergeColumnsIntoTable,
	mergeSchemasIntoDatabase,
	mergeTablesIntoSchema,
	patchNodeInTree,
} from '@/lib/data/datasource-tree-merge';
import { datasources } from '@/api/datasources';

export type DataWorkspaceViewProps = Record<string, never>;

type CatalogNodePatch = Partial<Database> & Partial<Schema> & Partial<Table> & Partial<Column>;

/**
 * Which kind of tag target the focused node is, or null for the kinds that
 * carry no tags.
 *
 * Read off the focus path — `dbId|schemaId|tableId|columnId` truncated to the
 * depth opened — rather than off the tree, so it also answers before the branch
 * has been hydrated, which is when the page decides whether to read the tags.
 */
const focusedTagType = (focusId: string | null): TagItemType | null => {
	const depth = focusId?.split('|').filter((segment) => segment.length > 0).length ?? 0;
	if (depth === 3) return TagItemType.Table;
	if (depth === 4) return TagItemType.Column;
	return null;
};

export function DataWorkspaceView() {
	const searchParams = useSearchParams();
	const rawFocus = searchParams.get('focus');
	const treeFocusId = rawFocus != null && rawFocus.trim() !== '' ? rawFocus.trim() : null;

	const [databases, setDatabases] = useState<Database[]>([]);
	const [loadError, setLoadError] = useState<string | null>(null);
	const [certError, setCertError] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);

	const databasesRef = useRef<Database[]>([]);
	const [treeDatabases, setTreeDatabases] = useState<Database[]>([]);
	const [treeDataEpoch, setTreeDataEpoch] = useState(0);
	const treeEpochFlushRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const inFlightRef = useRef(false);

	const workspaceDb = databases[0];
	const workspaceDataId = workspaceDb?.id ?? '';

	const fetchDatabases = useCallback(async () => {
		if (inFlightRef.current) return;
		inFlightRef.current = true;
		setLoading(true);
		try {
			const res = await datasources.getDBs();
			if (res.error) {
				setLoadError(res.message ?? 'Failed to load databases');
				return;
			}
			const next = res.data ?? [];
			setLoadError(null);
			const prevIds = new Set(databasesRef.current.map((d) => d.id));
			const nextIds = new Set(next.map((d) => d.id));
			const sameWorkspace =
				prevIds.size === nextIds.size && [...nextIds].every((id) => prevIds.has(id));
			databasesRef.current = next;
			setDatabases(next);
			if (!sameWorkspace) {
				setTreeDatabases(next);
				setTreeDataEpoch((n) => n + 1);
			}
		} finally {
			inFlightRef.current = false;
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		void fetchDatabases();
	}, [fetchDatabases]);

	// Auto-retry whenever the tab regains focus while we're in an error state
	// (typical case: backend was briefly down on a different terminal).
	useEffect(() => {
		if (loadError == null) return undefined;
		const onFocus = () => {
			void fetchDatabases();
		};
		window.addEventListener('focus', onFocus);
		return () => window.removeEventListener('focus', onFocus);
	}, [loadError, fetchDatabases]);

	useEffect(
		() => () => {
			if (treeEpochFlushRef.current != null) {
				clearTimeout(treeEpochFlushRef.current);
				treeEpochFlushRef.current = null;
			}
		},
		[],
	);

	const handleTreeDataUpdated = useCallback((dbs: Database[]) => {
		databasesRef.current = dbs;
		if (treeEpochFlushRef.current != null) return;
		treeEpochFlushRef.current = setTimeout(() => {
			treeEpochFlushRef.current = null;
			setTreeDatabases(databasesRef.current);
			setTreeDataEpoch((n) => n + 1);
		}, 0);
	}, []);

	const hydrateBranchForFocus = useCallback(async (focusId: string | null) => {
		if (!focusId) return;
		const parts = focusId.split('|').filter((p) => p.length > 0);
		const dbId = parts[0];
		if (!dbId) return;

		const dbs = databasesRef.current;
		if (!dbs.some((d) => d.id === dbId)) return;
		if (isCatalogBranchLoadedForFocus(dbs, focusId)) return;

		const schemaId = parts[1];
		const tableId = parts[2];

		let next = databasesRef.current;
		const db = next.find((d) => d.id === dbId);
		if (db && db.schemas.length === 0) {
			const r = await datasources.getSchemasForDatabase(dbId);
			if (r.error || !r.data) return;
			next = mergeSchemasIntoDatabase(next, dbId, r.data);
		}

		if (schemaId !== undefined && schemaId !== '') {
			const sch = next.find((d) => d.id === dbId)?.schemas.find((s) => s.id === schemaId);
			if (sch && sch.tables.length === 0 && (sch.tables_count ?? 0) > 0) {
				const r = await datasources.getTablesForSchema(schemaId);
				if (r.error || !r.data) return;
				next = mergeTablesIntoSchema(next, schemaId, r.data);
			}
		}

		if (parts.length >= 4 && tableId !== undefined && tableId !== '') {
			const sch2 = next.find((d) => d.id === dbId)?.schemas.find((s) => s.id === schemaId);
			const tbl = sch2?.tables.find((t) => t.id === tableId);
			if (tbl && tbl.columns.length === 0 && tbl.columns_count > 0) {
				const r = await datasources.getColumnsForTable(tableId);
				if (r.error || !r.data) return;
				next = mergeColumnsIntoTable(next, tableId, r.data);
			}
		}

		databasesRef.current = next;
	}, []);

	const getSinglePage = useCallback(
		async (_dataId: string, treeFocus: string | null): Promise<SinglePageFormat> => {
			// The picker's options are read only for the kinds that have a tags
			// section, and beside the hydration rather than after it.
			const [, tagOptions] = await Promise.all([
				hydrateBranchForFocus(treeFocus),
				focusedTagType(treeFocus) == null ? [] : fetchTagOptions(),
			]);
			return buildTreeFocusPageFormat(treeFocus, databasesRef.current, tagOptions);
		},
		[hydrateBranchForFocus],
	);

	const focusedEntityId = useMemo(() => {
		if (!treeFocusId) return null;
		const segments = treeFocusId.split('|').filter((s) => s.length > 0);
		return segments[segments.length - 1] ?? null;
	}, [treeFocusId]);

	// Merges a patch into the matching catalog node anywhere in the tree and
	// bumps the epoch so the detail page rebuilds with the new values.
	const applyNodePatch = useCallback((entityId: string, patch: CatalogNodePatch) => {
		if (!entityId || Object.keys(patch).length === 0) return;
		const [updated, found] = patchNodeInTree(databasesRef.current, entityId, patch);
		if (!found) return;
		databasesRef.current = updated;
		setTreeDatabases(updated);
		setTreeDataEpoch((n) => n + 1);
	}, []);

	/**
	 * Saves the Edit toolbar's pending edits.
	 *
	 * Written out rather than left to the composer's default `updateNode` path
	 * because tags do not go through that endpoint: they are their own writes,
	 * against a different one, and the rest of the edits are keyed by section
	 * id, which for the editable catalog sections is already `description` /
	 * `sample_values`.
	 *
	 * Nothing is re-read afterwards — the catalog page renders the tree this
	 * component holds, so each write is followed by patching the node it
	 * changed, which is what redraws the page.
	 */
	const handleSave = useCallback(
		async (edits: Record<string, ComposerEditValue>) => {
			if (focusedEntityId == null) {
				return { error: true, message: 'No catalog node is selected' };
			}
			const { [TAGS_SECTION_ID]: tagEdit, ...nodeEdits } = edits;

			if (Object.keys(nodeEdits).length > 0) {
				const res = await datasources.updateNode(focusedEntityId, nodeEdits as NodePatch);
				if (res.error) {
					return { error: true, message: res.message ?? 'Failed to save changes' };
				}
				applyNodePatch(focusedEntityId, nodeEdits as CatalogNodePatch);
			}

			const nextTagIds = stagedTagIds(tagEdit);
			const tagType = focusedTagType(treeFocusId);
			if (nextTagIds != null && tagType != null) {
				const focused = resolveTreeNode(treeFocusId, databasesRef.current);
				const current =
					focused.type === DataModels.TABLE
						? focused.table.tags
						: focused.type === DataModels.COLUMN
							? focused.column.tags
							: [];
				const { error, tags } = await syncTags({
					type: tagType,
					itemId: focusedEntityId,
					current: current ?? [],
					nextIds: nextTagIds,
				});
				// Whatever landed is what the node carries now, so the tree is
				// brought up to it either way.
				applyNodePatch(focusedEntityId, { tags });
				if (error != null) {
					return { error: true, message: error };
				}
			}

			return { error: false };
		},
		[applyNodePatch, focusedEntityId, treeFocusId],
	);

	// Certification saves immediately, independent of the description Save
	// toolbar. Only the Description card carries certification for catalog
	// nodes, so the composer's field id is unused.
	const handleCertificationChange = useCallback(
		async (_id: string, certified: boolean) => {
			if (focusedEntityId == null) return;
			const res = await datasources.updateNode(focusedEntityId, {
				description_certified: certified,
			});
			if (res.error) {
				setCertError(res.message ?? 'Failed to update certification');
				return;
			}
			setCertError(null);
			applyNodePatch(focusedEntityId, { description_certified: certified });
		},
		[applyNodePatch, focusedEntityId],
	);

	const handleChildCertificationChange = useCallback(
		async (_sectionId: string, rowId: string, certified: boolean) => {
			const res = await datasources.updateNode(rowId, {
				description_certified: certified,
			});
			if (res.error) {
				setCertError(res.message ?? 'Failed to update certification');
				return;
			}
			setCertError(null);
			applyNodePatch(rowId, { description_certified: certified });
		},
		[applyNodePatch],
	);

	if (loadError) {
		return (
			<div className="mx-auto flex w-full max-w-lg flex-1 flex-col items-center justify-center px-4 py-16 text-center">
				<div className="rounded-2xl border border-red-200/80 bg-white/90 px-8 py-10 shadow-xl shadow-red-100/50 dark:border-red-900/50 dark:bg-zinc-950/80 dark:shadow-none">
					<div
						className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-red-100 text-2xl dark:bg-red-950/80"
						aria-hidden
					>
						⚠
					</div>
					<h2 className="text-lg font-semibold tracking-tight text-red-800 dark:text-red-300">
						Couldn&apos;t load databases
					</h2>
					<pre className="mt-4 max-w-full overflow-x-auto rounded-lg border border-red-100 bg-red-50/80 p-3 text-left text-xs text-red-900/80 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
						{loadError}
					</pre>
				</div>
			</div>
		);
	}

	if (loading && databases.length === 0) {
		return (
			<div className="flex h-full flex-1" role="status" aria-label="Loading databases">
				<aside className="w-72 border-r border-zinc-200 p-4 dark:border-zinc-800">
					<SkeletonBlock className="mb-5 h-6 w-24" />
					<SkeletonRows rows={8} />
				</aside>
				<main className="flex flex-1 flex-col gap-4 p-6">
					<SkeletonCard rows={4} />
					<SkeletonCard rows={4} />
				</main>
			</div>
		);
	}

	if (!workspaceDb) {
		return (
			<EmptyState
				variant={EmptyStateVariant.Borderless}
				icon={IconName.Database}
				title="No Databases found"
			/>
		);
	}

	return (
		<BackPanelLayout
			panelAriaLabel="Datasource tree"
			expandAriaLabel="Expand explorer"
			collapseAriaLabel="Collapse explorer"
			panel={
				<DataTree
					key="data-catalog-tree"
					initialDatabases={treeDatabases}
					selectedId={treeFocusId ?? undefined}
					pathBase="/data"
					onTreeDataUpdated={handleTreeDataUpdated}
				/>
			}
		>
			<main className="flex min-w-0 flex-1 flex-col overflow-y-auto">
				<SinglePageView
					dataId={workspaceDataId}
					treeFocusId={treeFocusId}
					treeDataEpoch={treeDataEpoch}
					getSinglePage={getSinglePage}
					onPatchEdits={handleSave}
					onCertificationChange={handleCertificationChange}
					onDataTableCertificationChange={handleChildCertificationChange}
				/>
			</main>
			<Toast
				open={certError != null}
				message={certError ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setCertError(null)}
			/>
		</BackPanelLayout>
	);
}
