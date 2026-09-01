// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useState } from 'react';
import type { Database } from '@/types/datasources';
import { zonesApi } from '@/api/zones';
import { datasources } from '@/api/datasources';
import { ModalWithSteps, ConfirmModal } from '@/common/modal';
import { ColorPicker } from '@/components/settings/ColorPicker';
import type { ColorOption } from '@/components/settings/ColorPicker';
import { Button } from '@/common/Button';
import { EmptyState } from '@/common/EmptyState';
import { Toggle } from '@/common/Toggle';
import { Size, ButtonTheme } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import { Icon, IconName } from '@/common/icons';
import { SkeletonCard, SkeletonRows } from '@/common/Skeleton';
import { PopoverMenu } from '@/common/PopoverMenu';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import { ZonesDataTree } from '@/components/settings/ZonesDataTree';
import { mergeSchemasIntoDatabase, mergeTablesIntoSchema } from '@/lib/data/datasource-tree-merge';
import type { Zone, ZoneCreated, ZoneUpdateInput } from '@/types/zones';
import { useSession } from '@/auth/auth-client';
import { Role } from '@/enums/auth';

const ZONE_COLORS: readonly ColorOption[] = [
	{ value: '#0ea5e9', label: 'Sky', colorClass: 'bg-sky-500' },
	{ value: '#8b5cf6', label: 'Violet', colorClass: 'bg-violet-500' },
	{ value: '#ec4899', label: 'Pink', colorClass: 'bg-pink-500' },
	{ value: '#f97316', label: 'Orange', colorClass: 'bg-orange-500' },
	{ value: '#14b8a6', label: 'Teal', colorClass: 'bg-teal-500' },
	{ value: '#eab308', label: 'Amber', colorClass: 'bg-yellow-500' },
];
const DEFAULT_ZONE_COLOR = '#0ea5e9';

const normalizeDescription = (value: string): string | null => {
	const trimmed = value.trim();
	return trimmed === '' ? null : trimmed;
};

const setsAreEqual = (left: Set<string>, right: Set<string>): boolean => {
	if (left.size !== right.size) return false;
	for (const item of left) {
		if (!right.has(item)) return false;
	}
	return true;
};

const normalizeZoneItemSelection = (selected: Set<string>, databases: Database[]): Set<string> => {
	const normalized = new Set<string>();
	const known = new Set<string>();

	const isSchemaFullySelected = (database: Database, schemaId: string): boolean => {
		const schema = database.schemas.find((item) => item.id === schemaId);
		if (schema == null) return false;
		const tables = schema.tables ?? [];
		if (tables.length === 0) {
			return selected.has(schema.id);
		}
		return tables.every((table) => selected.has(table.id));
	};

	const isDatabaseFullySelected = (database: Database): boolean => {
		const schemas = database.schemas ?? [];
		if (schemas.length === 0) {
			return selected.has(database.id);
		}
		return schemas.every((schema) => isSchemaFullySelected(database, schema.id));
	};

	databases.forEach((database) => {
		known.add(database.id);
		if (isDatabaseFullySelected(database)) {
			normalized.add(database.id);
			database.schemas.forEach((schema) => {
				known.add(schema.id);
				schema.tables.forEach((table) => {
					known.add(table.id);
					table.columns.forEach((column) => known.add(column.id));
				});
			});
			return;
		}

		database.schemas.forEach((schema) => {
			known.add(schema.id);
			if (isSchemaFullySelected(database, schema.id)) {
				normalized.add(schema.id);
				schema.tables.forEach((table) => {
					known.add(table.id);
					table.columns.forEach((column) => known.add(column.id));
				});
				return;
			}

			schema.tables.forEach((table) => {
				known.add(table.id);
				if (selected.has(table.id)) {
					normalized.add(table.id);
				}
				table.columns.forEach((column) => known.add(column.id));
			});
		});
	});

	selected.forEach((itemId) => {
		if (!known.has(itemId)) {
			normalized.add(itemId);
		}
	});

	return normalized;
};

const expandInitialZoneItemSelection = (
	selected: Set<string>,
	databases: Database[],
): Set<string> => {
	const expanded = new Set<string>(selected);
	let changed = true;

	while (changed) {
		changed = false;
		databases.forEach((database) => {
			if (expanded.has(database.id)) {
				database.schemas.forEach((schema) => {
					if (!expanded.has(schema.id)) {
						expanded.add(schema.id);
						changed = true;
					}
				});
			}

			database.schemas.forEach((schema) => {
				if (!expanded.has(schema.id)) return;
				schema.tables.forEach((table) => {
					if (!expanded.has(table.id)) {
						expanded.add(table.id);
						changed = true;
					}
				});
			});
		});
	}

	return expanded;
};

// ---------------------------------------------------------------------------
// ZoneCard
// ---------------------------------------------------------------------------

const ZoneCard = ({
	zone,
	isAdmin,
	toggling,
	onEdit,
	onDelete,
	onToggleEnabled,
}: {
	zone: Zone;
	isAdmin: boolean;
	toggling: boolean;
	onEdit: (zone: Zone) => void;
	onDelete: (zone: Zone) => void;
	onToggleEnabled: (zone: Zone) => void;
}) => (
	<div
		className={`rounded-lg border border-zinc-200/90 bg-white/90 p-4 shadow-sm ring-1 ring-zinc-950/[0.04] transition-opacity dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06] ${zone.enabled ? '' : 'opacity-60'}`}
		style={{ borderTop: `4px solid ${zone.color ?? DEFAULT_ZONE_COLOR}` }}
	>
		<div className="flex items-start justify-between gap-4">
			<div className="min-w-0">
				<div className="flex items-center gap-2">
					<Text as="h2" text={zone.name} variant={TextVariant.Heading} />
					{!zone.enabled ? (
						<span className="shrink-0 rounded-full bg-zinc-200 px-2 py-0.5 text-[10px] font-medium tracking-wide text-zinc-600 uppercase dark:bg-zinc-700 dark:text-zinc-300">
							Disabled
						</span>
					) : null}
				</div>
				<p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">{zone.label}</p>
			</div>
			<div className="flex shrink-0 items-start gap-2">
				{isAdmin ? (
					<Toggle
						checked={zone.enabled}
						aria-label={zone.enabled ? `Disable ${zone.name}` : `Enable ${zone.name}`}
						disabled={toggling}
						onChange={() => onToggleEnabled(zone)}
					/>
				) : null}
				<div className="relative">
					<PopoverMenu
						items={[
							{
								label: 'Edit',
								icon: <Icon name={IconName.Pencil} className="h-3.5 w-3.5" />,
								onClick: () => onEdit(zone),
							},
							{
								label: 'Delete',
								icon: <Icon name={IconName.Trash} className="h-3.5 w-3.5" />,
								onClick: () => onDelete(zone),
								danger: true,
							},
						]}
						trigger={({ toggle }) => (
							<Button
								theme={ButtonTheme.IconNeutral}
								size={Size.SMALL}
								iconOnly
								type="button"
								onClick={toggle}
								aria-label="Zone actions"
							>
								<Icon name={IconName.DotsVertical} className="h-4 w-4" />
							</Button>
						)}
					/>
				</div>
			</div>
		</div>
		<p className="mt-3 text-sm text-zinc-700 dark:text-zinc-300">
			{zone.description?.trim() ? zone.description : 'No description'}
		</p>
	</div>
);

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ZonesSettingsPage() {
	const { data: session } = useSession();
	const currentUserId = session?.user?.id ?? '';
	const isAdmin = session?.user?.role === Role.Admin;

	const [modalMode, setModalMode] = useState<'create' | 'edit'>('create');
	const [editingZoneId, setEditingZoneId] = useState<string | null>(null);
	const [zones, setZones] = useState<Zone[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const [modalOpen, setModalOpen] = useState(false);
	const [activeStep, setActiveStep] = useState(0);
	const [name, setName] = useState('');
	const [description, setDescription] = useState('');
	const [color, setColor] = useState<string>(DEFAULT_ZONE_COLOR);
	const [treeDatabases, setTreeDatabases] = useState<Database[]>([]);
	const [treeLoading, setTreeLoading] = useState(false);
	const [treeError, setTreeError] = useState<string | null>(null);
	const [treeExpandSelectedOnlyKey, setTreeExpandSelectedOnlyKey] = useState(0);
	const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
	const [editItemsHydrated, setEditItemsHydrated] = useState(false);
	const [initialEditName, setInitialEditName] = useState('');
	const [initialEditDescription, setInitialEditDescription] = useState<string | null>(null);
	const [initialEditColor, setInitialEditColor] = useState<string | null>(null);
	const [initialEditSelectedItems, setInitialEditSelectedItems] = useState<Set<string>>(
		new Set(),
	);
	const [submitting, setSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState<string | null>(null);
	const [confirmDeleteZone, setConfirmDeleteZone] = useState<Zone | null>(null);
	const [deleteError, setDeleteError] = useState<string | null>(null);
	const [deletingZone, setDeletingZone] = useState(false);
	const [togglingZoneId, setTogglingZoneId] = useState<string | null>(null);
	const [toggleError, setToggleError] = useState<string | null>(null);

	const normalizedName = name.trim();
	const normalizedDescription = normalizeDescription(description);
	const nameExists = zones.some((zone) => {
		if (editingZoneId != null && zone.id === editingZoneId) {
			return false;
		}
		return zone.name.trim().toLowerCase() === normalizedName.toLowerCase();
	});
	const normalizedSelectedItems = normalizeZoneItemSelection(selectedItems, treeDatabases);
	const normalizedInitialEditSelectedItems = normalizeZoneItemSelection(
		expandInitialZoneItemSelection(initialEditSelectedItems, treeDatabases),
		treeDatabases,
	);
	const hasNormalizedSelectedData = normalizedSelectedItems.size > 0;
	const hasEditChanges =
		modalMode === 'edit' &&
		(normalizedName !== initialEditName ||
			normalizedDescription !== initialEditDescription ||
			color !== (initialEditColor ?? DEFAULT_ZONE_COLOR) ||
			!setsAreEqual(normalizedSelectedItems, normalizedInitialEditSelectedItems));

	const loadZones = useCallback(async () => {
		if (!currentUserId) return;
		setLoading(true);
		const response = await zonesApi.getAll(currentUserId);
		if (response.error) {
			setError(response.message ?? 'Failed to load zones.');
			setZones([]);
			setLoading(false);
			return;
		}
		setError(null);
		setZones(response.data ?? []);
		setLoading(false);
	}, [currentUserId]);

	useEffect(() => {
		if (!currentUserId) return;
		let cancelled = false;
		zonesApi.getAll(currentUserId).then((response) => {
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load zones.');
				setZones([]);
			} else {
				setError(null);
				setZones(response.data ?? []);
			}
			setLoading(false);
		});
		return () => {
			cancelled = true;
		};
	}, [currentUserId]);

	const openCreateModal = () => {
		setModalMode('create');
		setEditingZoneId(null);
		setName('');
		setDescription('');
		setColor(DEFAULT_ZONE_COLOR);
		setActiveStep(0);
		setTreeDatabases([]);
		setTreeError(null);
		setTreeExpandSelectedOnlyKey((prev) => prev + 1);
		setSelectedItems(new Set());
		setEditItemsHydrated(false);
		setInitialEditName('');
		setInitialEditDescription(null);
		setInitialEditColor(null);
		setInitialEditSelectedItems(new Set());
		setSubmitError(null);
		setModalOpen(true);
	};

	const openEditModal = async (zone: Zone) => {
		setModalMode('edit');
		setEditingZoneId(zone.id);
		setName(zone.name);
		setDescription(zone.description ?? '');
		setColor(zone.color ?? DEFAULT_ZONE_COLOR);
		setActiveStep(0);
		setTreeDatabases([]);
		setTreeError(null);
		setTreeExpandSelectedOnlyKey((prev) => prev + 1);
		setSelectedItems(new Set());
		setEditItemsHydrated(false);
		setInitialEditName(zone.name.trim());
		setInitialEditDescription(normalizeDescription(zone.description ?? ''));
		setInitialEditColor(zone.color ?? DEFAULT_ZONE_COLOR);
		setInitialEditSelectedItems(new Set());
		setSubmitError(null);
		setModalOpen(true);
	};

	const closeZoneModal = () => {
		if (submitting) return;
		setModalOpen(false);
		setSubmitError(null);
	};

	const canSubmit = !submitting && normalizedName.length > 0 && !nameExists;
	const canGoNext = normalizedName.length > 0 && !nameExists;

	const loadTreeDatabases = useCallback(async (): Promise<Database[]> => {
		setTreeLoading(true);
		const response = await datasources.getDBs();
		if (response.error) {
			setTreeError(response.message ?? 'Failed to load catalog data.');
			setTreeDatabases([]);
			setTreeLoading(false);
			return [];
		}
		const nextDatabases = response.data ?? [];
		setTreeError(null);
		setTreeDatabases(nextDatabases);
		setTreeLoading(false);
		return nextDatabases;
	}, []);

	const hydrateTreeForSelectedItems = useCallback(
		async (baseDatabases: Database[], selected: Set<string>): Promise<Database[]> => {
			if (selected.size === 0 || baseDatabases.length === 0) return baseDatabases;

			let nextDatabases = baseDatabases;
			const schemaResponses = await Promise.all(
				baseDatabases.map(async (database) => ({
					databaseId: database.id,
					response: await datasources.getSchemasForDatabase(database.id),
				})),
			);
			schemaResponses.forEach(({ databaseId, response }) => {
				if (response.error || !response.data) return;
				nextDatabases = mergeSchemasIntoDatabase(nextDatabases, databaseId, response.data);
			});

			const schemaNodes = nextDatabases.flatMap((database) =>
				database.schemas.map((schema) => ({
					schemaId: schema.id,
					databaseName: database.name,
				})),
			);
			const tableResponses = await Promise.all(
				schemaNodes.map(async (schemaNode) => ({
					schemaId: schemaNode.schemaId,
					response: await datasources.getTablesForSchema(schemaNode.schemaId, {
						databaseName: schemaNode.databaseName,
					}),
				})),
			);
			tableResponses.forEach(({ schemaId, response }) => {
				if (response.error || !response.data) return;
				nextDatabases = mergeTablesIntoSchema(nextDatabases, schemaId, response.data);
			});

			return nextDatabases;
		},
		[],
	);

	const loadSchemasForDatabase = useCallback(async (dbId: string) => {
		const response = await datasources.getSchemasForDatabase(dbId);
		if (response.error || !response.data) return [];
		setTreeDatabases((prev) => mergeSchemasIntoDatabase(prev, dbId, response.data ?? []));
		return response.data ?? [];
	}, []);

	const loadTablesForSchema = useCallback(async (schemaId: string) => {
		const response = await datasources.getTablesForSchema(schemaId);
		if (response.error || !response.data) return;
		setTreeDatabases((prev) => mergeTablesIntoSchema(prev, schemaId, response.data ?? []));
	}, []);

	const goToDataStep = async () => {
		if (!canGoNext || submitting) return;
		let effectiveSelectedItems = selectedItems;

		if (modalMode === 'edit' && editingZoneId != null && !editItemsHydrated) {
			const detailResponse = await zonesApi.getById(editingZoneId, currentUserId);
			if (detailResponse.error) {
				setSubmitError(detailResponse.message ?? 'Failed to load zone items.');
			} else {
				const itemIds = detailResponse.data?.items.map((item) => item.id) ?? [];
				effectiveSelectedItems = new Set(itemIds);
				setSelectedItems(effectiveSelectedItems);
				setInitialEditSelectedItems(new Set(itemIds));
			}
			setEditItemsHydrated(true);
		}

		setActiveStep(1);
		let nextDatabases = await loadTreeDatabases();

		if (modalMode === 'edit') {
			setTreeLoading(true);
			nextDatabases = await hydrateTreeForSelectedItems(
				nextDatabases,
				effectiveSelectedItems,
			);
			setTreeDatabases(nextDatabases);
			setTreeLoading(false);
		}
		setTreeExpandSelectedOnlyKey((prev) => prev + 1);
	};

	const handleSubmit = async () => {
		if (!canSubmit || activeStep !== 1) return;
		setSubmitting(true);
		setSubmitError(null);

		if (modalMode === 'create') {
			if (!hasNormalizedSelectedData) {
				setSubmitting(false);
				return;
			}
			const response = await zonesApi.create({
				name: normalizedName,
				description: normalizedDescription ?? undefined,
				color,
				items: Array.from(normalizedSelectedItems),
				created_by: currentUserId,
			});

			if (response.error) {
				setSubmitting(false);
				setSubmitError(response.message ?? 'Failed to create zone.');
				return;
			}

			const created: ZoneCreated | undefined = response.data;
			if (created != null) {
				const { id, name: n, label, description: d, color: c, enabled: en } = created;
				setZones((prev) => {
					const next = [
						...prev.filter((z) => z.id !== created.id),
						{ id, name: n, label, description: d, color: c, enabled: en },
					];
					return next.sort((a, b) => a.name.localeCompare(b.name));
				});
			} else {
				await loadZones();
			}
			setSubmitting(false);
			setModalOpen(false);
			return;
		}

		if (editingZoneId == null || !hasEditChanges) {
			setSubmitting(false);
			return;
		}

		const patch: ZoneUpdateInput = {};
		if (normalizedName !== initialEditName) {
			patch.name = normalizedName;
		}
		if (normalizedDescription !== initialEditDescription) {
			patch.description = normalizedDescription;
		}
		if (color !== (initialEditColor ?? DEFAULT_ZONE_COLOR)) {
			patch.color = color;
		}
		if (!setsAreEqual(normalizedSelectedItems, normalizedInitialEditSelectedItems)) {
			patch.items = Array.from(normalizedSelectedItems);
		}

		const hasFieldChanges = Object.keys(patch).length > 0;
		const response = hasFieldChanges ? await zonesApi.update(editingZoneId, patch) : null;

		setSubmitting(false);

		if (response && response.error) {
			setSubmitError(response.message ?? 'Failed to update zone.');
			return;
		}

		const updated = response?.data;
		if (updated != null) {
			setZones((prev) =>
				prev.map((zone) =>
					zone.id === updated.id
						? {
								...zone,
								name: updated.name,
								description: updated.description,
								color: updated.color,
							}
						: zone,
				),
			);
		} else {
			await loadZones();
		}
		setModalOpen(false);
	};

	const handleRequestZoneDelete = (zone: Zone) => {
		setDeleteError(null);
		setConfirmDeleteZone(zone);
	};

	const handleCancelZoneDelete = () => {
		if (deletingZone) return;
		setConfirmDeleteZone(null);
		setDeleteError(null);
	};

	const handleConfirmZoneDelete = async () => {
		if (!confirmDeleteZone) return;
		setDeletingZone(true);
		setDeleteError(null);
		const response = await zonesApi.delete(confirmDeleteZone.id);
		setDeletingZone(false);

		if (response.error) {
			setDeleteError(response.message ?? 'Failed to delete zone.');
			return;
		}

		setZones((prev) => prev.filter((zone) => zone.id !== confirmDeleteZone.id));
		setConfirmDeleteZone(null);
	};

	const handleToggleZoneEnabled = async (zone: Zone) => {
		if (togglingZoneId != null) return;
		const nextEnabled = !zone.enabled;
		setTogglingZoneId(zone.id);
		setToggleError(null);
		setZones((prev) =>
			prev.map((z) => (z.id === zone.id ? { ...z, enabled: nextEnabled } : z)),
		);

		const response = await zonesApi.setEnabled(zone.id, nextEnabled, currentUserId);
		setTogglingZoneId(null);

		if (response.error) {
			setZones((prev) =>
				prev.map((z) => (z.id === zone.id ? { ...z, enabled: zone.enabled } : z)),
			);
			setToggleError(response.message ?? 'Failed to update zone status.');
		}
	};

	return (
		<main className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] px-7 py-6 sm:px-10 sm:py-7 dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]">
			<div className="w-full space-y-5">
				<div className="flex items-center gap-3">
					<h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
						Zones
					</h1>
					<div className="ml-auto">
						<Button
							theme={ButtonTheme.Primary}
							size={Size.REGULAR}
							type="button"
							onClick={openCreateModal}
							iconPosition="left"
							shadow
						>
							<Icon name={IconName.Plus} className="h-4 w-4" />
							Create new zone
						</Button>
					</div>
				</div>

				{error ? (
					<div className="rounded-lg border border-red-200/90 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
						{error}
					</div>
				) : null}

				{toggleError ? (
					<div className="rounded-lg border border-red-200/90 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
						{toggleError}
					</div>
				) : null}

				{loading ? (
					<div
						className="grid grid-cols-1 gap-4 lg:grid-cols-2"
						role="status"
						aria-label="Loading zones"
					>
						{Array.from({ length: 4 }).map((_, index) => (
							<SkeletonCard key={index} rows={3} />
						))}
					</div>
				) : zones.length > 0 ? (
					<div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
						{zones.map((zone) => (
							<ZoneCard
								key={zone.id}
								zone={zone}
								isAdmin={isAdmin}
								toggling={togglingZoneId === zone.id}
								onEdit={(z) => {
									void openEditModal(z);
								}}
								onDelete={handleRequestZoneDelete}
								onToggleEnabled={(z) => {
									void handleToggleZoneEnabled(z);
								}}
							/>
						))}
					</div>
				) : (
					<EmptyState
						variant={EmptyStateVariant.Inline}
						icon={IconName.Key}
						title="No zones found"
						className="rounded-lg border border-dashed border-zinc-300/90 bg-white/70 dark:border-zinc-600 dark:bg-zinc-900/30"
					/>
				)}
			</div>
			<ModalWithSteps
				open={modalOpen}
				onClose={closeZoneModal}
				title={modalMode === 'edit' ? 'Edit zone' : 'Create new zone'}
				steps={['Info', 'Data to Connect']}
				activeStep={activeStep}
				onActiveStepChange={(step) => setActiveStep(step)}
				disabledSteps={activeStep === 0 ? [1] : []}
				alert={submitError ?? treeError}
				className="flex w-[760px] max-w-full flex-col"
				footerActions={
					activeStep === 0
						? [
								{
									label: 'Next',
									onClick: () => {
										void goToDataStep();
									},
									disabled: !canGoNext,
								},
							]
						: modalMode === 'create'
							? [
									{
										label: 'Back',
										onClick: () => setActiveStep(0),
										variant: 'outline' as const,
										disabled: submitting,
									},
									{
										label: 'Create zone',
										onClick: () => {
											void handleSubmit();
										},
										disabled: !hasNormalizedSelectedData,
										loading: submitting,
									},
								]
							: [
									{
										label: 'Back',
										onClick: () => setActiveStep(0),
										variant: 'outline' as const,
										disabled: submitting,
									},
									{
										label: 'Save changes',
										onClick: () => {
											void handleSubmit();
										},
										disabled: !canSubmit || !hasEditChanges,
										loading: submitting,
									},
								]
				}
			>
				{activeStep === 0 ? (
					<>
						<div className="flex items-end gap-3">
							<div className="flex-1">
								<label className="mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100">
									Name <span className="text-red-500">*</span>
								</label>
								<input
									type="text"
									value={name}
									onChange={(e) => setName(e.target.value)}
									placeholder="Zone name"
									className={`w-full rounded-lg border bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500 ${nameExists ? 'border-red-400 focus:border-red-500 focus:ring-2 focus:ring-red-500/30 dark:border-red-500 dark:focus:border-red-400 dark:focus:ring-red-400/30' : 'border-zinc-300 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600'}`}
								/>
								{nameExists ? (
									<p className="mt-1 text-xs text-red-500 dark:text-red-400">
										A zone with this name already exists
									</p>
								) : null}
							</div>
							<ColorPicker
								colors={ZONE_COLORS}
								value={color}
								onChange={setColor}
								fallbackColorClass="bg-sky-500"
							/>
						</div>
						<div>
							<label className="mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100">
								Description
							</label>
							<textarea
								value={description}
								onChange={(e) => setDescription(e.target.value)}
								placeholder="Optional description"
								rows={3}
								className="w-full resize-y rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500"
							/>
						</div>
					</>
				) : (
					<div className="space-y-3">
						<p className="text-sm text-zinc-600 dark:text-zinc-300">
							Choose data to connect for this zone.
						</p>
						{treeLoading ? (
							<div
								className="min-h-[220px] py-4"
								role="status"
								aria-label="Loading data tree"
							>
								<SkeletonRows rows={7} />
							</div>
						) : (
							<ZonesDataTree
								databases={treeDatabases}
								selectedItems={selectedItems}
								onSelectedItemsChange={setSelectedItems}
								onLoadSchemas={loadSchemasForDatabase}
								onLoadTables={loadTablesForSchema}
								expandSelectedOnlyKey={treeExpandSelectedOnlyKey}
							/>
						)}
					</div>
				)}
			</ModalWithSteps>
			<ConfirmModal
				open={confirmDeleteZone !== null}
				title="Delete zone"
				message="Are you sure you want to delete this zone? This action cannot be undone."
				onConfirm={handleConfirmZoneDelete}
				onCancel={handleCancelZoneDelete}
				confirming={deletingZone}
				error={deleteError}
			/>
		</main>
	);
}
