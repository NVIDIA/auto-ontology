// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

import { Placeholders } from '@/assets/images/placeholders';
import { Icon, IconName } from '@/common/icons';
import { ConfirmModal, ModalCreateNewItem } from '@/common/modal';
import { SearchInput } from '@/common/SearchInput';
import { termsApi } from '@/api/terms';
import { sqlAttributesApi } from '@/api/sqlAttributes';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import { type ComposerEditValue } from '@/common/SinglePageComposer';
import { Label } from '@/common/Label';
import { ComposerSectionKind } from '@/enums/datasources';
import { SinglePageView, type SinglePageFormat } from '@/common/SinglePageView';
import { SqlEditor } from '@/common/SqlBlock';
import type { ColumnAttribute, SqlAttribute, Term } from '@/types/terms';

type TermCardProps = {
	term: Term;
	columnAttributeCount: number;
	sqlAttributeCount: number;
	relatedCount: number;
	onClick: (term: Term) => void;
};

type SqlAttributeDeleteTarget = {
	id: string;
	name: string;
};

const TermCard = ({
	term,
	columnAttributeCount,
	sqlAttributeCount,
	relatedCount,
	onClick,
}: TermCardProps) => (
	<li
		role="button"
		tabIndex={0}
		onClick={() => onClick(term)}
		onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onClick(term)}
		className="cursor-pointer rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition-shadow hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
	>
		{/* Card header */}
		<div className="flex items-start justify-between gap-3">
			<div>
				<h2 className="text-base font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					{term.name}
				</h2>
				{term.synonyms.length > 0 && (
					<p className="mt-0.5 text-xs text-zinc-400 dark:text-zinc-500">
						{term.synonyms.join(', ')}
					</p>
				)}
			</div>
		</div>

		{term.description != null && term.description.trim() !== '' ? (
			<p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">{term.description}</p>
		) : (
			<p className="mt-2 text-sm italic text-zinc-400 dark:text-zinc-500">
				No Description Available
			</p>
		)}

		{/* Four-column section */}
		<div className="mt-4 grid grid-cols-4 gap-0 overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-700">
			{/* Column Attributes */}
			<div className="border-r border-zinc-200 dark:border-zinc-700">
				<div className="border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
					<span className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
						Column Attributes
					</span>
				</div>
				<div className="flex items-center">
					<span className="flex w-full items-center justify-between px-4 py-3 text-sm text-zinc-700 dark:text-zinc-300">
						<span>Column Attributes</span>
						<span className="ml-1.5 font-medium text-zinc-900 dark:text-zinc-100">
							{columnAttributeCount}
						</span>
					</span>
				</div>
			</div>

			{/* SQL Attributes */}
			<div className="border-r border-zinc-200 dark:border-zinc-700">
				<div className="border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
					<span className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
						SQL Attributes
					</span>
				</div>
				<div className="flex items-center">
					<span className="flex w-full items-center justify-between px-4 py-3 text-sm text-zinc-700 dark:text-zinc-300">
						<span>SQL Attributes</span>
						<span className="ml-1.5 font-medium text-zinc-900 dark:text-zinc-100">
							{sqlAttributeCount}
						</span>
					</span>
				</div>
			</div>

			{/* Related Terms */}
			<div className="border-r border-zinc-200 dark:border-zinc-700">
				<div className="border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
					<span className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
						Related Terms
					</span>
				</div>
				<div className="flex items-center justify-between px-4 py-3 text-sm text-zinc-700 dark:text-zinc-300">
					<span>Related Terms</span>
					<span className="font-medium text-zinc-900 dark:text-zinc-100">
						{relatedCount}
					</span>
				</div>
			</div>

			{/* Zones */}
			<div>
				<div className="border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
					<span className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
						Zones
					</span>
				</div>
				<div className="flex flex-wrap items-center gap-1.5 px-4 py-3">
					{term.zones.length > 0 ? (
						term.zones.map((zone) => (
							<Label
								key={zone.id}
								label={zone.name}
								color={zone.color}
								muted={!zone.enabled}
							/>
						))
					) : (
						<span className="text-sm text-zinc-500 dark:text-zinc-400">-</span>
					)}
				</div>
			</div>
		</div>
	</li>
);

const FIELD_INPUT_CLASSNAME =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500';

const FIELD_LABEL_CLASSNAME = 'mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100';

export const TermsView = () => {
	const router = useRouter();
	const searchParams = useSearchParams();
	const focusId = searchParams.get('focus');
	const sqlAttrId = searchParams.get('sqlAttr');
	const colAttrId = searchParams.get('colAttr');

	const [terms, setTerms] = useState<Term[]>([]);
	const [sqlAttrs, setSqlAttrs] = useState<SqlAttribute[]>([]);
	const [columnAttrs, setColumnAttrs] = useState<ColumnAttribute[]>([]);
	const [columnAttrCountsMap, setColumnAttrCountsMap] = useState<Map<string, number>>(new Map());
	const [sqlAttrCountsMap, setSqlAttrCountsMap] = useState<Map<string, number>>(new Map());
	const [relatedCountsMap, setRelatedCountsMap] = useState<Map<string, number>>(new Map());
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [searchQuery, setSearchQuery] = useState('');
	const debouncedSearchQuery = useDebouncedValue(searchQuery.trim(), 1000);
	const [createSqlAttrModalOpen, setCreateSqlAttrModalOpen] = useState(false);
	const [sqlAttrsEpoch, setSqlAttrsEpoch] = useState(0);
	const [columnAttrsEpoch, setColumnAttrsEpoch] = useState(0);
	const [sqlAttrName, setSqlAttrName] = useState('');
	const [sqlAttrDescription, setSqlAttrDescription] = useState('');
	const [sqlAttrSql, setSqlAttrSql] = useState('');
	const [sqlAttrValidating, setSqlAttrValidating] = useState(false);
	const [sqlAttrValidationMessage, setSqlAttrValidationMessage] = useState<string | null>(null);
	const [sqlAttrSqlValidated, setSqlAttrSqlValidated] = useState(false);
	const [sqlAttrSubmitting, setSqlAttrSubmitting] = useState(false);
	const [sqlAttrSubmitError, setSqlAttrSubmitError] = useState<string | null>(null);
	const [deletingSqlAttr, setDeletingSqlAttr] = useState<SqlAttributeDeleteTarget | null>(null);
	const [deletingSqlAttrBusy, setDeletingSqlAttrBusy] = useState(false);
	const [deleteSqlAttrError, setDeleteSqlAttrError] = useState<string | null>(null);
	const [termEditing, setTermEditing] = useState(false);
	const [sqlAttrEditing, setSqlAttrEditing] = useState(false);
	const [sqlAttrEditError, setSqlAttrEditError] = useState<string | null>(null);
	const [columnAttrEditing, setColumnAttrEditing] = useState(false);
	const [columnAttrEditError, setColumnAttrEditError] = useState<string | null>(null);

	const [prevFocusId, setPrevFocusId] = useState(focusId);
	const [prevSqlAttrId, setPrevSqlAttrId] = useState(sqlAttrId);
	const [prevColAttrId, setPrevColAttrId] = useState(colAttrId);
	if (focusId !== prevFocusId || sqlAttrId !== prevSqlAttrId || colAttrId !== prevColAttrId) {
		setPrevFocusId(focusId);
		setPrevSqlAttrId(sqlAttrId);
		setPrevColAttrId(colAttrId);
		setTermEditing(false);
		setSqlAttrEditing(false);
		setSqlAttrEditError(null);
		setColumnAttrEditing(false);
		setColumnAttrEditError(null);
	}
	const [sqlEditModalOpen, setSqlEditModalOpen] = useState(false);
	const [sqlEditValue, setSqlEditValue] = useState('');
	const [sqlEditOriginalValue, setSqlEditOriginalValue] = useState('');
	const [sqlEditValidating, setSqlEditValidating] = useState(false);
	const [sqlEditValidationMessage, setSqlEditValidationMessage] = useState<string | null>(null);
	const [sqlEditValidated, setSqlEditValidated] = useState(false);
	const [sqlEditSubmitting, setSqlEditSubmitting] = useState(false);
	const [sqlEditError, setSqlEditError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;

		(async () => {
			setLoading(true);
			const termsRes = await termsApi.list(
				debouncedSearchQuery ? { q: debouncedSearchQuery } : undefined,
			);
			if (cancelled) return;

			if (termsRes.error) {
				setError(termsRes.message ?? 'Failed to load terms');
				setTerms([]);
			} else {
				setError(null);
				setTerms(termsRes.terms ?? []);
				const map = new Map<string, number>();
				for (const { term_id, count } of termsRes.column_attribute_counts ?? []) {
					map.set(term_id, count);
				}
				setColumnAttrCountsMap(map);

				const sqlAttributeCountsMap = new Map<string, number>();
				for (const { term_id, count } of termsRes.sql_attribute_counts ?? []) {
					sqlAttributeCountsMap.set(term_id, count);
				}
				setSqlAttrCountsMap(sqlAttributeCountsMap);

				const relatedCountsMap = new Map<string, number>();
				for (const { term_id, count } of termsRes.related_counts ?? []) {
					relatedCountsMap.set(term_id, count);
				}
				setRelatedCountsMap(relatedCountsMap);
			}

			setLoading(false);
		})();

		return () => {
			cancelled = true;
		};
	}, [debouncedSearchQuery]);

	const handleCardClick = useCallback(
		(term: Term) => {
			router.push(`/terms?focus=${encodeURIComponent(term.id)}`);
		},
		[router],
	);

	const handleBack = useCallback(() => {
		router.push('/terms');
	}, [router]);

	const handleBackToTerm = useCallback(() => {
		if (focusId == null) {
			handleBack();
			return;
		}
		router.push(`/terms?focus=${encodeURIComponent(focusId)}`);
	}, [focusId, handleBack, router]);

	const handleSqlAttrCreated = useCallback((attribute: SqlAttribute) => {
		setSqlAttrs((prev) => [...prev, attribute]);
		// Re-fetch the term's single page so the new attribute shows up in the
		// "SQL Attributes" table without changing which term is focused.
		setSqlAttrsEpoch((prev) => prev + 1);
	}, []);

	const trimmedSqlAttrName = sqlAttrName.trim();
	const trimmedSqlAttrSql = sqlAttrSql.trim();
	const canCreateSqlAttr =
		trimmedSqlAttrName.length > 0 && trimmedSqlAttrSql.length > 0 && sqlAttrSqlValidated;

	const resetCreateSqlAttrForm = () => {
		setSqlAttrName('');
		setSqlAttrDescription('');
		setSqlAttrSql('');
		setSqlAttrValidating(false);
		setSqlAttrValidationMessage(null);
		setSqlAttrSqlValidated(false);
		setSqlAttrSubmitting(false);
		setSqlAttrSubmitError(null);
	};

	const handleCreateSqlAttrClose = () => {
		if (sqlAttrValidating || sqlAttrSubmitting) return;
		resetCreateSqlAttrForm();
		setCreateSqlAttrModalOpen(false);
	};

	const handleSqlAttrSqlChange = (value: string) => {
		setSqlAttrSql(value);
		setSqlAttrSqlValidated(false);
		setSqlAttrValidationMessage(null);
	};

	const handleValidateSqlAttrSql = async () => {
		if (trimmedSqlAttrSql.length === 0) return;
		setSqlAttrValidating(true);
		setSqlAttrValidationMessage(null);
		setSqlAttrSqlValidated(false);
		const res = await sqlAttributesApi.validate({
			expression: trimmedSqlAttrSql,
			term_id: focusId ?? undefined,
		});
		setSqlAttrValidating(false);
		if (res.error) {
			setSqlAttrValidationMessage(res.message ?? 'SQL validation failed');
			return;
		}
		const isValid = res.data.valid === true;
		setSqlAttrSqlValidated(isValid);
		setSqlAttrValidationMessage(isValid ? 'SQL is valid.' : 'SQL validation failed');
	};

	const handleCreateSqlAttr = async () => {
		if (!canCreateSqlAttr || focusId == null) return;
		setSqlAttrSubmitting(true);
		setSqlAttrSubmitError(null);
		const res = await sqlAttributesApi.create({
			name: trimmedSqlAttrName,
			description: sqlAttrDescription.trim(),
			expression: trimmedSqlAttrSql,
			term_id: focusId,
			source: 'manual',
		});
		setSqlAttrSubmitting(false);
		if (res.error) {
			setSqlAttrSubmitError(res.message ?? 'Failed to create SQL attribute');
			return;
		}
		handleSqlAttrCreated(res.data);
		resetCreateSqlAttrForm();
		setCreateSqlAttrModalOpen(false);
	};

	const handleDeleteSqlAttrClose = () => {
		if (deletingSqlAttrBusy) return;
		setDeletingSqlAttr(null);
		setDeleteSqlAttrError(null);
	};

	const handleDeleteSqlAttrConfirm = async () => {
		if (deletingSqlAttr == null) return;
		setDeletingSqlAttrBusy(true);
		setDeleteSqlAttrError(null);
		const res = await sqlAttributesApi.delete(deletingSqlAttr.id);
		setDeletingSqlAttrBusy(false);
		if (res.error) {
			setDeleteSqlAttrError(res.message ?? 'Failed to delete SQL attribute');
			return;
		}
		setSqlAttrs((prev) => prev.filter((attr) => attr.id !== deletingSqlAttr.id));
		setSqlAttrsEpoch((prev) => prev + 1);
		setDeletingSqlAttr(null);
		if (focusId != null) {
			router.push(`/terms?focus=${encodeURIComponent(focusId)}`);
		}
	};

	const updateSqlAttr = async (payload: {
		id: string;
		name: string;
		description: string;
		expression: string;
		termId: string;
	}) => {
		const res = await sqlAttributesApi.update(payload.id, {
			name: payload.name,
			description: payload.description,
			expression: payload.expression,
			term_id: payload.termId,
			source: 'manual',
		});
		if (res.error) {
			return res;
		}
		setSqlAttrs((prev) => {
			const exists = prev.some((attr) => attr.id === res.data.id);
			if (!exists) return [...prev, res.data];
			return prev.map((attr) => (attr.id === res.data.id ? res.data : attr));
		});
		setSqlAttrsEpoch((prev) => prev + 1);
		return res;
	};

	const handleSqlAttrEditSave = async (payload: Record<string, ComposerEditValue>) => {
		if (focusedSqlAttr == null || focusId == null) {
			return { error: true, message: 'SQL attribute not found' };
		}
		setSqlAttrEditError(null);
		const description =
			typeof payload.description === 'string'
				? payload.description
				: (focusedSqlAttr.description ?? '');
		const name = typeof payload.name === 'string' ? payload.name.trim() : focusedSqlAttr.name;
		if (!name) {
			return { error: true, message: 'SQL attribute name cannot be blank' };
		}

		const patch: { name?: string; description?: string } = {};
		if (name !== focusedSqlAttr.name.trim()) {
			patch.name = name;
		}
		if (description !== (focusedSqlAttr.description ?? '')) {
			patch.description = description;
		}
		if (Object.keys(patch).length === 0) {
			return { error: false };
		}

		const res = await sqlAttributesApi.patch(focusedSqlAttr.id, patch);
		if (res.error) {
			return { error: true, message: res.message ?? 'Failed to update SQL attribute' };
		}
		setSqlAttrs((prev) =>
			prev.map((attr) => (attr.id === focusedSqlAttr.id ? res.data : attr)),
		);
		setSqlAttrsEpoch((prev) => prev + 1);
		return { error: false };
	};

	const handleColumnAttrEditSave = async (payload: Record<string, ComposerEditValue>) => {
		if (focusedColAttr == null || focusId == null) {
			return { error: true, message: 'Column attribute not found' };
		}
		setColumnAttrEditError(null);
		const description =
			typeof payload.description === 'string'
				? payload.description
				: (focusedColAttr.description ?? '');
		const name =
			typeof payload.name === 'string' ? payload.name.trim() : focusedColAttr.name.trim();
		if (!name) {
			return { error: true, message: 'Column attribute name cannot be blank' };
		}
		const sampleValues = Array.isArray(payload.sample_values)
			? payload.sample_values.map((value) => String(value))
			: (focusedColAttr.sample_values ?? []);
		const previousSampleValues = focusedColAttr.sample_values ?? [];

		const patch: { name?: string; description?: string; sample_values?: string[] } = {};
		if (name !== focusedColAttr.name.trim()) {
			patch.name = name;
		}
		if (description !== (focusedColAttr.description ?? '')) {
			patch.description = description;
		}
		const sampleValuesChanged =
			sampleValues.length !== previousSampleValues.length ||
			sampleValues.some((value, index) => value !== previousSampleValues[index]);
		if (sampleValuesChanged) {
			patch.sample_values = sampleValues;
		}
		if (Object.keys(patch).length === 0) {
			return { error: false };
		}

		const res = await termsApi.updateColumnAttribute(focusId, focusedColAttr.id, patch);
		if (res.error) {
			return { error: true, message: res.message ?? 'Failed to update column attribute' };
		}
		setColumnAttrs((prev) =>
			prev.map((attr) =>
				attr.id === focusedColAttr.id
					? {
							...attr,
							name: res.data.name,
							description: res.data.description,
							sample_values: res.data.sample_values ?? sampleValues,
						}
					: attr,
			),
		);
		setColumnAttrsEpoch((prev) => prev + 1);
		return { error: false };
	};

	const handleTermEditSave = async (payload: Record<string, ComposerEditValue>) => {
		if (focusedTerm == null || focusId == null) {
			return { error: true, message: 'Term not found' };
		}

		const name =
			typeof payload.name === 'string' ? payload.name.trim() : focusedTerm.name.trim();
		if (!name) {
			return { error: true, message: 'Term name cannot be blank' };
		}

		const description =
			typeof payload.description === 'string'
				? payload.description
				: (focusedTerm.description ?? '');

		const patch: { name?: string; description?: string } = {};
		if (name !== focusedTerm.name.trim()) {
			patch.name = name;
		}
		if (description !== (focusedTerm.description ?? '')) {
			patch.description = description;
		}
		if (Object.keys(patch).length === 0) {
			return { error: false };
		}

		const res = await termsApi.update(focusId, patch);
		if (res.error) {
			return { error: true, message: res.message ?? 'Failed to update term' };
		}

		setTerms((prev) =>
			prev.map((term) =>
				term.id === focusId
					? {
							...term,
							name: res.data.name,
							description: res.data.description,
						}
					: term,
			),
		);
		setSqlAttrsEpoch((prev) => prev + 1);
		return { error: false };
	};

	const trimmedSqlEditValue = sqlEditValue.trim();
	const sqlEditUnchanged = trimmedSqlEditValue === sqlEditOriginalValue.trim();
	const canSaveSqlEdit =
		trimmedSqlEditValue.length > 0 &&
		!sqlEditUnchanged &&
		sqlEditValidated &&
		!sqlEditValidating &&
		!sqlEditSubmitting;

	const resetSqlEditModal = () => {
		setSqlEditValue('');
		setSqlEditOriginalValue('');
		setSqlEditValidating(false);
		setSqlEditValidationMessage(null);
		setSqlEditValidated(false);
		setSqlEditSubmitting(false);
		setSqlEditError(null);
	};

	const handleSqlEditClose = () => {
		if (sqlEditValidating || sqlEditSubmitting) return;
		resetSqlEditModal();
		setSqlEditModalOpen(false);
	};

	const handleSqlEditChange = (value: string) => {
		setSqlEditValue(value);
		setSqlEditValidated(false);
		setSqlEditValidationMessage(null);
	};

	const handleValidateSqlEdit = async () => {
		if (trimmedSqlEditValue.length === 0) return;
		setSqlEditValidating(true);
		setSqlEditValidationMessage(null);
		setSqlEditValidated(false);
		const res = await sqlAttributesApi.validate({
			expression: trimmedSqlEditValue,
			term_id: focusId ?? undefined,
			attribute_id: focusedSqlAttr?.id,
		});
		setSqlEditValidating(false);
		if (res.error) {
			setSqlEditValidationMessage(res.message ?? 'SQL validation failed');
			return;
		}
		const isValid = res.data.valid === true;
		setSqlEditValidated(isValid);
		setSqlEditValidationMessage(isValid ? 'SQL is valid.' : 'SQL validation failed');
	};

	const handleSaveSqlEdit = async () => {
		if (!canSaveSqlEdit || focusedSqlAttr == null || focusId == null) return;
		setSqlEditSubmitting(true);
		setSqlEditError(null);
		const res = await updateSqlAttr({
			id: focusedSqlAttr.id,
			name: focusedSqlAttr.name,
			description: focusedSqlAttr.description ?? '',
			expression: trimmedSqlEditValue,
			termId: focusedSqlAttr.term_id,
		});
		setSqlEditSubmitting(false);
		if (res.error) {
			setSqlEditError(res.message ?? 'Failed to save SQL');
			return;
		}
		resetSqlEditModal();
		setSqlEditModalOpen(false);
		setSqlAttrEditing(false);
	};

	const handleDataTableRowClick = useCallback(
		(sectionId: string, rowId: string) => {
			if (focusId == null) return;
			if (sectionId === 'sql_attributes') {
				router.push(
					`/terms?focus=${encodeURIComponent(focusId)}&sqlAttr=${encodeURIComponent(rowId)}`,
				);
			} else if (sectionId === 'column_attributes') {
				router.push(
					`/terms?focus=${encodeURIComponent(focusId)}&colAttr=${encodeURIComponent(rowId)}`,
				);
			}
		},
		[focusId, router],
	);

	const getColumnAttributeSinglePage = useCallback(
		async (attrId: string): Promise<SinglePageFormat> => {
			if (focusId == null) {
				return {
					sections: [],
					header: { header: { title: 'Column Attribute not found', withBorder: true } },
				};
			}
			// TODO: viewer zone-scoping handled in a separate PR — for now the
			// client treats a viewer the same as an admin here (no zone fetch,
			// userZoneIds = null → all zones accessible).
			const res = await termsApi.getColumnAttributes(focusId);
			const attrs = res.error ? [] : (res.data ?? []);
			const attr = attrs.find((a) => a.id === attrId);
			if (attr == null) {
				return {
					sections: [],
					header: { header: { title: 'Column Attribute not found', withBorder: true } },
				};
			}
			setColumnAttrs(attrs);

			const primaryColumn = attr.primary_column ?? null;
			const referencedColumns = attr.referenced_columns ?? [];
			const userZoneIds: string[] | null = null;

			return {
				header: {
					header: {
						title: attr.name,
						withBorder: true,
						showContentHeader: true,
						titleEditable: true,
					},
				},
				sections: [
					{
						type: ComposerSectionKind.TEXT_CARD,
						id: 'description',
						title: 'Description',
						body: attr.description ?? '',
						editable: true,
					},
					{
						type: ComposerSectionKind.TAG_LIST,
						id: 'sample_values',
						title: 'Sample Values',
						values: Array.isArray(attr.sample_values) ? attr.sample_values : [],
						editable: true,
					},
					{
						type: ComposerSectionKind.ZONES_CHIPS,
						id: 'zones',
						title: 'Zones',
						zones: (attr.zones ?? []).map((zone) => ({
							id: zone.id,
							name: zone.name,
							color: zone.color,
							enabled: zone.enabled,
						})),
						userZoneIds,
					},
					{
						type: ComposerSectionKind.ENTITY_CHIPS,
						id: 'primary_column',
						title: 'Primary Column',
						entities: primaryColumn
							? [
									{
										id: primaryColumn.id,
										name: `${primaryColumn.table_name}.${primaryColumn.column_name}`,
										focusId: [
											primaryColumn.db_id,
											primaryColumn.schema_id,
											primaryColumn.table_id,
											primaryColumn.id,
										].join('|'),
									},
								]
							: [],
					},
					{
						type: ComposerSectionKind.ENTITY_CHIPS,
						id: 'referenced_columns',
						title: 'Referenced Columns',
						entities: referencedColumns.map((col) => ({
							id: col.id,
							name: `${col.table_name}.${col.column_name}`,
							focusId: [col.db_id, col.schema_id, col.table_id, col.id].join('|'),
						})),
					},
				],
			};
		},
		[focusId],
	);

	const getSqlAttributeSinglePage = useCallback(
		async (attrId: string): Promise<SinglePageFormat> => {
			const res = await sqlAttributesApi.get(attrId);
			if (res.error || !res.data) {
				return {
					sections: [],
					header: { header: { title: 'SQL Attribute not found', withBorder: true } },
				};
			}
			const attr = res.data;

			// Upsert into the sqlAttrs cache so the breadcrumb title (derived
			// from that state) is correct even on a direct deep-link, when
			// this is the first sql-attribute fetch of the session (the Terms
			// list no longer preloads every SqlAttribute node).
			setSqlAttrs((prev) => {
				const exists = prev.some((a) => a.id === attr.id);
				if (!exists) return [...prev, attr];
				return prev.map((a) => (a.id === attr.id ? attr : a));
			});

			return {
				header: {
					header: {
						title: attr.name,
						withBorder: true,
						showContentHeader: true,
						titleEditable: true,
					},
				},
				sections: [
					{
						type: ComposerSectionKind.TEXT_CARD,
						id: 'description',
						title: 'Description',
						body: attr.description ?? '',
						editable: true,
						suggestable: true,
					},
					{
						type: ComposerSectionKind.SQL_BLOCK,
						id: 'sql',
						title: 'SQL',
						sql: attr.expression ?? '',
						editable: true,
					},
					{
						type: ComposerSectionKind.ZONES_CHIPS,
						id: 'zones',
						title: 'Zones',
						zones: (attr.zones ?? []).map((z) => ({
							id: z.id,
							name: z.name,
							color: z.color,
							enabled: z.enabled,
						})),
					},
				],
			};
		},
		[],
	);

	const getSinglePage = useCallback(async (termId: string): Promise<SinglePageFormat> => {
		const [res, attrsRes, sqlAttrsRes] = await Promise.all([
			termsApi.get(termId),
			termsApi.getColumnAttributes(termId),
			termsApi.getSqlAttributes(termId),
		]);
		if (res.error || !res.data) {
			return {
				sections: [],
				header: { header: { title: 'Term not found', withBorder: true } },
			};
		}
		const term = res.data;
		const termAttrs = attrsRes?.data ?? [];
		const termSqlAttrs = sqlAttrsRes?.data ?? [];
		const relatedTerms = term.related_terms ?? [];
		// Populate the sqlAttrs/columnAttrs caches from this term's own
		// attributes (fetched per-term above) rather than a global list —
		// the Terms list endpoint only returns counts, not the attribute
		// nodes themselves, to avoid pulling every attribute in the graph
		// on every page load. This keeps breadcrumb titles working once
		// the user has opened this term's detail page.
		setSqlAttrs(termSqlAttrs);
		setColumnAttrs(termAttrs);

		return {
			header: {
				header: {
					title: term.name,
					withBorder: true,
					showContentHeader: true,
					titleEditable: true,
				},
			},
			sections: [
				{
					type: ComposerSectionKind.TEXT_CARD,
					id: 'description',
					title: 'Description',
					body: term.description ?? '',
					editable: true,
				},
				{
					type: ComposerSectionKind.TAG_LIST,
					id: 'synonyms',
					title: 'Synonyms',
					values: term.synonyms ?? [],
				},
				{
					type: ComposerSectionKind.ENTITY_CHIPS,
					id: 'entities',
					title: 'Entities',
					entities: (term.tables ?? []).map((table) => ({
						id: table.id,
						name: table.name,
						focusId: [table.db_id, table.schema_id, table.id].join('|'),
					})),
				},
				{
					type: ComposerSectionKind.ZONES_CHIPS,
					id: 'zones',
					title: 'Zones',
					zones: term.zones.map((z) => ({
						id: z.id,
						name: z.name,
						color: z.color,
						enabled: z.enabled,
					})),
				},
				{
					type: ComposerSectionKind.RELATED_TERMS_CHIPS,
					id: 'related_terms',
					title: 'Related Terms',
					terms: relatedTerms.map((t) => ({
						id: t.id,
						name: t.name,
						description: t.description,
					})),
				},
				{
					type: ComposerSectionKind.DATA_TABLE,
					id: 'column_attributes',
					title: 'Column Attributes',
					rowIdKey: 'id',
					columns: [
						{ key: 'name', label: 'Attribute Name' },
						{ key: 'description', label: 'Description', truncate: true },
						{ key: 'sample_values', label: 'Sample Values', kind: 'tags' },
					],
					rows: termAttrs.map((attr) => ({
						id: attr.id,
						name: attr.name,
						description: attr.description ?? '',
						sample_values: attr.sample_values ?? [],
					})),
				},
				{
					type: ComposerSectionKind.DATA_TABLE,
					id: 'sql_attributes',
					title: 'SQL Attributes',
					rowIdKey: 'id',
					columns: [{ key: 'name', label: 'Attribute Name' }],
					rows: termSqlAttrs.map((attr) => ({
						id: attr.id,
						name: attr.name,
					})),
					emptyMessage: 'SQL attribute does not exist',
				},
			],
		};
	}, []);

	const focusedTerm = focusId != null ? (terms.find((t) => t.id === focusId) ?? null) : null;
	const focusedSqlAttr =
		sqlAttrId != null ? (sqlAttrs.find((attr) => attr.id === sqlAttrId) ?? null) : null;
	const focusedColAttr =
		colAttrId != null ? (columnAttrs.find((attr) => attr.id === colAttrId) ?? null) : null;

	if (focusId != null && sqlAttrId != null) {
		const termTitle = focusedTerm?.name ?? focusId;
		const sqlAttrTitle = focusedSqlAttr?.name ?? sqlAttrId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
					<button
						type="button"
						onClick={handleBack}
						className="flex cursor-pointer items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
						aria-label="Back to terms list"
					>
						<Icon name={IconName.Terms} className="h-4 w-4" />
						Terms
					</button>
					<span className="text-zinc-300 dark:text-zinc-600">/</span>
					<button
						type="button"
						onClick={handleBackToTerm}
						className="cursor-pointer rounded-lg px-1.5 py-1 text-sm font-medium text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
					>
						{termTitle}
					</button>
					<span className="text-zinc-300 dark:text-zinc-600">/</span>
					<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
						{sqlAttrTitle}
					</span>
					<div className="ml-auto flex shrink-0 items-center gap-1">
						{sqlAttrEditing ? null : (
							<>
								<button
									type="button"
									onClick={() => {
										setSqlAttrEditError(null);
										setSqlAttrEditing(true);
									}}
									aria-label={`Edit ${sqlAttrTitle}`}
									title="Edit"
									className="cursor-pointer rounded-md p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:text-zinc-400 dark:hover:bg-zinc-800"
								>
									<Icon name={IconName.Pencil} className="h-4 w-4" />
								</button>
								<button
									type="button"
									onClick={() => {
										setDeletingSqlAttr({ id: sqlAttrId, name: sqlAttrTitle });
										setDeleteSqlAttrError(null);
									}}
									aria-label={`Delete ${sqlAttrTitle}`}
									title="Delete"
									className="cursor-pointer rounded-md p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-red-600 dark:text-zinc-400 dark:hover:bg-zinc-800"
								>
									<Icon name={IconName.Trash} className="h-4 w-4" />
								</button>
							</>
						)}
					</div>
				</header>
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						key={sqlAttrId}
						dataId={sqlAttrId}
						title={sqlAttrTitle}
						getSinglePage={getSqlAttributeSinglePage}
						treeDataEpoch={sqlAttrsEpoch}
						isEditing={sqlAttrEditing}
						onPatchEdits={handleSqlAttrEditSave}
						onSave={() => {
							setSqlAttrEditing(false);
							setSqlAttrEditError(null);
						}}
						onCancel={() => {
							setSqlAttrEditing(false);
							setSqlAttrEditError(null);
						}}
						onEditSql={(_sectionId, sql) => {
							resetSqlEditModal();
							setSqlEditValue(sql);
							setSqlEditOriginalValue(sql);
							setSqlEditModalOpen(true);
						}}
						onSuggestDescription={async (sectionId) => {
							if (sectionId !== 'description') return null;
							const res = await sqlAttributesApi.suggestDescription(sqlAttrId);
							if (res.error) return null;
							return res.data;
						}}
					/>
				</main>
				{sqlAttrEditError != null && (
					<div className="border-t border-red-200 bg-red-50 px-6 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
						{sqlAttrEditError}
					</div>
				)}
				<ConfirmModal
					open={deletingSqlAttr !== null}
					onCancel={handleDeleteSqlAttrClose}
					onConfirm={handleDeleteSqlAttrConfirm}
					title="Delete SQL attribute"
					message={[
						'Are you sure you want to delete ',
						<strong key="name">{deletingSqlAttr?.name}</strong>,
						'? This action cannot be undone.',
					]}
					confirming={deletingSqlAttrBusy}
					error={deleteSqlAttrError}
				/>
				<ModalCreateNewItem
					open={sqlEditModalOpen}
					onClose={handleSqlEditClose}
					title="SQL Code"
					submitLabel={sqlEditSubmitting ? 'Saving…' : 'Save'}
					onSubmit={handleSaveSqlEdit}
					canSubmit={canSaveSqlEdit}
					secondaryAction={{
						label: sqlEditValidating ? 'Validating…' : 'Validate SQL',
						onClick: handleValidateSqlEdit,
						disabled:
							trimmedSqlEditValue.length === 0 ||
							sqlEditValidating ||
							sqlEditUnchanged,
					}}
				>
					<SqlEditor
						value={sqlEditValue}
						onChange={handleSqlEditChange}
						label="SQL Code"
						rows={10}
					/>

					{sqlEditValidationMessage != null && (
						<p className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-300">
							{sqlEditValidationMessage}
						</p>
					)}

					{sqlEditError != null && (
						<p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
							{sqlEditError}
						</p>
					)}
				</ModalCreateNewItem>
			</div>
		);
	}

	if (focusId != null && colAttrId != null) {
		const termTitle = focusedTerm?.name ?? focusId;
		const colAttrTitle = focusedColAttr?.name ?? colAttrId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
					<button
						type="button"
						onClick={handleBack}
						className="flex cursor-pointer items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
						aria-label="Back to terms list"
					>
						<Icon name={IconName.Terms} className="h-4 w-4" />
						Terms
					</button>
					<span className="text-zinc-300 dark:text-zinc-600">/</span>
					<button
						type="button"
						onClick={handleBackToTerm}
						className="cursor-pointer rounded-lg px-1.5 py-1 text-sm font-medium text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
					>
						{termTitle}
					</button>
					<span className="text-zinc-300 dark:text-zinc-600">/</span>
					<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
						{colAttrTitle}
					</span>
					<div className="ml-auto flex shrink-0 items-center gap-1">
						{columnAttrEditing ? null : (
							<button
								type="button"
								onClick={() => {
									setColumnAttrEditError(null);
									setColumnAttrEditing(true);
								}}
								aria-label={`Edit ${colAttrTitle}`}
								title="Edit"
								className="cursor-pointer rounded-md p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:text-zinc-400 dark:hover:bg-zinc-800"
							>
								<Icon name={IconName.Pencil} className="h-4 w-4" />
							</button>
						)}
					</div>
				</header>
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						key={colAttrId}
						dataId={colAttrId}
						title={colAttrTitle}
						getSinglePage={getColumnAttributeSinglePage}
						treeDataEpoch={columnAttrsEpoch}
						isEditing={columnAttrEditing}
						onPatchEdits={handleColumnAttrEditSave}
						onSave={() => {
							setColumnAttrEditing(false);
							setColumnAttrEditError(null);
						}}
						onCancel={() => {
							setColumnAttrEditing(false);
							setColumnAttrEditError(null);
						}}
					/>
				</main>
				{columnAttrEditError != null && (
					<div className="border-t border-red-200 bg-red-50 px-6 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
						{columnAttrEditError}
					</div>
				)}
			</div>
		);
	}

	if (focusId != null) {
		const termTitle = focusedTerm?.name ?? focusId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
					<button
						type="button"
						onClick={handleBack}
						className="flex cursor-pointer items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
						aria-label="Back to terms list"
					>
						<Icon name={IconName.Terms} className="h-4 w-4" />
						Terms
					</button>
					<span className="text-zinc-300 dark:text-zinc-600">/</span>
					<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
						{termTitle}
					</span>
					<div className="ml-auto flex items-center gap-2">
						{termEditing ? null : (
							<button
								type="button"
								onClick={() => {
									setTermEditing(true);
								}}
								className="cursor-pointer rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-300 dark:hover:bg-zinc-800"
							>
								Edit term
							</button>
						)}
						<button
							type="button"
							onClick={() => setCreateSqlAttrModalOpen(true)}
							className="flex cursor-pointer items-center gap-2 rounded-lg bg-[#76b900] px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[#5e9400]"
						>
							<svg
								className="h-4 w-4"
								viewBox="0 0 20 20"
								fill="currentColor"
								aria-hidden
							>
								<path d="M10 3.75a.75.75 0 0 1 .75.75v4.75h4.75a.75.75 0 0 1 0 1.5h-4.75v4.75a.75.75 0 0 1-1.5 0V10.75H4.5a.75.75 0 0 1 0-1.5h4.75V4.5a.75.75 0 0 1 .75-.75Z" />
							</svg>
							Create new sql attribute
						</button>
					</div>
				</header>
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						key={focusId}
						dataId={focusId}
						title={termTitle}
						getSinglePage={getSinglePage}
						treeDataEpoch={sqlAttrsEpoch}
						isEditing={termEditing}
						onPatchEdits={handleTermEditSave}
						onSave={() => {
							setTermEditing(false);
						}}
						onCancel={() => {
							setTermEditing(false);
						}}
						onDataTableRowClick={handleDataTableRowClick}
					/>
				</main>
				<ModalCreateNewItem
					open={createSqlAttrModalOpen}
					onClose={handleCreateSqlAttrClose}
					title="Create New SQL Attribute"
					submitLabel={sqlAttrSubmitting ? 'Creating…' : 'Create'}
					onSubmit={handleCreateSqlAttr}
					canSubmit={canCreateSqlAttr}
					secondaryAction={{
						label: sqlAttrValidating ? 'Validating…' : 'Validate SQL',
						onClick: handleValidateSqlAttrSql,
						disabled: trimmedSqlAttrSql.length === 0 || sqlAttrValidating,
					}}
				>
					<div>
						<label className={FIELD_LABEL_CLASSNAME}>Attribute Name</label>
						<input
							type="text"
							value={sqlAttrName}
							onChange={(e) => setSqlAttrName(e.target.value)}
							placeholder="Attribute Name"
							className={FIELD_INPUT_CLASSNAME}
						/>
					</div>

					<div>
						<label className={FIELD_LABEL_CLASSNAME}>Description (Optional)</label>
						<textarea
							value={sqlAttrDescription}
							onChange={(e) => setSqlAttrDescription(e.target.value)}
							placeholder="Add Description"
							rows={3}
							className={`resize-y ${FIELD_INPUT_CLASSNAME}`}
						/>
					</div>

					<div>
						<SqlEditor value={sqlAttrSql} onChange={handleSqlAttrSqlChange} rows={10} />
					</div>

					{sqlAttrValidationMessage != null && (
						<p className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-300">
							{sqlAttrValidationMessage}
						</p>
					)}

					{sqlAttrSubmitError != null && (
						<p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
							{sqlAttrSubmitError}
						</p>
					)}
				</ModalCreateNewItem>
			</div>
		);
	}

	return (
		<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
			<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
				<Icon name={IconName.Terms} className="h-5 w-5 text-[#76b900]" />
				<h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					Terms
				</h1>
			</header>

			<div className="flex-1 overflow-y-auto px-6 py-6">
				<SearchInput
					value={searchQuery}
					onChange={setSearchQuery}
					placeholder="Search terms…"
					aria-label="Search terms"
					className="mb-6 w-full"
				/>

				{loading && (
					<div className="flex h-full items-center justify-center">
						<div
							className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
							role="status"
							aria-label="Loading terms"
						/>
					</div>
				)}

				{!loading && error != null && (
					<div className="mx-auto max-w-lg rounded-2xl border border-red-200/80 bg-white/90 px-8 py-10 text-center shadow-xl shadow-red-100/50 dark:border-red-900/50 dark:bg-zinc-950/80 dark:shadow-none">
						<h2 className="text-lg font-semibold tracking-tight text-red-800 dark:text-red-300">
							Couldn&apos;t load terms
						</h2>
						<pre className="mt-4 max-w-full overflow-x-auto rounded-lg border border-red-100 bg-red-50/80 p-3 text-left text-xs text-red-900/80 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
							{error}
						</pre>
					</div>
				)}

				{!loading && error == null && terms.length === 0 && (
					<div className="flex h-full min-h-[40dvh] flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-zinc-300/80 bg-white/60 p-12 text-center dark:border-zinc-600 dark:bg-zinc-950/40">
						<Placeholders.NoTerms />
						<p className="mt-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
							{debouncedSearchQuery
								? 'No Terms Match Your Search'
								: 'No Terms Created Yet'}
						</p>
					</div>
				)}

				{!loading && error == null && terms.length > 0 && (
					<ul className="flex flex-col gap-4">
						{terms.map((term) => (
							<TermCard
								key={term.id}
								term={term}
								columnAttributeCount={columnAttrCountsMap.get(term.id) ?? 0}
								sqlAttributeCount={sqlAttrCountsMap.get(term.id) ?? 0}
								relatedCount={relatedCountsMap.get(term.id) ?? 0}
								onClick={handleCardClick}
							/>
						))}
					</ul>
				)}
			</div>
		</div>
	);
};
