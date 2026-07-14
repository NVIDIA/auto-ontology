// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

import { Placeholders } from '@/assets/images/placeholders';
import { Icon, IconName } from '@/components/icons';
import { ConfirmModal } from '@/components/ConfirmModal';
import { SearchInput } from '@/components/SearchInput';
import { termsApi } from '@/api/terms';
import { sqlAttributesApi } from '@/api/sqlAttributes';
import { zonesApi } from '@/api/zones';
import { useSession } from '@/auth/auth-client';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import type { ComposerEditValue } from '@/common/SinglePageComposer';
import { ComposerSectionKind } from '@/enums/datasources';
import { ModalCreateNewItem } from '@/components/ModalCreateNewItem';
import { SinglePageView, type SinglePageFormat } from '@/components/SinglePageView';
import { SqlEditor } from '@/components/SqlBlock';
import type { ColumnAttribute, SqlAttribute, Term } from '@/types/terms';

type TermCardProps = {
	term: Term;
	attributes: ColumnAttribute[];
	sqlAttributes: SqlAttribute[];
	relatedCount: number;
	onClick: (term: Term) => void;
};

type SqlAttributeDeleteTarget = {
	id: string;
	name: string;
};

const TermCard = ({ term, attributes, sqlAttributes, relatedCount, onClick }: TermCardProps) => (
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

		{/* Three-column section */}
		<div className="mt-4 grid grid-cols-3 gap-0 overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-700">
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
							{attributes.length}
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
							{sqlAttributes.length}
						</span>
					</span>
				</div>
			</div>

			{/* Related Terms */}
			<div>
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

	const { data: session } = useSession();
	const sessionUserId = session?.user?.id ?? null;
	const sessionRole = session?.user?.role ?? null;

	const [terms, setTerms] = useState<Term[]>([]);
	const [attrs, setAttrs] = useState<ColumnAttribute[]>([]);
	const [sqlAttrs, setSqlAttrs] = useState<SqlAttribute[]>([]);
	const [relatedCountsMap, setRelatedCountsMap] = useState<Map<string, number>>(new Map());
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [searchQuery, setSearchQuery] = useState('');
	const debouncedSearchQuery = useDebouncedValue(searchQuery.trim(), 1000);
	const loadedAuxRef = useRef(false);
	const [createSqlAttrModalOpen, setCreateSqlAttrModalOpen] = useState(false);
	const [sqlAttrsEpoch, setSqlAttrsEpoch] = useState(0);
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

	const [prevFocusId, setPrevFocusId] = useState(focusId);
	const [prevSqlAttrId, setPrevSqlAttrId] = useState(sqlAttrId);
	if (focusId !== prevFocusId || sqlAttrId !== prevSqlAttrId) {
		setPrevFocusId(focusId);
		setPrevSqlAttrId(sqlAttrId);
		setTermEditing(false);
		setSqlAttrEditing(false);
		setSqlAttrEditError(null);
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
			// Attributes/SQL attributes/related counts don't depend on the term
			// search — only fetch them once, up front, and just re-fetch `terms`
			// as the (debounced) search query changes.
			const loadAux = !loadedAuxRef.current;
			const [termsRes, attrsRes, sqlAttrsRes, countsRes] = await Promise.all([
				termsApi.list(debouncedSearchQuery ? { q: debouncedSearchQuery } : undefined),
				loadAux ? termsApi.listColumnAttributes() : Promise.resolve(null),
				loadAux ? termsApi.listSqlAttributes() : Promise.resolve(null),
				loadAux ? termsApi.listRelatedCounts() : Promise.resolve(null),
			]);
			if (cancelled) return;

			if (termsRes.error) {
				setError(termsRes.message ?? 'Failed to load terms');
				setTerms([]);
			} else {
				setError(null);
				setTerms(termsRes.data ?? []);
			}

			if (attrsRes != null && !attrsRes.error) {
				setAttrs(attrsRes.data ?? []);
			}

			if (sqlAttrsRes != null && !sqlAttrsRes.error) {
				setSqlAttrs(sqlAttrsRes.data ?? []);
			}

			if (countsRes != null && !countsRes.error) {
				const map = new Map<string, number>();
				for (const { term_id, count } of countsRes.data ?? []) {
					map.set(term_id, count);
				}
				setRelatedCountsMap(map);
			}

			loadedAuxRef.current = true;
			setLoading(false);
		})();

		return () => {
			cancelled = true;
		};
	}, [debouncedSearchQuery]);

	const attrsByTerm = useMemo(() => {
		const map = new Map<string, ColumnAttribute[]>();
		for (const attr of attrs) {
			const list = map.get(attr.term_name) ?? [];
			list.push(attr);
			map.set(attr.term_name, list);
		}
		return map;
	}, [attrs]);

	const sqlAttrsByTerm = useMemo(() => {
		const map = new Map<string, SqlAttribute[]>();
		for (const attr of sqlAttrs) {
			const list = map.get(attr.term_name) ?? [];
			list.push(attr);
			map.set(attr.term_name, list);
		}
		return map;
	}, [sqlAttrs]);

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

	const handleSqlAttrClick = useCallback(
		(sectionId: string, rowId: string) => {
			if (sectionId !== 'sql_attributes' || focusId == null) return;
			router.push(
				`/terms?focus=${encodeURIComponent(focusId)}&sqlAttr=${encodeURIComponent(rowId)}`,
			);
		},
		[focusId, router],
	);

	const getSqlAttributeSinglePage = useCallback(
		async (attrId: string): Promise<SinglePageFormat> => {
			const isViewer = sessionRole !== null && sessionRole !== 'admin';
			const [res, userZonesRes] = await Promise.all([
				sqlAttributesApi.get(attrId),
				isViewer && sessionUserId ? zonesApi.getAll(sessionUserId) : Promise.resolve(null),
			]);
			if (res.error || !res.data) {
				return {
					sections: [],
					header: { header: { title: 'SQL Attribute not found', withBorder: true } },
				};
			}
			const attr = res.data;

			// null = admin (no zone restriction), string[] = viewer's accessible zone IDs
			const userZoneIds: string[] | null =
				userZonesRes !== null && !userZonesRes.error
					? (userZonesRes.data ?? []).map((z) => z.id)
					: null;

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
						})),
						userZoneIds,
					},
				],
			};
		},
		[sessionUserId, sessionRole],
	);

	const getSinglePage = useCallback(
		async (termId: string): Promise<SinglePageFormat> => {
			const isViewer = sessionRole !== null && sessionRole !== 'admin';
			const [res, attrsRes, sqlAttrsRes, relatedRes, userZonesRes] = await Promise.all([
				termsApi.get(termId),
				termsApi.getColumnAttributes(termId),
				termsApi.getSqlAttributes(termId),
				termsApi.getRelatedTerms(termId),
				isViewer && sessionUserId ? zonesApi.getAll(sessionUserId) : Promise.resolve(null),
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
			const relatedTerms = relatedRes?.data ?? [];

			// null = admin (no zone restriction), string[] = viewer's accessible zone IDs
			const userZoneIds: string[] | null =
				userZonesRes !== null && !userZonesRes.error
					? (userZonesRes.data ?? []).map((z) => z.id)
					: null;

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
						})),
						userZoneIds,
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
						columns: [{ key: 'name', label: 'Attribute Name' }],
						rows: termAttrs.map((attr) => ({
							name: attr.name,
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
		},
		[sessionUserId, sessionRole],
	);

	const focusedTerm = focusId != null ? (terms.find((t) => t.id === focusId) ?? null) : null;
	const focusedSqlAttr =
		sqlAttrId != null
			? (sqlAttrs.find((attr) => attr.id === sqlAttrId) ??
				(focusedTerm != null
					? (sqlAttrsByTerm
							.get(focusedTerm.name)
							?.find((attr) => attr.id === sqlAttrId) ?? null)
					: null))
			: null;

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
						onDataTableRowClick={handleSqlAttrClick}
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
								attributes={attrsByTerm.get(term.name) ?? []}
								sqlAttributes={sqlAttrsByTerm.get(term.name) ?? []}
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
