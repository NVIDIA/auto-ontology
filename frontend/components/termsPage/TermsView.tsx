// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import type { BreadcrumbItem } from '@/common/Breadcrumbs';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { Icon, IconName } from '@/common/icons';
import { ConfirmModal, ModalCreateNewItem } from '@/common/modal';
import { termsApi } from '@/api/terms';
import { sqlAttributesApi } from '@/api/sqlAttributes';
import { type ComposerEditValue } from '@/common/SinglePageComposer';
import { entitySidePanel } from '@/common/SidePanel';
import { ComposerColumnType, ComposerSectionKind } from '@/enums/datasources';
import { CertificationStatus } from '@/enums/certification';
import { useBreadcrumbTrail } from '@/contexts/BreadcrumbContext';
import { TagItemType } from '@/enums/tags';
import { ToastVariant } from '@/enums/toast';
import { attributeStatus } from '@/lib/certification';
import {
	TAGS_SECTION_ID,
	entityTagsSection,
	fetchTagOptions,
	stagedTagIds,
	syncTags,
} from '@/lib/tags';
import { sampleValuesEditable, sampleValuesReadOnlyHint } from '@/lib/column-types';
import { canEditEntity, changedOwnerId, entityOwnerSection } from '@/lib/owners';
import { fetchOwnerDirectory, ownerQueries } from '@/lib/queries/owners';
import { OwnerPermission } from '@/types/owners';
import { useSession } from '@/auth/auth-client';
import { SinglePageView, type SinglePageFormat } from '@/common/SinglePageView';
import { SqlEditor } from '@/common/SqlBlock';
import { Toast } from '@/common/Toast';
import type { ColumnAttribute, SqlAttribute, Term, TermDetail } from '@/types/terms';
import { termHref } from '@/lib/termHref';

const termFocusTrail = ({
	focusId,
	sqlAttrId,
	colAttrId,
	termName,
	sqlAttrName,
	colAttrName,
}: {
	focusId: string | null;
	sqlAttrId: string | null;
	colAttrId: string | null;
	termName?: string;
	sqlAttrName?: string;
	colAttrName?: string;
}): BreadcrumbItem[] => {
	if (focusId == null) return [];

	const term: BreadcrumbItem = {
		label: termName ?? focusId,
		href: termHref(focusId),
	};
	if (sqlAttrId != null) return [term, { label: sqlAttrName ?? sqlAttrId }];
	if (colAttrId != null) return [term, { label: colAttrName ?? colAttrId }];
	return [term];
};

type SqlAttributeDeleteTarget = {
	id: string;
	name: string;
};

const FIELD_INPUT_CLASSNAME =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-body outline-none transition-colors placeholder:text-secondary focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500';

const FIELD_LABEL_CLASSNAME = 'mb-1.5 block text-sm font-medium text-heading dark:text-zinc-100';

/**
 * The selected term's detail — description, certification, column and SQL
 * attributes, tags, owner — for the Exploration page's side panel.
 *
 * The term is the one selected in the graph (`semanticId`); `sqlAttr` or
 * `colAttr` open one of its attributes in the same panel instead.
 */
export const TermsView = () => {
	const router = useRouter();
	// The tag picker's options come from the shared cache rather than a read of
	// this page's own, so opening a term after a column attribute doesn't ask
	// for the vocabulary twice.
	const queryClient = useQueryClient();
	const { data: session } = useSession();
	const { data: owners = [] } = useQuery(ownerQueries.all());
	const searchParams = useSearchParams();
	const focusId = searchParams.get('semanticId');
	const sqlAttrId = searchParams.get('sqlAttr');
	const colAttrId = searchParams.get('colAttr');

	const [sqlAttrs, setSqlAttrs] = useState<SqlAttribute[]>([]);
	const [columnAttrs, setColumnAttrs] = useState<ColumnAttribute[]>([]);
	/**
	 * The focused term as the single-term endpoint returned it. A term opened by
	 * link isn't necessarily on the pages the list has loaded, so the detail
	 * fetch below is what the header and the edit form fall back to.
	 */
	const [focusedTermDetail, setFocusedTermDetail] = useState<TermDetail | null>(null);
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
	// Every write that saves on its own rather than through the Save toolbar --
	// the certification dropdowns -- reports its failure here.
	const [writeError, setWriteError] = useState<string | null>(null);

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
		setWriteError(null);
		// Leaving the term drops the detail copy; moving between terms keeps the
		// old one on screen until the new term's fetch lands.
		if (focusId == null) setFocusedTermDetail(null);
	}
	const [sqlEditModalOpen, setSqlEditModalOpen] = useState(false);
	const [sqlEditValue, setSqlEditValue] = useState('');
	const [sqlEditOriginalValue, setSqlEditOriginalValue] = useState('');
	const [sqlEditValidating, setSqlEditValidating] = useState(false);
	const [sqlEditValidationMessage, setSqlEditValidationMessage] = useState<string | null>(null);
	const [sqlEditValidated, setSqlEditValidated] = useState(false);
	const [sqlEditSubmitting, setSqlEditSubmitting] = useState(false);
	const [sqlEditError, setSqlEditError] = useState<string | null>(null);

	// Attribute deep-links (`?focus=&colAttr=` / `?sqlAttr=`) skip the term
	// single-page fetch, so the header would otherwise fall back to the term
	// id. Load the term whenever an attribute page is open.
	useEffect(() => {
		if (focusId == null) return;
		if (colAttrId == null && sqlAttrId == null) return;

		const termId = focusId;
		let cancelled = false;
		void termsApi.get(termId).then((res) => {
			if (cancelled || res.error || res.data == null) return;
			setFocusedTermDetail(res.data);
		});
		return () => {
			cancelled = true;
		};
	}, [focusId, colAttrId, sqlAttrId]);

	/** Applies a saved change to the term the header and the edit form read. */
	const patchTerm = useCallback((termId: string, patch: Partial<Term>) => {
		setFocusedTermDetail((prev) => (prev?.id === termId ? { ...prev, ...patch } : prev));
	}, []);

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
			router.push(termHref(focusId));
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
		const nextTagIds = stagedTagIds(payload[TAGS_SECTION_ID]);
		if (Object.keys(patch).length === 0 && nextTagIds == null) {
			return { error: false };
		}

		if (Object.keys(patch).length > 0) {
			const res = await sqlAttributesApi.patch(focusedSqlAttr.id, patch);
			if (res.error) {
				return { error: true, message: res.message ?? 'Failed to update SQL attribute' };
			}
			setSqlAttrs((prev) =>
				prev.map((attr) => (attr.id === focusedSqlAttr.id ? res.data : attr)),
			);
		}

		if (nextTagIds != null) {
			const { error: tagError, tags } = await syncTags({
				type: TagItemType.SqlAttribute,
				itemId: focusedSqlAttr.id,
				current: focusedSqlAttr.tags ?? [],
				nextIds: nextTagIds,
			});
			// What the attribute carries now, which is what the *next* save has
			// to diff against. Applied here rather than left to the refetch
			// below, which lands a round trip later: re-opening Edit before it
			// does would otherwise measure the change against the tags this
			// save already replaced. On a partial failure this is whatever
			// landed, so it holds either way.
			setSqlAttrs((prev) =>
				prev.map((attr) => (attr.id === focusedSqlAttr.id ? { ...attr, tags } : attr)),
			);
			if (tagError != null) {
				setSqlAttrsEpoch((prev) => prev + 1);
				return { error: true, message: tagError };
			}
		}

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

		const patch: {
			name?: string;
			description?: string;
			sample_values?: string[];
		} = {};
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
		const nextTagIds = stagedTagIds(payload[TAGS_SECTION_ID]);
		if (Object.keys(patch).length === 0 && nextTagIds == null) {
			return { error: false };
		}

		if (Object.keys(patch).length > 0) {
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
		}

		if (nextTagIds != null) {
			const { error: tagError, tags } = await syncTags({
				type: TagItemType.ColumnAttribute,
				itemId: focusedColAttr.id,
				current: focusedColAttr.tags ?? [],
				nextIds: nextTagIds,
			});
			// As on the SQL attribute above: the baseline the next save diffs
			// against cannot wait for the refetch.
			setColumnAttrs((prev) =>
				prev.map((attr) => (attr.id === focusedColAttr.id ? { ...attr, tags } : attr)),
			);
			if (tagError != null) {
				setColumnAttrsEpoch((prev) => prev + 1);
				return { error: true, message: tagError };
			}
		}

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

		const patch: { name?: string; description?: string; owner_id?: string | null } = {};
		if (name !== focusedTerm.name.trim()) {
			patch.name = name;
		}
		if (description !== (focusedTerm.description ?? '')) {
			patch.description = description;
		}
		const nextOwner = changedOwnerId(payload.owner_id, focusedTerm.owner_id);
		if (nextOwner !== undefined) patch.owner_id = nextOwner;
		const nextTagIds = stagedTagIds(payload[TAGS_SECTION_ID]);
		if (Object.keys(patch).length === 0 && nextTagIds == null) {
			return { error: false };
		}

		if (Object.keys(patch).length > 0) {
			const res = await termsApi.update(focusId, patch);
			if (res.error) {
				return { error: true, message: res.message ?? 'Failed to update term' };
			}
			patchTerm(focusId, {
				name: res.data.name,
				description: res.data.description,
				owner_id: res.data.owner_id,
			});
		}

		if (nextTagIds != null) {
			const { error: tagError, tags } = await syncTags({
				type: TagItemType.Term,
				itemId: focusId,
				current: (focusedTermDetail?.id === focusId ? focusedTermDetail.tags : null) ?? [],
				nextIds: nextTagIds,
			});
			// The baseline the next save diffs against, and what the list card
			// draws. `patchTerm` writes both copies, so going back to the list
			// shows the chips this save just landed.
			patchTerm(focusId, { tags });
			if (tagError != null) {
				// Some of the writes may have landed, and the text edit — if
				// there was one — is already saved, so the refetch has to
				// happen even on this failure.
				setSqlAttrsEpoch((prev) => prev + 1);
				return { error: true, message: tagError };
			}
		}

		setSqlAttrsEpoch((prev) => prev + 1);
		return { error: false };
	};

	const applyTermCertification = (termId: string, certification: CertificationStatus | null) => {
		if (certification == null) return;
		patchTerm(termId, { certification });
	};

	// Certification changes save immediately (independent of the text Save
	// toolbar). `id` is `'name'` or `'description'`; both map to the matching
	// `*_certified` flag on the backend PATCH endpoints.
	const handleTermCertificationChange = async (id: string, certified: boolean) => {
		if (focusId == null) return;
		const payload =
			id === 'name' ? { name_certified: certified } : { description_certified: certified };
		const res = await termsApi.update(focusId, payload);
		if (res.error) {
			setWriteError(res.message ?? 'Failed to update certification');
			return;
		}
		setWriteError(null);
		patchTerm(focusId, {
			name_certified: res.data.name_certified,
			description_certified: res.data.description_certified,
			certification: res.data.certification,
		});
		setSqlAttrsEpoch((prev) => prev + 1);
	};

	// Attributes carry a single top-level certification flag, so the `id`
	// argument (name/description) from the composer control is ignored.
	const handleColumnAttrCertificationChange = async (_id: string, certified: boolean) => {
		if (focusId == null || focusedColAttr == null) return;
		const res = await termsApi.updateColumnAttribute(focusId, focusedColAttr.id, {
			certified,
		});
		if (res.error) {
			setWriteError(res.message ?? 'Failed to update certification');
			return;
		}
		setWriteError(null);
		setColumnAttrs((prev) =>
			prev.map((attr) =>
				attr.id === focusedColAttr.id
					? {
							...attr,
							certified: res.data.certified,
						}
					: attr,
			),
		);
		applyTermCertification(focusId, res.term_certification);
		setColumnAttrsEpoch((prev) => prev + 1);
	};

	const handleSqlAttrCertificationChange = async (_id: string, certified: boolean) => {
		if (focusedSqlAttr == null) return;
		const res = await sqlAttributesApi.patch(focusedSqlAttr.id, { certified });
		if (res.error) {
			setWriteError(res.message ?? 'Failed to update certification');
			return;
		}
		setWriteError(null);
		setSqlAttrs((prev) =>
			prev.map((attr) => (attr.id === focusedSqlAttr.id ? res.data : attr)),
		);
		applyTermCertification(res.data.term_id, res.term_certification);
		setSqlAttrsEpoch((prev) => prev + 1);
	};

	// Certification dropdowns in the attribute tables save immediately. Each
	// one is offered only when the caller may edit that attribute, and the
	// epoch bump refetches the term page (`treeDataEpoch={sqlAttrsEpoch}`).
	const handleTermTableCertificationChange = async (
		sectionId: string,
		rowId: string,
		certified: boolean,
	) => {
		if (sectionId === 'column_attributes') {
			if (focusId == null) return;
			const res = await termsApi.updateColumnAttribute(focusId, rowId, { certified });
			if (res.error) {
				setWriteError(res.message ?? 'Failed to update certification');
				return;
			}
			setWriteError(null);
			setColumnAttrs((prev) =>
				prev.map((attr) =>
					attr.id === rowId ? { ...attr, certified: res.data.certified } : attr,
				),
			);
			applyTermCertification(focusId, res.term_certification);
			setSqlAttrsEpoch((prev) => prev + 1);
		} else if (sectionId === 'sql_attributes') {
			const res = await sqlAttributesApi.patch(rowId, { certified });
			if (res.error) {
				setWriteError(res.message ?? 'Failed to update certification');
				return;
			}
			setWriteError(null);
			setSqlAttrs((prev) => prev.map((attr) => (attr.id === rowId ? res.data : attr)));
			applyTermCertification(res.data.term_id, res.term_certification);
			setSqlAttrsEpoch((prev) => prev + 1);
		}
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
				router.push(termHref(focusId, { sqlAttr: rowId }));
			} else if (sectionId === 'column_attributes') {
				router.push(termHref(focusId, { colAttr: rowId }));
			}
		},
		[focusId, router],
	);

	const getColumnAttributeSinglePage = useCallback(
		async (attrId: string): Promise<SinglePageFormat> => {
			if (focusId == null) {
				return {
					sections: [],
					header: { header: { title: 'Column Attribute Not Found' } },
				};
			}
			// TODO: viewer zone-scoping handled in a separate PR — for now the
			// client treats a viewer the same as an admin here (no zone fetch,
			// userZoneIds = null → all zones accessible).
			const [res, tagOptions, ownerDirectory, termRes] = await Promise.all([
				termsApi.getColumnAttributes(focusId),
				fetchTagOptions(queryClient),
				fetchOwnerDirectory(queryClient),
				termsApi.get(focusId),
			]);
			const attrs = res.error ? [] : (res.data ?? []);
			const attr = attrs.find((a) => a.id === attrId);
			if (attr == null) {
				return {
					sections: [],
					header: { header: { title: 'Column Attribute Not Found' } },
				};
			}
			setColumnAttrs(attrs);

			const primaryColumn = attr.primary_column ?? null;
			const referencedColumns = attr.referenced_columns ?? [];
			const userZoneIds: string[] | null = null;
			// Sample values live on the owning Column, so its declared SQL type is
			// what decides whether they can be edited here.
			const samplesEditable = sampleValuesEditable(attr.datatype);

			return {
				header: {
					header: {
						title: attr.name,
						icon: IconName.Key,
						titleEditable: true,
						certification: { certified: attr.certified, showLabel: true },
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
						editable: samplesEditable,
						hint: samplesEditable ? undefined : sampleValuesReadOnlyHint(attr.datatype),
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
				leftPanel: entitySidePanel({
					owner: entityOwnerSection(
						termRes.error ? null : termRes.data.owner_id,
						ownerDirectory,
						OwnerPermission.catalog,
						false,
						'Term Owner',
					),
					tags: entityTagsSection(attr.tags, tagOptions),
					activity: attr,
				}),
			};
		},
		[focusId, queryClient],
	);

	const getSqlAttributeSinglePage = useCallback(
		async (attrId: string): Promise<SinglePageFormat> => {
			const [res, tagOptions, ownerDirectory] = await Promise.all([
				sqlAttributesApi.get(attrId),
				fetchTagOptions(queryClient),
				fetchOwnerDirectory(queryClient),
			]);
			if (res.error || !res.data) {
				return {
					sections: [],
					header: { header: { title: 'SQL Attribute Not Found' } },
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

			const termRes = await termsApi.get(attr.term_id);

			return {
				header: {
					header: {
						title: attr.name,
						icon: IconName.ChartLine,
						titleEditable: true,
						certification: { certified: attr.certified, showLabel: true },
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
				leftPanel: entitySidePanel({
					owner: entityOwnerSection(
						termRes.error ? null : termRes.data.owner_id,
						ownerDirectory,
						OwnerPermission.catalog,
						false,
						'Term Owner',
					),
					tags: entityTagsSection(attr.tags, tagOptions),
					activity: attr,
					showCreatedBy: true,
				}),
			};
		},
		[queryClient],
	);

	const getSinglePage = useCallback(
		async (termId: string): Promise<SinglePageFormat> => {
			const [res, attrsRes, sqlAttrsRes, tagOptions, ownerDirectory] = await Promise.all([
				termsApi.get(termId),
				termsApi.getColumnAttributes(termId),
				termsApi.getSqlAttributes(termId),
				fetchTagOptions(queryClient),
				fetchOwnerDirectory(queryClient),
			]);
			if (res.error || !res.data) {
				return {
					sections: [],
					header: { header: { title: 'Term Not Found' } },
				};
			}
			const term = res.data;
			const termAttrs = attrsRes?.data ?? [];
			const termSqlAttrs = sqlAttrsRes?.data ?? [];
			// The header reads this when the term isn't on a page the list loaded,
			// which is the normal case for a link straight into a term.
			setFocusedTermDetail(term);
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
						icon: IconName.Terms,
						titleEditable: true,
						certification: { certified: term.name_certified },
						// Same server-rolled-up value the list card shows, so the two
						// can't disagree.
						status: term.certification,
					},
				},
				sections: [
					{
						type: ComposerSectionKind.TEXT_CARD,
						id: 'description',
						title: 'Description',
						body: term.description ?? '',
						editable: true,
						certification: { certified: term.description_certified },
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
						})),
					},
					{
						type: ComposerSectionKind.DATA_TABLE,
						id: 'column_attributes',
						title: 'Column Attributes',
						rowIdKey: 'id',
						layout: 'fixed',
						columns: [
							{ key: 'name', label: 'Attribute Name', width: 'w-1/3' },
							{ key: 'description', label: 'Description', truncate: true },
							{
								key: 'certification',
								label: 'Certification',
								type: ComposerColumnType.CERTIFICATION,
								align: 'center',
								width: 'w-44',
							},
						],
						rows: termAttrs.map((attr) => ({
							id: attr.id,
							name: attr.name,
							description: attr.description ?? '',
							certification: attributeStatus(attr),
						})),
					},
					{
						type: ComposerSectionKind.DATA_TABLE,
						id: 'sql_attributes',
						title: 'SQL Attributes',
						rowIdKey: 'id',
						layout: 'fixed',
						columns: [
							{ key: 'name', label: 'Attribute Name', width: 'w-1/3' },
							{ key: 'description', label: 'Description', truncate: true },
							{
								key: 'certification',
								label: 'Certification',
								type: ComposerColumnType.CERTIFICATION,
								align: 'center',
								width: 'w-44',
							},
						],
						rows: termSqlAttrs.map((attr) => ({
							id: attr.id,
							name: attr.name,
							description: attr.description ?? '',
							certification: attributeStatus(attr),
						})),
						emptyMessage: 'SQL Attribute Does Not Exist',
					},
				],
				leftPanel: entitySidePanel({
					owner: entityOwnerSection(
						term.owner_id,
						ownerDirectory,
						OwnerPermission.catalog,
						true,
						'Term Owner',
					),
					tags: entityTagsSection(term.tags, tagOptions),
					activity: term,
				}),
			};
		},
		[queryClient],
	);

	const focusedTerm =
		focusId != null && focusedTermDetail?.id === focusId ? focusedTermDetail : null;
	const focusedSqlAttr =
		sqlAttrId != null ? (sqlAttrs.find((attr) => attr.id === sqlAttrId) ?? null) : null;
	const focusedColAttr =
		colAttrId != null ? (columnAttrs.find((attr) => attr.id === colAttrId) ?? null) : null;

	useBreadcrumbTrail(
		termFocusTrail({
			focusId,
			sqlAttrId,
			colAttrId,
			termName: focusedTerm?.name ?? focusedSqlAttr?.term_name ?? focusedColAttr?.term_name,
			sqlAttrName: focusedSqlAttr?.name,
			colAttrName: focusedColAttr?.name,
		}),
	);

	const actor = session?.user ? { id: session.user.id, role: session.user.role ?? null } : null;
	const canEditCatalog = canEditEntity({
		user: actor,
		permission: { catalog: ['edit'] },
	});
	const canEditFocusedTerm = canEditEntity({
		user: actor,
		permission: { catalog: ['edit'] },
		ownerId: focusedTerm?.owner_id,
		owners,
	});
	// A SQL attribute shows the term's owner and does not grant edit through an
	// owner of its own. The button and the certification dropdown follow the
	// account role, the same way a column attribute does.
	const canEditFocusedSqlAttr = canEditCatalog;
	const canEditFocusedColAttr = canEditCatalog;
	const canCertifyAttributeRow = (): boolean => canEditCatalog;

	if (focusId != null && sqlAttrId != null) {
		const sqlAttrTitle = focusedSqlAttr?.name ?? sqlAttrId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						compact
						key={sqlAttrId}
						dataId={sqlAttrId}
						getSinglePage={getSqlAttributeSinglePage}
						treeDataEpoch={sqlAttrsEpoch}
						headerActions={
							<>
								{sqlAttrEditing || !canEditFocusedSqlAttr ? null : (
									<>
										<Button
											theme={ButtonTheme.Primary}
											size={Size.REGULAR}
											type="button"
											onClick={() => {
												setSqlAttrEditError(null);
												setSqlAttrEditing(true);
											}}
											aria-label={`Edit ${sqlAttrTitle}`}
											title="Edit"
											iconPosition="left"
										>
											<Icon name={IconName.Pencil} className="h-3.5 w-3.5" />
											Edit
										</Button>
										<Button
											theme={ButtonTheme.DangerSubtle}
											size={Size.REGULAR}
											type="button"
											onClick={() => {
												setDeletingSqlAttr({
													id: sqlAttrId,
													name: sqlAttrTitle,
												});
												setDeleteSqlAttrError(null);
											}}
											aria-label={`Delete ${sqlAttrTitle}`}
											title="Delete"
											iconPosition="left"
										>
											<Icon name={IconName.Trash} className="h-3.5 w-3.5" />
											Delete
										</Button>
									</>
								)}
							</>
						}
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
						onCertificationChange={handleSqlAttrCertificationChange}
					/>
				</main>
				<Toast
					open={sqlAttrEditError != null || writeError != null}
					message={sqlAttrEditError ?? writeError ?? ''}
					variant={ToastVariant.Error}
					onClose={() => {
						setSqlAttrEditError(null);
						setWriteError(null);
					}}
				/>
				<ConfirmModal
					open={deletingSqlAttr !== null}
					onCancel={handleDeleteSqlAttrClose}
					onConfirm={handleDeleteSqlAttrConfirm}
					title="Delete SQL Attribute"
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
						<p className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-body dark:border-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-300">
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
		const colAttrTitle = focusedColAttr?.name ?? colAttrId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						compact
						key={colAttrId}
						dataId={colAttrId}
						getSinglePage={getColumnAttributeSinglePage}
						treeDataEpoch={columnAttrsEpoch}
						headerActions={
							<>
								{columnAttrEditing || !canEditFocusedColAttr ? null : (
									<Button
										theme={ButtonTheme.Primary}
										size={Size.REGULAR}
										type="button"
										onClick={() => {
											setColumnAttrEditError(null);
											setColumnAttrEditing(true);
										}}
										aria-label={`Edit ${colAttrTitle}`}
										title="Edit"
										iconPosition="left"
									>
										<Icon name={IconName.Pencil} className="h-3.5 w-3.5" />
										Edit
									</Button>
								)}
							</>
						}
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
						onCertificationChange={handleColumnAttrCertificationChange}
					/>
				</main>
				<Toast
					open={columnAttrEditError != null || writeError != null}
					message={columnAttrEditError ?? writeError ?? ''}
					variant={ToastVariant.Error}
					onClose={() => {
						setColumnAttrEditError(null);
						setWriteError(null);
					}}
				/>
			</div>
		);
	}

	if (focusId != null) {
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						compact
						key={focusId}
						dataId={focusId}
						getSinglePage={getSinglePage}
						treeDataEpoch={sqlAttrsEpoch}
						headerActions={
							<>
								{termEditing || !canEditFocusedTerm ? null : (
									<Button
										theme={ButtonTheme.Primary}
										size={Size.REGULAR}
										type="button"
										onClick={() => {
											setTermEditing(true);
										}}
										iconPosition="left"
									>
										<Icon name={IconName.Pencil} className="h-3.5 w-3.5" />
										Edit
									</Button>
								)}
								{canEditCatalog ? (
									<Button
										theme={ButtonTheme.Primary}
										size={Size.REGULAR}
										type="button"
										onClick={() => setCreateSqlAttrModalOpen(true)}
										iconPosition="left"
										shadow
									>
										<Icon name={IconName.Plus} className="h-4 w-4" />
										SQL Attribute
									</Button>
								) : null}
							</>
						}
						isEditing={termEditing}
						onPatchEdits={handleTermEditSave}
						onSave={() => {
							setTermEditing(false);
						}}
						onCancel={() => {
							setTermEditing(false);
						}}
						onDataTableRowClick={handleDataTableRowClick}
						onCertificationChange={handleTermCertificationChange}
						onDataTableCertificationChange={handleTermTableCertificationChange}
						canCertifyDataTableRow={canCertifyAttributeRow}
					/>
				</main>
				<Toast
					open={writeError != null}
					message={writeError ?? ''}
					variant={ToastVariant.Error}
					onClose={() => setWriteError(null)}
				/>
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
						<p className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-body dark:border-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-300">
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

	return null;
};
