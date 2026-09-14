// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

import { Placeholders } from '@/assets/images/placeholders';
import { Breadcrumbs } from '@/common/Breadcrumbs';
import { Button } from '@/common/Button';
import { EmptyState } from '@/common/EmptyState';
import { Size, ButtonTheme } from '@/enums/button';
import { Icon, IconName } from '@/common/icons';
import { InfiniteScroll } from '@/common/InfiniteScroll';
import { SkeletonCard } from '@/common/Skeleton';
import { ConfirmModal, ModalCreateNewItem } from '@/common/modal';
import { SearchInput } from '@/common/SearchInput';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import { termsApi } from '@/api/terms';
import { sqlAttributesApi } from '@/api/sqlAttributes';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import { DEFAULT_PAGE_SIZE, useInfiniteList } from '@/hooks/useInfiniteList';
import { type ComposerEditValue } from '@/common/SinglePageComposer';
import { Label } from '@/common/Label';
import { CertificationBadge } from '@/common/CertificationBadge';
import { ComposerColumnType, ComposerSectionKind } from '@/enums/datasources';
import { CertificationStatus } from '@/enums/certification';
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
import { SinglePageView, type SinglePageFormat } from '@/common/SinglePageView';
import { SqlEditor } from '@/common/SqlBlock';
import { Toast } from '@/common/Toast';
import type { ColumnAttribute, SqlAttribute, Term, TermCount, TermDetail } from '@/types/terms';

type TermCardProps = {
	term: Term;
	columnAttributeCount: number;
	sqlAttributeCount: number;
	relatedCount: number;
	certificationStatus: CertificationStatus;
	onClick: (term: Term) => void;
};

type SqlAttributeDeleteTarget = {
	id: string;
	name: string;
};

const LOADING_SKELETON_CLASSNAMES = [
	'',
	'',
	'hidden [@media(min-height:760px)]:block',
	'hidden [@media(min-height:960px)]:block',
] as const;

export const TermsLoadingSkeleton = () => (
	<div
		className="flex min-h-[calc(100dvh-11rem)] flex-col gap-4"
		role="status"
		aria-label="Loading terms"
	>
		{LOADING_SKELETON_CLASSNAMES.map((className, index) => (
			<SkeletonCard key={index} className={className} rows={4} />
		))}
	</div>
);

const TermCard = ({
	term,
	columnAttributeCount,
	sqlAttributeCount,
	relatedCount,
	certificationStatus,
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
			<div className="min-w-0 space-y-0.5">
				<Text as="h2" text={term.name} variant={TextVariant.CardTitle} />
				{term.synonyms && term.synonyms.length > 0 && (
					<Text as="p" text={term.synonyms.join(', ')} variant={TextVariant.Caption} />
				)}
			</div>
			<CertificationBadge status={certificationStatus} />
		</div>

		<div className="mt-2">
			{term.description != null && term.description.trim() !== '' ? (
				<Text as="p" text={term.description} lines={3} variant={TextVariant.Body} />
			) : (
				<p className="text-sm italic text-zinc-400 dark:text-zinc-500">
					No Description Available
				</p>
			)}
		</div>

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

/**
 * Per-card badge numbers, keyed by term id. The list endpoint sends them
 * alongside each page and only for that page's terms, so they accumulate as
 * pages load rather than replacing one another.
 */
type TermBadgeCounts = {
	columnAttributes: Map<string, number>;
	sqlAttributes: Map<string, number>;
	related: Map<string, number>;
};

const EMPTY_BADGE_COUNTS: TermBadgeCounts = {
	columnAttributes: new Map(),
	sqlAttributes: new Map(),
	related: new Map(),
};

const withCounts = (held: Map<string, number>, rows: TermCount[] | undefined) => {
	const merged = new Map(held);
	for (const { term_id: termId, count } of rows ?? []) merged.set(termId, count);
	return merged;
};

const FIELD_INPUT_CLASSNAME =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500';

const FIELD_LABEL_CLASSNAME = 'mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100';

export const TermsView = () => {
	const router = useRouter();
	const searchParams = useSearchParams();
	const focusId = searchParams.get('focus');
	const sqlAttrId = searchParams.get('sqlAttr');
	const colAttrId = searchParams.get('colAttr');

	const [sqlAttrs, setSqlAttrs] = useState<SqlAttribute[]>([]);
	const [columnAttrs, setColumnAttrs] = useState<ColumnAttribute[]>([]);
	const [badgeCounts, setBadgeCounts] = useState<TermBadgeCounts>(EMPTY_BADGE_COUNTS);
	const [hasLoadedTerms, setHasLoadedTerms] = useState(false);
	const [searchQuery, setSearchQuery] = useState('');
	const debouncedSearchQuery = useDebouncedValue(searchQuery.trim(), 1000);
	// Read inside `fetchTermsPage` to tell a response for a query that has
	// since been superseded apart from one that still describes what's on
	// screen — `useInfiniteList` already discards a stale `items`/`total`
	// response by its own request id, but the badge counts below are a side
	// effect of that same callback and need the same guard applied by hand.
	const debouncedSearchQueryRef = useRef(debouncedSearchQuery);
	useEffect(() => {
		debouncedSearchQueryRef.current = debouncedSearchQuery;
	}, [debouncedSearchQuery]);
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

	const fetchTermsPage = useCallback(
		async (skip: number, limit: number) => {
			const query = debouncedSearchQuery;
			const res = await termsApi.list({
				...(query ? { query } : {}),
				skip,
				limit,
			});
			if (res.error) return { error: res.message ?? 'Failed to load terms' };

			// A newer search may have started while this request was in flight.
			// `useInfiniteList` already drops a late `items`/`total` response by
			// request id; the badge counts have no such guard of their own, so a
			// stale response merged in here would show counts for a search that
			// is no longer on screen.
			if (debouncedSearchQueryRef.current === query) {
				setBadgeCounts((held) => {
					const base = skip === 0 ? EMPTY_BADGE_COUNTS : held;
					return {
						columnAttributes: withCounts(
							base.columnAttributes,
							res.column_attribute_counts,
						),
						sqlAttributes: withCounts(base.sqlAttributes, res.sql_attribute_counts),
						related: withCounts(base.related, res.related_counts),
					};
				});
			}
			setHasLoadedTerms(true);
			return { items: res.terms ?? [], total: res.total ?? 0 };
		},
		[debouncedSearchQuery],
	);

	const {
		items: terms,
		setItems: setTerms,
		isLoading: loading,
		isLoadingMore: loadingMoreTerms,
		error,
		hasMore: hasMoreTerms,
		loadMore: loadMoreTerms,
	} = useInfiniteList(fetchTermsPage, {
		pageSize: DEFAULT_PAGE_SIZE,
		itemKey: (term) => term.id,
	});

	/**
	 * Applies a saved change to both copies of a term: the card in the list
	 * behind this page, and the detail copy the header falls back to when the
	 * term isn't on a page the list has loaded.
	 */
	const patchTerm = useCallback(
		(termId: string, patch: Partial<Term>) => {
			setTerms((prev) =>
				prev.map((term) => (term.id === termId ? { ...term, ...patch } : term)),
			);
			setFocusedTermDetail((prev) => (prev?.id === termId ? { ...prev, ...patch } : prev));
		},
		[setTerms],
	);

	const handleCardClick = useCallback(
		(term: Term) => {
			router.push(`/terms?focus=${encodeURIComponent(term.id)}`);
		},
		[router],
	);

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

		const patch: { name?: string; description?: string } = {};
		if (name !== focusedTerm.name.trim()) {
			patch.name = name;
		}
		if (description !== (focusedTerm.description ?? '')) {
			patch.description = description;
		}
		const nextTagIds = stagedTagIds(payload[TAGS_SECTION_ID]);
		if (Object.keys(patch).length === 0 && nextTagIds == null) {
			return { error: false };
		}

		if (Object.keys(patch).length > 0) {
			const res = await termsApi.update(focusId, patch);
			if (res.error) {
				return { error: true, message: res.message ?? 'Failed to update term' };
			}
			patchTerm(focusId, { name: res.data.name, description: res.data.description });
		}

		if (nextTagIds != null) {
			const { error: tagError, tags } = await syncTags({
				type: TagItemType.Term,
				itemId: focusId,
				current: (focusedTermDetail?.id === focusId ? focusedTermDetail.tags : null) ?? [],
				nextIds: nextTagIds,
			});
			// The baseline the next save diffs against, for the reason on the
			// attribute handlers above. Not through `patchTerm`, which patches
			// both copies of a `Term`: only the detail read resolves tags, so
			// the list card has no `tags` to bring up to date.
			setFocusedTermDetail((prev) => (prev?.id === focusId ? { ...prev, tags } : prev));
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

	// Certification dropdowns inside the term page's Column/SQL attribute
	// tables save immediately and update the relevant attribute cache so the
	// table cell reflects the change. The epoch bump refetches the term page
	// (which passes `treeDataEpoch={sqlAttrsEpoch}`) for both branches.
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
					header: { header: { title: 'Column Attribute not found' } },
				};
			}
			// TODO: viewer zone-scoping handled in a separate PR — for now the
			// client treats a viewer the same as an admin here (no zone fetch,
			// userZoneIds = null → all zones accessible).
			const [res, tagOptions] = await Promise.all([
				termsApi.getColumnAttributes(focusId),
				fetchTagOptions(),
			]);
			const attrs = res.error ? [] : (res.data ?? []);
			const attr = attrs.find((a) => a.id === attrId);
			if (attr == null) {
				return {
					sections: [],
					header: { header: { title: 'Column Attribute not found' } },
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
					entityTagsSection(attr.tags, tagOptions),
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
			const [res, tagOptions] = await Promise.all([
				sqlAttributesApi.get(attrId),
				fetchTagOptions(),
			]);
			if (res.error || !res.data) {
				return {
					sections: [],
					header: { header: { title: 'SQL Attribute not found' } },
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
					entityTagsSection(attr.tags, tagOptions),
				],
			};
		},
		[],
	);

	const getSinglePage = useCallback(async (termId: string): Promise<SinglePageFormat> => {
		const [res, attrsRes, sqlAttrsRes, tagOptions] = await Promise.all([
			termsApi.get(termId),
			termsApi.getColumnAttributes(termId),
			termsApi.getSqlAttributes(termId),
			fetchTagOptions(),
		]);
		if (res.error || !res.data) {
			return {
				sections: [],
				header: { header: { title: 'Term not found' } },
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
					titleEditable: true,
					certification: { certified: term.name_certified },
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
				entityTagsSection(term.tags, tagOptions),
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
					emptyMessage: 'SQL attribute does not exist',
				},
			],
		};
	}, []);

	const listedTerm = focusId != null ? (terms.find((t) => t.id === focusId) ?? null) : null;
	const focusedTerm =
		listedTerm ??
		(focusId != null && focusedTermDetail?.id === focusId ? focusedTermDetail : null);
	const focusedSqlAttr =
		sqlAttrId != null ? (sqlAttrs.find((attr) => attr.id === sqlAttrId) ?? null) : null;
	const focusedColAttr =
		colAttrId != null ? (columnAttrs.find((attr) => attr.id === colAttrId) ?? null) : null;

	// Same server-rolled-up value the list card shows, so the two can't disagree.
	const termCertificationStatus = focusedTerm?.certification ?? CertificationStatus.Pending;

	if (focusId != null && sqlAttrId != null) {
		const termTitle = focusedTerm?.name ?? focusedSqlAttr?.term_name ?? focusId;
		const sqlAttrTitle = focusedSqlAttr?.name ?? sqlAttrId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
					<Breadcrumbs
						items={[
							{ label: 'Terms', href: '/terms' },
							{
								label: termTitle,
								href: `/terms?focus=${encodeURIComponent(focusId)}`,
							},
							{ label: sqlAttrTitle },
						]}
					/>
					<div className="ml-auto flex shrink-0 items-center gap-1">
						{sqlAttrEditing ? null : (
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
										setDeletingSqlAttr({ id: sqlAttrId, name: sqlAttrTitle });
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
					</div>
				</header>
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						key={sqlAttrId}
						dataId={sqlAttrId}
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
		const termTitle = focusedTerm?.name ?? focusedColAttr?.term_name ?? focusId;
		const colAttrTitle = focusedColAttr?.name ?? colAttrId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
					<Breadcrumbs
						items={[
							{ label: 'Terms', href: '/terms' },
							{
								label: termTitle,
								href: `/terms?focus=${encodeURIComponent(focusId)}`,
							},
							{ label: colAttrTitle },
						]}
					/>
					<div className="ml-auto flex shrink-0 items-center gap-1">
						{columnAttrEditing ? null : (
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
					</div>
				</header>
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						key={colAttrId}
						dataId={colAttrId}
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
		const termTitle = focusedTerm?.name ?? focusId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
					<Breadcrumbs
						items={[{ label: 'Terms', href: '/terms' }, { label: termTitle }]}
					/>
					<CertificationBadge status={termCertificationStatus} />
					<div className="ml-auto flex items-center gap-2">
						{termEditing ? null : (
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
						<Button
							theme={ButtonTheme.Primary}
							size={Size.REGULAR}
							type="button"
							onClick={() => setCreateSqlAttrModalOpen(true)}
							iconPosition="left"
							shadow
						>
							<Icon name={IconName.Plus} className="h-4 w-4" />
							Create new sql attribute
						</Button>
					</div>
				</header>
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						key={focusId}
						dataId={focusId}
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
						onCertificationChange={handleTermCertificationChange}
						onDataTableCertificationChange={handleTermTableCertificationChange}
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

			<InfiniteScroll
				className="flex-1 px-6 py-6"
				onLoadMore={loadMoreTerms}
				isLoading={loadingMoreTerms}
				hasMore={hasMoreTerms}
				// Only a failed *first* page is shown below — a failed later page
				// keeps the cards already loaded and gets its own retry control
				// instead (see `error` on `InfiniteScroll`).
				error={terms.length > 0 ? error : null}
			>
				{hasLoadedTerms && (
					<SearchInput
						value={searchQuery}
						onChange={setSearchQuery}
						placeholder="Search terms…"
						aria-label="Search terms"
						className="mb-6 w-full"
					/>
				)}

				{loading && <TermsLoadingSkeleton />}

				{!loading && error != null && terms.length === 0 && (
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
					<EmptyState
						illustration={<Placeholders.NoTerms />}
						title={
							debouncedSearchQuery
								? 'No Terms Match Your Search'
								: 'No Terms Created Yet'
						}
					/>
				)}

				{!loading && terms.length > 0 && (
					<ul className="flex flex-col gap-4">
						{terms.map((term) => (
							<TermCard
								key={term.id}
								term={term}
								columnAttributeCount={
									badgeCounts.columnAttributes.get(term.id) ?? 0
								}
								sqlAttributeCount={badgeCounts.sqlAttributes.get(term.id) ?? 0}
								relatedCount={badgeCounts.related.get(term.id) ?? 0}
								certificationStatus={term.certification}
								onClick={handleCardClick}
							/>
						))}
					</ul>
				)}
			</InfiniteScroll>
		</div>
	);
};
